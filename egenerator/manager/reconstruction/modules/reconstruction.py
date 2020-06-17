import numpy as np
import tensorflow as tf


class Reconstruction:

    def __init__(self, manager, loss_module, function_cache,
                 fit_paramater_list,
                 seed_tensor_name,
                 seed_from_previous_module=False,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',
                 reco_optimizer_interface='scipy',
                 scipy_optimizer_settings={'method': 'BFGS'},
                 tf_optimizer_settings={'method': 'bfgs_minimize',
                                        'x_tolerance': 0.001},
                 ):
        """Initialize reconstruction module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        fit_paramater_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        seed_tensor_name : str
            This defines the seed for the reconstruction. Depending on the
            value of `seed_from_previous_module` this defines one of 2 things:
            `seed_from_previous_module` == True:
                Name of the previous module from which the 'result' field will
                be used as the seed tensor.
            `seed_from_previous_module` == False:
                Name of the seed tensor which defines that the tensor from the
                input data batch that will be used as the seed.
        seed_from_previous_module : bool, optional
            This defines what `seed_tensor_name` refers to.
            `seed_from_previous_module` == True:
                Name of the previous module from which the 'result' field will
                be used as the seed tensor.
            `seed_from_previous_module` == False:
                Name of the seed tensor which defines that the tensor from the
                input data batch that will be used as the seed.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
        reco_optimizer_interface : str, optional
            The reconstruction interface to use. Options are 'scipy' and 'tfp'
            for the scipy minimizer and tensorflow probability minimizers.
        scipy_optimizer_settings : dict, optional
            Settings that will be passed on to the scipy.optmize.minimize
            function.
        tf_optimizer_settings : dict, optional
            Settings that will be passed on to the tensorflow probability
            minimizer.

        Raises
        ------
        ValueError
            Description
        """

        # store settings
        self.manager = manager
        self.fit_paramater_list = fit_paramater_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.seed_from_previous_module = seed_from_previous_module

        if self.seed_from_previous_module:
            self.seed_index = manager.data_handler.tensors.get_index(
                seed_tensor_name)

        # parameter input signature
        param_dtype = getattr(tf, manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_paramater_list, dtype=int)],
            dtype=param_dtype)
        param_signature_full = tf.TensorSpec(
            shape=[None, len(fit_paramater_list)],
            dtype=param_dtype)

        data_batch_signature = []
        for tensor in manager.data_handler.tensors.list:
            if tensor.exists:
                shape = tf.TensorShape(tensor.shape)
            else:
                shape = tf.TensorShape(None)
            data_batch_signature.append(tf.TensorSpec(
                shape=shape,
                dtype=getattr(tf, tensor.dtype)))

        data_batch_signature = tuple(data_batch_signature)

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------

        # Get loss function
        loss_settings = dict(
            input_signature=(param_signature_full, data_batch_signature),
            loss_module=loss_module,
            fit_paramater_list=[True for f in fit_paramater_list],
            minimize_in_trafo_space=False,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )
        self.parameter_loss_function = function_cache.get(
            'parameter_loss_function', loss_settings)

        if self.parameter_loss_function is None:
            self.parameter_loss_function = manager.get_parameter_loss_function(
                **loss_settings)
            function_cache.add(self.parameter_loss_function, loss_settings)

        # Get loss and gradients function
        function_settings = dict(
            input_signature=(
                param_signature, data_batch_signature, param_signature_full),
            loss_module=loss_module,
            fit_paramater_list=fit_paramater_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        loss_and_gradients_function = function_cache.get(
            'loss_and_gradients_function', function_settings)

        if loss_and_gradients_function is None:
            loss_and_gradients_function = \
                manager.get_loss_and_gradients_function(**function_settings)
            function_cache.add(loss_and_gradients_function, function_settings)

        # choose reconstruction method depending on the optimizer interface
        if reco_optimizer_interface.lower() == 'scipy':
            def reconstruction_method(data_batch, seed_tensor):
                return manager.reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor,
                    parameter_tensor_name=parameter_tensor_name,
                    **scipy_optimizer_settings)

        elif reco_optimizer_interface.lower() == 'tfp':
            def reconstruction_method(data_batch, seed_tensor):
                return manager.tf_reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_paramater_list=fit_paramater_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor,
                    parameter_tensor_name=parameter_tensor_name,
                    **tf_optimizer_settings)
        else:
            raise ValueError('Unknown interface {!r}. Options are {!r}'.format(
                reco_optimizer_interface, ['scipy', 'tfp']))

        self.reconstruction_method = reconstruction_method

    def execute(self, data_batch, results):
        """Execute reconstruction for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of tf.Tensors
            A data batch which consists of a tuple of tf.Tensors.
        results : dict
            A dictrionary with the results of previous modules.

        Returns
        -------
        TYPE
            Description
        """

        # get seed: either from seed tensor or from previous results
        if self.seed_from_previous_module:
            seed_tensor = results[self.seed_tensor_name]['result']
        else:
            seed_tensor = data_batch[self.seed_index]

        result_trafo, result_object = self.reconstruction_method(
            data_batch, seed_tensor)

        # invert possible transformation and put full hypothesis together
        result = self.get_reco_result_batch(
            result_trafo=result_trafo,
            data_batch=data_batch,
            seed_tensor=seed_tensor,
            fit_paramater_list=self.fit_paramater_list,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            parameter_tensor_name=self.parameter_tensor_name)

        loss_seed = self.parameter_loss_function(
            seed_tensor, data_batch).numpy()
        loss_reco = self.parameter_loss_function(result, data_batch).numpy()

        result_dict = {
            'result': result,
            'result_trafo': result_trafo,
            'result_object': result_object,
            'loss_seed': loss_seed,
            'loss_reco': loss_reco,
        }
        return result_dict

    def get_reco_result_batch(self, result_trafo,
                              data_batch,
                              seed_tensor,
                              fit_paramater_list,
                              minimize_in_trafo_space,
                              parameter_tensor_name='x_parameters'):
        """Get the reco result batch.

        This inverts a possible transformation if minimize_in_trafo_space is
        True and also puts the full hypothesis back together if only parts
        of it were fitted

        Parameters
        ----------
        result_trafo : TYPE
            Description
        data_batch : TYPE
            Description
        seed_tensor : TYPE
            Description
        fit_paramater_list : TYPE
            Description
        minimize_in_trafo_space : TYPE
            Description
        parameter_tensor_name : str, optional
            Description

        Returns
        -------
        tf.Tensor
            The full result batch.

        """
        if minimize_in_trafo_space:
            cascade_seed_batch_trafo = self.manager.data_trafo.transform(
                        data=seed_tensor,
                        tensor_name=parameter_tensor_name).numpy()
        else:
            cascade_seed_batch_trafo = seed_tensor

        if np.all(fit_paramater_list):
            cascade_reco_batch = result_trafo
        else:
            # get seed parameters
            cascade_reco_batch = []
            result_counter = 0
            for i, fit in enumerate(fit_paramater_list):
                if fit:
                    cascade_reco_batch.append(result_trafo[:, result_counter])
                    result_counter += 1
                else:
                    cascade_reco_batch.append(cascade_seed_batch_trafo[:, i])
            cascade_reco_batch = np.array(cascade_reco_batch).T

        # transform back if minimization was performed in trafo space
        if minimize_in_trafo_space:
            cascade_reco_batch = self.manager.data_trafo.inverse_transform(
                data=cascade_reco_batch,
                tensor_name=parameter_tensor_name)
        return cascade_reco_batch


class SelectBestReconstruction:

    def __init__(self, reco_names):
        """Initialize reconstruction module and setup tensorflow functions.

        Parameters
        ----------
        reco_names : list of str
            A list of reco names from previous reconstruction modules. The
            best reconstruction, e.g. lowest reco loss, will be chosen from
            these reconstructions.

        Raises
        ------
        ValueError
            Description
        """

        # store settings
        self.reco_names = reco_names

    def execute(self, data_batch, results):
        """Execute selection.

        Choose best reconstruction from a list of results

        Parameters
        ----------
        data_batch : tuple of tf.Tensors
            A data batch which consists of a tuple of tf.Tensors.
        results : dict
            A dictrionary with the results of previous modules.

        Returns
        -------
        TYPE
            Description
        """
        min_loss = float('inf')
        min_results = None

        for reco_name in self.reco_names:
            results = results[reco_name]

            # check if it has a lower loss
            if results['loss_reco'] < min_loss:
                min_loss = results['loss_reco']
                min_results = results['loss_reco']

        return min_results

import numpy as np
import tensorflow as tf


class Visualize1DLikelihoodScan:

    def __init__(self, manager, loss_module, function_cache,
                 fit_parameter_list,
                 seed_tensor_name,
                 reco_key,
                 plot_file_template,
                 covariance_key=None,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',):
        """Initialize module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        fit_parameter_list : TYPE
            Description
        seed_tensor_name : TYPE
            Description
        reco_key : TYPE
            Description
        minimize_in_trafo_space : bool, optional
            Description
        parameter_tensor_name : str, optional
            Description

        Raises
        ------
        NotImplementedError
            Description

        """

        # store settings
        self.event_counter = 0
        self.manager = manager
        self.fit_parameter_list = fit_parameter_list
        self.reco_key = reco_key
        self.covariance_key = covariance_key
        self.plot_file_template = plot_file_template
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.param_index = manager.data_handler.tensors.get_index(
            parameter_tensor_name)

        param_dtype = manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype_tf
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_parameter_list, dtype=int)],
            dtype=param_dtype)

        # define data batch tensor specification
        data_batch_signature = manager.data_handler.get_data_set_signature()

        # define function settings
        func_settings = dict(
            input_signature=(param_signature, data_batch_signature),
            loss_module=loss_module,
            fit_parameter_list=fit_parameter_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=seed_tensor_name,
            parameter_tensor_name=parameter_tensor_name,
        )

        # Get parameter loss function
        self.loss_function = function_cache.get(
            'parameter_loss_function', func_settings)

        if self.loss_function is None:
            self.loss_function = manager.get_parameter_loss_function(
                **func_settings)
            function_cache.add(self.loss_function, func_settings)

    def execute(self, data_batch, results, **kwargs):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of array_like
            A batch of data consisting of a tuple of data arrays.
        results : dict
            A dictrionary with the results of previous modules.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        TYPE
            Description
        """
        result_trafo = results[self.reco_key]['result_trafo']
        if self.covariance_key is not None:
            covariances = results[self.covariance_key]

        self.make_1d_llh_scans(
            data_batch=data_batch,
            result_trafo=result_trafo,
            truth_trafo=self.manager.data_trafo.transform(
                data=data_batch[self.param_index].numpy(),
                tensor_name=self.parameter_tensor_name),
            # covariance_list=[cov, cov_min, cov_sand, cov_sand_fit],
            # covariance_names=['cov', 'cov_min', 'cov_sand',
            #                   'cov_sand_fit'],
            loss_function=self.loss_function,
            plot_file=self.plot_file_template,
            event_counter=self.event_counter,
            parameter_tensor_name=self.parameter_tensor_name)

        self.event_counter += 1
        return None

    def make_1d_llh_scans(self, data_batch, result_trafo, truth_trafo,
                          loss_function,
                          covariance_list=[],
                          covariance_names=[],
                          plot_file='llh_scan_{parameter}_{event_counter}',
                          event_counter=0,
                          parameter_tensor_name='x_parameters'):
        """Make 1D LLh scans for each parameter

        Parameters
        ----------
        data_batch : list of tf.Tensor
            The tensorflow data batch from the tf.Dataset.
        result_trafo : np.array
            The best fit result in transformed and normalized coordinates.
            Shape: [1, num_parameters]
        truth_trafo : np.array
            The true parameters in transformed and normalized coordinates.
            Shape: [1, num_parameters]
        loss_function : tf.function
            The function which computes the llh loss for a given parameter
            tensor.
        covariance_list : list of array_like, optional
            A list of covariance matrices.
            Shape of each matrix: [num_parameters, num_parameters]
        covariance_names : list of str, optional
            A list of names for each of the provided covariance matrices.
        plot_file : str, optional
            The name of the output file. The plot will be saved to
            plot_file.format(parameter=parameter) where parameter is the name
            of the current parameter.
        event_counter : int, optional
            The current event id.
        parameter_tensor_name : str, optional
            The name of the parameter tensor.
        """
        from matplotlib import pyplot as plt

        assert len(result_trafo) == 1, 'Expects one event at a time'
        assert len(truth_trafo) == 1, 'Expects one event at a time'
        assert len(covariance_names) == len(covariance_list)

        truth = self.manager.data_trafo.inverse_transform(
            data=truth_trafo,
            tensor_name=parameter_tensor_name,
        )
        result = self.manager.data_trafo.inverse_transform(
            data=result_trafo,
            tensor_name=parameter_tensor_name,
        )

        range_dict = {
            # 'x': 0.05,
            # 'y': 0.05,
            # 'z': 0.05,
        }

        # now go through each of the parameters and make the scan
        for param_i, parameter in enumerate(
                self.manager.models[0].parameter_names):

            # make llh scan
            diff = range_dict.get(parameter, 1.0)
            values_trafo = np.linspace(truth_trafo[0, param_i] - diff,
                                       truth_trafo[0, param_i] + diff, 1000)
            llh_vals_true = []
            llh_vals_reco = []
            values = []
            for value_trafo in values_trafo:

                # create a parameter tensor based on the best fit result
                param_tensor_trafo = np.array(result_trafo)
                param_tensor_trafo[0, param_i] = value_trafo
                llh_vals_reco.append(loss_function(
                    parameters_trafo=param_tensor_trafo,
                    data_batch=data_batch).numpy()
                )

                # create a parameter tensor based on the truth
                param_tensor_trafo = np.array(truth_trafo)
                param_tensor_trafo[0, param_i] = value_trafo
                llh_vals_true.append(loss_function(
                    parameters_trafo=param_tensor_trafo,
                    data_batch=data_batch).numpy()
                )

                # undo transformation in values
                param_tensor = self.manager.data_trafo.inverse_transform(
                                data=param_tensor_trafo,
                                tensor_name=parameter_tensor_name)
                values.append(param_tensor[0, param_i])

            offset_truth = loss_function(parameters_trafo=truth_trafo,
                                         data_batch=data_batch).numpy()
            offset_reco = loss_function(parameters_trafo=result_trafo,
                                        data_batch=data_batch).numpy()

            print('parameter', parameter)
            print('values_trafo', values_trafo)
            print('llh_vals_true', llh_vals_true)
            print('llh_vals_reco', llh_vals_reco)

            fig, axes = plt.subplots(2, 1, sharex=True, sharey=False,
                                     figsize=(9, 8))

            # add covariance parabolas
            offsets = [offset_truth, offset_reco]
            mu_list = [truth[0, param_i], result[0, param_i]]
            for name, covariance in zip(covariance_names, covariance_list):
                sigma = np.sqrt(covariance[param_i, param_i])
                for i, ax in enumerate(axes):
                    offset = -np.log(sigma) - np.log(np.sqrt(2*np.pi))
                    parabola = offset - basis_functions.log_gauss(
                        values, mu=mu_list[i], sigma=sigma) + offsets[i]
                    ax.plot(values, parabola, label=name)

            # plot landscapes
            axes[0].plot(values, llh_vals_true,
                         label='LLh (Truth)', color='0.2')
            axes[0].axhline(offset_truth, color='0.8', linestyle='--')
            axes[1].plot(values, llh_vals_reco,
                         label='LLh (Reco)', color='0.2')
            axes[1].axhline(offset_reco, color='0.8', linestyle='-')

            # plot lines at best fit and truth
            for ax in axes:
                ax.axvline(truth[0, param_i], label='Truth',
                           linestyle='--', color='red')
                ax.axvline(result[0, param_i], label='Reco',
                           linestyle='-', color='red')
                ax.legend(ncol=2)

            if False:
                axes[0].set_ylim(min(offset_truth - 0.1, min(llh_vals_true)),
                                 offset_truth + 10)
                axes[1].set_ylim(min(offset_reco - 0.1, min(llh_vals_reco)),
                                 offset_reco + 10)

            axes[0].set_ylabel('LLH Value')
            axes[1].set_ylabel('LLH Value')
            axes[1].set_xlabel('{}'.format(parameter))
            fig.savefig(plot_file.format(
                parameter=parameter, event_counter=event_counter))
            plt.close(fig)

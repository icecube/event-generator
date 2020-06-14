from __future__ import division, print_function
import os
import logging
import tensorflow as tf
import numpy as np
import timeit

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.model.base import Model


class BaseModelManager(Model):

    @property
    def optimizer(self):
        if self.untracked_data is not None and \
                'optimizer' in self.untracked_data:
            return self.untracked_data['optimizer']
        else:
            return None

    @property
    def data_handler(self):
        if self.sub_components is not None and \
                'data_handler' in self.sub_components:
            return self.sub_components['data_handler']
        else:
            return None

    @property
    def data_trafo(self):
        if self.models is not None:
            return self.models[0].data_trafo
        else:
            return None

    @property
    def models(self):

        if self.sub_components is not None and \
                'models_0000' in self.sub_components:

            # cache model list in untracked data
            if 'models' not in self.untracked_data:
                model_list = []
                for key in sorted([k for k in self.sub_components.keys()
                                   if 'models_' == k[:7]]):
                    model_list.append(self.sub_components[key])
                self._untracked_data['models'] = model_list

            return self.untracked_data['models']
        else:
            return None

    def __init__(self, logger=None):
        """Initializes ModelManager object.

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(BaseModelManager, self).__init__(logger=self._logger)

    def _configure(self, config, data_handler, models):
        """Configure the ModelManager component instance.

        Parameters
        ----------
        config : dict
            Configuration of the ModelManager object.
        data_handler : DataHandler object
            The data handler object.
        models : List of Models
            The list of model objects to use. An ensemble of models will be
            created. All of the models must define the same hypothesis and use
            the same data transformation object.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed directly as parameters into the configure method,
            as these are automatically gathered. Components passed as lists,
            tuples, and dicts are also collected, unless they are nested
            deeper (list of list of components will not be detected).
            The dependent_sub_components may also be left empty for these
            passed and detected sub components.
            Deeply nested sub components or sub components created within
            (and not directly passed as an argument to) this component
            must be added manually.
            Settings that need to be defined are:
                class_string:
                    misc.get_full_class_string_of_object(self)
                settings: dict
                    The settings of the component.
                mutable_settings: dict, default={}
                    The mutable settings of the component.
                check_values: dict, default={}
                    Additional check values.
        dict
            The data of the component.
            This must at least contain the tensor list which must be
            stored under the key 'tensors'.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """

        sub_components = {
            'data_handler': data_handler,
        }

        for i, m in enumerate(models):
            sub_components['models_{:04d}'.format(i)] = m

        # check for compatibilities of sub components
        self._check_sub_component_compatibility(sub_components)

        # check if all models define the same hypothesis and use the same
        # data transformation object
        self._check_model_ensemble_compatibility(models)

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config))

        return configuration, {}, sub_components

    def _check_model_ensemble_compatibility(self, models):
        """Check compatibility of models in an ensemble.

        Models need to define the same hypothesis and ordering of parameter
        names as well as use the same data transformation object
        (values must all match exactly).

        Parameters
        ----------
        models : list of Model objects
            A list of model objects that define the model ensemble.
        """

        # get parameter names and data trafo object of the first model
        parameter_names = models[0].parameter_names
        data_trafo = models[0].data_trafo

        # Now go through models and check if all define the same
        # data transformation and parameters
        for i, model in enumerate(models):

            # check parameter names
            if parameter_names != models[i].parameter_names:
                msg = 'Parameter names of model {:04d} do not match: {} != {}'
                raise ValueError(msg.format(
                    i, models[i].parameter_names, parameter_names))

            # check data trafo object
            data_trafo_i = models[i].data_trafo
            if not data_trafo.configuration.is_compatible(
                    data_trafo_i.configuration):
                msg = 'Data Trafo of model {:04d} is not compatible: {}, {}'
                raise ValueError(msg.format(
                    i, data_trafo_i.configuration, data_trafo.configuration))

            # (The following are probably unnecessary, since it should already
            #  be checked in the compatibility check.)
            if set(data_trafo.data.keys()) != set(data_trafo_i.data.keys()):
                msg = 'Data Trafo keys of model {:04d} do not match: {} != {}'
                raise ValueError(msg.format(
                    i, set(data_trafo_i.keys()), set(data_trafo.keys())))

            for key in data_trafo.data.keys():
                if np.any(data_trafo.data[key] != data_trafo_i.data[key]):
                    msg = 'Data trafo key {} of model {:04d} does not match: '
                    msg += '{} != {}'
                    raise ValueError(msg.format(
                        key, i, data_trafo.data[key], data_trafo_i.data[key]))

    def _check_sub_component_compatibility(self, sub_components):
        """Check compatibility of sub components.

        The model manager class takes care of handling and putting together
        multiple sub components such as the data_handler,
        data_trafo (part of model), and model components.
        Before using these components together, they must be checked for
        compatibility.

        Parameters
        ----------
        sub_components : dict of sub components
            A dictionary of the sub components which are to be checked.
            This dictionary must consist of the 'data_handler' and 'model'.
            sub_compontens = {
                'data_handler': data_handler,
                'model': model,
            }
        """
        for name, model in sub_components.items():

            # skip data handler sub component, since that is what we are
            # checking compatiblity against
            if name == 'data_handler':
                continue

            # check compatibility of data_handler configurations of
            # data_trafo (model) and the data_handler component
            model_config = model.configuration
            trafo_config = Configuration(
                **model_config.sub_component_configurations['data_trafo'])
            data_handler_config = Configuration(
                **trafo_config.sub_component_configurations['data_handler'])

            if not sub_components['data_handler'].configuration.is_compatible(
                    data_handler_config):
                msg = 'Model {} and data handler are not compatible: {}, {}'
                raise ValueError(msg.format(
                        name,
                        sub_components['data_handler'].configuration.dict,
                        data_handler_config.dict))

    def _update_sub_components(self, names):
        """Update settings which are based on the modified sub component.

        During loading of a component, sub components may be changed with a
        new and modified (but compatible) version. This allows the alteration
        of mutable settings.
        Some settings or data of a component may depend on mutable settings
        of a sub component. If these are not saved and retrieved directly from
        the sub component, they will not be automatically updated.
        This method is triggered when a sub component with the name 'name'
        is updated. It allows to update settings and data that depend on the
        modified sub component.

        Enforcing a derived class to implement this method (even if it is a
        simple 'pass' in the case of no dependent settings and data)
        will ensure that the user is aware of the issue.

        A good starting point to obtain an overview of which settings may need
        to be modified, is to check the _configure method. Any settings and
        data set there might need to be updated.

        Parameters
        ----------
        names : list of str
            The names of the sub components that were modified.
        """
        for name in names:
            if name not in ['data_handler']:
                msg = 'Can not update {!r}.'
                raise ValueError(msg.format(name))

        self._check_sub_component_compatibility(self.sub_components)

    def save_weights(self, dir_path, max_keep=3, protected=False,
                     description=None, num_training_steps=None):
        """Save the model weights.

        Metadata on the checkpoints is stored in a model_checkpoints.yaml
        in the output directory. If it does not exist yet, a new one will be
        created. Otherwise, its values will be updated
        The file contains meta data on the checkpoints and keeps track
        of the most recents files. The structure  and content of meta data:

            latest_checkpoint: int
                The number of the latest checkpoint.

            unprotected_checkpoints:
                '{checkpoint_number}':
                    'creation_date': str
                        Time of model creation in human-readable format
                    'time_stamp': int
                        Time of model creation in seconds since
                        1/1/1970 at midnight (time.time()).
                    'file_basename': str
                        Path to the model
                    'description': str
                        Optional description of the checkpoint.

            protected_checkpoints:
                (same as unprotected_checkpoints)

        Parameters
        ----------
        dir_path : str
            Path to the output directory.
        max_keep : int, optional
            The maximum number of unprotectd checkpoints to keep.
            If there are more than this amount of unprotected checkpoints,
            the oldest checkpoints will be deleted.
        protected : bool, optional
            If True, this checkpoint will not be considered for deletion
            for max_keep.
        description : str, optional
            An optional description string that describes the checkpoint.
            This will be saved in the checkpoints meta data.
        num_training_steps : int, optional
            The number of training steps with the current training settings.
            This will be used to update the training_steps.yaml file to
            account for the correct number of training steps for the most
            recent training step.

        Raises
        ------
        IOError
            If the model checkpoint file already exists.
        KeyError
            If the model checkpoint meta data already exists.
        ValueError
            If the model has changed since it was configured.

        """
        for name, sub_component in self.sub_components.items():

            # get directory of sub component
            sub_dir_path = os.path.join(dir_path, name)

            if issubclass(type(sub_component), Model):
                # save weights of Model sub component
                sub_component.save_weights(
                    dir_path=sub_dir_path, max_keep=max_keep,
                    protected=protected, description=description,
                    num_training_steps=num_training_steps)

    def save_training_settings(self, dir_path, new_training_settings):
        """Save a new training step with its components and settings.

        Parameters
        ----------
        dir_path : str
            Path to the output directory.
        new_training_settings : dict, optional
            If provided, a training step will be created.
            A dictionary containing the settings of the new training step.
            This dictionary must contain the following keys:

                config: dict
                    The configuration settings used to train.
                components: dict
                    The components used during training. These typically
                    include the Loss and Evaluation components.
        """
        for name, sub_component in self.sub_components.items():

            # get directory of sub component
            sub_dir_path = os.path.join(dir_path, name)

            if issubclass(type(sub_component), Model):
                # save weights of Model sub component
                sub_component.save_training_settings(
                    dir_path=sub_dir_path,
                    new_training_settings=new_training_settings)

    def load_weights(self, dir_path, checkpoint_number=None):
        """Load the model weights.

        Parameters
        ----------
        dir_path : str
            Path to the input directory.
        checkpoint_number : None, optional
            Optionally specify a certain checkpoint number that should be
            loaded. If checkpoint_number is None (default), then the latest
            checkpoint will be loaded.

        Raises
        ------
        IOError
            If the checkpoint meta data cannot be found in the input directory.
        """
        for name, sub_component in self.sub_components.items():

            # get directory of sub component
            sub_dir_path = os.path.join(dir_path, name)

            if issubclass(type(sub_component), Model):
                # load weights of Model sub component
                sub_component.load_weights(dir_path=sub_dir_path,
                                           checkpoint_number=checkpoint_number)

    def _save(self, dir_path, **kwargs):
        """Virtual method for additional save tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to save the component.
        This can for instance be saving of tensorflow model weights.

        The MultiSource only contains weights in its submodules which are
        automatically saved via recursion. Therefore, it does not need
        to explicitly save anything here.

        Parameters
        ----------
        dir_path : str
            The path to the output directory to which the component will be
            saved.
        **kwargs
            Additional keyword arguments that may be used by the derived
            class.
        """
        pass

    def _load(self, dir_path, **kwargs):
        """Virtual method for additional load tasks by derived class

        This is a virtual method that may be overwritten by derived class
        to perform additional tasks necessary to load the component.
        This can for instance be loading of tensorflow model weights.

        The MultiSource only contains weights in its submodules which are
        automatically loaded via recursion. Therefore, it does not need
        to explicitly load anything here.

        Parameters
        ----------
        dir_path : str
            The path to the input directory from which the component will be
            loaded.
        **kwargs
            Additional keyword arguments that may be used by the derived
            class.
        """

        # check for compatibilities of sub components
        self._check_sub_component_compatibility(self.sub_components)

    def regularization_loss(self, variables, opt_config):
        """Get L1 and L2 regularization terms.

        Parameters
        ----------
        variables : List of tensors
            The variables for which to compute the regularization.
        opt_config : config
            The optimization config defining the settings.

        Returns
        -------
        tf.Tensor
            Scalar regularization loss
        """
        reg_loss = 0.

        # apply regularization
        if opt_config['l1_regularization'] > 0.:
            reg_loss += tf.add_n([tf.reduce_sum(tf.abs(v)) for v in variables])

        if opt_config['l2_regularization'] > 0.:
            reg_loss += tf.add_n([tf.reduce_sum(v**2) for v in variables])

        return reg_loss

    @tf.function
    def get_loss(self, data_batch, loss_module, opt_config, is_training,
                 add_summaries=False, parameter_tensor_name='x_parameters'):
        """Get the scalar loss for a batch of data and a given loss component.

        Parameters
        ----------
        data_batch : tuple of tf.Tensor
            A tuple of tensors. This is the batch received from the tf.Dataset.
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors)
            method.
        opt_config : config
            The optimization config defining the settings.
        is_training : bool, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        add_summaries : bool, optional
            If True, tensorflow summaries will be calculated and added.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tf.Tensor
            The scalar loss.

        Deleted Parameters
        ------------------
        step : int, optional
            The current training step.
        """
        data_batch_dict = {}
        for i, name in enumerate(self.data_handler.tensors.names):
            data_batch_dict[name] = data_batch[i]

        # get loss for each model
        combined_loss = None
        for i, model in enumerate(self.models):
            result_tensors = model.get_tensors(
                data_batch_dict,
                is_training=is_training,
                parameter_tensor_name=parameter_tensor_name)

            loss_value = loss_module.get_loss(
                data_batch_dict, result_tensors,
                self.data_handler.tensors,
                model=model,
                parameter_tensor_name=parameter_tensor_name)

            reg_loss = self.regularization_loss(
                                        variables=model.trainable_variables,
                                        opt_config=opt_config)

            if combined_loss is None:
                combined_loss = loss_value + reg_loss
            else:
                combined_loss += loss_value + reg_loss

            # create summaries if a writer is provided
            if add_summaries:
                tf.summary.scalar(
                    'loss_{:04d}'.format(i), loss_value,
                    step=tf.cast(model.step, dtype=tf.int64))
                if (opt_config['l1_regularization'] > 0. or
                        opt_config['l2_regularization'] > 0.):
                    tf.summary.scalar(
                        'reg_loss_{:04d}'.format(i), reg_loss,
                        step=tf.cast(model.step, dtype=tf.int64))

        return combined_loss

    @tf.function
    def perform_training_step(self, data_batch, loss_module, opt_config,
                              parameter_tensor_name='x_parameters'):
        """Perform one training step

        Parameters
        ----------
        data_batch : tuple of tf.Tensor
            A tuple of tensors. This is the batch received from the tf.Dataset.
        loss_module : LossComponent
            A loss component that is used to compute the loss. The component
            must provide a
            loss_module.get_loss(data_batch_dict, result_tensors, tensors)
            method.
        opt_config : config
            The optimization config defining the settings.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'

        Returns
        -------
        tf.Tensor
            The scalar loss.
        """
        with tf.GradientTape() as tape:
            combined_loss = self.get_loss(
                                data_batch, loss_module, opt_config,
                                is_training=True,
                                parameter_tensor_name=parameter_tensor_name)

        variables = []
        for model in self.models:
            variables.extend(model.trainable_variables)
        gradients = tape.gradient(combined_loss, variables)

        # remove nans in gradients and replace these with zeros
        if 'remove_nan_gradients' in opt_config:
            remove_nan_gradients = opt_config['remove_nan_gradients']
        else:
            remove_nan_gradients = False
        if remove_nan_gradients:
            gradients = [tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)
                         for grad in gradients if grad is not None]

        if 'clip_gradients_value' in opt_config:
            clip_gradients_value = opt_config['clip_gradients_value']
        else:
            clip_gradients_value = None
        if clip_gradients_value is not None:
            capped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                         clip_gradients_value)
        else:
            capped_gradients = gradients

        self.optimizer.apply_gradients(zip(capped_gradients, variables))

        return combined_loss

    def get_concrete_function(self, function, input_signature,
                              **fixed_objects):
        """Get a concrete tensorflow function with a fixed input signature

        Parameters
        ----------
        function : function
            The function for which to obtain a concrete version.
        input_signature : tf.TensorSpec or nested tf.TensorSpec
            The input signature for tensors.
        **fixed_objects
            These are python objects which are held constant.

        Returns
        -------
        tf.function
            A concrete tensorflow function with a fixed input_signature.
        """
        @tf.function(input_signature=input_signature)
        def concrete_function(data_batch):
            return function(data_batch, **fixed_objects)

        return concrete_function

    def train(self, config, loss_module, num_training_iterations,
              evaluation_module=None):
        """Train the model.

        Parameters
        ----------
        config: dict
            A config describing all of the settings for the training script.
            Amongst others, this config must contain:

            train_iterator_settings : dict
                The settings for the training data iterator that will be
                created from the data handler.
            validation_iterator_settings : dict
                The settings for the validation data iterator that will be
                created from the data handler.
            training_settings : dict
                Optimization configuration with settings for the optimizer
                and regularization.

        loss_module : LossComponent
            A loss component that defines the loss function. The loss component
            must provide the method
                loss_module.get_loss(data_batch_dict, result_tensors)
        num_training_iterations : int
            Number of training iterations to perform.
        evaluation_module : EvaluationComponent, optional
            An evaluation component that can be used to calculate and log
            evaluation metrics on validation batches. The evaluation module
            must implement a method
                evaluation_module.evaluate(data_batch, result_tensors, tensors)
        """
        self.assert_configured(True)

        # deine directories
        save_dir = self.configuration.config['config']['manager_dir']
        train_log_dir = os.path.join(save_dir, 'logs/training')
        val_log_dir = os.path.join(save_dir, 'logs/validation')
        eval_log_dir = os.path.join(save_dir, 'logs/evaluation')

        # create optimizer from config
        opt_config = config['training_settings']
        optimizer = getattr(tf.optimizers, opt_config['optimizer_name'])(
                                **opt_config['optimizer_settings']
                                )
        self._untracked_data['optimizer'] = optimizer

        # save new training step to model
        training_components = {'loss_module': loss_module}
        if evaluation_module is not None:
            training_components['evaluation_module'] = evaluation_module

        new_training_settings = {
            'config': config,
            'components': training_components,
        }
        self.save(dir_path=save_dir,
                  description='Starting Training',
                  new_training_settings=new_training_settings,
                  num_training_steps=None,
                  protected=True,
                  overwrite=True)

        # create writers
        training_writer = tf.summary.create_file_writer(train_log_dir)
        validation_writer = tf.summary.create_file_writer(val_log_dir)
        evaluation_writer = tf.summary.create_file_writer(eval_log_dir)

        train_dataset = iter(self.data_handler.get_tf_dataset(
            **config['data_iterator_settings']['training']))
        validation_dataset = iter(self.data_handler.get_tf_dataset(
            **config['data_iterator_settings']['validation']))

        # -------------------------------------------------------
        # get concrete functions for training and loss evaluation
        # -------------------------------------------------------
        perform_training_step = self.get_concrete_function(
            function=self.perform_training_step,
            input_signature=(train_dataset.element_spec,),
            loss_module=loss_module,
            opt_config=opt_config)

        get_loss = self.get_concrete_function(
            function=self.get_loss,
            input_signature=(train_dataset.element_spec,),
            loss_module=loss_module,
            opt_config=opt_config,
            is_training=False,
            add_summaries=True)

        # --------------------------------
        # start loop over training batches
        # --------------------------------
        #   every n batches:
        #       - calculate loss on validation
        #       - run evaluation module
        #       - save model weights
        start_time = timeit.default_timer()
        validation_time = start_time
        for step in range(num_training_iterations):
            # --------------------------
            # perform one training step
            # --------------------------

            # increment step counter
            for model in self.models:
                model.step.assign_add(1)

            # get new batch of training data
            training_data_batch = next(train_dataset)

            # perform one training iteration
            perform_training_step(data_batch=training_data_batch)

            # --------------------------
            # evaluate on validation set
            # --------------------------
            if step % opt_config['validation_frequency'] == 0:
                new_validation_time = timeit.default_timer()
                time_diff = new_validation_time - validation_time
                validation_time = new_validation_time

                # get new batch of training data
                training_data_batch = next(train_dataset)

                # compute loss on training data
                with training_writer.as_default():
                    loss_training = get_loss(data_batch=training_data_batch)

                # get new batch of validation data
                val_data_batch = next(train_dataset)

                # compute loss on validation data
                with validation_writer.as_default():
                    loss_validation = get_loss(data_batch=val_data_batch)

                # check if there is a nan
                if np.isnan(loss_training) or np.isnan(loss_validation):
                    raise ValueError('Aborting training due to invalid loss')

                # write to file
                training_writer.flush()
                validation_writer.flush()

                # print out loss to console
                msg = 'Step: {:08d}, Runtime: {:2.2f}s, Time/Step: {:3.3f}s'
                print(msg.format(step, validation_time - start_time,
                                 time_diff /
                                 opt_config['validation_frequency']))
                print('\t[Train]      {:3.3f}'.format(loss_training))
                print('\t[Validation] {:3.3f}'.format(loss_validation))

            # ------------------------------------------------
            # Perform additional evaluations on validation set
            # ------------------------------------------------
            if step % opt_config['evaluation_frequency'] == 0:

                # call evaluation component if it exists
                if evaluation_module is not None:

                    # get new batch of validation data
                    val_data_batch = next(train_dataset)

                    evaluation_module.evaluate(
                        data_batch=val_data_batch,
                        loss_module=loss_module,
                        tensors=self.data_handler.tensors,
                        step=tf.convert_to_tensor(step, dtype=tf.int64),
                        writer=evaluation_writer)

                    # write to file
                    evaluation_writer.flush()

            # ----------
            # save model
            # ----------
            if step % opt_config['save_frequency'] == 0 and step != 0:
                self.save_weights(dir_path=save_dir,
                                  num_training_steps=step)

        # save model
        self.save_weights(dir_path=save_dir,
                          num_training_steps=step,
                          description='End of training step',
                          protected=True)

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
    def model(self):
        if self.sub_components is not None and \
                'model' in self.sub_components:
            return self.sub_components['model']
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

    def _configure(self, config, data_handler, model):
        """Configure the ModelManager component instance.

        Parameters
        ----------
        config : dict
            Configuration of the ModelManager object.
        data_handler : TYPE
            Description
        model : TYPE
            Description

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
            'model': model,
        }

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config))

        return configuration, {}, sub_components

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
        pass

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
                 step=None):
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
        step : int, optional
            The current training step.

        Returns
        -------
        tf.Tensor
            The scalar loss.
        """
        data_batch_dict = {}
        for i, name in enumerate(self.data_handler.tensors.names):
            data_batch_dict[name] = data_batch[i]

        result_tensors = self.model.get_tensors(data_batch_dict,
                                                is_training=is_training)

        loss_value = loss_module.get_loss(data_batch_dict, result_tensors,
                                          self.data_handler.tensors)

        reg_loss = self.regularization_loss(
                                    variables=self.model.trainable_variables,
                                    opt_config=opt_config)

        combined_loss = loss_value + reg_loss

        # create summaries if a writer is provided
        if step is not None:
            tf.summary.scalar('loss', loss_value, step=step)
            if (opt_config['l1_regularization'] > 0. or
                    opt_config['l2_regularization'] > 0.):
                tf.summary.scalar('reg_loss', reg_loss)

        return combined_loss

    @tf.function
    def perform_training_step(self, data_batch, loss_module, opt_config):
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

        Returns
        -------
        tf.Tensor
            The scalar loss.
        """
        with tf.GradientTape() as tape:
            combined_loss = self.get_loss(data_batch, loss_module, opt_config,
                                          is_training=True)

        variables = self.model.trainable_variables
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
                  overwrite=True)

        # create writers
        training_writer = tf.summary.create_file_writer(train_log_dir)
        validation_writer = tf.summary.create_file_writer(val_log_dir)
        evaluation_writer = tf.summary.create_file_writer(eval_log_dir)

        train_dataset = iter(self.data_handler.get_tf_dataset(
            **config['data_iterator_settings']['training']))
        validation_dataset = iter(self.data_handler.get_tf_dataset(
            **config['data_iterator_settings']['validation']))

        # start loop over training batches
        #   every n batches:
        #       - calculate loss on validation
        #       - run evaluation module
        #       - save model weights
        start_time = timeit.default_timer()
        validation_time = start_time
        for step in range(num_training_iterations):
            tf_step = tf.convert_to_tensor(step, dtype=tf.int64)
            # --------------------------
            # perform one training step
            # --------------------------

            # increment step counter
            self.model.step.assign_add(1)

            # get new batch of training data
            training_data_batch = next(train_dataset)

            # perform one training iteration
            self.perform_training_step(data_batch=training_data_batch,
                                       loss_module=loss_module,
                                       opt_config=opt_config)

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
                    loss_training = self.get_loss(data_batch=training_data_batch,
                                                  loss_module=loss_module,
                                                  opt_config=opt_config,
                                                  is_training=False,
                                                  step=tf_step,
                                                  )

                # get new batch of validation data
                val_data_batch = next(train_dataset)

                # compute loss on validation data
                with validation_writer.as_default():
                    loss_validation = self.get_loss(data_batch=val_data_batch,
                                                    loss_module=loss_module,
                                                    opt_config=opt_config,
                                                    is_training=False,
                                                    step=tf_step,
                                                    )

                # write to file
                training_writer.flush()
                validation_writer.flush()

                # print out loss to console
                msg = 'Step: {:08d}, Runtime: {:2.2f}s, Time/Step: {:3.3f}s'
                print(msg.format(step, validation_time - start_time,
                                 time_diff / opt_config['validation_frequency']))
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
                        step=tf_step,
                        writer=evaluation_writer)

                    # write to file
                    evaluation_writer.flush()

            # ----------
            # save model
            # ----------
            if step % opt_config['save_frequency'] == 0:
                self.save_weights(dir_path=save_dir,
                                  num_training_steps=step)

        # save model
        self.save_weights(dir_path=save_dir,
                          num_training_steps=step,
                          description='End of training step',
                          protected=True)

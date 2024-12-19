import os
import tensorflow as tf
import numpy as np
import timeit

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.model.base import Model
from egenerator.utils import tf_helpers


class BaseModelManager(Model):

    @property
    def optimizer(self):
        if (
            self.untracked_data is not None
            and "optimizer" in self.untracked_data
        ):
            return self.untracked_data["optimizer"]
        else:
            return None

    @property
    def data_handler(self):
        if (
            self.sub_components is not None
            and "data_handler" in self.sub_components
        ):
            return self.sub_components["data_handler"]
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

        if (
            self.sub_components is not None
            and "models_0000" in self.sub_components
        ):

            # cache model list in untracked data
            if "models" not in self.untracked_data:
                model_list = []
                for key in sorted(
                    [
                        k
                        for k in self.sub_components.keys()
                        if "models_" == k[:7]
                    ]
                ):
                    model_list.append(self.sub_components[key])
                self._untracked_data["models"] = model_list

            return self.untracked_data["models"]
        else:
            return None

    @tf.function
    def _compile_optimizer(self):
        """Compile the optimizer by running it with zero gradients."""
        variables = []
        for model in self.models:
            variables.extend(model.trainable_variables)
        zero_grads = [tf.zeros_like(w) for w in variables]

        self._untracked_data["optimizer"].apply_gradients(
            zip(zero_grads, variables)
        )

    def _set_optimizer(self, opt_config):
        """Get the optimizer object.

        Creates the optimizer object from the optimizer config
        and sets it as an untracked data variable. Also (re-)creates
        a checkpoint object to keep track of the optimizer variables.

        Parameters
        ----------
        opt_config : dict
            The optimization config defining the settings.
            Must contain the following keys:
                optimizer_name : str
                    The name of the optimizer to use.
                optimizer_settings : dict
                    The settings for the optimizer.
        """

        # create a tensorflow optimizer object
        optimizer_settings = dict(opt_config["optimizer_settings"])

        # create learning rate schedule if learning rate is a dict
        if "learning_rate" in optimizer_settings:
            if isinstance(optimizer_settings["learning_rate"], dict):

                # assume that the learning rate dictionary defines a schedule
                # In this case the dictionary must have the following keys:
                #   full_class_string: str
                #       The full class string of the scheduler class to use.
                #   settings: dict
                #       keyword arguments that are passed on to the scheduler
                #       class.
                lr_cfg = optimizer_settings.pop("learning_rate")
                scheduler_class = misc.load_class(lr_cfg["full_class_string"])
                scheduler = scheduler_class(**lr_cfg["settings"])
                optimizer_settings["learning_rate"] = scheduler

        self._untracked_data["optimizer"] = getattr(
            tf.optimizers, opt_config["optimizer_name"]
        )(**optimizer_settings)

        # run optimizer with zero gradients to create optimizer variables
        if self.models is not None:
            self._compile_optimizer()

        # create a tensorflow checkpoint object and keep track of variables
        checkpoint_vars = {
            "step": self._untracked_data["step"],
            "optimizer": self._untracked_data["optimizer"],
        }
        self._untracked_data["checkpoint"] = tf.train.Checkpoint(
            **checkpoint_vars
        )

    def _configure(self, config, opt_config, data_handler, models):
        """Configure the ModelManager component instance.

        Parameters
        ----------
        config : dict
            Configuration of the ModelManager object.
        opt_config : dict
            Configuration defining the settings for the optimizer.
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

        # get name of model
        if "name" in self._untracked_data:
            name = self._untracked_data["name"]
        else:
            name = self.__class__.__name__

        sub_components = {
            "data_handler": data_handler,
        }

        for i, m in enumerate(models):
            sub_components["models_{:04d}".format(i)] = m

        # check for compatibilities of sub components
        self._check_sub_component_compatibility(sub_components)

        # check if all models define the same hypothesis and use the same
        # data transformation object
        self._check_model_ensemble_compatibility(models)

        # create step counter for this object
        self._untracked_data["step"] = tf.Variable(
            0, trainable=False, dtype=tf.int64, name=name + "_step"
        )

        # create a tensorflow optimizer and checkpoint object
        self._set_optimizer(opt_config=opt_config)

        # collect any variables from sub_components if they aren't already
        all_variables = list(self.variables)
        for name, sub_component in sorted(sub_components.items()):
            if issubclass(type(sub_component), tf.Module):
                all_variables.extend(
                    [
                        v
                        for v in sub_component.variables
                        if not tf_helpers.is_in(v, all_variables)
                    ]
                )
        self._untracked_data["variables"] = tuple(all_variables)

        num_vars, num_total_vars = self._count_number_of_variables()
        msg = f"\nNumber of Model Variables for {name}:\n"
        msg = f"\tFree: {num_vars}\n"
        msg += f"\tTotal: {num_total_vars}"
        self._logger.info(msg)

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(config=config),
            mutable_settings=dict(opt_config=opt_config),
        )

        return configuration, {}, sub_components

    def _rebuild_computation_graph(self):
        """Rebuild the computation graph of the model.

        When constructing the model via the BaseComponent.load() method,
        the computation graph is not automatically created. This method
        rebuilds the computation graph of the model.
        It is therefore typically only needed when the model is loaded
        via the BaseComponent.load() method.
        """
        # rebuild the tensorflow graph if it does not exist yet
        if not self.is_configured:

            # save temporary values to make sure these aren't modified
            configuration_id = id(self.configuration)
            sub_components_id = id(self.sub_components)
            configuration = Configuration(**self.configuration.dict)
            data = dict(self.data)
            sub_components = dict(self.sub_components)

            # rebuild graph
            config_dict = self.configuration.config

            models = []
            for name in sorted(sub_components.keys()):
                if name.startswith("models_"):
                    models.append(sub_components[name])
                else:
                    config_dict[name] = sub_components[name]
            config_dict["models"] = models

            self._logger.debug(f"[Model] Rebuilding {self.__class__.__name__}")

            self._configure(**config_dict)

            # make sure that no additional class attributes are created
            # apart from untracked ones
            self._check_member_attributes()

            # make sure the other values weren't overwritten
            if (
                not configuration.is_compatible(self.configuration)
                or configuration_id != id(self.configuration)
                or data != self.data
                or sub_components != self.sub_components
                or sub_components_id != id(self.sub_components)
            ):
                raise ValueError("Tracked components were changed!")

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
                msg = "Parameter names of model {:04d} do not match: {} != {}"
                raise ValueError(
                    msg.format(i, models[i].parameter_names, parameter_names)
                )

            # check data trafo object
            data_trafo_i = models[i].data_trafo
            if not data_trafo.configuration.is_compatible(
                data_trafo_i.configuration
            ):
                msg = "Data Trafo of model {:04d} is not compatible: {}, {}"
                raise ValueError(
                    msg.format(
                        i, data_trafo_i.configuration, data_trafo.configuration
                    )
                )

            # (The following are probably unnecessary, since it should already
            #  be checked in the compatibility check.)
            if set(data_trafo.data.keys()) != set(data_trafo_i.data.keys()):
                msg = "Data Trafo keys of model {:04d} do not match: {} != {}"
                raise ValueError(
                    msg.format(
                        i, set(data_trafo_i.keys()), set(data_trafo.keys())
                    )
                )

            for key in data_trafo.data.keys():
                if np.any(data_trafo.data[key] != data_trafo_i.data[key]):
                    msg = "Data trafo key {} of model {:04d} does not match: "
                    msg += "{} != {}"
                    raise ValueError(
                        msg.format(
                            key,
                            i,
                            data_trafo.data[key],
                            data_trafo_i.data[key],
                        )
                    )

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
            # checking compatibility against
            if name == "data_handler":
                continue

            # check compatibility of data_handler configurations of
            # data_trafo (model) and the data_handler component
            model_config = model.configuration
            trafo_config = Configuration(
                **model_config.sub_component_configurations["data_trafo"]
            )
            data_handler_config = Configuration(
                **trafo_config.sub_component_configurations["data_handler"]
            )

            if not sub_components["data_handler"].configuration.is_compatible(
                data_handler_config
            ):
                msg = "Model {} and data handler are not compatible: {}, {}"
                raise ValueError(
                    msg.format(
                        name,
                        sub_components["data_handler"].configuration.dict,
                        data_handler_config.dict,
                    )
                )

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
            if name not in ["data_handler"] and not isinstance(
                self.sub_components[name], Model
            ):
                msg = "Can not update {!r}."
                raise ValueError(msg.format(name))

        self._check_sub_component_compatibility(self.sub_components)

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

        super()._load(dir_path, **kwargs)

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
        reg_loss = 0.0

        # apply regularization
        if opt_config["l1_regularization"] > 0.0:
            reg_loss += tf.add_n([tf.reduce_sum(tf.abs(v)) for v in variables])

        if opt_config["l2_regularization"] > 0.0:
            reg_loss += tf.add_n([tf.reduce_sum(v**2) for v in variables])

        return reg_loss

    @tf.function
    def get_loss(
        self,
        data_batch,
        loss_module,
        opt_config,
        is_training,
        summary_writer=None,
        parameter_tensor_name="x_parameters",
        **kwargs
    ):
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
        summary_writer : tf.summary.SummaryWriter, optional
            If provided, tensorflow summaries will be calculated and written
            to the specified summary writer.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'
        **kwargs
            Arbitrary keyword arguments. These will be passed on to
            the get_loss function of the loss module.

        Returns
        -------
        tf.Tensor
            The scalar loss.
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
                parameter_tensor_name=parameter_tensor_name,
            )

            loss_value = loss_module.get_loss(
                data_batch_dict,
                result_tensors,
                self.data_handler.tensors,
                model=model,
                parameter_tensor_name=parameter_tensor_name,
                **kwargs
            )

            reg_loss = self.regularization_loss(
                variables=model.trainable_variables, opt_config=opt_config
            )

            if combined_loss is None:
                combined_loss = loss_value + reg_loss
            else:
                combined_loss += loss_value + reg_loss

            # create summaries if a writer is provided
            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.scalar(
                        "loss_{:04d}".format(i),
                        loss_value,
                        step=model.step,
                    )
                    if (
                        opt_config["l1_regularization"] > 0.0
                        or opt_config["l2_regularization"] > 0.0
                    ):
                        tf.summary.scalar(
                            "reg_loss_{:04d}".format(i),
                            reg_loss,
                            step=model.step,
                        )

        return combined_loss

    @tf.function
    def perform_training_step(
        self,
        data_batch,
        loss_module,
        opt_config,
        parameter_tensor_name="x_parameters",
        **kwargs
    ):
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
        **kwargs
            Arbitrary keyword arguments. These will be passed on to
            the get_loss function of the loss module.

        Returns
        -------
        tf.Tensor
            The scalar loss.
        """
        with tf.GradientTape() as tape:
            combined_loss = self.get_loss(
                data_batch,
                loss_module,
                opt_config,
                is_training=True,
                parameter_tensor_name=parameter_tensor_name,
                **kwargs
            )

        variables = []
        for model in self.models:
            variables.extend(model.trainable_variables)
        gradients = tape.gradient(combined_loss, variables)

        # remove nans in gradients and replace these with zeros
        if "remove_nan_gradients" in opt_config:
            remove_nan_gradients = opt_config["remove_nan_gradients"]
        else:
            remove_nan_gradients = False
        if remove_nan_gradients:
            gradients = [
                tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
                for grad in gradients
                if grad is not None
            ]

        if "clip_gradients_value" in opt_config:
            clip_gradients_value = opt_config["clip_gradients_value"]
        else:
            clip_gradients_value = None
        if clip_gradients_value is not None:
            capped_gradients, _ = tf.clip_by_global_norm(
                gradients, clip_gradients_value
            )
        else:
            capped_gradients = gradients

        # Ensure finite values
        asserts = []
        for gradient in capped_gradients:
            assert_finite = tf.Assert(
                tf.math.is_finite(tf.reduce_mean(gradient)),
                [
                    tf.reduce_min(gradient),
                    tf.reduce_mean(gradient),
                    tf.reduce_max(gradient),
                ],
            )
            asserts.append(assert_finite)
        with tf.control_dependencies(asserts):
            self.optimizer.apply_gradients(zip(capped_gradients, variables))

        return combined_loss

    def get_concrete_function(
        self, function, input_signature, **fixed_objects
    ):
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
        def concrete_function(data_batch, **kwargs):
            return function(data_batch, **kwargs, **fixed_objects)

        return concrete_function

    def train(
        self,
        config,
        loss_module,
        num_training_iterations,
        evaluation_module=None,
        profile_training=False,
    ):
        """Train the model.

        Parameters
        ----------
        config : dict
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
        profile_training : bool, optional
            If true, trainings teps 90 to 100 will be profiled.

        Raises
        ------
        ValueError
            Description
        """
        self.assert_configured(True)

        # define directories
        save_dir = self.configuration.config["config"]["manager_dir"]
        train_log_dir = os.path.join(save_dir, "logs/training")
        val_log_dir = os.path.join(save_dir, "logs/validation")
        eval_log_dir = os.path.join(save_dir, "logs/evaluation")

        # check if we got a different optimizer definition
        opt_config = config["training_settings"]
        self_opt_config = self.configuration.config["opt_config"]
        is_new_optimizer = False
        if opt_config["optimizer_name"] != self_opt_config["optimizer_name"]:
            is_new_optimizer = True
        elif (
            opt_config["optimizer_settings"]
            != self_opt_config["optimizer_settings"]
        ):
            is_new_optimizer = True
        if is_new_optimizer:
            self._logger.warning(
                "Found new optimizer settings. Reconfiguring optimizer. "
                "Note that no checks for compatibility are performed! "
                "Training with an incompatible change in optimizer settings "
                "will result in unability to load saved checkpoints. "
            )

            # create a tensorflow optimizer and checkpoint object
            self._set_optimizer(opt_config)
            self.optimizer.iterations.assign(self.step)
            file_path = self.retreive_weight_file_path(
                dir_path=self.configuration.config["config"]["manager_dir"],
                checkpoint_number=None,
            )
            self._logger.debug(f"[Model] Loading checkpoint: {file_path}")
            self._untracked_data["checkpoint"].read(
                file_path
            ).assert_consumed()

        # save new training step to model
        training_components = {"loss_module": loss_module}
        if evaluation_module is not None:
            training_components["evaluation_module"] = evaluation_module

        new_training_settings = {
            "config": config,
            "components": training_components,
        }
        self.save(
            dir_path=save_dir,
            description="Starting Training",
            new_training_settings=new_training_settings,
            num_training_steps=None,
            protected=True,
            overwrite=True,
        )

        # create writers
        training_writer = tf.summary.create_file_writer(train_log_dir)
        validation_writer = tf.summary.create_file_writer(val_log_dir)
        evaluation_writer = tf.summary.create_file_writer(eval_log_dir)

        train_dataset = iter(
            self.data_handler.get_tf_dataset(
                **config["data_iterator_settings"]["training"]
            )
        )
        validation_dataset = iter(
            self.data_handler.get_tf_dataset(
                **config["data_iterator_settings"]["validation"]
            )
        )

        # -------------------------------------------------------
        # get concrete functions for training and loss evaluation
        # -------------------------------------------------------
        perform_training_step = self.get_concrete_function(
            function=self.perform_training_step,
            input_signature=(train_dataset.element_spec,),
            loss_module=loss_module,
            opt_config=opt_config,
            **opt_config["additional_loss_module_kwargs"]
        )

        get_loss_train = self.get_concrete_function(
            function=self.get_loss,
            input_signature=(train_dataset.element_spec,),
            loss_module=loss_module,
            opt_config=opt_config,
            is_training=False,
            **opt_config["additional_loss_module_kwargs"]
        )

        get_loss_val = self.get_concrete_function(
            function=self.get_loss,
            input_signature=(train_dataset.element_spec,),
            loss_module=loss_module,
            opt_config=opt_config,
            is_training=False,
            summary_writer=validation_writer,
            **opt_config["additional_loss_module_kwargs"]
        )

        # --------------------------------
        # start loop over training batches
        # --------------------------------
        #   every n batches:
        #       - calculate loss on validation
        #       - run evaluation module
        #       - save model weights
        start_time = timeit.default_timer()
        validation_time = start_time
        num_training_steps = 0
        for step in range(self.step.numpy() + 1, num_training_iterations + 1):
            # --------------------------
            # perform one training step
            # --------------------------

            # increment step counter
            self.increment_step()
            num_training_steps += 1

            # get new batch of training data
            training_data_batch = next(train_dataset)

            # perform one training iteration
            perform_training_step(data_batch=training_data_batch)

            # --------------------------
            # evaluate on validation set
            # --------------------------
            if step % opt_config["validation_frequency"] == 0:
                new_validation_time = timeit.default_timer()
                time_diff = new_validation_time - validation_time
                validation_time = new_validation_time

                # get new batch of training data
                training_data_batch = next(train_dataset)

                # compute loss on training data
                loss_training = get_loss_train(data_batch=training_data_batch)

                # get new batch of validation data
                val_data_batch = next(validation_dataset)

                # compute loss on validation data
                loss_validation = get_loss_val(data_batch=val_data_batch)

                # check if there is a nan
                if np.isnan(loss_training) or np.isnan(loss_validation):
                    raise ValueError("Aborting training due to invalid loss")

                # write to file
                training_writer.flush()
                validation_writer.flush()

                # print out loss to console
                msg = "Step: {:08d}, Runtime: {:2.2f}s, Time/Step: {:3.3f}s"
                print(
                    msg.format(
                        step,
                        validation_time - start_time,
                        time_diff / opt_config["validation_frequency"],
                    )
                )
                print("\t[Train]      {:3.3f}".format(loss_training))
                print("\t[Validation] {:3.3f}".format(loss_validation))

            # ------------------------------------------------
            # Perform additional evaluations on validation set
            # ------------------------------------------------
            if step % opt_config["evaluation_frequency"] == 0:

                # call evaluation component if it exists
                if evaluation_module is not None:

                    # get new batch of validation data
                    val_data_batch = next(train_dataset)

                    evaluation_module.evaluate(
                        data_batch=val_data_batch,
                        loss_module=loss_module,
                        tensors=self.data_handler.tensors,
                        step=tf.convert_to_tensor(step, dtype=tf.int64),
                        writer=evaluation_writer,
                    )

                    # write to file
                    evaluation_writer.flush()

            # ----------
            # save model
            # ----------
            if step % opt_config["save_frequency"] == 0 and step != 0:
                self.save_weights(
                    dir_path=save_dir,
                    num_training_steps=num_training_steps,
                )

            # -----------------------
            # Profile steps 90 to 100
            # -----------------------
            if profile_training:
                if step == 90:
                    # start profiler
                    tf.profiler.experimental.start(train_log_dir)
                if step == 100:
                    # stop profiler
                    tf.profiler.experimental.stop()

        # save model
        self.save_weights(
            dir_path=save_dir,
            num_training_steps=num_training_steps,
            description="End of training step",
            protected=True,
        )

"""Helper Functions to build components.

Functions:
    - buildManager(config)

"""

from egenerator import misc
from egenerator.data.trafo import DataTransformer
from egenerator.loss.multi_loss import MultiLossModule
from egenerator.data.tensor import DataTensorList


def build_data_handler(data_handler_settings):
    """Build a data handler object

    Parameters
    ----------
    data_handler_settings : dict
        A dictionary containing the settings for the data handler.

    Returns
    -------
    DataHandler object
        The data handler object.
    """
    DataHandlerClass = misc.load_class(
        "egenerator.data.handler.{}".format(
            data_handler_settings["data_handler"]
        )
    )
    data_handler = DataHandlerClass()
    data_handler.configure(config=data_handler_settings)
    return data_handler


def build_data_transformer(
    data_trafo_settings,
    modified_tensors=None,
):
    """Build a data transformer object

    Parameters
    ----------
    data_trafo_settings : dict
        A dictionary containing the settings for the data transformer.
    modifed_tensors : DataTensorList object, optional
        A modified tensorlist to use.
        Will update information on the tensorlist including:
            - exists fields of each tensor

    Returns
    -------
    DataTransformer object
        The data transformer object.
    """
    data_trafo = DataTransformer()
    data_trafo.load(data_trafo_settings["model_dir"])

    if modified_tensors is not None:

        tensors = []
        for tensor_i in data_trafo.data["tensors"].list:

            tensor_to_add = tensor_i

            # only update tensors that have no transformation
            # i.e. which the trafo model does not rely on
            if (
                not tensor_i.trafo
            ) and tensor_i.name in modified_tensors.names:
                t_mod_i = modified_tensors.list[
                    modified_tensors.get_index(tensor_i.name)
                ]
                diff = tensor_i.compare(t_mod_i)

                # only update tensor if the only difference
                # is whether the tensor exists or not
                if len(diff) == 0 or diff == ["exists"]:
                    tensor_to_add = t_mod_i

            tensors.append(tensor_to_add)

        # update the tensor list
        data_trafo._data["tensors"] = DataTensorList(tensors)

    return data_trafo


def build_loss_module(loss_module_settings):
    """Build a loss module

    Parameters
    ----------
    loss_module_settings : dict or list of dict
        A dictionary containing the settings for the loss module.
        This may also be a list of dictionaries, where each dictionary
        defines the settings for one loss module. These are then all combined
        to a MultiLossModule which simply accumulates the losses of all
        sub modules.

    Returns
    -------
    LossModule object
        The loss module object.
    """

    # If a dictionary is provided, then this is just a single loss module
    if isinstance(loss_module_settings, dict):
        LossModuleClass = misc.load_class(loss_module_settings["loss_module"])
        loss_module = LossModuleClass()
        loss_module.configure(config=loss_module_settings["config"])

    # A list of dictionaries is provided which each define a loss module
    else:
        loss_modules = []
        loss_modules_weights = []
        for settings in loss_module_settings:
            LossModuleClass = misc.load_class(settings["loss_module"])
            if "loss_module_weight" in settings:
                loss_modules_weights.append(settings["loss_module_weight"])
            else:
                loss_modules_weights.append(1.0)
            loss_module_i = LossModuleClass()
            loss_module_i.configure(config=settings["config"])
            loss_modules.append(loss_module_i)

        # create a multi loss module from a list of given loss modules
        loss_module = MultiLossModule()
        loss_module.configure(
            loss_modules=loss_modules,
            loss_modules_weights=loss_modules_weights,
        )

    return loss_module


def build_model(
    model_settings, data_transformer, allow_rebuild_base_models=False
):
    """Build a Model object

    Parameters
    ----------
    model_settings : dict
        A dictionary containing the model settings. Must at least contain
        `model_class`, `config` and if this is a multi-source:
        `multi_source_bases`.
    data_transformer : DataTransformer object
        The data transformer object to use for the model.
    allow_rebuild_base_models : bool, optional
        If True, the base model is allowed to be rebuild, otherwise it
        will raise an error if a base model is not loaded, but attempted to
        be rebuild from scratch.

    Returns
    -------
    Model object
        The model object.

    Raises
    ------
    ValueError
        Description
    """

    # check if base sources need to be built:
    base_models = {}
    if "multi_source_bases" in model_settings:

        multi_source_bases = model_settings["multi_source_bases"]

        # loop through defined multi-source bases and create them
        for name, settings in multi_source_bases.items():

            # create base source
            BaseSourceClass = misc.load_class(settings["model_class"])
            base_source = BaseSourceClass()

            # load model
            if "load_dir" in settings and settings["load_dir"] is not None:
                base_source.load(settings["load_dir"])

            # configure model if we are not loading it new
            else:
                if not allow_rebuild_base_models:
                    msg = "Model is not allowed to be rebuild! To change this "
                    msg += "setting, set 'allow_rebuild_base_models' to True."
                    raise ValueError(msg)

                # if this multi source base is a nested multi source
                # with sub sources, we need to recursively build them
                if "multi_source_bases" in settings:
                    base_source = build_model(
                        model_settings=settings,
                        data_transformer=data_transformer,
                        allow_rebuild_base_models=allow_rebuild_base_models,
                    )
                else:

                    # check if the base model has its own data transformer defined
                    if "data_trafo_settings" in settings:
                        data_transformer_base = build_data_transformer(
                            settings["data_trafo_settings"],
                            modified_tensors=data_transformer.data["tensors"],
                        )
                    else:
                        data_transformer_base = data_transformer
                    base_source.configure(
                        config=settings["config"],
                        data_trafo=data_transformer_base,
                    )

            base_models[name] = base_source

    ModelClass = misc.load_class(model_settings["model_class"])
    model = ModelClass()

    arguments = dict(
        config=model_settings["config"], data_trafo=data_transformer
    )
    if base_models != {}:
        arguments["base_models"] = base_models

    model.configure(**arguments)
    return model


def build_manager(
    config,
    restore,
    data_handler=None,
    data_transformer=None,
    models=None,
    modified_sub_components={},
    allow_rebuild_base_models=False,
):
    """Build the Manager Component.

    Parameters
    ----------
    config : dict
        The configuration settings.
    restore : bool
        If True, the manager will be restored from file.
    data_handler : DataHandler object, optional
        A data handler object to use.
    data_transformer : DataTransformer object, optional
        A data transformer object to use.
    models : List of Model objects, optional
        The model objects to use. If more than one model object is provided,
        an ensemble of models is created. Models must be compatible, define
        the same hypothesis and use the same data trafo model.
    modified_sub_components : dict, optional
        A dictionary of modified sub-components that will be passed on to
        ModelManager load method.
    allow_rebuild_base_models : bool, optional
        If True, the model is allowed to be rebuild, otherwise it will raise
        an error if a model is not loaded, but rebuild from scratch.

    Returns
    -------
    Manager object
        Returns the created or loaded manager object.
    Model object
        Returns the created or passed model object.
    DataHandler object
        Returns the created or passed data handler object.
    DataTransformer object
        Returns the created or passed data transformer object.
    """

    manager_config = config["model_manager_settings"]
    manager_dir = manager_config["config"]["manager_dir"]

    # ---------------------------------------
    # Create and configure/load Model Manager
    # ---------------------------------------
    ModelManagerClass = misc.load_class(manager_config["model_manager_class"])
    manager = ModelManagerClass()

    if restore:
        manager.load(
            manager_dir, modified_sub_components=modified_sub_components
        )

    else:
        # --------------------------
        # Create Data Handler object
        # --------------------------
        if data_handler is None:
            data_handler = build_data_handler(config["data_handler_settings"])

        # --------------------------
        # create and load TrafoModel
        # --------------------------
        if data_transformer is None:
            data_transformer = build_data_transformer(
                config["data_trafo_settings"],
                modified_tensors=data_handler.tensors,
            )

        # -----------------------
        # create and Model object
        # -----------------------
        if models is None:
            model = build_model(
                config["model_settings"],
                data_transformer=data_transformer,
                allow_rebuild_base_models=allow_rebuild_base_models,
            )
            models = [model]

        # -----------------------
        # configure model manager
        # -----------------------
        manager.configure(
            config=manager_config["config"],
            data_handler=data_handler,
            models=models,
        )

    return manager, manager.models, manager.data_handler, manager.data_trafo

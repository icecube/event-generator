import os
import tensorflow as tf
import ruamel.yaml as yaml

from egenerator import misc


def extract_model_class(manager_dir):
    """Extract the class of a model.

    Parameters
    ----------
    manager_dir : str
        The directory of model manager. This is the root directory of the
        exported model.

    Returns
    -------
    str
        The name of the class. This can be loaded via
        egenerator.misc.load_class().
    """
    model_config_file = os.path.join(
        manager_dir, 'models_0000/configuration.yaml')

    with open(model_config_file, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.Loader)
    return config_dict['class_string']


def extract_model_parameters(manager_dir):
    """Extract the model parameters of a model.

    Parameters
    ----------
    manager_dir : str
        The directory of model manager. This is the root directory of the
        exported model.

    Returns
    -------
    list of str
        The ordered model parameter names.
    """
    model = misc.load_class(extract_model_class(manager_dir))()
    model.load(os.path.join(manager_dir, 'models_0000/'))
    return list(model.parameter_names)

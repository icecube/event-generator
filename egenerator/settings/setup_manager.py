from __future__ import division, print_function
import os
import logging
import numpy as np
from copy import deepcopy
import tensorflow as tf
import ruamel.yaml as yaml

from egenerator import misc
from egenerator.settings import version_control


class SetupManager:
    """Setup Manager for Event-Generator project

    Handles loading and merging of yaml config files, sets up directories.

    Config Keys
    -----------

    Automatically created settings

        config_name : str
            An automatically created config name.
            Base names of yaml config files are concatenated with '__'.
            This can be used to create subdirectories for logging and
            checkpoints.

    General settings

        test_data_file : str
            Path to test data file.


    General Training settings

        num_training_iterations : int
            Number of training iterations to perform.


    Trafo settings

        trafo_model_path : str
            Path to trafo model file.

    Attributes
    ----------
    config : dictionary
        Dictionary with defined settings.
    """

    # define default config
    _default_config = {
                        'float_precision': 'float32',
                     }

    def __init__(self, config_files):
        """Initializes the Event-Generator Setup Manager

        Loads and merges yaml config files, sets up necessary directories.

        Parameters
        ----------
        config_files : list of strings
            List of yaml config files.
        program_options : str
            A string defining the program options.
        """
        self.logger = logging.getLogger(__name__)
        self._config_files = config_files

        # load and combine configs
        self._setup_config()

    def _setup_config(self):
        """Loads and merges config

        Raises
        ------
        ValueError
            If no config files are given.
            If a setting is defined in multiplie config files.
        """
        if not isinstance(self._config_files, (list, tuple)):
            raise ValueError(
                'Wrong data type: {!r}. Must be list of file paths'.format(
                    self._config_files))

        if len(self._config_files) == 0:
            raise ValueError('You must specify at least one config file!')

        # ----------------------------------
        # load config
        # ----------------------------------
        new_config = {}
        config_name = None
        for config_file in self._config_files:

            # append yaml file to config_name
            file_base_name = os.path.basename(config_file).replace('.yaml', '')
            if config_name is None:
                config_name = file_base_name
            else:
                config_name += '__' + file_base_name

            config_update = yaml.safe_load(open(config_file))
            duplicates = set(new_config.keys()).intersection(
                                                    set(config_update.keys()))

            # make sure no options are defined multiple times
            if duplicates:
                raise ValueError('Keys are defined multiple times {!r}'.format(
                                                                duplicates))
            # update config
            new_config.update(config_update)
        config = dict(self._default_config)
        config.update(new_config)

        # define numpy and tensorflow float precision
        config['tf_float_precision'] = getattr(tf, config['float_precision'])
        config['np_float_precision'] = getattr(np, config['float_precision'])
        try:
            import tfscripts as tfs
            tfs.FLOAT_PRECISION = config['tf_float_precision']
        except ImportError:
            self.logger.warning('Could not import tfscripts package.')

        # get git repo information
        config['git_short_sha'] = str(version_control.short_sha)
        config['git_sha'] = str(version_control.sha)
        config['git_origin'] = str(version_control.origin)
        config['git_uncommited_changes'] = version_control.uncommitted_changes
        config['pip_installed_packages'] = version_control.installed_packages

        # ----------------------------------
        # expand all strings with variables
        # ----------------------------------
        config['config_name'] = str(config_name)
        for key in config:
            if isinstance(config[key], str):
                config[key] = config[key].format(**config)

        self.config = config

    def get_config(self):
        """Returns config

        Returns
        -------
        dictionary
            Dictionary with defined settings.
        """
        return dict(deepcopy(self.config))

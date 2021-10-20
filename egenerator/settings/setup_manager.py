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
            checkpoints..

    Attributes
    ----------
    config : dictionary
        Dictionary with defined settings.
    """

    # define default config
    _default_config = {}

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

            # config_update = yaml.round_trip_load(open(config_file),
            #                                      preserve_quotes=True)
            yaml_loader = yaml.YAML(typ='safe')
            config_update = yaml_loader.load(open(config_file))
            # config_update = yaml.load(open(config_file), loader=yaml.Loader())
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
        config['egenerator_dir'] = os.path.join(os.path.dirname(__file__),
                                                '../..')
        config = self.expand_strings(config, config)

        self.config = config

    def expand_strings(self, config, expand_config):
        """Recursively expands all strings of dictionary or nested dictionaries

        Parameters
        ----------
        config : dict
            A dictionary of settings
        expand_config : dict
            The definitions of strings which are to be expanded.

        Returns
        -------
        dict
            The dictionary with expanded strings.
        """
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = value.format(**expand_config)
            elif isinstance(value, dict):
                config[key] = self.expand_strings(value, expand_config)
        return config

    def get_config(self):
        """Returns config

        Returns
        -------
        dictionary
            Dictionary with defined settings.
        """
        return dict(deepcopy(self.config))

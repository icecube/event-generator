#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helper Functions to build components.

Functions:
    - buildManager(config)

"""
from __future__ import division, print_function
import os
import shutil
import logging

from egenerator import misc
from egenerator.data.trafo import DataTransformer


def build_manager(config, restore,
                  data_handler=None,
                  data_transformer=None,
                  model=None,
                  modified_sub_components={}):
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
    model : Model object, optional
        A model object to use.
    modified_sub_components : dict, optional
        A dictionary of modified sub-components that will be passed on to
        ModelManager load method.

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

    manager_config = config['model_manager_settings']
    manager_dir = manager_config['config']['manager_dir']

    # ---------------------------------------
    # Create and configure/load Model Manager
    # ---------------------------------------
    ModelManagerClass = misc.load_class(manager_config['model_manager_class'])
    manager = ModelManagerClass()

    if restore:
        manager.load(manager_dir,
                     modified_sub_components=modified_sub_components)

    else:
        # --------------------------
        # Create Data Handler object
        # --------------------------
        if data_handler is None:
            DataHandlerClass = misc.load_class(
                'egenerator.data.handler.{}'.format(
                            config['data_handler_settings']['data_handler']))
            data_handler = DataHandlerClass()
            data_handler.configure(config=config['data_handler_settings'])

        # --------------------------
        # create and load TrafoModel
        # --------------------------
        data_transformer = DataTransformer()
        data_transformer.load(config['data_trafo_settings']['model_dir'])

        # -----------------------
        # create and Model object
        # -----------------------
        if model is None:
            model_settings = config['model_settings']

            # check if base sources need to be built:
            base_sources = {}
            if 'multi_source_bases' in model_settings:

                multi_source_bases = model_settings['multi_source_bases']

                # loop through defined multi-source bases and create them
                for name, settings in multi_source_bases.items():

                    # create base source
                    BaseSourceClass = misc.load_class(settings['model_class'])
                    base_source = BaseSourceClass()

                    # load model
                    if ('load_dir' in settings and
                            settings['load_dir'] is not None):
                        base_source.load(settings['load_dir'])

                    # configure model if we are not loading it new
                    else:
                        base_source.configure(config=settings['config'],
                                              data_trafo=data_transformer)

                    base_sources[name] = base_source

            ModelClass = misc.load_class(model_settings['model_class'])
            model = ModelClass()

            arguments = dict(config=model_settings['config'],
                             data_trafo=data_transformer)
            if base_sources != {}:
                arguments['base_sources'] = base_sources

            model.configure(**arguments)

        # -----------------------
        # configure model manager
        # -----------------------
        manager.configure(config=manager_config['config'],
                          data_handler=data_handler,
                          model=model)

    return manager, model, data_handler, data_transformer

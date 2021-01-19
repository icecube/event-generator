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
from egenerator.loss.multi_loss import MultiLossModule


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
    DataHandlerClass = misc.load_class('egenerator.data.handler.{}'.format(
                    data_handler_settings['data_handler']))
    data_handler = DataHandlerClass()
    data_handler.configure(config=data_handler_settings)
    return data_handler


def build_loss_module(loss_module_settings):
    """Build a loss module

    Parameters
    ----------
    loss_module_settings : dict or list of dict
        A dictionary containg the settings for the loss module.
        This may also be a list of dictionaries, where each dictionary
        defines the settings for one loss module. Thes are then all combined
        to a MultiLossModule which simply accumulates the losses of all
        sub modules.

    Returns
    -------
    LossModule object
        The loss module object.
    """

    # If a dictionary is provided, then this is just a single loss module
    if isinstance(loss_module_settings, dict):
        LossModuleClass = misc.load_class(loss_module_settings['loss_module'])
        loss_module = LossModuleClass()
        loss_module.configure(config=loss_module_settings['config'])

    # A list of dictionaries is provided which each define a loss module
    else:
        loss_modules = []
        for settings in loss_module_settings:
            LossModuleClass = misc.load_class(settings['loss_module'])
            loss_module_i = LossModuleClass()
            loss_module_i.configure(config=settings['config'])
            loss_modules.append(loss_module_i)

        # create a multi loss module from a list of given loss modules
        loss_module = MultiLossModule()
        loss_module.configure(loss_modules=loss_modules)

    return loss_module


def build_model(model_settings, data_transformer,
                allow_rebuild_base_sources=False):
    """Build a Model object

    Parameters
    ----------
    model_settings : dict
        A dictionary containg the model settings. Must at least contain
        `model_class`, `config` and if this is a multi-source:
        `multi_source_bases`.
    data_transformer : DataTransformer object
        The data transformer object to use for the model.
    allow_rebuild_base_sources : bool, optional
        If True, the base source model is allowed to be rebuild, otherwise it
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
    base_sources = {}
    if 'multi_source_bases' in model_settings:

        multi_source_bases = model_settings['multi_source_bases']

        # loop through defined multi-source bases and create them
        for name, settings in multi_source_bases.items():

            # create base source
            BaseSourceClass = misc.load_class(settings['model_class'])
            base_source = BaseSourceClass()

            # load model
            if ('load_dir' in settings and settings['load_dir'] is not None):
                base_source.load(settings['load_dir'])

            # configure model if we are not loading it new
            else:
                if not allow_rebuild_base_sources:
                    msg = 'Model is not allowed to be rebuild! To change this '
                    msg += "setting, set 'allow_rebuild_base_sources' to True."
                    raise ValueError(msg)

                # if this multi source base is a nested multi source
                # with sub sources, we need to recursively build them
                if 'multi_source_bases' in settings:
                    base_source = build_model(
                        model_settings=settings,
                        data_transformer=data_transformer,
                        allow_rebuild_base_sources=allow_rebuild_base_sources,
                    )
                else:

                    # check if the base model has its own data transformer defined
                    if 'data_trafo_settings' in settings:
                        data_transformer_base = DataTransformer()
                        data_transformer_base.load(
                            settings['data_trafo_settings']['model_dir'])
                    else:
                        data_transformer_base = data_transformer
                    base_source.configure(config=settings['config'],
                                          data_trafo=data_transformer_base)

            base_sources[name] = base_source

    ModelClass = misc.load_class(model_settings['model_class'])
    model = ModelClass()

    arguments = dict(config=model_settings['config'],
                     data_trafo=data_transformer)
    if base_sources != {}:
        arguments['base_sources'] = base_sources

    model.configure(**arguments)
    return model


def build_manager(config, restore,
                  data_handler=None,
                  data_transformer=None,
                  models=None,
                  modified_sub_components={},
                  allow_rebuild_base_sources=False):
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
    allow_rebuild_base_sources : bool, optional
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
            data_handler = build_data_handler(config['data_handler_settings'])

        # --------------------------
        # create and load TrafoModel
        # --------------------------
        if data_transformer is None:
            data_transformer = DataTransformer()
            data_transformer.load(config['data_trafo_settings']['model_dir'])

        # -----------------------
        # create and Model object
        # -----------------------
        if models is None:
            model = build_model(
                config['model_settings'],
                data_transformer=data_transformer,
                allow_rebuild_base_sources=allow_rebuild_base_sources,
            )
            models = [model]

        # -----------------------
        # configure model manager
        # -----------------------
        manager.configure(config=manager_config['config'],
                          data_handler=data_handler,
                          models=models)

    return manager, manager.models, manager.data_handler, manager.data_trafo

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import shutil
import logging
import click
import ruamel.yaml as yaml

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.data.trafo import DataTransformer


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
def main(config_files):
    """Script to train model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    """

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # # Check if path already exists
    # if os.path.exists(config['data_trafo_settings']['model_path']):
    #     if not click.confirm(
    #             'File already exists at destination. Overwrite {!r}?'.format(
    #                 config['data_trafo_settings']['model_path']),
    #             default=False):
    #         raise ValueError('Aborting!')

    # ------------------
    # Create loss module
    # ------------------
    LossModuleClass = \
        misc.load_class(config['loss_module_settings']['loss_module'])
    loss_module = LossModuleClass()
    loss_module.configure(config=config['loss_module_settings']['config'])

    # -------------------------------------
    # create evaluation module if specified
    # -------------------------------------
    if 'evaluation_module_settings' in config and \
            config['evaluation_module_settings'][
                                        'evaluation_module'] is not None:

        EvaluationModuleClass = misc.load_class(
                    config['evaluation_module_settings']['evaluation_module'])
        evaluation_module = EvaluationModuleClass()
        evaluation_module.configure(
            config=config['evaluation_module_settings']['config'])
    else:
        evaluation_module = None

    # ---------------------------------------
    # Create and configure/load Model Manager
    # ---------------------------------------
    ModelManagerClass = misc.load_class(
        config['model_manager_settings']['model_manager_class'])
    manager = ModelManagerClass()

    manager_config = config['model_manager_settings']
    manager_dir = manager_config['config']['manager_dir']

    if manager_config['model_restore']:
        if os.path.exists(os.path.join(manager_dir, 'configuration.yaml')):
            restore = True

        else:
            logger = logging.getLogger(__name__)
            msg = 'Could not find a saved model at {!r}. '
            msg += 'Starting from scratch.'
            logger.warning(msg.format(manager_dir))

            restore = False
    else:
        # start from scratch
        if os.path.exists(os.path.join(manager_dir, 'configuration.yaml')):

            # check if director y already exists
            msg = 'A saved model already exists at destination. Delete {!r}?'
            if not click.confirm(msg.format(manager_dir), default=False):
                raise ValueError('Aborting!')

            # delete directory
            shutil.rmtree(manager_dir)

        restore = False

    if restore:
        manager.load(manager_dir)

    else:
        # --------------------------
        # Create Data Handler object
        # --------------------------
        DataHandlerClass = misc.load_class('egenerator.data.handler.{}'.format(
                            config['data_handler_settings']['data_handler']))
        data_handler = DataHandlerClass()
        data_handler.configure(config=config['data_handler_settings'])

        # --------------------------
        # create and load TrafoModel
        # --------------------------
        data_transformer = DataTransformer()
        data_transformer.load_trafo_model(
            config['data_trafo_settings']['model_path'])

        # -----------------------
        # create and Model object
        # -----------------------
        ModelClass = misc.load_class(config['model_settings']['model_class'])
        model = ModelClass()
        model.configure(config=config['model_settings']['config'],
                        data_trafo=data_transformer)

        # -----------------------
        # configure model manager
        # -----------------------
        manager.configure(config=manager_config['config'],
                          data_handler=data_handler,
                          model=model)

    # --------------
    # start training
    # --------------
    num_training_iterations = manager_config['num_training_iterations']
    manager.train(config=config,
                  loss_module=loss_module,
                  num_training_iterations=num_training_iterations,
                  evaluation_module=evaluation_module)

    # kill multiprocessing queues and workers
    data_handler.kill()


if __name__ == '__main__':
    main()

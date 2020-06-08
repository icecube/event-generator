#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import shutil
import logging
import click
import tensorflow as tf

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager
from egenerator.utils.build_components import build_loss_module
from egenerator.data.trafo import DataTransformer


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
@click.option('-c', '--nn_minmimizer_config', type=click.Path(exists=True))
def main(config_files, nn_minmimizer_config):
    """Script to train model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    nn_minmimizer_config : str
        The nn_minimizer_config

    Raises
    ------
    ValueError
        Description
    """

    # limit GPU usage
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # load Event-Generator model manager and loss module
    model_manager, model_loss_module, data_handler, data_transformer = \
        setup_egenerator_manager(config_files)

    # read in and combine config files and set up
    setup_manager = SetupManager([nn_minmimizer_config])
    config = setup_manager.get_config()

    # ------------------
    # Create loss module
    # ------------------
    loss_module = build_loss_module(config)

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
    manager_config = config['model_manager_settings']
    manager_dir = manager_config['config']['manager_dir']

    if manager_config['restore_model']:
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

            # check if directory already exists
            msg = 'A saved model already exists at destination. Delete {!r}?'
            if not click.confirm(msg.format(manager_dir), default=False):
                raise ValueError('Aborting!')

            # delete directory
            shutil.rmtree(manager_dir)

        restore = False

    # ------------------------
    # build NN minimizer model
    # ------------------------
    if restore:
        models = None
    else:
        model_settings = config['model_settings']
        ModelClass = misc.load_class(model_settings['model_class'])
        model = ModelClass()
        model.configure(
            config=model_settings['config'],
            data_trafo=data_transformer,
            model_parameter_names=model_manager.models[0].parameter_names,
        )
        model.setup_model_loss_function(model_manager, model_loss_module)
        models = [model]

    # build manager object
    manager, model, data_handler, data_transformer = build_manager(
        config, models=models, data_handler=data_handler,
        data_transformer=data_transformer, restore=restore)

    # --------------
    # start training
    # --------------
    num_training_iterations = \
        config['training_settings']['num_training_iterations']
    manager.train(config=config,
                  loss_module=loss_module,
                  num_training_iterations=num_training_iterations,
                  evaluation_module=evaluation_module)

    # kill multiprocessing queues and workers
    data_handler.kill()


def setup_egenerator_manager(config_files):
    """Setup the Event-Generator mangager and loss module.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.

    Raises
    ------
    ValueError
        Description
    """
    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # Create loss module
    loss_module = build_loss_module(config)

    # Load Model Manager
    manager_config = config['model_manager_settings']
    manager_dir = manager_config['config']['manager_dir']
    manager, model, data_handler, data_transformer = build_manager(
                                                    config, restore=True)

    return manager, loss_module, data_handler, data_transformer


if __name__ == '__main__':
    main()

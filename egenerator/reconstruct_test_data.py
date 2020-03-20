#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import shutil
import logging
import click
import ruamel.yaml as yaml
import tensorflow as tf

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.data.trafo import DataTransformer


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
def main(config_files):
    """Script to reconstruct test data.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    """

    # limit GPU usage
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # ------------------
    # Create loss module
    # ------------------
    LossModuleClass = \
        misc.load_class(config['loss_module_settings']['loss_module'])
    loss_module = LossModuleClass()
    loss_module.configure(config=config['loss_module_settings']['config'])

    # -----------------------------
    # Create and load Model Manager
    # -----------------------------
    ModelManagerClass = misc.load_class(
        config['model_manager_settings']['model_manager_class'])
    manager = ModelManagerClass()

    manager_config = config['model_manager_settings']
    manager_dir = manager_config['config']['manager_dir']

    if not os.path.exists(os.path.join(manager_dir, 'configuration.yaml')):
        msg = 'Could not find a saved model at {!r}!'
        raise ValueError(msg.format(manager_dir))

    manager.load(manager_dir)

    # --------------------
    # start reconstruction
    # --------------------
    manager.reconstruct_testdata(config=config,
                                 loss_module=loss_module)

    # kill multiprocessing queues and workers
    data_handler.kill()


if __name__ == '__main__':
    main()

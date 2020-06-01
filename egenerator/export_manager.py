#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
from copy import deepcopy
import shutil
import click
import tensorflow as tf

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager
from egenerator.utils.build_components import build_loss_module
from egenerator.data.modules.misc.seed_loader import SeedLoaderMiscModule


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
@click.option('-o', '--output_dir', type=click.Path(),
              help='The reconstruction config file to use. If None, '
                   'then the first provided config file will be used.')
@click.option('-r', '--reco_config_file', default=None,
              type=click.Path(exists=True),
              help='The reconstruction config file to use. If None, '
                   'then the first provided config file will be used.')
def main(config_files, output_dir, reco_config_file=None):
    """Script to export model manager.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
        Each config file defines a model. If more than one config file is
        passed, an ensemble of models will be used.
    reco_config_file : str, optional
        Name of config file which defines the reconstruction settings.

    Raises
    ------
    ValueError
        Description
    """

    if os.path.exists(output_dir):
        if not click.confirm(
                'Directory already exists at destination. Delete {!r}?'.format(
                    output_dir),
                default=False):
            raise ValueError('Aborting!')
        else:
            print('Deleting directory {}'.format(output_dir))
            shutil.rmtree(output_dir)

    if reco_config_file is None:
        reco_config_file = [config_files[0]]
    else:
        reco_config_file = [reco_config_file]

    # limit GPU usage
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # read in reconstruction config file
    setup_manager = SetupManager(reco_config_file)
    config = setup_manager.get_config()

    # ------------------
    # Create loss module
    # ------------------
    loss_module = build_loss_module(config)

    # --------------
    # Create manager
    # --------------

    # load models from config files
    models = []
    for config_file in config_files:

        # load manager objects and extract models and a data_handler
        model_manger,  _, data_handler, data_transformer = build_manager(
            SetupManager([config_file]).get_config(),
            restore=True,
            modified_sub_components={},
            allow_rebuild_base_sources=False,
        )
        models.extend(model_manger.models)

    # build manager object
    manager, models, data_handler, data_transformer = build_manager(
                            config,
                            restore=False,
                            models=models,
                            data_handler=data_handler,
                            data_transformer=data_transformer,
                            allow_rebuild_base_sources=False)

    # --------------
    # Export Manager
    # --------------
    manager.save(output_dir)

    # copy over reco config to output directory
    shutil.copy(reco_config_file[0],
                os.path.join(output_dir, 'reco_config.yaml'))

    print('\n====================================')
    print('= Successfully exported model to:  =')
    print('====================================')
    print('{!r}\n'.format(output_dir))


if __name__ == '__main__':
    main()

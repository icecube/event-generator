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
from egenerator.utils.build_components import build_manager
from egenerator.data.trafo import DataTransformer
from egenerator.data.modules.misc.seed_loader import SeedLoaderMiscModule


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

    # ------------------
    # Create misc module
    # ------------------
    reco_config = config['reconstruction_settings']
    if reco_config['seed'] == 'x_parameters':
        # The parameter labels are being used as a seed, so we do not need
        # to create a modified misc module
        modified_sub_components = {}
    else:
        misc_module = SeedLoaderMiscModule()
        misc_module.configure(
            config_data=None,
            seed_names=[reco_config['seed']],
            seed_parameter_names=reco_config['seed_parameter_names'],
            float_precision=reco_config['seed_float_precision'],
            missing_value=reco_config['seed_missing_value'],
            )

        # create nested dictionary of modified sub components in order to
        # swap out the loaded misc_module of the data_handler sub component
        modified_sub_components = {'data_handler': {
            'misc_module': misc_module,
        }}

    if 'modified_label_module' in reco_config:
        label_config = reco_config['modified_label_module']
        LabelModuleClass = misc.load_class(
            'egenerator.data.modules.labels.{}'.format(
                        label_config['label_module']))
        label_module = LabelModuleClass()
        label_module.configure(config=label_config['label_settings'])

        if 'data_handler' in modified_sub_components:
            modified_sub_components['data_handler']['label_module'] = \
                label_module
        else:
            modified_sub_components['data_handler'] = {
                'label_module': label_module
            }

    # -----------------------------
    # Create and load Model Manager
    # -----------------------------
    manager_config = config['model_manager_settings']
    manager_dir = manager_config['config']['manager_dir']

    if manager_config['restore_model']:
        if not os.path.exists(os.path.join(manager_dir, 'configuration.yaml')):
            msg = 'Could not find a saved model at {!r}!'
            raise ValueError(msg.format(manager_dir))

    else:
        if reco_config['seed'] != 'x_parameters':
            # Model Manager is being built from scratch, so we need to pass
            # the data handler settings with the modified misc module
            config['data_handler_settings']['misc_module'] = \
                'seed_loader.SeedLoaderMiscModule'

            config['data_handler_settings']['misc_settings'] = {
                'seed_names': [reco_config['seed']],
                'seed_parameter_names': reco_config['seed_parameter_names'],
                'float_precision': reco_config['seed_float_precision'],
                'missing_value': reco_config['seed_missing_value'],
            }

    # build manager object
    manager, model, data_handler, data_transformer = build_manager(
                            config,
                            restore=manager_config['restore_model'],
                            modified_sub_components=modified_sub_components,
                            allow_rebuild_base_sources=False)

    # --------------------
    # start reconstruction
    # --------------------
    manager.reconstruct_testdata(config=config,
                                 loss_module=loss_module)


if __name__ == '__main__':
    main()

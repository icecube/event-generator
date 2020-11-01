#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
from copy import deepcopy
import click
import tensorflow as tf

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager
from egenerator.utils.build_components import build_loss_module
from egenerator.data.modules.misc.seed_loader import SeedLoaderMiscModule


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
@click.option('-r', '--reco_config_file', default=None, type=str,
              help='The reconstruction config file to use. If None, '
                   'then the first provided config file will be used.')
def main(config_files, reco_config_file=None):
    """Script to train model.

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

    # ------------------
    # Create misc module
    # ------------------
    reco_config = config['reconstruction_settings']
    if isinstance(reco_config['seed'], str):
        seed_names = [reco_config['seed']]
    else:
        seed_names = reco_config['seed']

    # remove x_parameters from seed names:
    seed_names = [name for name in seed_names if name != 'x_parameters']

    if len(seed_names) == 0:
        # The parameter labels are being used as a seed, so we do not need
        # to create a modified misc module
        modified_sub_components = {}
    else:
        misc_module = SeedLoaderMiscModule()
        misc_module.configure(
            config_data=None,
            seed_names=seed_names,
            seed_parameter_names=reco_config['seed_parameter_names'],
            float_precision=reco_config['seed_float_precision'],
            missing_value=reco_config['seed_missing_value'],
            missing_value_dict=reco_config['seed_missing_value_dict'],
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
        label_module.configure(config_data=None,
                               **label_config['label_settings'])

        if 'data_handler' in modified_sub_components:
            modified_sub_components['data_handler']['label_module'] = \
                label_module
        else:
            modified_sub_components['data_handler'] = {
                'label_module': label_module
            }

    if 'modified_data_module' in reco_config:
        data_config = reco_config['modified_data_module']
        DataModuleClass = misc.load_class(
            'egenerator.data.modules.data.{}'.format(
                        data_config['data_module']))
        data_module = DataModuleClass()
        data_module.configure(config_data=None, **data_config['data_settings'])

        if 'data_handler' in modified_sub_components:
            modified_sub_components['data_handler']['data_module'] = \
                data_module
        else:
            modified_sub_components['data_handler'] = {
                'data_module': data_module
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
        if len(seed_names) > 0:
            # Model Manager is being built from scratch, so we need to pass
            # the data handler settings with the modified misc module
            config['data_handler_settings']['misc_module'] = \
                'seed_loader.SeedLoaderMiscModule'

            config['data_handler_settings']['misc_settings'] = {
                'seed_names': seed_names,
                'seed_parameter_names': reco_config['seed_parameter_names'],
                'float_precision': reco_config['seed_float_precision'],
                'missing_value': reco_config['seed_missing_value'],
            }

    # load models from config files
    if 'restore_manager' in reco_config:
        print('Restoring Manager from file')
        restore_manager = reco_config['restore_manager']
    else:
        print('Rebuilding Manager from scratch (hopefully loading models)')
        restore_manager = False

    models = []
    for config_file in config_files:

        # load manager objects and extract models and a data_handler
        model_manger,  _, data_handler, data_transformer = build_manager(
            SetupManager([config_file]).get_config(),
            restore=restore_manager,
            modified_sub_components=deepcopy(modified_sub_components),
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

    # --------------------
    # start reconstruction
    # --------------------
    manager.reconstruct_testdata(config=config,
                                 loss_module=loss_module)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import shutil
import click
import tensorflow as tf

from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager


@click.command()
@click.argument("config_files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "-o",
    "--output_dir",
    type=click.Path(),
    help="The reconstruction config file to use. If None, "
    "then the first provided config file will be used.",
)
@click.option(
    "-r",
    "--reco_config_file",
    default=None,
    type=click.Path(exists=True),
    help="The reconstruction config file to use. If None, "
    "then the first provided config file will be used.",
)
@click.option(
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="WARNING",
)
def main(config_files, output_dir, reco_config_file=None, log_level="WARNING"):
    """Script to export model manager.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
        Each config file defines a model. If more than one config file is
        passed, an ensemble of models will be used.
    reco_config_file : str, optional
        Name of config file which defines the reconstruction settings.
        If None, then the first provided config file will be used.
    log_level : str, optional
        The logging level.

    Raises
    ------
    ValueError
        Description
    """

    # set up logging
    logging.basicConfig(level=log_level)

    if os.path.exists(output_dir):
        if not click.confirm(
            "Directory already exists at destination. Delete {!r}?".format(
                output_dir
            ),
            default=False,
        ):
            raise ValueError("Aborting!")
        else:
            print("Deleting directory {}".format(output_dir))
            shutil.rmtree(output_dir)

    if reco_config_file is None:
        reco_config_file = [config_files[0]]
    else:
        reco_config_file = [reco_config_file]

    # limit GPU usage
    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # read in reconstruction config file
    setup_manager = SetupManager(reco_config_file)
    config = setup_manager.get_config()
    manager_config = config["model_manager_settings"]

    # --------------
    # Create manager
    # --------------

    # load models from config files
    models = []
    for config_file in config_files:

        # load manager objects and extract models and a data_handler
        model_manger, _, data_handler, data_transformer = build_manager(
            SetupManager([config_file]).get_config(),
            restore=manager_config["restore_model"],
            modified_sub_components={},
            allow_rebuild_base_models=not manager_config["restore_model"],
            allow_rebuild_base_decoders=not manager_config["restore_model"],
        )
        models.extend(model_manger.models)

    # build manager object
    manager, models, data_handler, data_transformer = build_manager(
        config,
        restore=False,
        models=models,
        data_handler=data_handler,
        data_transformer=data_transformer,
        allow_rebuild_base_models=False,
        allow_rebuild_base_decoders=False,
    )

    # --------------
    # Export Manager
    # --------------
    manager.save(output_dir)

    # copy over reco config to output directory
    shutil.copy(
        reco_config_file[0], os.path.join(output_dir, "reco_config.yaml")
    )

    print("\n====================================")
    print("= Successfully exported model to:  =")
    print("====================================")
    print("{!r}\n".format(output_dir))


if __name__ == "__main__":
    main()

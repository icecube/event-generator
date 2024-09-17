#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import logging
import click
import tensorflow as tf

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager
from egenerator.utils.build_components import build_loss_module


@click.command()
@click.argument("config_files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="WARNING",
)
def main(config_files, log_level):
    """Script to train model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    log_level : str
        The logging level.
    """

    # set up logging
    logging.basicConfig(level=log_level)

    # limit GPU usage
    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # ------------------
    # Create loss module
    # ------------------
    loss_module = build_loss_module(config["loss_module_settings"])

    # -------------------------------------
    # create evaluation module if specified
    # -------------------------------------
    if (
        "evaluation_module_settings" in config
        and config["evaluation_module_settings"]["evaluation_module"]
        is not None
    ):

        EvaluationModuleClass = misc.load_class(
            config["evaluation_module_settings"]["evaluation_module"]
        )
        evaluation_module = EvaluationModuleClass()
        evaluation_module.configure(
            config=config["evaluation_module_settings"]["config"]
        )
    else:
        evaluation_module = None

    # ---------------------------------------
    # Create and configure/load Model Manager
    # ---------------------------------------
    manager_config = config["model_manager_settings"]
    manager_dir = manager_config["config"]["manager_dir"]

    if manager_config["restore_model"]:
        if os.path.exists(os.path.join(manager_dir, "configuration.yaml")):
            restore = True

        else:
            logger = logging.getLogger(__name__)
            msg = "Could not find a saved model at {!r}. "
            msg += "Starting from scratch."
            logger.warning(msg.format(manager_dir))

            restore = False
    else:
        # start from scratch
        if os.path.exists(os.path.join(manager_dir, "configuration.yaml")):

            # check if directory already exists
            msg = "A saved model already exists at destination. Delete {!r}?"
            if not click.confirm(msg.format(manager_dir), default=False):
                raise ValueError("Aborting!")

            # delete directory
            shutil.rmtree(manager_dir)

        restore = False

    # build manager object
    manager, model, data_handler, data_transformer = build_manager(
        config, restore=restore, allow_rebuild_base_sources=True
    )

    # --------------
    # start training
    # --------------
    num_training_iterations = config["training_settings"][
        "num_training_iterations"
    ]
    manager.train(
        config=config,
        loss_module=loss_module,
        num_training_iterations=num_training_iterations,
        evaluation_module=evaluation_module,
    )

    # kill multiprocessing queues and workers
    data_handler.kill()


if __name__ == "__main__":
    main()

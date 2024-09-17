#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import click

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.data.trafo import DataTransformer


@click.command()
@click.argument("config_files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="WARNING",
)
def main(config_files, log_level="INFO"):
    """Script to generate trafo model.

    Creates the desired trafo model as defined in the yaml configuration files
    and saves the trafo model to disc.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    log_level : str
        The logging level.
    """

    # set up logging
    logging.basicConfig(level=log_level)

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # Check if path already exists
    model_dir = config["data_trafo_settings"]["model_dir"]
    if os.path.exists(os.path.splitext(model_dir)[0]):
        if not click.confirm(
            "File already exists at destination. Overwrite {!r}?".format(
                os.path.splitext(model_dir)[0]
            ),
            default=False,
        ):
            raise ValueError("Aborting!")

    # Create Data Handler object
    DataHandlerClass = misc.load_class(
        "egenerator.data.handler.{}".format(
            config["data_handler_settings"]["data_handler"]
        )
    )
    data_handler = DataHandlerClass()
    data_handler.configure(config=config["data_handler_settings"])

    # create TrafoModel
    data_transformer = DataTransformer()

    data_transformer.configure(
        data_handler=data_handler,
        data_iterator_settings=config["data_iterator_settings"]["trafo"],
        num_batches=config["data_trafo_settings"]["num_batches"],
        float_precision=config["data_trafo_settings"]["float_precision"],
        norm_constant=config["data_trafo_settings"]["norm_constant"],
    )

    # save trafo model to file
    data_transformer.save(model_dir, overwrite=True)

    # kill multiprocessing queues and workers
    data_handler.kill()

    logger = logging.getLogger(__name__)
    logger.info("\n=======================================")
    logger.info("= Successfully saved trafo model to:  =")
    logger.info("=======================================")
    logger.info("{!r}\n".format(model_dir))


if __name__ == "__main__":
    main()

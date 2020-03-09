#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import logging
import click
import ruamel.yaml as yaml

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.data.trafo import DataTransformer


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
def main(config_files):
    """Script to generate trafo model.

    Creates the desired trafo model as defined in the yaml configuration files
    and saves the trafo model to disc.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    """

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # Check if path already exists
    if os.path.exists(config['data_trafo_settings']['model_path']):
        if not click.confirm(
                'File already exists at destination. Overwrite {!r}?'.format(
                    config['data_trafo_settings']['model_path']),
                default=False):
            raise ValueError('Aborting!')

    # Create Data Handler object
    DataHandlerClass = misc.load_class('egenerator.data.handler.{}'.format(
                            config['data_handler_settings']['data_handler']))
    data_handler = DataHandlerClass()
    data_handler.setup(config['data_handler_settings'])

    trafo_data_generator = data_handler.get_batch_generator(
        **config['data_iterator_settings']['trafo'])

    # create TrafoModel
    data_transformer = DataTransformer(data_handler=data_handler)

    data_transformer.create_trafo_model_iteratively(
        data_iterator=trafo_data_generator,
        num_batches=config['data_trafo_settings']['num_batches'],
        float_precision=config['data_trafo_settings']['float_precision'],
        norm_constant=config['data_trafo_settings']['norm_constant'])

    # save trafo model to file
    data_transformer.save_trafo_model(
        config['data_trafo_settings']['model_path'], overwrite=True)

    # kill multiprocessing queues and workers
    data_handler.kill()

    logger = logging.getLogger(__name__)
    logger.info('\n=======================================')
    logger.info('= Successfully saved trafo model to:  =')
    logger.info('=======================================')
    logger.info('{!r}\n'.format(config['data_trafo_settings']['model_path']))


if __name__ == '__main__':
    main()

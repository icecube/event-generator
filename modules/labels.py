#!/usr/bin/env python
# -*- coding: utf-8 -*
from __future__ import print_function, division
from copy import deepcopy

from ic3_labels.labels.utils import general


def add_mc_energy_loss(tray, cfg, name='get_significant_energy_loss'):
    """Add most significant MC energy loss based on a pulse series.

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray module.
    """
    if 'mc_energy_loss_config' in cfg:
        loss_config = dict(deepcopy(cfg['mc_energy_loss_config']))

        if 'pulse_key' not in loss_config:
            loss_config['pulse_key'] = 'InIceDSTPulses'

        if 'output_key' not in loss_config:
            loss_config['output_key'] = 'MCEnergyLoss'

        def get_significant_energy_loss(frame, pulse_key='InIceDSTPulses',
                                        output_key='MCEnergyLoss'):
            if output_key in frame:
                del frame[output_key]
            frame[output_key] = \
                general.get_significant_energy_loss(frame, pulse_key)
        tray.AddModule(get_significant_energy_loss, name,
                       pulse_key=loss_config['pulse_key'],
                       output_key=loss_config['output_key'])

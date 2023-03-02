from copy import deepcopy


def bias_mesc_hotspot_muons(tray, cfg, name='BiasedMESCHotspotWeighter'):
    """Bias Corridor MuonGun Events.

    Config setup:

        ApplyBiasedMESCHotspotWeighter: True
        BiasedMESCHotspotWeighterConfig: {
            hotspots: , [optional]
            sigmoid_scaling: , [optional]
            sigmoid_offset: , [optional]
            lower_probability_bound: , [optional]
            keep_all_events: , [optional]
        }

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray segment.
    """
    if ('ApplyBiasedMESCHotspotWeighter' in cfg and
            cfg['ApplyBiasedMESCHotspotWeighter']):

        from egenerator.addons.muon_bias.mesc_hotspot import (
            BiasedMESCHotspotWeighter
        )
        bias_cfg = deepcopy(cfg['BiasedMESCHotspotWeighterConfig'])

        if 'output_key' in bias_cfg:
            output_key = bias_cfg.pop('output_key')
        else:
            output_key = name

        tray.AddModule(
            BiasedMESCHotspotWeighter, name,
            output_key=output_key,
            mc_tree_name=mc_tree_name,
            **bias_cfg
        )


def bias_corridor_muons(tray, cfg, name='BiasedMuonCorridorWeighter'):
    """Bias Corridor MuonGun Events.

    Config setup:

        ApplyBiasedMuonCorridorWeighter: True
        BiasedMuonCorridorWeighterConfig: {
            sigmoid_scaling: , [optional]
            sigmoid_offset: , [optional]
            lower_probability_bound: , [optional]
            bad_doms_key: , [optional]
            track_step_size: , [optional]
            keep_all_events: , [optional]
        }

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray segment.
    """
    if ('ApplyBiasedMuonCorridorWeighter' in cfg and
            cfg['ApplyBiasedMuonCorridorWeighter']):

        from egenerator.addons.muon_bias.corridor_weighter import (
            BiasedMuonCorridorWeighter
        )
        bias_cfg = deepcopy(cfg['BiasedMuonCorridorWeighterConfig'])

        if 'output_key' in bias_cfg:
            output_key = bias_cfg.pop('output_key')
        else:
            output_key = name

        tray.AddModule(
            BiasedMuonCorridorWeighter, name,
            output_key=output_key,
            **bias_cfg
        )


def bias_muongun_events(tray, cfg, name='BiasedMuonWeighter'):
    """Bias MuonGun Events.

    Config setup:

        ApplyBiasedMuonGun: True
        BiasedMuonGunConfig: {
            bias_function: 'dummy',
            model_name: 'cascade_7param_charge_only_BFRv1Spice321_01',
            bias_function_settings: {},
            model_base_dir: , [optional]
            save_debug_info: , [optional]
            bad_doms_key: , [optional]
            track_bin_size: , [optional]
            keep_all_events: , [optional]
        }

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray segment.
    """
    if 'ApplyBiasedMuonGun' in cfg and cfg['ApplyBiasedMuonGun']:

        from egenerator.addons.muon_bias.weighter import BiasedMuonWeighter
        from . import bias_functions

        bias_cfg = deepcopy(cfg['BiasedMuonGunConfig'])

        if 'output_key' in bias_cfg:
            output_key = bias_cfg.pop('output_key')
        else:
            output_key = name

        if output_key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(output_key)

        bias_function_name = bias_cfg.pop('bias_function')
        bias_function_settings = bias_cfg.pop('bias_function_settings')
        bias_function = getattr(bias_functions, bias_function_name)(
            **bias_function_settings)

        tray.AddModule(
            BiasedMuonWeighter, name,
            bias_function=bias_function,
            output_key=output_key,
            **bias_cfg
        )

# Since this is just an ugly collection of scripts rather than a proper
# python package, we need to manually add the directories of imports to PATH
import sys
sys.path.insert(0, '../../..')

import numpy as np
import healpy as hp
import re
from copy import deepcopy

from icecube import icetray, dataclasses
from icecube.icetray.i3logging import log_warn

from modules.reco import apply_event_generator_reco
from modules.utils.combine_exclusions import get_combined_exclusions
from modules.utils.combine_i3_particle import add_combined_i3_particle


def get_direction_seeds(azimuth_seed, zenith_seed, add_reverse=True, nside=1):
    """Get a list of direction seeds

    Parameters
    ----------
    azimuth_seed : float
        The original azimuth seed.
        This will be appended to the seed list.
    zenith_seed : float
        The original zenith seed.
        This will be appended to the seed list.
    add_reverse : bool, optional
        If true, the reverse direction of the provided direction seed will be
        added.
    nside : int, optional
        The nside parameter of the healpy pixels to add. If None, no seeds
        based on healpy directions are added.

    Returns
    -------
    list of float
        The list of zenith seeds.
    list of float
        The list of azimuth seeds.
    list of str
        The list of seed names.
    """
    azimuth_seeds = [azimuth_seed]
    zenith_seeds = [zenith_seed]
    seed_names = ['original']

    if nside is not None:
        n_pixels = hp.nside2npix(nside)

        for ipix in range(n_pixels):
            theta, phi = hp.pix2ang(nside, ipix=ipix)

            zenith_seeds.append(theta)
            azimuth_seeds.append(phi)
            seed_names.append('healpix_{:04d}_{:08d}'.format(nside, ipix))

    if add_reverse:
        i3_dir_rev = -dataclasses.I3Direction(zenith_seed, azimuth_seed)

        zenith_seeds.append(i3_dir_rev.zenith)
        azimuth_seeds.append(i3_dir_rev.azimuth)
        seed_names.append('reverse')

    return zenith_seeds, azimuth_seeds, seed_names


def get_seed_map(base_seed_obj):
    """Get I3MapStringDouble from a frame seed object

    Parameters
    ----------
    base_seed_obj : I3Particle or I3MapStringDouble
        The seed object from the I3Frame. This is either the particle
        or a I3MapStringDouble containing the seeds.

    Raises
    ------
    ValueError
        Description
    """
    if isinstance(base_seed_obj, dataclasses.I3Particle):
        base_seed_map = dataclasses.I3MapStringDouble({
            'x': base_seed_obj.pos.x,
            'y': base_seed_obj.pos.y,
            'z': base_seed_obj.pos.z,
            'zenith': base_seed_obj.dir.zenith,
            'azimuth': base_seed_obj.dir.azimuth,
            'energy': base_seed_obj.energy,
            'time': base_seed_obj.time,
        })

    elif isinstance(base_seed_obj, dataclasses.I3MapStringDouble):
        base_seed_map = dataclasses.I3MapStringDouble()
        for key in ['x', 'y', 'z', 'zenith', 'azimuth', 'energy', 'time']:
            if key in base_seed_obj:
                base_seed_map[key] = base_seed_obj[key]
            else:
                base_seed_map[key] = base_seed_obj['cascade_'+key]

    else:
        raise ValueError('Unknown seed type:', base_seed_obj)

    return base_seed_map


def add_cascade_model_seeds(
        frame, seed_base, add_reverse=True, nside=1,
        min_energy=None, prefix=''):
    """Add a list of I3MapStringDouble seeds to the frame

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame to which the seed particles will be aded.
    seed_base : str
        The name of the base seed particle. This particle will be used as a
        basis from which the seeds will be constructed. This base seed will
        also be included in the added seed particles.
    add_reverse : bool, optional
        If true, the reverse direction of the provided direction seed will be
        added.
    nside : int, optional
        The nside parameter of the healpy pixels to add. If None, no seeds
        based on healpy directions are added.
    min_energy : float, optional
        The minimal seed energy. Energies below this value will be set to this
        threshold instead.
    prefix : str, optional
        An optional prefix to the frame keys can be provided.

    Returns
    -------
    list of str
        The list of added I3Particles.

    No Longer Raises
    ----------------
    ValueError
        Description

    """
    base_seed_obj = frame[seed_base]
    base_seed_map = get_seed_map(base_seed_obj)

    azimuth_seed = base_seed_map['azimuth']
    zenith_seed = base_seed_map['zenith']

    zenith_seeds, azimuth_seeds, seed_names = get_direction_seeds(
        azimuth_seed=azimuth_seed,
        zenith_seed=zenith_seed,
        add_reverse=add_reverse,
        nside=nside,
    )

    # add seed particles
    added_seed_names = []
    for zen, azi, name in zip(zenith_seeds, azimuth_seeds, seed_names):
        new_seed_map = dataclasses.I3MapStringDouble(base_seed_map)
        new_seed_map['zenith'] = zen
        new_seed_map['azimuth'] = azi
        if min_energy is not None:
            new_seed_map['energy'] = max(new_seed_map['energy'], min_energy)

        seed_name = prefix + seed_base + '_' + name
        frame[seed_name] = new_seed_map
        added_seed_names.append(seed_name)

    return added_seed_names


def add_2_cascade_model_seeds(
        frame, seed_base, seed_distances,
        additional_seeds=[],
        add_reverse=True, nside=1, cluster_settings=None, min_energy=None,
        prefix=''):
    """Add a list of I3MapStringDouble seeds to the frame

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame to which the seed particles will be aded.
    seed_base : str
        The name of the base seed particle. This particle will be used as a
        basis from which the seeds will be constructed. This base seed will
        also be included in the added seed particles.
    seed_distances : list of float
        A list of seed distances
    additional_seeds : list of str, optional
        Additional seeds for which to create seeds for each specified distance.
    add_reverse : bool, optional
        If true, the reverse direction of the provided direction seed will be
        added.
    nside : int, optional
        The nside parameter of the healpy pixels to add. If None, no seeds
        based on healpy directions are added.
    cluster_settings : dict, optional
        If provided, charge clusters will be computed and also used as seeds.
        'cluster_settings': {
            'pulse_key': 'SplitInIceDSTPulses',
            'n_clusters': 5,
            'min_dist': 200,
            'min_cluster_charge': 3,
            'min_hit_doms': 3,
        },
    min_energy : float, optional
        The minimal seed energy. Energies below this value will be set to this
        threshold instead.
    prefix : str, optional
        An optional prefix to the frame keys can be provided.

    Returns
    -------
    list of str
        The list of added I3Particles.

    No Longer Raises
    ----------------
    ValueError
        Description
    """
    from egenerator.addons.multi_cascade_seed import CascadeClusterSearchModule

    seed_names = add_cascade_model_seeds(
        frame=frame,
        seed_base=seed_base,
        add_reverse=add_reverse,
        nside=nside,
        min_energy=min_energy,
        prefix=prefix,
    )
    if additional_seeds is not None:
        seed_names += additional_seeds

    # add seeds for distances of 2nd cascade
    added_seed_names = []
    for seed_name in seed_names:

        seed_map = get_seed_map(frame[seed_name])

        for distance in seed_distances:
            new_seed_map = dataclasses.I3MapStringDouble(seed_map)
            new_seed_map['cascade_00001_distance'] = distance
            new_seed_map['cascade_00001_energy'] = seed_map['energy'] * 0.2
            new_seed_map['energy'] = seed_map['energy'] * 0.8

            new_seed_name = seed_name + '_dist_{:04.0f}'.format(distance)
            frame[new_seed_name] = new_seed_map
            added_seed_names.append(new_seed_name)

        if additional_seeds is None or seed_name not in additional_seeds:
            del frame[seed_name]

    if cluster_settings is not None:
        cluster_seeds = [seed_base]
        if additional_seeds is not None:
            cluster_seeds += additional_seeds
        for cluster_seed in cluster_seeds:

            seed_map = get_seed_map(frame[cluster_seed])

            initial_clusters = [[
                seed_map['x'], seed_map['y'], seed_map['z'], seed_map['time']
            ]]

            pulse_series = frame[cluster_settings['pulse_key']]
            if isinstance(
                    pulse_series,
                    (dataclasses.I3RecoPulseSeriesMapMask,
                     dataclasses.I3RecoPulseSeriesMapUnion)):
                pulse_series = pulse_series.apply(frame)

            # compute clusters based off of seed
            n_found, clusters = CascadeClusterSearchModule.calculate_clusters(
                pulse_series=pulse_series,
                n_clusters=cluster_settings['n_clusters'],
                min_dist=cluster_settings['min_dist'],
                min_cluster_charge=cluster_settings['min_cluster_charge'],
                min_hit_doms=cluster_settings['min_hit_doms'],
                initial_clusters=initial_clusters,
            )
            nan_distances = np.linspace(
                -cluster_settings['min_dist'] / 2,
                cluster_settings['min_dist'] / 2,
                cluster_settings['n_clusters'] - n_found + 1)
            nan_counter = 0
            for i, cluster in enumerate(clusters[1:]):

                new_seed_map = dataclasses.I3MapStringDouble(seed_map)

                if np.isfinite(cluster).all():

                    # compute direction and distance
                    dx = cluster[0] - seed_map['x']
                    dy = cluster[1] - seed_map['y']
                    dz = cluster[2] - seed_map['z']
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    direction = dataclasses.I3Direction(dx, dy, dz)

                    new_seed_map['zenith'] = direction.zenith
                    new_seed_map['azimuth'] = direction.azimuth
                    new_seed_map['cascade_00001_distance'] = distance
                    new_seed_map['cascade_00001_energy'] = (
                        seed_map['energy'] * 0.2)
                    new_seed_map['energy'] = seed_map['energy'] * 0.8

                else:
                    # did not find a cluster, so lets pick a distance instead
                    # since the reco module expects a constant number of seeds
                    distance = nan_distances[nan_counter]
                    nan_counter += 1
                    new_seed_map['cascade_00001_distance'] = distance
                    new_seed_map['cascade_00001_energy'] = (
                        seed_map['energy'] * 0.2)
                    new_seed_map['energy'] = seed_map['energy'] * 0.8

                new_seed_name = cluster_seed + '_cluster_{:03d}'.format(i)
                frame[new_seed_name] = new_seed_map
                added_seed_names.append(new_seed_name)

    return added_seed_names


def get_circ_unc(frame, reco_name, cov_name='_cov_matrix_cov_sand'):
    """Get the circularized uncertainty estimate from the covariance matrix.

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame
    reco_name : str
        The name of the event-generator reco.
    cov_name : str
        The name of the covariance matrix to use.

    Returns
    -------
    float
        The circularized uncertainty estimate.
    """
    zenith = frame[reco_name + '_I3Particle'].dir.zenith
    var_azimuth = frame[reco_name+cov_name]['cascade_azimuth_cascade_azimuth']
    var_zenith = frame[reco_name+cov_name]['cascade_zenith_cascade_zenith']
    unc = np.sqrt(
        (var_zenith + (np.sin(zenith)**2)*var_azimuth) / 2.
    )
    return unc


def select_best_reco(frame, reco_names, output_key):
    """Select Best Reconstruction result

    Parameters
    ----------
    frame : I3Frame
        The I3Frame which contains the recos.
    reco_names : list of str
        The list of reconstruction base strings from which the best
        reconstruction will be chosen.
    """
    min_loss = float('inf')
    best_reco = None

    for reco_name in reco_names:
        loss = frame[reco_name]['loss_reco']

        print('Reco Loss: {:3.3f} | {}'.format(loss, reco_name))

        # check if it has a lower loss
        if loss < min_loss:
            min_loss = loss
            best_reco = reco_name

    if best_reco is None:

        # something went wrong, did the fit return NaN or inf loss?
        # For now: just choose first reco
        best_reco = reco_names[0]
        log_warn('No best reco found, choosing first reco: {}!'.format(
            best_reco))

    # collect reco
    frame[output_key + 'KeyName'] = dataclasses.I3String(best_reco)
    # frame[output_key] = dataclasses.I3MapStringDouble(frame[best_reco])

    # collect I3Particle
    if best_reco + '_I3Particle' in frame:
        frame[output_key + '_I3Particle'] = dataclasses.I3Particle(
            frame[best_reco + '_I3Particle'])

    # collect covariance matrices and circularized error
    cov_matrices = [
        '_cov_matrix_cov',
        '_cov_matrix_cov_fit',
        '_cov_matrix_cov_fit_trafo',
        '_cov_matrix_cov_sand',
        '_cov_matrix_cov_sand_fit',
        '_cov_matrix_cov_sand_fit_trafo',
        '_cov_matrix_cov_sand_trafo',
        '_cov_matrix_cov_trafo',
    ]
    for suffix in cov_matrices:

        # if best_reco + suffix in frame:
        #     frame[output_key + suffix] = dataclasses.I3MapStringDouble(
        #         frame[best_reco + suffix])

        if best_reco + suffix + '_circular_unc' in frame:
            frame[output_key + suffix + '_circular_unc'] = (
                dataclasses.I3Double(
                    frame[best_reco + suffix + '_circular_unc'])
            )


def run_egen_1_cascade_reco(tray, cfg, name='EGen1CascadeReco'):
    """Run EventGenerator 1-Cascade Model Reconstruction

    Config setup:

        EGen_1Cascade_Reco_config: {
            'seed_base': 'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01__bfgs_gtol_10',
            'seed_settings': {
                'additional_seeds': ['event_selection_cascade'],
                'add_reverse': False,
                'nside': 1,
                'min_energy': ,
            },
            'add_circular_err': False,
            'add_covariances': True,
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

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    reco_settings = cfg_copy['EGen_1Cascade_Reco_config']
    seed_base = reco_settings['seed_base']
    seed_settings = reco_settings['seed_settings']

    # get a list of seed keys (call with dummy data)
    _, _, seed_names = get_direction_seeds(
        azimuth_seed=2.1,
        zenith_seed=1.0,
        add_reverse=seed_settings['add_reverse'],
        nside=seed_settings['nside'],
    )
    new_seed_keys = [seed_base + '_' + n for n in seed_names]

    if seed_settings['additional_seeds'] is not None:
        seed_keys = new_seed_keys + seed_settings['additional_seeds']
    else:
        seed_keys = new_seed_keys

    # add keys to frame
    def add_seed_keys(frame):
        add_cascade_model_seeds(
            frame=frame,
            seed_base=seed_base,
            add_reverse=seed_settings['add_reverse'],
            nside=seed_settings['nside'],
            min_energy=seed_settings['min_energy'],
        )

    tray.AddModule(add_seed_keys, name+'add_seed_keys')

    if 'output_key' in reco_settings:
        output_key = reco_settings['output_key']
    else:
        output_key = 'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01'

    # update config
    cfg_copy.update({
        'add_egenerator_reco': True,
        'egenerator_configs': [
            {
                'seed_keys': seed_keys,
                'output_key': output_key,
                'model_names': 'cascade_7param_noise_tw_BFRv1Spice321_01',
                'model_base_dir':
                    '/data/user/mhuennefeld/exported_models/egenerator',
                'pulse_key': 'SplitInIceDSTPulses',
                'dom_exclusions_key': [
                    'SaturationWindows', 'BadDomsList', 'CalibrationErrata'],
                'partial_exclusion': True,
                'add_circular_err': reco_settings['add_circular_err'],
                'add_covariances': reco_settings['add_covariances'],
                'add_goodness_of_fit': False,
                'num_threads': 1,
                'scipy_optimizer_settings': {
                },
            },
        ]
    })

    # run event-generator
    apply_event_generator_reco(tray, cfg_copy, name)

    # Add Event-Generator circularized error
    def add_circ_unc(frame):
        cov_name = '_cov_matrix_cov_sand'
        circ_unc = get_circ_unc(frame, reco_name=output_key, cov_name=cov_name)
        frame[output_key + cov_name + '_circular_unc'] = dataclasses.I3Double(
            float(circ_unc))

    if reco_settings['add_covariances']:
        tray.AddModule(add_circ_unc, name+'add_circ_unc')

    # clean up seed keys
    tray.AddModule('Delete', name+'Cleanup', Keys=new_seed_keys)

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)


def run_egen_2_cascade_reco(tray, cfg, name='EGen2CascadeReco'):
    """Run EventGenerator 2-Cascade Model Reconstruction

    Config setup:

        EGen_2Cascade_Reco_config: {
            'seed_base': 'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01',
            'seed_settings': {
                'seed_distances': [5, 50, 300],
                'additional_seeds': ['event_selection_cascade'],
                'add_reverse': False,
                'nside': 1,
                'min_energy': ,
                'cluster_settings': {
                    'pulse_key': 'SplitInIceDSTPulses',
                    'n_clusters': 5,
                    'min_dist': 200,
                    'min_cluster_charge': 3,
                    'min_hit_doms': 3,
                },
            },
            'add_circular_err': False,
            'add_covariances': True,
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

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    reco_settings = cfg_copy['EGen_2Cascade_Reco_config']
    seed_base = reco_settings['seed_base']
    seed_settings = reco_settings['seed_settings']

    # get a list of seed keys (call with dummy data)
    _, _, seed_names = get_direction_seeds(
        azimuth_seed=2.1,
        zenith_seed=1.0,
        add_reverse=seed_settings['add_reverse'],
        nside=seed_settings['nside'],
    )
    new_seed_keys = [seed_base + '_' + n for n in seed_names]
    seed_keys = []
    for seed_name in new_seed_keys + seed_settings['additional_seeds']:
        for distance in seed_settings['seed_distances']:
            seed_keys.append(seed_name + '_dist_{:04.0f}'.format(distance))

    if seed_settings['cluster_settings'] is not None:
        cluster_seeds = [seed_base] + seed_settings['additional_seeds']
        for cluster_seed in cluster_seeds:
            seed_keys += [
                cluster_seed + '_cluster_{:03d}'.format(i) for i in
                range(seed_settings['cluster_settings']['n_clusters'] - 1)]

    # add keys to frame
    def add_seed_keys(frame):
        add_2_cascade_model_seeds(
            frame=frame,
            seed_base=seed_base,
            additional_seeds=seed_settings['additional_seeds'],
            seed_distances=seed_settings['seed_distances'],
            add_reverse=seed_settings['add_reverse'],
            nside=seed_settings['nside'],
            min_energy=seed_settings['min_energy'],
            cluster_settings=seed_settings['cluster_settings'],
        )

    tray.AddModule(add_seed_keys, name+'add_seed_keys')

    if 'output_key' in reco_settings:
        output_key = reco_settings['output_key']
    else:
        output_key = (
            'EventGenerator_starting_multi_cascade_7param_noise_tw_'
            'BFRv1Spice321_low_mem_n002_01'
        )

    model_names = (
        'starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002_01'
    )

    # update config
    cfg_copy.update({
        'add_egenerator_reco': True,
        'egenerator_configs': [
            {
                'seed_keys': seed_keys,
                'output_key': output_key,
                'model_names': model_names,
                'model_base_dir':
                    '/data/user/mhuennefeld/exported_models/egenerator',
                'pulse_key': 'SplitInIceDSTPulses',
                'dom_exclusions_key': [
                    'SaturationWindows', 'BadDomsList', 'CalibrationErrata'],
                'partial_exclusion': True,
                'add_circular_err': reco_settings['add_circular_err'],
                'add_covariances': reco_settings['add_covariances'],
                'add_goodness_of_fit': False,
                'num_threads': 1,
                'scipy_optimizer_settings': {
                },
            },
        ]
    })

    # run event-generator
    apply_event_generator_reco(tray, cfg_copy, name)

    # Add Event-Generator circularized error
    def add_circ_unc(frame):
        cov_name = '_cov_matrix_cov_sand'
        circ_unc = get_circ_unc(frame, reco_name=output_key, cov_name=cov_name)
        frame[output_key + cov_name + '_circular_unc'] = dataclasses.I3Double(
            float(circ_unc))

    if reco_settings['add_covariances']:
        tray.AddModule(add_circ_unc, name+'add_circ_unc')

    # clean up seed keys
    tray.AddModule('Delete', name+'Cleanup', Keys=seed_keys)

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)


def run_egen_select_reco(tray, cfg, name='EGenSelectReco'):
    """Select best Event-Generator reco based on its Likelihood value

    Config setup:

        EGen_Select_Reco_config: {
            'reco_base_names': [
                'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01_unc_patch',
                'EventGenerator_starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002_01',
            ]
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

    sel_settings = cfg['EGen_Select_Reco_config']

    if 'output_key' in sel_settings:
        output_key = sel_settings['output_key']
    else:
        output_key = 'EventGeneratorSelectedReco'

    tray.AddModule(
        select_best_reco, name,
        reco_names=sel_settings['reco_base_names'],
        output_key=output_key,
    )

    # update HDF5 output keys
    if output_key not in cfg['HDF_keys']:
        cfg['HDF_keys'].append(output_key)


def run_egen_cascade_recos(tray, cfg, name='EGenCascadeRecos'):
    """Run EventGenerator Cascade Reconstructions

    Config setup:

        apply_egen_cascade_recos: True,
        EGen_1Cascade_Reco_config: {
            'seed_base': 'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01__bfgs_gtol_10',
            'seed_settings': {
                'additional_seeds': ['event_selection_cascade'],
                'add_reverse': False,
                'nside': 1,
                'min_energy': ,
            },
            'add_circular_err': False,
            'add_covariances': True,
        }

        EGen_2Cascade_Reco_config: {
            'seed_base': 'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01',
            'seed_settings': {
                'seed_distances': [5, 50, 300],
                'additional_seeds': ['event_selection_cascade'],
                'add_reverse': False,
                'nside': ,
                'min_energy': ,
                'cluster_settings': {
                    'pulse_key': 'SplitInIceDSTPulses',
                    'n_clusters': 5,
                    'min_dist': 200,
                    'min_cluster_charge': 3,
                    'min_hit_doms': 3,
                },
            },
            'add_circular_err': False,
            'add_covariances': True,
        }

        EGen_Select_Reco_config: {
            'reco_base_names': [
                'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01',
                'EventGenerator_starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002_01',
            ]
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

    if 'EGen_1Cascade_Reco_config' in cfg:
        run_egen_1_cascade_reco(tray, cfg, name=name+'EGen1CascadeReco')

    if 'EGen_2Cascade_Reco_config' in cfg:
        run_egen_2_cascade_reco(tray, cfg, name=name+'EGen2CascadeReco')

    if 'EGen_Select_Reco_config' in cfg:
        run_egen_select_reco(tray, cfg, name=name+'EGenSelectReco')

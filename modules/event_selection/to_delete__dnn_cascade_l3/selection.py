# Since this is just an ugly collection of scripts rather than a proper
# python package, we need to manually add the directories of imports to PATH
import sys
sys.path.insert(0, '../../..')

import numpy as np
import re
from copy import deepcopy

from icecube import icetray, dataclasses

from .selection_utils import discard_events_not_passing_l2_cascade_filter
from modules.reco import apply_dnn_reco, apply_event_generator_reco, apply_bdts
from modules.cscdSBU.cscdSBU_vars import addvars
from modules.utils.combine_exclusions import get_combined_exclusions
from modules.utils.combine_i3_particle import add_combined_i3_particle
from modules.misc import create_cascade_classification_base_cascades
from modules.cscd_l3.cscd_l3_cuts import CascadeL3Cuts


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
    var_azimuth = frame[reco_name + cov_name]['azimuth_azimuth']
    var_zenith = frame[reco_name + cov_name]['zenith_zenith']
    unc = np.sqrt(
        (var_zenith +
         (np.sin(zenith)**2)*var_azimuth) / 2.
    )
    return unc


def filter_streams(frame, sub_event_streams):
    """Filter Events based on the SubEventStream

    Parameters
    ----------
    frame : I3Frame
        The current I3Frame
    sub_event_streams : list of str
        The list of sub event streams to accept.

    Returns
    -------
    bool
        True, if event is kept, False if not.
    """
    if frame.Stop == icetray.I3Frame.Physics:
        if frame['I3EventHeader'].sub_event_stream not in sub_event_streams:
            return False
    return True


@icetray.traysegment
def apply_dnn_cuts(tray, name, cut_dict):
    """Apply Cuts.

    Parameters
    ----------
    tray : I3Tray
        The I3Tray object.
    name : str
        Name of the tray segment to add
    cut_dict : dict
        Dictionary defining cuts. Content must be of format:
        {
            'frame_key': (col_name, cut_value),
        }
    """
    def apply_cuts(frame, cut_dict):
        for reco, (col, cut) in cut_dict.items():
            if frame[reco][col] < cut:
                return False
        return True

    tray.AddModule(apply_cuts, name+'apply_cut', cut_dict=cut_dict)
    tray.AddModule('I3OrphanQDropper', name+'drop_orphan_q_frames')


def run_dnn_cascade_level3(tray, cfg, name='DNNCascadeLevel3',
                           sub_event_streams=['InIceSplit']):
    """Run the DNN Cascade Level 3 selection

    Config setup:

        'DNN_Cascade_Level3_config': {

            'precut_DNN': {
                'pulse_map_string': 'SplitInIceDSTPulses',
                'excluded_doms': [
                    'SaturationWindows', 'BadDomsList', 'CalibrationErrata',
                ],
                'partial_exclusion': True,
                'models_dir': '/mnt/lfs7/user/mhuennefeld/DNN_reco/models/exported_models',
                'ignore_misconfigured_settings_list': ['pulse_key'],
            },

            'basic_DNN': {
                'pulse_map_string': 'SplitInIceDSTPulses',
                'excluded_doms': [
                    'BrightDOMs', 'SaturationWindows',
                    'BadDomsList', 'CalibrationErrata',
                ],
                'partial_exclusion': True,
                'models_dir': '/mnt/lfs7/user/mhuennefeld/DNN_reco/models/exported_models',
                'ignore_misconfigured_settings_list': ['pulse_key'],
            },

            'cascade_DNN': {
                'pulse_map_string': 'SplitInIceDSTPulses',
                'excluded_doms': [
                    'SaturationWindows', 'BadDomsList', 'CalibrationErrata'],
                'partial_exclusion': True,
                'models_dir': '/mnt/lfs7/user/mhuennefeld/DNN_reco/models/exported_models',
                'ignore_misconfigured_settings_list': ['pulse_key'],
            },

            'egenerator_fast': {
                'excluded_doms': [
                    'BadDomsList', 'CalibrationErrata', 'SaturationWindows'],
            },
        }

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray segment.
    sub_event_streams : list, optional
        The streams on which to run. Physics frames of other streams are
        discarded. Pass None run on all frames.

    Returns
    -------
    TYPE
        Description
    """

    if 'up_to_step' in cfg['DNN_Cascade_Level3_config']:
        up_to_step = cfg['DNN_Cascade_Level3_config']['up_to_step']
    else:
        up_to_step = None

    cfg_dnn_precut = cfg['DNN_Cascade_Level3_config']['precut_DNN']
    cfg_dnn_basic = cfg['DNN_Cascade_Level3_config']['basic_DNN']
    cfg_dnn_cascade = cfg['DNN_Cascade_Level3_config']['cascade_DNN']
    cfg_egen_fast = cfg['DNN_Cascade_Level3_config']['egenerator_fast']

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    # ------------------------------------------
    # Remove events from other sub event streams
    # ------------------------------------------
    if sub_event_streams is not None:
        tray.AddModule(filter_streams, 'filter_streams',
                       sub_event_streams=sub_event_streams)
        tray.AddModule('I3OrphanQDropper', name+'drop_orphan_q_frames_streams')

    # -------------------------------
    # Remove non-CascadeFilter events
    # -------------------------------
    tray.AddModule(discard_events_not_passing_l2_cascade_filter,
                   name + 'discard_events_not_passing_l2_cascade_filter')
    tray.AddModule('I3OrphanQDropper', name+'drop_orphan_q_frames_dnn_cscdl2')

    # ------------------------
    # Run Pre-Selection DNN 01
    # ------------------------

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    # define settings
    cfg_copy.update({
        'add_dnn_reco': True,
        'DNN_batch_size': 64,
        'pulse_map_string': cfg_dnn_precut['pulse_map_string'],
        'DNN_excluded_doms': cfg_dnn_precut['excluded_doms'],
        'DNN_partial_exclusion': cfg_dnn_precut['partial_exclusion'],
        'DNN_models_dir': cfg_dnn_precut['models_dir'],
        'DNN_ignore_misconfigured_settings_list': cfg_dnn_precut[
            'ignore_misconfigured_settings_list'],
    })
    model_names = [
        'event_selection_dnn_cscd_l3a_starting_events_300m'
        '_red_summary_stats_fast_02'
    ]
    cfg_copy['DNN_reco_configs'] = [
        {
            'DNN_model_names': model_names,
        }
    ]

    # run DNN classifiers
    apply_dnn_reco(tray, cfg_copy, name=name+'ApplyDNNRecosPreCut')

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)

    # Apply pre-cut
    key = ('DeepLearningReco_event_selection_dnn_cscd_l3a_'
           'starting_events_300m_red_summary_stats_fast_02')
    cut_dict = {key: ('p_starting_300m', 0.95)}
    tray.AddSegment(apply_dnn_cuts, 'apply_precut_01', cut_dict=cut_dict)

    # -------------------------------
    # Run Pre-Selection DNN 02 and 03
    # -------------------------------

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    # define settings
    cfg_copy.update({
        'add_dnn_reco': True,
        'DNN_batch_size': 64,
        'pulse_map_string': cfg_dnn_precut['pulse_map_string'],
        'DNN_excluded_doms': cfg_dnn_precut['excluded_doms'],
        'DNN_partial_exclusion': cfg_dnn_precut['partial_exclusion'],
        'DNN_models_dir': cfg_dnn_precut['models_dir'],
        'DNN_ignore_misconfigured_settings_list': cfg_dnn_precut[
            'ignore_misconfigured_settings_list'],
    })
    model_names = [
        'event_selection_dnn_cscd_l3a_starting_'
        'events_150m_red_summary_stats_fast_02',
        'event_selection_dnn_cscd_l3a_starting_'
        'cascades_150m_red_summary_stats_fast_02',
    ]
    cfg_copy['DNN_reco_configs'] = [
        {
            'DNN_model_names': model_names,
        }
    ]

    # run DNN classifiers
    apply_dnn_reco(tray, cfg_copy, name=name+'ApplyDNNRecosPreCuts_02')

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)

    # Apply pre-cut
    cut_dict = {
        ('DeepLearningReco_event_selection_dnn_cscd_l3a_starting_'
         'events_150m_red_summary_stats_fast_02'): ('p_starting', 0.95),
        ('DeepLearningReco_event_selection_dnn_cscd_l3a_starting_'
         'cascades_150m_red_summary_stats_fast_02'): (
            'p_starting_cascade_L100_D150', 0.95),
    }
    tray.AddSegment(apply_dnn_cuts, 'apply_precut_02', cut_dict=cut_dict)

    if up_to_step == 0:
        return
    # -------------------------
    # Run Basic DNN Classifiers
    # -------------------------

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    # define settings
    cfg_copy.update({
        'add_dnn_reco': True,
        'DNN_batch_size': 32,
        'pulse_map_string': cfg_dnn_basic['pulse_map_string'],
        'DNN_excluded_doms': cfg_dnn_basic['excluded_doms'],
        'DNN_partial_exclusion': cfg_dnn_basic['partial_exclusion'],
        'DNN_models_dir': cfg_dnn_basic['models_dir'],
        'DNN_ignore_misconfigured_settings_list': cfg_dnn_basic[
            'ignore_misconfigured_settings_list'],
    })
    model_names = [
        # 'mese_v2__all_gl_both2',
        # 'event_selection_track_length_02',
        # 'dnn_export_cscdID',

        # # Basic DNN classifiers
        # 'event_selection_03',
        # 'event_selection_cscdl3_02',
        # 'event_selection_cscdl3_big_03',
        # 'event_selection_cscdl3_300m_01',  # Used for 1. Cut
        # 'event_selection_dnn_cscd_l3a_starting_events_03',  # Used for 1. Cut
        # 'event_selection_dnn_cscd_l3b_starting_events_02',

        # Used for 1. Cut
        'event_selection_dnn_cscd_l3b_cut2_starting_events_300m_fast_medium_01',

        # # newer models on Manuel's MuonGun
        # 'event_selection_dnn_cscd_l3b_starting_cascades_big_kernel_01',
        # 'event_selection_dnn_cscd_l3b_starting_cascades_big_kernel_strict_01',
        # 'event_selection_dnn_cscd_l3b_starting_events_big_kernel_01',
        # 'event_selection_dnn_cscd_l3b_starting_cascades_big_kernel_02',
        # 'event_selection_dnn_cscd_l3b_starting_cascades_big_kernel_strict_02',
        # 'event_selection_dnn_cscd_l3b_starting_events_big_kernel_02',

        # newer models after precuts dnn_cscdl3
        'event_selection_dnn_cscd_l3c_cut2_starting_events_300m_fast_medium_01',
        'event_selection_dnn_cscd_l3c_track_numu_cc_vs_nue_cc_01',
        'event_selection_dnn_cscd_l3c_track_numu_cc_vs_starting_01',

        # Necessary for egenerator seed
        'event_selection_cascade_pos_01',
        'event_selection_cascade_dir_01',
        'event_selection_cascade_time_01',
        'event_selection_cascade_energy_01',
    ]
    cfg_copy['DNN_reco_configs'] = [
        {
            'DNN_model_names': model_names,
        }
    ]

    # run DNN classifiers
    apply_dnn_reco(tray, cfg_copy, name=name+'ApplyDNNRecos')

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)

    # -------------------------------
    # Apply first cuts to dnn_cscdl3a
    # -------------------------------
    cut_dict = {
        ('DeepLearningReco_event_selection_dnn_cscd_l3b_cut2_'
         'starting_events_300m_fast_medium_01'): ('p_starting_300m', 0.90),
    }
    tray.AddSegment(apply_dnn_cuts, 'dnn_cscdl3a_cuts', cut_dict=cut_dict)

    if up_to_step == 1:
        return

    # -------------------
    # Run Fast Egenerator
    # -------------------

    # # get combined DOM exclusions
    # tray.AddModule(
    #     get_combined_exclusions, name+'get_combined_exclusions',
    #     dom_exclusions=cfg_egen_fast['excluded_doms'],
    #     partial_exclusion=False,
    #     output_key=name+'combined_exclusions',
    # )

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    # create a new combined I3Particle from the DNN estimates
    prefix = 'DeepLearningReco_event_selection_cascade_'
    cfg_copy.update({
        'combine_i3_particle_output_name': 'event_selection_cascade',
        'combine_i3_particle': {
            'pos_x_name': prefix + 'pos_01_I3Particle',
            'pos_y_name': prefix + 'pos_01_I3Particle',
            'pos_z_name': prefix + 'pos_01_I3Particle',
            'dir_name': prefix + 'dir_01_I3Particle',
            'time_name': prefix + 'time_01_I3Particle',
            'energy_name': prefix + 'energy_01_I3Particle',
        },
    })
    add_combined_i3_particle(tray, cfg_copy, name + 'CombinedI3Particle')

    # update config
    cfg_copy.update({
        'add_egenerator_reco': True,
        'egenerator_configs': [
            {
                'seed_keys': [
                    'event_selection_cascade',
                ],
                'output_key': 'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01__bfgs_gtol_10',
                'model_names': 'cascade_7param_noise_tw_BFRv1Spice321_01',
                'model_base_dir':
                    '/data/user/mhuennefeld/exported_models/egenerator',
                'pulse_key': 'SplitInIceDSTPulses',
                'dom_exclusions_key': [
                    'SaturationWindows', 'BadDomsList', 'CalibrationErrata'],
                'partial_exclusion': True,
                'add_circular_err': False,
                'add_covariances': False,
                'add_goodness_of_fit': False,
                'num_threads': 1,
                'scipy_optimizer_settings': {
                    'options': {'gtol': 10},
                },
            },
        ]
    })

    # run event-generator
    apply_event_generator_reco(tray, cfg_copy, name + 'FastReco')

    # # Add Event-Generator circularized error
    # def add_circ_unc(frame):
    #     reco_name = 'EventGenerator_cascade_7param_BFRv1Spice321__small_01'
    #     cov_name = '_cov_matrix_cov_sand'
    #     circ_unc = get_circ_unc(frame, reco_name=reco_name, cov_name=cov_name)
    #     frame[reco_name + cov_name + '_circular_unc'] = dataclasses.I3Double(
    #         float(circ_unc))

    # tray.AddModule(add_circ_unc, name+'add_circ_unc')

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)

    if up_to_step == 2:
        return

    # ---------------------------------
    # Run Cascade-Based DNN Classifiers
    # ---------------------------------

    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    egen_name = ('EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01__'
                 'bfgs_gtol_10_I3Particle')

    # Add cascade classification model base cascade
    cfg_copy.update({
        'create_cascade_classification_base_cascades': [
            {
                'I3ParticleBase': egen_name,
                'VertexX_unc': 5,
                'VertexY_unc': 5,
                'VertexZ_unc': 5,
                'VertexTime_unc': 15,
            },
        ]
    })
    create_cascade_classification_base_cascades(
        tray, cfg_copy, name+'BaseCascade')

    # define settings
    cfg_copy.update({
        'add_dnn_reco': True,
        'DNN_batch_size': 32,
        'DNN_cascade_key': 'cscd_classification_base_' + egen_name,
        'pulse_map_string': cfg_dnn_cascade['pulse_map_string'],
        'DNN_excluded_doms': cfg_dnn_cascade['excluded_doms'],
        'DNN_partial_exclusion': cfg_dnn_cascade['partial_exclusion'],
        'DNN_models_dir': cfg_dnn_cascade['models_dir'],
        'DNN_ignore_misconfigured_settings_list': cfg_dnn_cascade[
            'ignore_misconfigured_settings_list'],
    })
    model_names = [
        'event_selection_egen_vertex_starting_events_300m_fast_medium_01',
        'event_selection_egen_vertex_starting_nue_300m_fast_medium_01',
        'event_selection_egen_vertex_track_numu_cc_vs_nue_cc_01',
        'event_selection_egen_vertex_track_numu_cc_vs_starting_01',
    ]
    cfg_copy['DNN_reco_configs'] = [
        {
            'DNN_model_names': model_names,
        }
    ]

    # run DNN classifiers
    apply_dnn_reco(tray, cfg_copy, name=name+'ApplyCascadeBasedDNNRecos')

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)

    if up_to_step == 3:
        return

    # --------------------------
    # Apply DNN Cascade L3b Cuts
    # --------------------------
    pass

    # # ---------------------------------------
    # # Add CscdSBU variables (fast egenerator)
    # # ---------------------------------------
    # tray.AddSegment(addvars, name+'SBU_vars_' + egen_name,
    #                 pulses=cfg_dnn_cascade['pulse_map_string'],
    #                 vertex=egen_name)

    # cfg['HDF_keys'].extend([
    #     'cscdSBU_I3Double_' + egen_name,
    #     'cscdSBU_I3Bool_' + egen_name,
    #     'cscdSBU_I3Int_' + egen_name,
    # ])

    # # ----------------
    # # Add Cscd L3 cuts
    # # ----------------

    # if 'year' in cfg:
    #     year = re.findall(r'\d+', str(cfg['year']))[-1]
    # else:
    #     year = '2013'
    # tray.AddSegment(CascadeL3Cuts, name + 'CascadeL3Cuts',
    #                 year=year, discard_non_l2=False,
    #                 discard_non_cscdl2=False,
    #                 discard_non_cscdl3=False)

    # ------------------
    # Add Selection BDTs
    # ------------------
    # make a copy of the cfg to not mess anything up
    cfg_copy = deepcopy(cfg)

    # define BDTs to add
    bdt_configs = [
        {
            'model_path': (
                '/data/user/ssclafani/data/cscd/final_bdt/final_reduced_bdt/'
                'seed_3/corsika/bdt/n_cv_0/bdt_max_depth_4_n_est_2000lr_0_02_'
                'seed_3_train_size_50'
            ),
            'batch_size': 32,
        },
    ]

    # define settings
    cfg_copy.update({
        'add_bdt': True,
        'bdt_configs': bdt_configs,
    })

    # run BDT classifiers
    apply_bdts(tray, cfg_copy, name=name+'ApplyBDTs')

    # update HDF5 output keys
    for key in cfg_copy['HDF_keys']:
        if key not in cfg['HDF_keys']:
            cfg['HDF_keys'].append(key)

    # --------------------------
    # Apply DNN Cascade L3c Cuts
    # --------------------------
    pass


    # -------------------
    # Run Event-Generator
    # -------------------
    pass









# class CascadeSelectionBDT(icetray.I3ConditionalModule):
#     """Class to perform BDT cuts for selection for enhanced cascades,
#        and rejection of throughgoing-muons.
#     """
#     def __init__(self, context):
#         """Class to apply BDT

#         Parameters
#         ----------
#         context : TYPE
#             Description
#         """
#         icetray.I3ConditionalModule.__init__(self, context)
#         self.AddOutBox('OutBox')
#         self.AddParameter('model_path',
#                           'The path to the BDT model that will be loaded.',
#                           None)
#         self.AddParameter('ouput_key',
#                           'The key to which the output will be written',
#                           'CascadeBDTOutput')
#     def Configure(self):
#         """Configures Module and loads BDT from file.
#         """
#         self.model_path = self.GetParameter('model_path')
#         self.ouput_key = self.GetParameter('ouput_key')

#     def Physics(self):
#         """Apply BDT to physics frames
#         """

#         # push frame to next modules
#         self.PushFrame(frame)



# @icetray.traysegment
# def DNNCascadeLevel3(
#         tray, name,
#         pulse_key=None,
#         dom_exclusions=None,
#         partial_exclusion=None,
#         output_keys=None,
#         models_dir='/data/user/mhuennefeld/DNN_reco/models/exported_models',
#         measure_time=True,
#         batch_size=1,
#         num_cpus=1,
#         verbose=True,
#         ):
#     """Apply DNN reco

#     Parameters
#     ----------
#     tray : icecube.icetray
#         Description
#     name : str
#         Name of module
#     pulse_key : str
#         Name of pulses to use.
#         If None is passed, the model's default settings will be used.
#     dom_exclusions : list of str, optional
#         List of frame keys that define DOMs or TimeWindows that should be
#         excluded. Typical values for this are:
#         ['BrightDOMs','SaturationWindows','BadDomsList','CalibrationErrata']
#         If None is passed, the model's default settings will be used.
#     partial_exclusion : bool, optional
#         If True, partially exclude DOMS, e.g. only omit pulses from excluded
#         TimeWindows defined in 'dom_exclusions'.
#         If False, all pulses from a DOM will be excluded if the omkey exists
#         in the dom_exclusions.
#         If None is passed, the model's default settings will be used.
#     output_keys : None, optional
#         A list of output keys for the reco results.
#         If None, the output will be saved as dnn_reco_{ModelName}.
#     models_dir : str, optional
#         The main model directory. The final model directory will be:
#             os.path.join(models_dir, ModelName)
#     measure_time : bool, optional
#         If True, the run-time will be measured.
#     batch_size : int, optional
#         The number of events to accumulate and pass through the network in
#         parallel. A higher batch size than 1 can usually improve recontruction
#         runtime, but will also increase the memory footprint.
#     num_cpus : int, optional
#         Number of CPU cores to use if CPUs are used instead of a GPU.
#     verbose : bool, optional
#         If True, output pulse masking information.

#     """
#     if isinstance(model_names, str):
#         model_names = [model_names]

#     if output_keys is None:
#         output_keys = ['DeepLearningReco_{}'.format(m) for m in model_names]



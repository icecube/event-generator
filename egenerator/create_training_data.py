import os
import click
import numpy as np
import timeit

from I3Tray import I3Tray
from icecube import icetray, dataio, hdfwriter, dataclasses
from icecube.weighting import get_weighted_primary

from ic3_labels.labels.modules.event_generator.muon_track_labels import EventGeneratorMuonTrackLabels
from ic3_labels.labels.modules import MCLabelsCascades, MCLabelsCascadeParameters#/Split
from label_functions import create_hdf5_time_window_series_map_hack
import yaml

from modules.utils.tray_timer import TimerStart, TimerStop
from modules.utils.recreate_and_add_mmc_tracklist import RerunProposal

from combined import CombineAndApplyExclusions

@click.command()
@click.argument('cfg', type=click.Path(exists=True))
@click.argument('in_number', type=int)
@click.argument('run_number', type=int)
def main(cfg, in_number, run_number):

    with open(cfg, 'r') as stream:
        cfg = yaml.full_load(stream) 
    
    start_time = timeit.default_timer()
    # loop through original data files
    for n in range(in_number, run_number):
        n_start_time = timeit.default_timer()
        cfg['run_number'] = n
        infile = cfg['in_file_pattern'].format(**cfg)
        outfile = os.path.join(cfg['data_folder'], 
                cfg['out_dir_pattern'].format(**cfg), 
                cfg['out_file_pattern'].format(**cfg))
        if cfg['gcd']:
            infiles = [cfg['gcd'], infile]

        #print(infiles)
        # 
        tray = I3Tray()
        
        tray.AddModule('I3Reader', Filenamelist = infiles)

        # ---------------------------------------------------------
        # Copied from $DNN_HOME/processing/scripts/dnn_reco_create_data.py
        # ---------------------------------------------------------

        # --------------------------------------------------
        # Add MC labels
        # --------------------------------------------------
       
        if 'RerunProposal_kwargs' not in cfg:
            cfg['RerunProposal_kwargs'] = {}
            
        # Add I3MCTree/MMCTrackList if missing
        tray.AddSegment(
                RerunProposal, 'RerunProposal',
                mctree_name='I3MCTree',
                # Merging causes issues with MuonGunWeighter (needs investigation)
                # merge_trees=['BackgroundI3MCTree'],
                **cfg['RerunProposal_kwargs']
                )
            
        if 'I3MCTree_recreation_meta_info' not in cfg['HDF_keys']:
                cfg['HDF_keys'].append('I3MCTree_recreation_meta_info')

        # get weighted primary
        tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
                           If=lambda f: not f.Has('MCPrimary'),
                           )
        # ---------------------------------------------------------------
        # Dirty Hack to write I3MCWeightDict in custom cascade simulation
        # ---------------------------------------------------------------
        if 'I3MCWeightDict' in cfg:
            log_info('Adding the following I3MCWeightDict to frame:',
                         cfg['I3MCWeightDict'])

            def add_pseudo_I3MCWeightDict(frame, dict):
                frame['I3MCWeightDict'] = dataclasses.I3MapStringDouble(dict)

            tray.AddModule(add_pseudo_I3MCWeightDict,
                           'add_pseudo_I3MCWeightDict',
                           dict=cfg['I3MCWeightDict'],
                           # If=lambda f: not f.Has('I3MCWeightDict'),
                           Streams=[icetray.I3Frame.DAQ],
                           )
        # ---------------------------------------------------------------

        tray.AddModule("Copy", "copy_series", keys=['I3MCPulseSeriesMap', 'I3MCPESeriesMap'])

        tray.AddModule(create_hdf5_time_window_series_map_hack,
                      tws_map_name='SaturationWindows',
                      output_key='time_exclusion')

        def has_muon(frame):
            x = dataclasses.get_most_energetic_muon(frame['I3MCTree'])
            if x:
                return True
            else:
                return False

        tray.Add(has_muon)

        tray.AddModule(MCLabelsCascadeParameters, 'MCLabelsDeepLearning', 
                                                  PrimaryKey = 'MCPrimary',
                                                  OutputKey = 'LabelsDeepLearning')
        

        # ----------------------------------------------------------------------------------------------
        # End of copy from Add MC Labels section of $DNN_HOME/precessing/scripts/dnn_reco_create_data.py
        # ----------------------------------------------------------------------------------------------
 
        tray.AddSegment(
                    CombineAndApplyExclusions, 'CombinedExclusions',
                    pulse_key='HVInIcePulses',
                    dom_and_tw_exclusions=['SaturationWindows'],
                    partial_exclusion=True,
                    exclude_bright_doms=False,
                    bright_doms_threshold_fraction=0.4,
                    bright_doms_threshold_charge=100.,
                    merge_pulses_time_threshold=None,
                    )

        # at least 3 as above
        tray.Add(lambda fr: len(fr['masked_pulses']) > 3)

        tray.AddSegment(hdfwriter.I3HDFWriter, 'hdf',
                                Output='{}.hdf5'.format(outfile),
                                CompressionLevel=9,
                                Keys=cfg['HDF_keys'],
                                SubEventStreams=cfg['HDF_SubEventStreams']
                                )
        #print(cfg['HDF_Keys']
        tray.AddModule('TrashCan', 'YesWeCan')
        tray.Execute()
        tray.Finish()


        time_now = timeit.default_timer()
        print(n, " is finished, duration: {:5.3f}s".format(time_now - n_start_time))
        
    end_time = timeit.default_timer()
    print('Duration: {:5.3f}s'.format(end_time - start_time))

if __name__ == '__main__':
    main() 



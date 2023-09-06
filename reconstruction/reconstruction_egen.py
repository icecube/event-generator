#!/data/user/jvara/egenerator_tutorial/egen_tf.2.8_env/bin/python
from __future__ import print_function, division
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from icecube import dataclasses, icetray, dataio, simclasses, hdfwriter
from icecube import icetray
from I3Tray import *
from os.path import expandvars
from icecube.weighting import get_weighted_primary
from egenerator.ic3.segments import ApplyEventGeneratorReconstruction
import click
import yaml
import glob
from tqdm import tqdm
import re
from mcpe_label_function import mcpe_label_module
import traceback
'''
This is a code to run the i3 segment for reconstructions with event generator. It needs as input a cfg file. 
'''

class count_pulses(icetray.I3Module):

    """
    Class to count pulses of icecube gen2-mcpes
    """

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)

    def Physics(self,frame):

        Gen2MCPE = simclasses.I3MCPESeriesMap()
        try:
            mcpes = frame['I3MCPESeriesMapWithoutNoise']
        except:
            mcpes = frame["I3MCPESeriesMap"]
        
        count = 0
        for omkey, pulse_list in mcpes.items():
            if omkey.string>1000:
                Gen2MCPE[omkey]=pulse_list
                count+=len(pulse_list)

        frame['Gen2-mcpes']=Gen2MCPE
        frame['Num_mcpes'] = icetray.I3Int(count)
        #At least 3 mcpes
        if count <=3:
            return False
        else:
            self.PushFrame(frame)

@click.command()
@click.argument('cfg', type=click.Path(exists=True))
def main(cfg):

    with open(cfg,'r') as stream:
        cfg=yaml.full_load(stream)


    files = glob.glob(f"{cfg['infile_pattern']}")
    files.sort()

    for infile in tqdm(files):
        
        outdir = cfg['reco_dir']

        out_name = infile.split("/")[-1]
        out_name = out_name.split(".i3.zst")[0]
        outfile = outdir + "/" + out_name

        try:
            HDF_keys = ['LabelsDeepLearning','Num_mcpes','EventGenerator','EventGenerator_I3Particle']
            tray = I3Tray()

            tray.AddModule('I3Reader', 'reader', Filename=infile)

            #At least 10 pulses
            tray.AddModule(count_pulses, 'number_of_pulses')

            tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
                                        If=lambda f: not f.Has('MCPrimary'))

            # Add LabelsDeepLearning key, just in case
            tray.AddModule(mcpe_label_module, 'MCLabelsDeepLearning')

            tray.AddSegment(
                ApplyEventGeneratorReconstruction, 'ApplyEventGeneratorReconstruction',
                pulse_key='I3RecoPulseSeriesMapGen2',
                dom_and_tw_exclusions=None,
                partial_exclusion=False,
                exclude_bright_doms=False,
                model_names=['lom_pulses_negative_memory_september'],
                seed_keys=['LabelsDeepLearning'],
                model_base_dir='/data/user/jvara/exported_models/event-generator/lom',
                num_threads = 0,

            )

            tray.AddSegment(hdfwriter.I3HDFWriter, 'hdf',
                                            Output='{}.hdf5'.format(outfile),
                                            CompressionLevel=9,
                                            Keys=HDF_keys,
                                            SubEventStreams=['SplitInIceGen2'])

        
            tray.Execute()
            tray.Finish()

        except Exception as e:
            print("The exception is: ",e)
            traceback.print_exc()
            print("Error")

if __name__ == '__main__':

    main()


from __future__ import print_function, division
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from icecube import dataclasses, icetray, dataio, simclasses
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

@click.command()
@click.argument('cfg', type=click.Path(exists=True))
def main(cfg):
    
    print("hey")

    if cfg.startswith('"') or cfg.endswith('"'):
        cfg = cfg.strip('"')
    with open(cfg,'r') as stream:
        cfg=yaml.full_load(stream)


    files = glob.glob(f"{cfg['infile_directory']}/*destim*custom.i3.zst")
    files.sort()

    files = ['/data/user/jvara/simulations/lom_sim/iceprod/local_sim_test/NuE_0_out_LOM16_detsim_custom.i3.zst']

    print("files",len(files))

    for infile in tqdm(files):

        #file_index = int(re.search(cfg["infile_pattern"], infile).group(1))

        outfile = os.path.join(cfg['out_dir_pattern'],cfg['out_file_pattern'].format(6))

        try:

            print("first check")

            tray = I3Tray()

            tray.AddModule('I3Reader', 'reader', Filenamelist=files)

            #At least 10 pulses
            tray.Add(lambda fr: len(fr['I3MCPESeriesMap']) > 9)

            tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
                                        If=lambda f: not f.Has('MCPrimary'))

            # Add LabelsDeepLearning key, just in case
            tray.AddModule(mcpe_label_module, 'MCLabelsDeepLearning')

            print("second check")

            tray.AddSegment(
                ApplyEventGeneratorReconstruction, 'ApplyEventGeneratorReconstruction',
                pulse_key='I3RecoPulseSeriesMapGen2',
                dom_and_tw_exclusions=None,
                partial_exclusion=False,
                exclude_bright_doms=False,
                model_names=['lom_pulses_negative_july_2023'],
                seed_keys=['LabelsDeepLearning'],
                model_base_dir='/data/user/jvara/exported_models/event-generator/lom',
                #snowstorm_key = '',
                num_threads = 4,

            )

            print("third check")

            #check keys are on the frame              

            tray.AddModule("I3Writer", "EventWriter", filename='{}.i3.bz2'.format(outfile))

            #tray.AddSegment(hdfwriter.I3HDFWriter, 'hdf',
            #                                Output='{}.hdf5'.format(outfile),
            #                                CompressionLevel=9,
            #                                Keys=HDF_keys,
            #                                SubEventStreams=['GEN2_IC86_InIceSplit'])

            tray.Execute()
            tray.Finish()

        except Exception as e:
            print("The exception is: ",e)
            traceback.print_exc()
            print("Error")
        finally:
            print("Fin")
            break

if __name__ == '__main__':

    main()


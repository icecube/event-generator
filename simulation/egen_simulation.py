from __future__ import print_function, division
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from icecube import dataclasses, icetray, dataio
from icecube import icetray
from I3Tray import *
import click
import yaml
import glob
from tqdm import tqdm
import re
import traceback
from egenerator.ic3.MCPE_simulation_single_cascade import EventGeneratorSimulation
from mcpe_label_function import mcpe_label_module
from icecube.weighting import get_weighted_primary


@click.command()
@click.argument('cfg', type=click.Path(exists=True))
def main(cfg):
    
    if cfg.startswith('"') or cfg.endswith('"'):
        cfg = cfg.strip('"')
    with open(cfg,'r') as stream:
        cfg=yaml.full_load(stream)


    files = glob.glob(f"{cfg['infile_directory']}/NuE_0_out_hits_custom.i3.zst")
    files.sort()


    for infile in tqdm(files):

        infiles = [infile]

        try:
            tray = I3Tray()

            tray.AddModule('I3Reader', 'reader', Filenamelist=infiles)

            tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
                   If=lambda f: not f.Has('MCPrimary'), Streams = [icetray.I3Frame.DAQ])

            tray.AddModule(mcpe_label_module, 'MCLabelsDeepLearning')

            tray.AddModule(
                EventGeneratorSimulation,
                'EventGeneratorSimulation',
                model_name = 'lom_mcpe_june_2023',
                num_threads = 0,
                 )

            tray.AddModule("I3Writer", "EventWriter", filename='{}.i3.bz2'.format(cfg['outfile']))

            tray.Execute(10)
            tray.Finish()

        except Exception as e:
            print("The exception is: ",e)
            traceback.print_exc()
            print("Error")
        finally:
            print("Fin")
            break

if __name__ == '__main__':
    #this is a simple code to pass the Event-generator simulation module. The input should be a (i3) file with I3MCTree
    main()

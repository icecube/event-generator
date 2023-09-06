#!/data/user/jvara/egenerator_tutorial/egen_tf.2.8_env/bin/python
from __future__ import print_function, division
import numpy as np
import os
import sys
from icecube import dataclasses, icetray, dataio, simclasses, hdfwriter
from I3Tray import *
import click
import yaml
import glob
from tqdm import tqdm
from mcpe_label_function import mcpe_label_module
from icecube.weighting import get_weighted_primary
import tensorflow as tf
from scipy import stats
from egenerator.ic3.goodness_of_fit import goodness_of_fit
import glob
from tqdm import tqdm
'''
This code is use to run the i3 module to evaluate the goodness of fit. It needs a .yaml file as input
'''
class ExtractGen2mcpes(icetray.I3Module):

    """
    Class to extract the mcpes of IceCube-Gen2 strings, i.e string>=1001
    """

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)

    def DAQ(self,frame):

        Gen2MCPE = simclasses.I3MCPESeriesMap()
        try:
            mcpes = frame['I3MCPESeriesMapWithoutNoise']
        except:
            mcpes = frame["I3MCPESeriesMap"]
        for omkey, pulse_list in mcpes.items():
            if omkey.string>1000:
                Gen2MCPE[omkey]=pulse_list

        frame['Gen2-mcpes']=Gen2MCPE
        self.PushFrame(frame)

@click.command()
@click.argument('cfg', type=click.Path(exists=True))
def main(cfg):
    with open(cfg,"r") as stream:
        cfg = yaml.full_load(stream)

    hdf_keys = ["ks_test","charge_test","ks_sampled"]

    infiles = glob.glob(cfg["infile_pattern"])
    infiles = sorted(infiles)
    for file in tqdm(infiles):
        filename = file.split("/")[-1]
        outfile = filename.split(".i3.zst")[0]
        outfile = cfg["out_dir"]+outfile
        tray = I3Tray()

        tray.AddModule('I3Reader', 'reader', Filename=file)

        tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
            If=lambda f: not f.Has('MCPrimary'), Streams = [icetray.I3Frame.DAQ])

        tray.AddModule(mcpe_label_module, 'MCLabelsDeepLearning')

        tray.AddModule(ExtractGen2mcpes, "mcpes")

        tray.AddModule(goodness_of_fit, calculate_truth=True,min_charge_pred=0.05,num_pulse_threshold=5)

        tray.AddSegment(hdfwriter.I3SimHDFWriter, 'hdf',
                            Output='{}.hdf5'.format(outfile),
                            CompressionLevel=9,
                            Keys=hdf_keys,
                            )

        tray.Execute()
        tray.Finish()


if __name__ == "__main__":
    main()

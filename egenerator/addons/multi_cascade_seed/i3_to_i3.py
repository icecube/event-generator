#!/usr/bin/env python
# encoding: utf-8

from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube import hdfwriter, dataclasses
from icecube import icetray, dataio
from icecube.phys_services import I3Calculator

import click
import glob
import numpy as np

from egenerator.addons.multi_cascade_seed import CascadeClusterSearchModule


@click.command()
@click.argument('input_file_pattern', type=click.Path(exists=True),
                required=True)
@click.option('-o', '--outfile', default='dnn_output',
              help='Name of output file without file ending.')
@click.option('-g', '--gcd', default=None, type=click.Path(),
              help='Path to GCD file.')
@click.option('-n', '--number', default=None,
              help='Number of frames to pass to execute method.')
def main(input_file_pattern, outfile, gcd, number):

    tray = I3Tray()

    file_name_list = glob.glob(input_file_pattern)
    if gcd is not None:
        file_name_list = [gcd] + file_name_list

    tray.AddModule('I3Reader', 'reader', Filenamelist=file_name_list)

    tray.AddModule(
        CascadeClusterSearchModule, 'CascadeClusterSearchModule',
        n_clusters=10,
        min_dist=200,
        min_cluster_charge=3,
        min_hit_doms=3,
        add_to_frame=True,
        initial_clusters_particles=[
            'EventGenerator_cascade_7param_noise_tw_BFRv1Spice321_01_I3Particle',
        ],
    )

    tray.AddModule("I3Writer", "EventWriter",
                   filename='{}.i3.bz2'.format(outfile),
                   Streams=[icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics,
                            # icetray.I3Frame.Geometry,
                            # icetray.I3Frame.Calibration,
                            # icetray.I3Frame.DetectorStatus,
                            #icetray.I3Frame.TrayInfo,
                            #icetray.I3Frame.Simulation,
                            #icetray.I3Frame.Stream('S'),
                            ])
    tray.AddModule('TrashCan', 'YesWeCan')
    if number is not None:
        tray.Execute(int(number))
    else:
        tray.Execute()

    del tray


if __name__ == '__main__':
    main()

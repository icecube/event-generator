from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray, dataio
from I3Tray import *
import os
import pickle
from tqdm import tqdm
from generate_geo_pickle import generate_geometry_pickle
from generate_geo_pickle_added import generate_geometry_pickle_added
import click


@click.command()
@click.option('--gcd', default="/cvmfs/icecube.opensciencegrid.org/users/gen2-optical-sim/gcd/IceCubeHEX_Sunflower_240m_v4.0beta_ExtendedDepthRange_LOM16.GCD.i3.bz2",
              help='Path to the gcd file.')
@click.option('--om_name', default="LOM16", help='Name of optical module to store data for')
@click.option('--tray_module', default="default", help='Name of the icetray module to use')
@click.option('--base_dir', default="/data/user/jvara/egenerator_tutorial/repositories/event-generator/gcd_preprocessing/geometry_pickles", help='Base directory to save the pickle files')
def main(gcd, om_name, tray_module, base_dir):
    file_name = str(gcd)

    tray = I3Tray()

    tray.AddModule('I3Reader', 'reader', Filename=file_name)
    if tray_module=="default":
        tray.AddModule(generate_geometry_pickle, 'Generate_geometry', om_name=om_name, base_dir=base_dir)
    elif tray_module=="added":
        tray.AddModule(generate_geometry_pickle_added, 'Generate_geometry', om_name=om_name, base_dir=base_dir)

    tray.Execute()
    tray.Finish()


if __name__ == '__main__':
    main()

from __future__ import print_function, division
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from icecube import dataclasses, icetray, dataio, simclasses
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
from egenerator.utils.configurator import ManagerConfigurator
import tensorflow as tf

def get_labels_from_frame(frame,label_key='LabelsDeepLearning', *args, **kwargs):

    '''
    For some reason the labels are not obtained from the get_from_frame thing so I implement it here
    '''
    cascade_parameters = []

    cascade = frame[label_key]
    
    _labels = {"cascade_x":cascade.pos.x,"cascade_y":cascade.pos.y,"cascade_z":cascade.pos.z,
                "cascade_zenith":cascade.pos.theta,"cascade_azimuth":cascade.pos.phi,
                "cascade_energy":cascade.total_energy,"cascade_t":cascade.time
                }

    try:
        
        for l in ['cascade_x', 'cascade_y', 'cascade_z', 'cascade_zenith',
                    'cascade_azimuth', 'cascade_energy', 'cascade_t']:
            cascade_parameters.append(np.atleast_1d(_labels[l]))

    except Exception as e:
        
        print(f"There was an exception {e}")
    

    
    cascade_parameters = np.array(cascade_parameters,
                                    dtype='float32').T
    num_events = len(cascade_parameters)

    return num_events, (cascade_parameters,)

class ExtractGen2mcpes(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)

    def DAQ(self,frame):

        Gen2MCPE = simclasses.I3MCPESeriesMap()
        mcpes = frame["I3MCPESeriesMap"]
        for omkey, pulse_list in mcpes.items():
            if omkey.string>1000:
                Gen2MCPE[omkey]=pulse_list

        frame['Gen2-mcpes']=Gen2MCPE
        self.PushFrame(frame)

class calculate_loss(icetray.I3Module):
    """
    This code is to calculate the loss (-log likelihood) using model expectations
    to test the performance of the model. On the same dataset, the lower this value 
    the better. This code needs to be documented.
    """
    def __init__(self,context):
        icetray.I3Module.__init__(self,context)

    def DAQ(self,frame):

        n, data = data_handler.get_data_from_frame(frame)

        n, labels = get_labels_from_frame(frame)
        
        dataset = data + labels
        
        
        if "x_parameters" not in data_handler.tensors.names:
            data_handler.tensors.names.append("x_parameters") # this one seems to be missing

        loss = manager.get_loss(data_batch = dataset, loss_module = loss_module, is_training = False, opt_config = config['training_settings'])
        
        main_array.append(float(loss))


if __name__ == "__main__":

    main_array = []

    infiles = ["/data/user/jvara/simulations/lom_sim/iceprod/local_sim_test/NuE_0_out_hits_custom_primero.i3.zst"]

    model_dir = "/data/user/jvara/exported_models/event-generator/lom/lom_mcpe_june_2023"

    manager_configurator = ManagerConfigurator(
            manager_dirs=[model_dir],
            num_threads=0,
        )
    
    config = manager_configurator.config
   
    loss_module = manager_configurator.loss_module
    
    manager = manager_configurator.manager

    data_handler = manager.data_handler

    print(data_handler.tensors)

    tray = I3Tray()

    tray.AddModule('I3Reader', 'reader', Filenamelist=infiles)

    tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
        If=lambda f: not f.Has('MCPrimary'), Streams = [icetray.I3Frame.DAQ])

    tray.AddModule(mcpe_label_module, 'MCLabelsDeepLearning')

    tray.AddModule(ExtractGen2mcpes, "mcpes")

    #here is where the loss is calculated
    tray.AddModule(calculate_loss, 'printing')

    tray.Execute(6)
    tray.Finish()

    print("end")
    print(main_array)

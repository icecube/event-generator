"""
Module for generating pickle files with detector information.
"""

from __future__ import print_function, division
import numpy as np
from icecube import dataclasses, icetray
import os
import pickle
from tqdm import tqdm

class generate_geometry_pickle_added(icetray.I3ConditionalModule):
    """
    Module to generate the pickle files with detector information.
    This class is meant to use module global information instead
    of PMT.
    """
    def __init__(self, context):
        """
        Initialize the module.

        Args:
            context: The context in which the module is being run.
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("om_name", "Name of optical module to store data for", "LOM16")
        self.AddParameter("base_dir", "The directory where the pickle file is saved", ".")
        
    def Configure(self):
        """
        Configure the module.
        """
        self.om_name = self.GetParameter("om_name")
        self.base_dir = self.GetParameter("base_dir")
    
    def Calibration(self,frame):
        """
        Generate the pickle file with detector information.

        Args:
            frame: The I3Frame object containing the detector information.

        Returns:
            None
        """
          
        om_name = self.om_name
        
        om_dict = {}
        
        print(f"Storing {om_name} data")
    
        try:
            #New gcds
            geomap = frame['I3OMGeoMap']
            
        except:
            #Old gcds
            geomap = frame['I3Geometry'].omgeo
            
            
        #Define the number of strings, modules per string and pmts per module
        
        set_strings = set()
        set_modules_per_string = set()
        set_pmts_per_module = set()
        omtypes_names=set()
        
        
        for omkey, omgeo in geomap:
            if omgeo.omtype.name==om_name:
                om_dict["DOM_name"]=omgeo.omtype.name
                set_strings.add(omkey[0])
                set_modules_per_string.add(omkey[1])
                set_pmts_per_module.add(omkey[2])
            omtypes_names.add(omgeo.omtype.name)
            #geomap = frame['I3OMGeoMap']    
        set_strings = sorted(set_strings)
        set_modules_per_string=sorted(set_modules_per_string)
        set_pmts_per_module=sorted(set_pmts_per_module)
        
        
        assert om_name in omtypes_names, f"{om_name} not found in omtype names: {omtypes_names}"
        
        om_dict["string_offset"]=set_strings[0]-1
        om_dict["num_strings"]=len(set_strings)
        om_dict["doms_per_string"]=len(set_modules_per_string)
        num_pmts=len(set_pmts_per_module)
        om_dict["num_pmts"]=1 
        
        
        #Define the dom coordinates
        
        dom_coordinates = np.zeros((om_dict["num_strings"],om_dict["doms_per_string"],num_pmts,3))
        

        omgeo_names = [omkey for omkey, omgeo in geomap if omgeo.omtype.name == om_name]
        indices = np.array([(omkey[0]-1-om_dict["string_offset"], omkey[1]-1, omkey[2]) for omkey in omgeo_names])
        positions = np.array([omgeo.position for omkey, omgeo in geomap if omgeo.omtype.name == om_name])
        dom_coordinates[indices[:,0], indices[:,1], indices[:,2], :] = positions
        
        
        dom_coordinates = np.mean(dom_coordinates, axis=2)
        
        dom_coordinates = dom_coordinates.reshape(om_dict["num_strings"], om_dict["doms_per_string"], 3)
        
        om_dict["dom_coordinates"]=dom_coordinates
        
        om_dict['coordinates_mean']=round(np.mean(om_dict['dom_coordinates']))
        om_dict['coordinates_std']=round(np.std(om_dict['dom_coordinates']))
        
        
        #Define azimuth and zenith orientation of the pmts
        
        dom_azimuths = np.zeros((om_dict["num_strings"],om_dict["doms_per_string"],num_pmts,))
        dom_zeniths = np.zeros((om_dict["num_strings"],om_dict["doms_per_string"],num_pmts,))
        
        azimuths = np.array([omgeo.direction.azimuth for omkey, omgeo in geomap if omgeo.omtype.name == om_name])
        zeniths = np.array([omgeo.direction.zenith for omkey, omgeo in geomap if omgeo.omtype.name == om_name])
        
        dom_azimuths[indices[:,0], indices[:,1], indices[:,2], ] = azimuths
        dom_zeniths[indices[:,0], indices[:,1], indices[:,2], ] = zeniths
        
        
        dom_azimuths = np.mean(dom_azimuths, axis=2)
        dom_zeniths = np.mean(dom_zeniths, axis=2)
        
        om_dict["dom_azimuths"]=dom_azimuths
        om_dict["dom_zeniths"]=dom_zeniths
        

        #Define simulation noise and relative eff
        
        
        
        Calibration_values = frame["I3Calibration"].dom_cal
        print("This may take some minutes")
        dom_rel_eff = np.zeros((om_dict["num_strings"], om_dict["doms_per_string"], num_pmts,))
        dom_noise_rates = np.zeros((om_dict["num_strings"], om_dict["doms_per_string"], num_pmts,))
        
        # This for loop is extremely slow for multi-PMT modules (to do -> improve it)
        for omkey, dom_data in tqdm(Calibration_values):
            if omkey in omgeo_names:
                dom_noise_rates[omkey[0]-1-om_dict["string_offset"],omkey[1]-1,omkey[2]]=dom_data.dom_noise_rate
                dom_rel_eff[omkey[0]-1-om_dict["string_offset"],omkey[1]-1,omkey[2]]=dom_data.relative_dom_eff
                
  
        dom_rel_eff = np.mean(dom_rel_eff, axis=2)
        dom_noise_rates = np.sum(dom_noise_rates, axis=2)

        om_dict["dom_rel_eff"] = dom_rel_eff
        om_dict["dom_noise_rates"] = dom_noise_rates
        
        om_dict['bad_doms_mask'] = dom_rel_eff != 0
     
        file_name = f"{om_name}.pickle"
        directory = self.base_dir

        # Check if a file with the same name already exists in the directory
        
        if os.path.exists(os.path.join(directory, file_name)):
            # If the file exists, ask user if they want to delete it
            response = input("File already exists. Do you want to delete it? (y/n): ")
            if response.lower() == "y":
                # Delete the existing file
                os.remove(os.path.join(directory, file_name))
            else:
                # If the user does not want to delete the file, exit the program
                print("File not written.")
                exit()

        # Write the dictionary to a pickle file
        
        with open(os.path.join(directory, file_name), "wb") as f:
            pickle.dump(om_dict, f)
 
        print("File written successfully.")
        
        
        print(om_dict.keys())
        
        for key, value in om_dict.items():
            try:
                print(key,value.shape)
            except:
                print(key,value)
                
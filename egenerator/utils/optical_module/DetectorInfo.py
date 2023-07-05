from dataclasses import dataclass
import numpy as np
import pickle
import os


@dataclass
class DetectorInfoModule:
    dom_name: str
    string_offset: int
    num_strings: int
    doms_per_string: int
    num_pmts: int
    coordinates_mean: int
    coordinates_std: int
    dom_coordinates: np.ndarray
    dom_azimuths: np.ndarray
    dom_zeniths: np.ndarray
    dom_rel_eff: np.ndarray
    dom_noise_rates: np.ndarray
    bad_doms_mask: np.ndarray

    def __init__(self, file_path):
        """
        Load the data from a pickle file and create an instance of the data class.

        Args:
            file_path: The path to the pickle file.
        """
        try:
            assert os.path.exists(file_path), f"File {file_path} does not exist"
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except:
            file_path = "/data/user/jvara/egenerator_tutorial/repositories/event-generator/gcd_preprocessing/geometry_pickles/LOM16.pickle"
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
        self.dom_name = data["DOM_name"]
        self.string_offset = data["string_offset"]
        self.num_strings = data["num_strings"]
        self.doms_per_string = data["doms_per_string"]
        self.coordinates_mean = data["coordinates_mean"]
        self.coordinates_std = data["coordinates_std"]
        self.num_pmts = data["num_pmts"]
        self.dom_coordinates = data["dom_coordinates"]
        self.dom_azimuths = data["dom_azimuths"]
        self.dom_zeniths = data["dom_zeniths"]
        self.dom_rel_eff = data["dom_rel_eff"]
        self.dom_noise_rates = data["dom_noise_rates"]
        self.bad_doms_mask = data["bad_doms_mask"]

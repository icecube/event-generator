#!/usr/local/bin/python3

import unittest
import os
import shutil
import numpy as np
import ruamel.yaml as yaml

from egenerator.data.trafo import DataTransformer
from egenerator.data.handler.modular import ModuleDataHandler
from egenerator import create_trafo_model


class TestCreateTrafoModel(unittest.TestCase):
    """Test initialization of setup manager class.
    Make sure correct exceptions are raised.
    """

    def setUp(self):

        # Test Data
        self.file_path = os.path.join(
            os.path.dirname(__file__), "test_data/cascade_mesc_l5_nue_low.hdf5"
        )

        self.times_618_to_627 = [
            11737.0,
            10813.0,
            10743.0,
            10419.0,
            10430.0,
            10439.0,
            10481.0,
            10440.0,
            11055.0,
            10487.0,
        ]
        self.charges_618_to_627 = [
            0.824999988079071,
            1.024999976158142,
            1.774999976158142,
            0.42500001192092896,
            1.774999976158142,
            0.22499999403953552,
            1.1749999523162842,
            0.32499998807907104,
            0.875,
            1.4249999523162842,
        ]
        self.total_event_charge = np.array([1080.0499, 668.4249])

        self.dom_exclusions = np.ones([2, 86, 60, 1], dtype=bool)
        self.dom_exclusions[0, 25, 47] = False
        self.dom_exclusions[0, 25, 48] = False
        self.dom_exclusions[0, 25, 49] = False
        self.dom_exclusions[1, 34, 53] = False
        self.dom_exclusions[1, 34, 54] = False
        self.dom_exclusions[1, 34, 55] = False

        self.cascades_true = np.array(
            [
                [
                    -13.094963629354766,
                    -197.0847391208472,
                    -322.0192710148053,
                    1.0771952265275238,
                    4.601483747646196,
                    2360.8997600199427,
                    9663.551318717717,
                ],
                [
                    -70.78487964475926,
                    -32.47261211840669,
                    -426.5132607462965,
                    1.586083894785944,
                    1.5573642249002815,
                    924.7251046427211,
                    9789.880753474426,
                ],
            ]
        )

        # setup module
        data_handler_settings = {
            "data_handler": "modular.ModuleDataHandler",
            # settings for the data module
            "data_module": "pulse_data.PulseDataModule",
            "data_settings": {
                "pulse_key": "InIceDSTPulses",
                "dom_exclusions_key": "BrightDOMs",
                "time_exclusions_key": None,
                "float_precision": "float32",
                "add_charge_quantiles": False,
            },
            # settings for the label module
            "label_module": "cascades.CascadeGeneratorLabelModule",
            "label_settings": {
                "shift_cascade_vertex": False,
                # logarithm on labels (x, y, z, zenith, azimuth, energy, time)?
                "trafo_log": [False, False, False, False, False, False, False],
                "label_key": "LabelsDeepLearning",
                "float_precision": "float32",
            },
            # settings for the weight module
            "weight_module": "dummy.DummyWeightModule",
            "weight_settings": {},
            # settings for the misc module
            "misc_module": "dummy.DummyMiscModule",
            "misc_settings": {},
            # settings for the filter module
            "filter_module": "dummy.DummyFilterModule",
            "filter_settings": {},
        }
        data_trafo_settings = {
            "float_precision": "float64",
            "norm_constant": 1e-6,
            "num_batches": 20,
            "model_dir": os.path.join(
                os.path.dirname(__file__),
                "../data/temp_test_files/test_create_trafo_model",
            ),
        }

        data_iterator_settings = {
            "trafo": {
                "batch_size": 3,
                "num_splits": None,
                "file_capacity": 1,
                "batch_capacity": 1,
                "num_jobs": 1,
                "num_add_files": 1,
                "num_repetitions": 1,
                "pick_random_files_forever": True,
                "input_data": [self.file_path],
            }
        }

        # put it all together
        self.config = {
            "unique_name": "unittest_create_trafo_model_config",
            "data_iterator_settings": data_iterator_settings,
            "data_trafo_settings": data_trafo_settings,
            "data_handler_settings": data_handler_settings,
        }

        # create temp yaml config
        self.yaml_file_path = os.path.join(
            os.path.dirname(__file__),
            "../data/temp_test_files/unittest_created_trafo_model_config.yaml",
        )

        with open(self.yaml_file_path, "w") as yaml_file:
            yaml.dump(self.config, yaml_file)

    def test_create_trafo_model(self):
        """Call create trafo model script"""

        # remove trafo file if it already exists
        if os.path.exists(self.config["data_trafo_settings"]["model_dir"]):
            shutil.rmtree(self.config["data_trafo_settings"]["model_dir"])

        create_trafo_model.main.callback([self.yaml_file_path])

        # check if created trafo model seems correct
        data_handler = ModuleDataHandler()
        data_handler.configure(config=self.config["data_handler_settings"])
        data_trafo = DataTransformer(data_handler)

        data_trafo.load(self.config["data_trafo_settings"]["model_dir"])

        self.assertTrue(
            np.allclose(
                data_trafo.data["x_parameters_mean"],
                np.mean(self.cascades_true, axis=0),
            )
        )


if __name__ == "__main__":
    unittest.main()

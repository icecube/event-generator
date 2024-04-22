import unittest
import os
import numpy as np
import tensorflow as tf
import pickle

from egenerator.utils.configurator import ManagerConfigurator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestModelPrediction(unittest.TestCase):
    """Test the model prediction of a 7-cascade model.

    The model is loaded from a directory and the prediction is tested.
    """

    def definte_settings(self):
        self.model_name = "cascade_7param_noise_tw_BFRv1Spice321_01"
        self.params_dict = {
            "cascade_x": 100,
            "cascade_y": -100,
            "cascade_z": -300,
            "cascade_zenith": 150,
            "cascade_azimuth": 177,
            "cascade_energy": 100000,
            "cascade_time": 0,
        }

    def setUp(self):

        # define settings
        self.definte_settings()

        # define model to load
        self.model_dir = os.path.join(
            SCRIPT_DIR,
            f"test_data/models/{self.model_name}/",
        )

        # ----------
        # setup data
        # ----------
        self.dom_list = []
        for s in range(1, 87):
            for d in range(1, 61):
                self.dom_list.append((s, d))

        self.t_min = 0.0
        self.t_max = 8000.0
        self.t_width = self.t_max - self.t_min
        self.x_time_exclusions = [
            [-50000, self.t_min],
            [self.t_max, 50000],
        ] * len(self.dom_list)

        n_pulses = 2
        self.x_pulses = [
            np.transpose(
                [
                    np.linspace(
                        self.t_min - self.t_width,
                        self.t_max + self.t_width,
                        n_pulses,
                    ),
                    [1.0] * n_pulses,
                ],
                (1, 0),
            )
        ] * len(self.dom_list)
        self.x_pulses = np.reshape(
            self.x_pulses, (len(self.dom_list) * n_pulses, 2)
        )
        self.x_time_exclusions_ids = []
        self.x_pulses_ids = []
        for s, d in self.dom_list:
            self.x_time_exclusions_ids.append([0, s - 1, d - 1])
            self.x_time_exclusions_ids.append([0, s - 1, d - 1])

            for i in range(n_pulses):
                self.x_pulses_ids.append([0, s - 1, d - 1])
        self.x_time_window = np.array([[self.t_min, self.t_max]])

        # ----------
        # load model
        # ----------
        self.load_model(self.model_dir)

    def get_params(self):
        """Get parameters for model prediction."""

        # define parameters of model (order + names via model.parameters)
        params = []
        for param in self.model.parameter_names:
            params.append(self.params_dict[param])
        params = [params]
        return params

    def load_model(self, model_dir):
        """Load model from directory."""
        # load and build model
        # model_dir: path to an exported event-generator model
        self.configurator = ManagerConfigurator(model_dir)
        self.manager = self.configurator.manager
        self.model = self.manager.models[0]

        # trace Model
        self.get_dom_expectation = self.manager.get_model_tensors_function()

    def check_keys(self, result_tensors):
        """Check keys in result_tensors."""
        keys = [
            "dom_cdf_exclusion_sum",
            "dom_charges",
            "dom_charges_variance",
            "nested_results",
            "pulse_pdf",
        ]
        self.assertEqual(sorted(list(result_tensors.keys())), keys)

    def check_result_tensors(self, result_tensors, result_tensors_true):
        """Check result_tensors against true values."""
        keys = [
            "dom_cdf_exclusion_sum",
            "dom_charges",
            "dom_charges_variance",
            "pulse_pdf",
        ]

        for key in keys:
            self.assertTrue(
                np.allclose(result_tensors[key], result_tensors_true[key])
            )

        # now check nested results
        self._recursive_check(
            result_tensors["nested_results"],
            result_tensors_true["nested_results"],
        )

    def _recursive_check(self, dict1, dict2, atol=5e-6, rtol=5e-4):
        keys1 = sorted(list(dict1.keys()))
        keys2 = sorted(list(dict2.keys()))

        self.assertEqual(keys1, keys2)

        for key in keys1:
            if isinstance(dict1[key], dict):
                self._recursive_check(dict1[key], dict2[key])
            elif isinstance(dict1[key], tf.Tensor):
                if not np.allclose(
                    dict1[key].numpy(),
                    dict2[key].numpy(),
                    atol=atol,
                    rtol=rtol,
                ):
                    print(f"key: {key}")
                    print(f"dict1[key]: {dict1[key].numpy()}")
                    print(f"dict2[key]: {dict2[key].numpy()}")
                self.assertTrue(
                    np.allclose(
                        dict1[key].numpy(),
                        dict2[key].numpy(),
                        atol=atol,
                        rtol=rtol,
                    )
                )
            elif dict1[key] is None:
                self.assertTrue(dict2[key] is None)
            else:
                if not np.allclose(
                    dict1[key], dict2[key], atol=atol, rtol=rtol
                ):
                    print(f"key: {key}")
                    print(f"dict1[key]: {dict1[key].numpy()}")
                    print(f"dict2[key]: {dict2[key].numpy()}")
                self.assertTrue(
                    np.allclose(dict1[key], dict2[key], atol=atol, rtol=rtol)
                )

    def get_test_data_file_path(self, tw_exclusions):
        """Get test data file path."""
        if tw_exclusions:
            file_name = "tw_exclusions.pkl"
        else:
            file_name = "no_exclusions.pkl"

        return os.path.join(
            SCRIPT_DIR,
            "test_data/result_tensors",
            self.model_name,
            file_name,
        )

    def write_test_data(self, result_tensors, tw_exclusions):
        """Write test data to file."""
        file_path = self.get_test_data_file_path(tw_exclusions)

        # save result_tensors to file
        out_dir = os.path.dirname(file_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(file_path, "wb") as f:
            pickle.dump(result_tensors, f, protocol=2)

    def load_test_data(self, tw_exclusions):
        """Load test data from file."""
        file_path = self.get_test_data_file_path(tw_exclusions)

        with open(file_path, "rb") as f:
            result_tensors = pickle.load(f)

        return result_tensors

    def test_model_prediction__no_exclusions(self):
        """Test model prediction without exclusions."""

        # get parameters for event hypothesis
        params = self.get_params()

        # run TF and get model expectation
        result_tensors = self.get_dom_expectation(
            params,
            x_pulses=self.x_pulses,
            x_pulses_ids=self.x_pulses_ids,
        )

        # check keys in result_tensors
        self.check_keys(result_tensors)

        # # save result_tensors to file
        # self.write_test_data(result_tensors, tw_exclusions=False)

        # load true values
        result_tensors_true = self.load_test_data(tw_exclusions=False)

        # check against true values
        self.check_result_tensors(result_tensors, result_tensors_true)

        # check correct values
        self.assertEqual(result_tensors["dom_charges"].shape, (1, 86, 60, 1))
        self.assertEqual(
            result_tensors["dom_charges_variance"].shape, (1, 86, 60, 1)
        )
        self.assertEqual(np.sum(result_tensors["dom_cdf_exclusion_sum"]), 0.0)

    def test_model_prediction__tw_exclusions(self):
        """Test model prediction without exclusions."""

        # get parameters for event hypothesis
        params = self.get_params()

        # run TF and get model expectation
        result_tensors = self.get_dom_expectation(
            params,
            x_pulses=self.x_pulses,
            x_pulses_ids=self.x_pulses_ids,
            x_time_window=self.x_time_window,
            x_time_exclusions=self.x_time_exclusions,
            x_time_exclusions_ids=self.x_time_exclusions_ids,
        )

        # check keys in result_tensors
        self.check_keys(result_tensors)

        # # save result_tensors to file
        # self.write_test_data(result_tensors, tw_exclusions=True)

        # load true values
        result_tensors_true = self.load_test_data(tw_exclusions=True)

        # check against true values
        self.check_result_tensors(result_tensors, result_tensors_true)

        # check correct values
        self.assertEqual(result_tensors["dom_charges"].shape, (1, 86, 60, 1))
        self.assertEqual(
            result_tensors["dom_charges_variance"].shape, (1, 86, 60, 1)
        )
        self.assertTrue(np.sum(result_tensors["dom_cdf_exclusion_sum"]) > 0.0)


class TestModelPredictionMultiSource(TestModelPrediction):
    """Test the model prediction of a multi-cascade  model.

    The model is loaded from a directory and the prediction is tested.
    """

    def definte_settings(self):
        self.model_name = "starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002_01"
        self.params_dict = {
            "cascade_x": 100,
            "cascade_y": -100,
            "cascade_z": -300,
            "cascade_zenith": 150,
            "cascade_azimuth": 177,
            "cascade_energy": 100000,
            "cascade_time": 0,
            "cascade_cascade_00001_energy": 100000,
            "cascade_cascade_00001_distance": 100,
        }

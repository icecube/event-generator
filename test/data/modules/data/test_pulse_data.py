import unittest
import os
import importlib
import numpy as np
from copy import deepcopy

from egenerator.data.modules.data.pulse_data import PulseDataModule
from egenerator.data.tensor import DataTensorList, DataTensor


class TestPulseDataModule(unittest.TestCase):
    """Test pulse data module."""

    def setUp(self):
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            "../../../test_data/cascade_mesc_l5_nue_low.hdf5",
        )

        self.times_0_to_9 = [
            6697.0,
            6722.0,
            4481.0,
            12141.0,
            23446.0,
            23471.0,
            11632.0,
            21256.0,
            9313.0,
            12314.0,
        ]
        self.charges_0_to_9 = [
            1.4249999523162842,
            0.5249999761581421,
            0.925000011920929,
            1.975000023841858,
            2.125,
            0.625,
            0.5249999761581421,
            1.4249999523162842,
            0.9750000238418579,
            1.1749999523162842,
        ]
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
        self.times_last_5 = [19050.0, 10525.0, 10784.0, 11563.0, 10585.0]
        self.charges_last_5 = [
            0.574999988079071,
            1.475000023841858,
            1.375,
            0.7749999761581421,
            1.0750000476837158,
        ]

        # Note: string and DOM ids are zero based here
        self.charges_dict = {
            (0, 25, 48): np.array(
                [
                    0.72500002,
                    1.52499998,
                    26.92499924,
                    23.02499962,
                    4.7249999,
                    0.625,
                    3.57500005,
                    1.42499995,
                    3.42499995,
                    2.625,
                    2.125,
                    1.07500005,
                    1.82500005,
                    0.32499999,
                    0.57499999,
                    0.22499999,
                    3.32500005,
                    0.22499999,
                    0.97500002,
                    0.32499999,
                    1.32500005,
                    0.57499999,
                    0.22499999,
                    1.47500002,
                    0.92500001,
                    1.125,
                    1.625,
                    0.82499999,
                    0.92500001,
                    0.82499999,
                ]
            ),
            (0, 85, 39): np.array([0.92500001]),
            (0, 85, 38): np.array([1.22500002, 1.47500002]),
            (1, 34, 53): np.array(
                [
                    0.27500001,
                    1.77499998,
                    4.9749999,
                    11.27499962,
                    3.32500005,
                    1.52499998,
                    2.07500005,
                    4.2750001,
                    2.0250001,
                    1.875,
                    0.47499999,
                    1.42499995,
                    1.875,
                    1.57500005,
                    0.77499998,
                    1.125,
                    1.52499998,
                    0.175,
                    0.22499999,
                    1.02499998,
                    0.22499999,
                    0.77499998,
                    1.07500005,
                    0.32499999,
                    1.57500005,
                    1.375,
                ]
            ),
            (1, 34, 50): np.array([0.47499999, 0.875]),
            (1, 34, 45): np.array([0.52499998]),
        }

        self.total_event_charge = np.array([1080.0499, 668.4249])

        self.dom_exclusions_key = np.ones([2, 86, 60, 1], dtype=bool)
        self.dom_exclusions_key[0, 25, 47] = False
        self.dom_exclusions_key[0, 25, 48] = False
        self.dom_exclusions_key[0, 25, 49] = False
        self.dom_exclusions_key[1, 34, 53] = False
        self.dom_exclusions_key[1, 34, 54] = False
        self.dom_exclusions_key[1, 34, 55] = False

        self.config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": "BrightDOMs",
            "time_exclusions_key": None,
            "float_precision": "float64",
            "add_charge_quantiles": False,
        }

    def test_class_initialization_parameters(self):
        """Check that initializer only tak"""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "pulse_key",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float32",
            "add_charge_quantiles": False,
        }

        with self.assertRaises(TypeError) as context:
            module = PulseDataModule()
            mod_config = dict(deepcopy(config))
            mod_config["pulse_key"] = 4
            module.configure(**mod_config)

        with self.assertRaises(ValueError) as context:
            module = PulseDataModule()
            mod_config = dict(deepcopy(config))
            mod_config["float_precision"] = "float31"
            module.configure(**mod_config)
        self.assertTrue("Invalid dtype str" in str(context.exception))

        module = PulseDataModule()
        module.configure(**config)

    def test_member_variables(self):
        """Test if member variables have correct values."""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "pulse_key",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float32",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        self.assertEqual(module.configuration.config, config)

    def test_configuration_check(self):
        """Check whether passed tensor list in configuration is checked and
        found to be wrong.
        """
        config = {
            "pulse_key": "pulse_key",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float32",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            module.configure(config_data=DataTensorList([]), **config)
        self.assertTrue(" != " in str(context.exception))

        # passing a file path should be fine
        module.configure(config_data="file_path_string", **config)

    def test_correct_configuration(self):
        """Check if the module correctly creates the tensors"""
        for dom_exclusions_key in [None, "exclusion_key"]:
            for float_precision in ["float32", "float64"]:
                config = {
                    "config_data": "file_path_string",
                    "pulse_key": "pulse_key",
                    "dom_exclusions_key": dom_exclusions_key,
                    "time_exclusions_key": None,
                    "float_precision": float_precision,
                    "add_charge_quantiles": False,
                }
                module = PulseDataModule()
                module.configure(**config)

                tensors_true = DataTensorList(
                    [
                        DataTensor(
                            name="x_dom_charge",
                            shape=[None, 86, 60, 1],
                            tensor_type="data",
                            dtype=float_precision,
                            trafo=True,
                            trafo_reduce_axes=(1, 2),
                            trafo_log=True,
                            trafo_batch_axis=0,
                        ),
                        DataTensor(
                            name="x_dom_exclusions",
                            shape=[None, 86, 60, 1],
                            tensor_type="data",
                            dtype="bool",
                            exists=dom_exclusions_key is not None,
                        ),
                        DataTensor(
                            name="x_pulses",
                            shape=[None, 2],
                            tensor_type="data",
                            vector_info={
                                "type": "value",
                                "reference": "x_pulses_ids",
                            },
                            dtype=float_precision,
                        ),
                        DataTensor(
                            name="x_pulses_ids",
                            shape=[None, 3],
                            tensor_type="data",
                            vector_info={
                                "type": "index",
                                "reference": "x_pulses",
                            },
                            dtype="int32",
                        ),
                        DataTensor(
                            name="x_time_exclusions",
                            shape=[None, 2],
                            tensor_type="data",
                            vector_info={
                                "type": "value",
                                "reference": "x_time_exclusions_ids",
                            },
                            dtype=float_precision,
                            exists=False,
                        ),
                        DataTensor(
                            name="x_time_exclusions_ids",
                            shape=[None, 3],
                            tensor_type="data",
                            vector_info={
                                "type": "index",
                                "reference": "x_time_exclusions",
                            },
                            dtype="int32",
                            exists=False,
                        ),
                    ]
                )
                self.assertTrue(module.data["data_tensors"] == tensors_true)

                # make sure the internal check also works
                config = {
                    "config_data": tensors_true,
                    "pulse_key": "pulse_key",
                    "dom_exclusions_key": dom_exclusions_key,
                    "time_exclusions_key": None,
                    "float_precision": float_precision,
                    "add_charge_quantiles": False,
                }
                module = PulseDataModule()
                module.configure(**config)

    def test_get_data_from_hdf_skip_file(self):
        """Check if file is skipped correctly if a label does not exist."""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "pulse_key",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float32",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)

        self.assertEqual(num_events, None)
        self.assertEqual(data, None)

    def test_get_data_from_hdf_missing_exclusion_key(self):
        """Check if missing exclusion key is handled as if there were no
        exclusions.
        """
        config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": "missing_key",
            "time_exclusions_key": None,
            "float_precision": "float32",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertTrue((data[1] == np.ones([2, 86, 60, 1], dtype=bool)).all())

    def test_get_data_from_hdf_check_configured(self):
        """Check if error is raised when not configured"""
        module = PulseDataModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.get_data_from_hdf("wrong_file_path")
        self.assertTrue("Module not configured yet!" in str(context.exception))

    def test_get_data_from_frame_check_configured(self):
        """Check if error is raised when not configured"""

        # This test requires IceCube metaproject to be loaded
        if not importlib.util.find_spec("dataclasses"):
            raise unittest.SkipTest("IceCube metaproject not loaded")

        module = PulseDataModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.get_data_from_frame("wrong_file_path")
        self.assertTrue("Module not configured yet!" in str(context.exception))

    def test_write_data_to_frame_check_configured(self):
        """Check if error is raised when not configured"""
        module = PulseDataModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.write_data_to_frame(None, None)
        self.assertTrue("Module not configured yet!" in str(context.exception))

    def test_create_data_from_frame_check_configured(self):
        """Check if error is raised when not configured"""

        # This test requires IceCube metaproject to be loaded
        if not importlib.util.find_spec("dataclasses"):
            raise unittest.SkipTest("IceCube metaproject not loaded")

        module = PulseDataModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.create_data_from_frame("wrong_file_path")
        self.assertTrue("Module not configured yet!" in str(context.exception))

    def test_get_data_from_hdf_wrong_file_name(self):
        """Check if IOError is raised if file does not exist"""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float32",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        with self.assertRaises(IOError) as context:
            num_events, data = module.get_data_from_hdf("wrong_file_path")
        self.assertTrue("does not exist" in str(context.exception))

    def test_get_data_from_hdf(self):
        """Test if loaded data is correct"""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float64",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertEqual(len(data), 6)
        self.assertEqual(data[1], None)
        self.assertEqual(data[4], None)
        self.assertEqual(data[5], None)

        # check specific values for pulse times
        self.assertListEqual(list(data[2][0:10, 1]), self.times_0_to_9)
        self.assertListEqual(list(data[2][618:628, 1]), self.times_618_to_627)
        self.assertListEqual(list(data[2][-5:, 1]), self.times_last_5)

        # check specific values for pulse charges
        self.assertListEqual(list(data[2][0:10, 0]), self.charges_0_to_9)
        self.assertListEqual(
            list(data[2][618:628, 0]), self.charges_618_to_627
        )
        self.assertListEqual(list(data[2][-5:, 0]), self.charges_last_5)

        # check total event charge
        event_sum = np.sum(data[0], axis=(1, 2, 3))
        self.assertTrue(np.allclose(self.total_event_charge, event_sum))

        # collect all pulses of an event and accumulate charge
        pulses = data[2]
        pulses_ids = data[3]
        total_charge = [
            np.sum(pulses[pulses_ids[:, 0] == 0][:, 0]),
            np.sum(pulses[pulses_ids[:, 0] == 1][:, 0]),
        ]
        self.assertTrue(np.allclose(self.total_event_charge, total_charge))

    def check_quantiles_of_dom(self, pulses, pulses_ids, event, string, dom):
        """Helper function to check if the charge quantiles are correct.

        Parameters
        ----------
        pulses : array_like
            The pulses. Shape: [n_pulses, 3]
        pulses_ids : array_like
            The pulses ids. Shape: [n_pulses, 3]
        event : int
            The (zero-based) event id.
        string : int
            The (zero-based) string number.
        dom : int
            The (zero-based) dom number.
        """

        # get true quantiles
        charges_true = self.charges_dict[(event, string, dom)]
        quantiles_true = np.cumsum(charges_true) / np.sum(charges_true)

        # get loaded quantiles
        pulse_mask = np.logical_and(
            pulses_ids[:, 0] == event, pulses_ids[:, 1] == string
        )
        pulse_mask = np.logical_and(pulse_mask, pulses_ids[:, 2] == dom)

        quantiles = pulses[pulse_mask][:, 2]

        self.assertTrue(np.allclose(quantiles_true, quantiles))

    def test_get_data_from_hdf_with_quantiles(self):
        """Test if loaded data is correct when loading quantiles as well"""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": None,
            "time_exclusions_key": None,
            "float_precision": "float64",
            "add_charge_quantiles": True,
        }
        module = PulseDataModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertEqual(len(data), 6)
        self.assertEqual(data[1], None)
        self.assertEqual(data[4], None)
        self.assertEqual(data[5], None)

        # check specific values for pulse times
        self.assertListEqual(list(data[2][0:10, 1]), self.times_0_to_9)
        self.assertListEqual(list(data[2][618:628, 1]), self.times_618_to_627)
        self.assertListEqual(list(data[2][-5:, 1]), self.times_last_5)

        # check specific values for pulse charges
        self.assertListEqual(list(data[2][0:10, 0]), self.charges_0_to_9)
        self.assertListEqual(
            list(data[2][618:628, 0]), self.charges_618_to_627
        )
        self.assertListEqual(list(data[2][-5:, 0]), self.charges_last_5)

        # check total event charge
        event_sum = np.sum(data[0], axis=(1, 2, 3))
        self.assertTrue(np.allclose(self.total_event_charge, event_sum))

        # collect all pulses of an event and accumulate charge
        pulses = data[2]
        pulses_ids = data[3]
        total_charge = [
            np.sum(pulses[pulses_ids[:, 0] == 0][:, 0]),
            np.sum(pulses[pulses_ids[:, 0] == 1][:, 0]),
        ]
        self.assertTrue(np.allclose(self.total_event_charge, total_charge))

        # check specific values for pulse charge quantiles
        for key in self.charges_dict.keys():
            self.check_quantiles_of_dom(pulses, pulses_ids, *key)

        # no quantile should be zero or less, or greater than 1:
        self.assertTrue(np.all(pulses[:, 2] > 0.0))
        self.assertTrue(np.all(pulses[:, 2] <= 1.0))

    def test_get_data_from_hdf_with_dom_exclusions_and_float64(self):
        """Test if loaded data is correct"""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": "BrightDOMs",
            "time_exclusions_key": None,
            "float_precision": "float64",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertEqual(len(data), 6)
        self.assertEqual(data[4], None)
        self.assertEqual(data[5], None)

        # check specific values for pulse times
        self.assertListEqual(list(data[2][0:10, 1]), self.times_0_to_9)
        self.assertListEqual(list(data[2][618:628, 1]), self.times_618_to_627)
        self.assertListEqual(list(data[2][-5:, 1]), self.times_last_5)

        # check specific values for pulse charges
        self.assertListEqual(list(data[2][0:10, 0]), self.charges_0_to_9)
        self.assertListEqual(
            list(data[2][618:628, 0]), self.charges_618_to_627
        )
        self.assertListEqual(list(data[2][-5:, 0]), self.charges_last_5)

        # check total event charge
        event_sum = np.sum(data[0], axis=(1, 2, 3))
        self.assertTrue(np.allclose(self.total_event_charge, event_sum))

        # collect all pulses of an event and accumulate charge
        pulses = data[2]
        pulses_ids = data[3]
        total_charge = [
            np.sum(pulses[pulses_ids[:, 0] == 0][:, 0]),
            np.sum(pulses[pulses_ids[:, 0] == 1][:, 0]),
        ]
        self.assertTrue(np.allclose(self.total_event_charge, total_charge))

        # check dom exclusions
        self.assertTrue((data[1] == self.dom_exclusions_key).all())

    def test_get_data_from_hdf_not_implemented_time_exlcusions(self):
        """Test if not implemented error is thrown if time exclusions are used"""
        config = {
            "config_data": "dummy_data",
            "pulse_key": "InIceDSTPulses",
            "dom_exclusions_key": "BrightDOMs",
            "time_exclusions_key": "BrightDOMs",
            "float_precision": "float64",
            "add_charge_quantiles": False,
        }
        module = PulseDataModule()
        module.configure(**config)

        with self.assertRaises(NotImplementedError):
            module.get_data_from_hdf(self.file_path)


if __name__ == "__main__":
    unittest.main()

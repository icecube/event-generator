import unittest

from egenerator.data.modules.weights.dummy import DummyWeightModule
from egenerator.data.tensor import DataTensorList


class TestDummyWeightModule(unittest.TestCase):
    """Test dummy weight module."""

    def test_member_variables(self):
        """Test if member variables have correct values."""
        dummy_module = DummyWeightModule()
        self.assertEqual(dummy_module.data, None)
        self.assertEqual(dummy_module.is_configured, False)
        self.assertEqual(dummy_module.configuration, None)
        self.assertEqual(dummy_module.untracked_data, {})
        self.assertEqual(dummy_module.sub_components, {})

    def test_configuration_data_type_check(self):
        dummy_module = DummyWeightModule()

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            dummy_module.configure(config_data=4)
        self.assertTrue("Unknown type:" in str(context.exception))

    def test_configuration(self):
        dummy_module = DummyWeightModule()

        # check if model is unconfigured
        self.assertEqual(dummy_module.is_configured, False)

        # now configure model
        dummy_module.configure(config_data=None)
        data_tensors = dummy_module.data["weight_tensors"]

        # check if model is now configured
        self.assertEqual(dummy_module.is_configured, True)

        # check if configured data_tensors are correct
        self.assertEqual(data_tensors, DataTensorList([]))
        self.assertEqual(
            dummy_module.configuration.config, {"config_data": None}
        )
        self.assertEqual(
            dummy_module.configuration.settings, {"config_data": None}
        )

    def test_dummy_data_method(self):
        """Test the dummy weight data loading method"""
        dummy_module = DummyWeightModule()
        dummy_module.configure(config_data=None)
        num_events, values = dummy_module.get_data_from_hdf(None)
        self.assertEqual(num_events, None)
        self.assertEqual(values, (None,))


if __name__ == "__main__":
    unittest.main()

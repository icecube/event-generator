#!/usr/local/bin/python3

import unittest

from egenerator.data.modules.misc.dummy import DummyMiscModule
from egenerator.data.tensor import DataTensorList


class TestDummyMiscModule(unittest.TestCase):

    """Test dummy misc module.
    """

    def test_member_variables(self):
        """Test if member variables have correct values.
        """
        dummy_module = DummyMiscModule()
        self.assertEqual(dummy_module.skip_check_keys, [])
        self.assertEqual(dummy_module.settings, {})

    def test_configuration_data_type_check(self):
        dummy_module = DummyMiscModule()

        # check if error is correctly rasied when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            dummy_module.configure(config_data=4)
        self.assertTrue('Unknown type:' in str(context.exception))

    def test_configuration(self):
        dummy_module = DummyMiscModule()

        # check if model is unconfigured
        self.assertEqual(dummy_module.is_configured, False)

        # check if error is correctly used when trying to use unconfigured
        with self.assertRaises(ValueError) as context:
            dummy_module.get_skip_check_keys()
        self.assertTrue('Module must first be configured!' in
                        str(context.exception))

        # now configure model
        data_tensors = dummy_module.configure(config_data=None)

        # check if model is now configured
        self.assertEqual(dummy_module.is_configured, True)

        # check if configured data_tensors are correct
        self.assertEqual(data_tensors, DataTensorList([]))
        self.assertEqual(dummy_module.get_skip_check_keys(), [])
        self.assertEqual(dummy_module.skip_check_keys, [])
        self.assertEqual(dummy_module.settings, {})

    def test_dummy_data_method(self):
        """Test the dummy weight data loading method
        """
        dummy_module = DummyMiscModule()
        data_tensors = dummy_module.configure(None)
        num_events, values = dummy_module.get_data_from_hdf(None)
        self.assertEqual(num_events, None)
        self.assertEqual(values, (None,))


if __name__ == '__main__':
    unittest.main()

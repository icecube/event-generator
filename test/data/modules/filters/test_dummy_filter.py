#!/usr/local/bin/python3
import unittest
import numpy as np

from egenerator.data.modules.filters.dummy import DummyFilterModule


class TestDummyFilterModule(unittest.TestCase):

    """Test dummy filter module.
    """

    def test_member_variables(self):
        """Test if member variables have correct values.
        """
        dummy_module = DummyFilterModule()
        self.assertEqual(dummy_module.data, None)
        self.assertEqual(dummy_module.is_configured, False)
        self.assertEqual(dummy_module.configuration, None)
        self.assertEqual(dummy_module.untracked_data, {})
        self.assertEqual(dummy_module.sub_components, {})

    def test_configuration_data_type_check(self):
        dummy_module = DummyFilterModule()

        # check that dummy filter module does not care what it gets passed
        dummy_module.configure(config_data=4)

        # check that it does not get configured twice
        with self.assertRaises(ValueError) as context:
            dummy_module.configure(config_data=None)
        self.assertTrue('Component is already configured!'
                        in str(context.exception))

    def test_dummy_filter_method(self):
        """Test the dummy filter method
        """
        dummy_module = DummyFilterModule()
        dummy_module.configure(config_data=None)
        mask = dummy_module.get_event_filter_mask_from_hdf(
            None, 'tensors', 42, 1337)
        self.assertEqual(np.sum(mask), 42)
        self.assertEqual(len(mask), 42)


if __name__ == '__main__':
    unittest.main()

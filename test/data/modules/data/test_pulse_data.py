#!/usr/local/bin/python3

import unittest
import os
import numpy as np

from egenerator.data.modules.data.pulse_data import PulseDataModule
from egenerator.data.tensor import DataTensorList, DataTensor


class TestPulseDataModule(unittest.TestCase):

    """Test pulse data module.
    """
    def test_class_initialization_parameters(self):
        """Check that initializer only tak
        """
        with self.assertRaises(TypeError) as context:
            PulseDataModule(config_data=4)

        with self.assertRaises(TypeError) as context:
            PulseDataModule(4, label_key='labels')

        with self.assertRaises(TypeError) as context:
            PulseDataModule(4, True)

        with self.assertRaises(AttributeError) as context:
            PulseDataModule('pulse_key', None, None, 'float31')
        self.assertTrue("has no attribute 'float31'" in str(context.exception))

        module = PulseDataModule('pulse_key', None, None, 'float32')

    def test_member_variables(self):
        """Test if member variables have correct values.
        """
        module = PulseDataModule('pulse_key', None, None, 'float32')
        self.assertEqual(module.skip_check_keys,
                         ['pulse_key', 'dom_exclusions_key',
                          'time_exclusions_key'])
        settings_true = {
            'pulse_key': 'pulse_key',
            'dom_exclusions_key': None,
            'time_exclusions_key': None,
            'float_precision': 'float32',
        }
        self.assertEqual(module.settings, settings_true)

    def test_configuration_check(self):
        """Check whether passed tensor list in confguration is checked and
        found to be wrong.
        """
        module = PulseDataModule('pulse_key', None, None, 'float32')

        # check if error is correctly rasied when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            module.configure(config_data=DataTensorList([]))
        self.assertTrue(' != ' in str(context.exception))

        # pasing a file path should be fine
        module.configure(config_data='file_path_string')

    def test_correct_configuration(self):
        """Check if the module correctly creates the tensors
        """
        for dom_exclusions_key in [None, 'exclusion_key']:
            for float_precision in ['float32', 'float64']:
                module = PulseDataModule('pulse_key', dom_exclusions_key, None,
                                         float_precision)
                tensors = module.configure(config_data='file_path_string')
                tensors_true = DataTensorList([
                    DataTensor(name='x_dom_charge',
                               shape=[None, 86, 60, 1],
                               tensor_type='data',
                               dtype=float_precision,
                               trafo=True,
                               trafo_reduce_axes=(1, 2),
                               trafo_log=True,
                               trafo_batch_axis=0),
                    DataTensor(name='x_dom_exclusions',
                               shape=[None, 86, 60, 1],
                               tensor_type='data',
                               dtype='bool',
                               exists=dom_exclusions_key is not None),
                    DataTensor(name='x_pulses',
                               shape=[None, 2],
                               tensor_type='data',
                               dtype=float_precision),
                    DataTensor(name='x_pulses_ids',
                               shape=[None, 3],
                               tensor_type='data',
                               dtype='int32'),
                    DataTensor(name='x_time_exclusions',
                               shape=[None, 2],
                               tensor_type='data',
                               dtype=float_precision,
                               exists=False),
                    DataTensor(name='x_time_exclusions_ids',
                               shape=[None, 3],
                               tensor_type='data',
                               dtype='int32',
                               exists=False),
                    ])
                self.assertTrue(tensors == tensors_true)

                # make sure the internal check also works
                module = PulseDataModule('pulse_key', dom_exclusions_key, None,
                                         float_precision)
                tensors = module.configure(config_data=tensors_true)

    def test_not_implemented_get_data_from_frame(self):
        """Check not implemented method
        """
        module = PulseDataModule('pulse_key', None, None, 'float32')

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(NotImplementedError) as context:
            module.get_data_from_frame(None)

    def test_not_implemented_create_data_from_frame(self):
        """Check not implemented method
        """
        module = PulseDataModule('pulse_key', None, None, 'float32')

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(NotImplementedError) as context:
            module.create_data_from_frame(None)

    def test_not_implemented_write_data_to_frame(self):
        """Check not implemented method
        """
        module = PulseDataModule('pulse_key', None, None, 'float32')

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(NotImplementedError) as context:
            module.write_data_to_frame(None, None)


if __name__ == '__main__':
    unittest.main()

#!/usr/local/bin/python3

import unittest
import os
import numpy as np
from copy import deepcopy

from egenerator.data.handler.modular import ModuleDataHandler
from egenerator.data.tensor import DataTensor, DataTensorList
from egenerator.data.modules.data.pulse_data import PulseDataModule
from egenerator.data.modules.labels.cascades import CascadeGeneratorLabelModule
from egenerator.data.modules.weights.dummy import DummyWeightModule
from egenerator.data.modules.misc.dummy import DummyMiscModule
from egenerator.data.modules.filters.dummy import DummyFilterModule


class TestModuleDataHandler(unittest.TestCase):

    """Test base data handler class.
    """
    def setUp(self):

        # setup module
        self.config = {
            # settings for the data module
            'data_module': 'pulse_data.PulseDataModule',
            'data_settings': {
                'pulse_key': 'AggregatedPulses_combined',
                'dom_exclusions_key': None,
                'time_exclusions_key': None,
                'float_precision': 'float32',
            },

            # settings for the label module
            'label_module': 'cascades.CascadeGeneratorLabelModule',
            'label_settings': {
                'shift_cascade_vertex': False,
                # logarithm on labels (x, y, z, zenith, azimuth, energy, time)?
                'trafo_log': [False, False, False, False, False, True, False],
                'label_key': 'LabelsDeepLearning',
                'float_precision': 'float32',
            },

            # settings for the weight module
            'weight_module': 'dummy.DummyWeightModule',
            'weight_settings': {},

            # settings for the misc module
            'misc_module': 'dummy.DummyMiscModule',
            'misc_settings': {},

            # settings for the filter module
            'filter_module': 'dummy.DummyFilterModule',
            'filter_settings': {},
        }

        # modify tensor data
        self.config2 = dict(deepcopy(self.config))
        self.config2['label_settings']['trafo_log'] = \
            [False, False, False, False, False, False, False]

        # # create handler object
        # data_handler = ModuleDataHandler()
        # data_handler.setup(config)
        # self.data_handler = data_handler

    def test_object_initialization(self):
        data_handler = ModuleDataHandler()
        self.assertEqual(data_handler.tensors, None)
        self.assertEqual(data_handler.config, None)
        self.assertEqual(data_handler.skip_check_keys, None)
        self.assertEqual(data_handler._mp_processes, [])
        self.assertEqual(data_handler._mp_managers, [])
        self.assertEqual(data_handler._is_setup, False)

        self.assertEqual(data_handler.modules_are_loaded, False)
        self.assertEqual(data_handler.data_module, None)
        self.assertEqual(data_handler.label_module, None)
        self.assertEqual(data_handler.weight_module, None)
        self.assertEqual(data_handler.misc_module, None)
        self.assertEqual(data_handler.filter_module, None)

        self.assertEqual(data_handler.data_tensors, None)
        self.assertEqual(data_handler.label_tensors, None)
        self.assertEqual(data_handler.weight_tensors, None)
        self.assertEqual(data_handler.misc_tensors, None)

    def test_method_load_modules(self):
        data_handler = ModuleDataHandler()

        # load modules
        data_handler._load_modules(self.config)

        self.assertTrue(isinstance(data_handler.data_module, PulseDataModule))
        self.assertTrue(isinstance(data_handler.label_module,
                                   CascadeGeneratorLabelModule))
        self.assertTrue(isinstance(data_handler.weight_module,
                                   DummyWeightModule))
        self.assertTrue(isinstance(data_handler.misc_module, DummyMiscModule))
        self.assertTrue(isinstance(data_handler.filter_module,
                                   DummyFilterModule))

        # load modules again
        with self.assertRaises(ValueError) as context:
            data_handler._load_modules(self.config)
        self.assertTrue('Modules have already been loaded!'
                        in str(context.exception))

    def test_method_assign_settings_of_derived_class_tensor_check(self):
        """Check if mismatches in the tensors are found
        """
        data_handler = ModuleDataHandler()
        data_handler.setup(self.config)
        print

        # emulate loading configuration from file
        # Since the settings match, this should work
        data_handler2 = ModuleDataHandler()
        data_handler2._assign_settings(
                data_handler.tensors, self.config,
                data_handler.skip_check_keys, True)

        # Now we are using the wrong config, so we have a mis-match and
        # and an error should be thrown
        with self.assertRaises(ValueError) as context:
            data_handler3 = ModuleDataHandler()
            data_handler3._assign_settings_of_derived_class(
                data_handler.tensors, self.config2, None)
        self.assertTrue(' != ' in str(context.exception))

    def test_method_setup_skip_keys(self):
        """Check if skip keys are correctly configured
        """
        data_handler = ModuleDataHandler()
        self.assertEqual(data_handler.skip_check_keys, None)

        data_handler.setup(self.config)

        self.assertListEqual(data_handler.skip_check_keys,
                             ['dom_exclusions_key', 'pulse_key',
                              'time_exclusions_key'])
        self.assertListEqual(data_handler.skip_check_keys,
                             data_handler.get_skip_check_keys())

    def test_method_check_if_setup(self):
        """Test if check if setup raises an error
        """
        data_handler = ModuleDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler.check_if_setup()
        self.assertTrue('Data handler needs to be set up first!'
                        in str(context.exception))

        # if we setup the data handler this shoudl run without any errors
        data_handler._is_setup = True
        data_handler.check_if_setup()

    def test_method_get_skip_check_keys_check_if_not_setup(self):
        """Test if check if setup raises an error
        """
        data_handler = ModuleDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler.get_skip_check_keys()
        self.assertTrue('Modules must first be loaded!'
                        in str(context.exception))

        # if we setup the data handler this should run without any errors
        data_handler.setup(self.config)
        data_handler.get_skip_check_keys()

    def test_method_configure_settings_already_setup_yet(self):
        data_handler = ModuleDataHandler()

        data_handler.setup(self.config)
        with self.assertRaises(ValueError) as context:
            data_handler.setup(self.config)
        self.assertTrue('The data handler is already set up!'
                        in str(context.exception))

    def test_methods_load_and_save(self):
        """Test the saving and loading of a previously created data handler obj.
        """
        data_handler = ModuleDataHandler()
        data_handler.setup(self.config)

        # save trafo model
        file_path = os.path.join(
            os.path.dirname(__file__),
            '../../../data/temp_test_files/pulse_data_handler/',
            'pulse_data_handler.yaml')

        # remove it if it already exists
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(os.path.dirname(file_path)):
            os.removedirs(os.path.dirname(file_path))

        data_handler.save(file_path, overwrite=False)
        data_handler.save(file_path, overwrite=True)

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            data_handler.save(file_path, overwrite=False)
        self.assertTrue(' already exists!' in str(context.exception))

        # check if loaded data handler has same settings
        data_handler_new = ModuleDataHandler()
        data_handler_new.load(file_path)

        self.assertListEqual(data_handler_new.skip_check_keys,
                             data_handler.skip_check_keys)
        self.assertDictEqual(data_handler_new.config,
                             data_handler.config)
        self.assertTrue(data_handler_new.tensors == data_handler.tensors)

    def test_not_implemented_method_get_data_from_frame(self):

        data_handler = ModuleDataHandler()
        data_handler.setup(self.config)

        with self.assertRaises(NotImplementedError) as context:
            data_handler.get_data_from_frame(None)

    def test_not_implemented_method_create_data_from_frame(self):

        data_handler = ModuleDataHandler()
        data_handler.setup(self.config)

        with self.assertRaises(NotImplementedError) as context:
            data_handler.create_data_from_frame(None)

    def test_not_implemented_method_write_data_to_frame(self):

        data_handler = ModuleDataHandler()
        data_handler.setup(self.config)

        with self.assertRaises(NotImplementedError) as context:
            data_handler.write_data_to_frame(None, None)

    # def test_method_batch_to_event_structure(self):
    #     """Test restructuring method which restructures a vector shape to
    #     a structure where the first dimension corresponds to the event id.
    #     """
    #     random_state = np.random.RandomState(42)
    #     values_list_true = []
    #     indices_list_true = []
    #     values = []
    #     indices = []
    #     num_events = 10
    #     for i in range(num_events):
    #         num_pulses_per_event = random_state.randint(0, 30)

    #         pulses = []
    #         index_list = []
    #         for p in range(num_pulses_per_event):
    #             pulse = random_state.uniform(size=3)
    #             index = [i, p]
    #             pulses.append(pulse)
    #             values.append(pulse)
    #             indices.append(index)
    #             index_list.append(index)
    #         values_list_true.append(pulses)
    #         indices_list_true.append(index_list)

    #     values = np.array(values)
    #     indices = np.array(indices)

    #     values_list, indices_list = self.data_handler.batch_to_event_structure(
    #         values, indices, num_events)

    #     for v1, v2 in zip(values_list, values_list_true):
    #         self.assertTrue(np.allclose(v1, v2))
    #     for v1, v2 in zip(indices_list, indices_list_true):
    #         self.assertTrue(np.allclose(v1, v2))


if __name__ == '__main__':
    unittest.main()

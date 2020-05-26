#!/usr/local/bin/python3

import unittest
import os
import shutil
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
            'config': {
                # settings for the data module
                'data_module': 'pulse_data.PulseDataModule',
                'data_settings': {
                    'pulse_key': 'AggregatedPulses_combined',
                    'dom_exclusions_key': None,
                    'time_exclusions_key': None,
                    'float_precision': 'float32',
                    'add_charge_quantiles': False,
                },

                # settings for the label module
                'label_module': 'cascades.CascadeGeneratorLabelModule',
                'label_settings': {
                    'shift_cascade_vertex': False,
                    # logarithm on labels
                    # (x, y, z, zenith, azimuth, energy, time)?
                    'trafo_log': [False, False, False, False, False, True,
                                  False],
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
        }

        # modify tensor data
        self.config2 = dict(deepcopy(self.config))
        self.config2['config']['label_settings']['trafo_log'] = \
            [False, False, False, False, False, False, False]

    def test_object_initialization(self):
        data_handler = ModuleDataHandler()
        self.assertEqual(data_handler.tensors, None)
        self.assertEqual(data_handler.configuration, None)
        self.assertEqual(data_handler._untracked_data['mp_processes'], [])
        self.assertEqual(data_handler._untracked_data['mp_managers'], [])
        self.assertEqual(data_handler.modules_are_loaded, False)
        self.assertEqual(data_handler._is_configured, False)

    def test_method_load_modules(self):
        data_handler = ModuleDataHandler()

        # load modules
        sub_components = data_handler._load_modules(**self.config)

        self.assertTrue(isinstance(sub_components['data_module'],
                                   PulseDataModule))
        self.assertTrue(isinstance(sub_components['label_module'],
                                   CascadeGeneratorLabelModule))
        self.assertTrue(isinstance(sub_components['weight_module'],
                                   DummyWeightModule))
        self.assertTrue(isinstance(sub_components['misc_module'],
                                   DummyMiscModule))
        self.assertTrue(isinstance(sub_components['filter_module'],
                                   DummyFilterModule))

        # load modules again
        with self.assertRaises(ValueError) as context:
            data_handler._load_modules(self.config)
        self.assertTrue('Modules have already been loaded!'
                        in str(context.exception))

    def test_method_check_if_configured(self):
        """Test if check if configured raises an error
        """
        data_handler = ModuleDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler.check_if_configured()
        self.assertTrue('Data handler needs to be set up first!'
                        in str(context.exception))

        # if we setup the data handler this shoudl run without any errors
        data_handler._is_configured = True
        data_handler.check_if_configured()

    def test_method_configure_settings_already_setup_yet(self):
        data_handler = ModuleDataHandler()

        data_handler.configure(**self.config)
        with self.assertRaises(ValueError) as context:
            data_handler.configure(**self.config)
        self.assertTrue('Component is already configured!'
                        in str(context.exception))

        with self.assertRaises(ValueError) as context:
            data_handler._configure(**self.config)
        self.assertTrue('The data handler is already set up!'
                        in str(context.exception))

    def test_methods_load_and_save(self):
        """Test the saving and loading of a previously created data handler obj.
        """
        data_handler = ModuleDataHandler()
        data_handler.configure(**self.config)

        # save data handler
        directory = os.path.join(
            os.path.dirname(__file__),
            '../../../data/temp_test_files/pulse_data_handler')

        # remove it if it already exists
        if os.path.exists(directory):
            shutil.rmtree(directory)

        data_handler.save(directory, overwrite=False)
        data_handler.save(directory, overwrite=True)

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            data_handler.save(directory, overwrite=False)
        self.assertTrue(' already exists!' in str(context.exception))

        # check if loaded data handler has same settings
        data_handler_new = ModuleDataHandler()
        data_handler_new.load(directory)

        self.assertDictEqual(data_handler_new.configuration.dict,
                             data_handler.configuration.dict)
        self.assertTrue(data_handler_new.tensors == data_handler.tensors)
        self.assertTrue(data_handler.is_compatible(data_handler_new))

    def test_method_batch_to_event_structure(self):
        """Test restructuring method which restructures a vector shape to
        a structure where the first dimension corresponds to the event id.
        """
        data_handler = ModuleDataHandler()
        data_handler.configure(**self.config)

        random_state = np.random.RandomState(42)
        values_list_true = []
        indices_list_true = []
        values = []
        indices = []
        num_events = 10
        for i in range(num_events):
            num_pulses_per_event = random_state.randint(0, 30)

            pulses = []
            index_list = []
            for p in range(num_pulses_per_event):
                pulse = random_state.uniform(size=3)
                index = [i, p]
                pulses.append(pulse)
                values.append(pulse)
                indices.append(index)
                index_list.append(index)
            values_list_true.append(pulses)
            indices_list_true.append(index_list)

        values = np.array(values)
        indices = np.array(indices)

        values_list, indices_list = data_handler.batch_to_event_structure(
            values, indices, num_events)

        for v1, v2 in zip(values_list, values_list_true):
            self.assertTrue(np.allclose(v1, v2))
        for v1, v2 in zip(indices_list, indices_list_true):
            self.assertTrue(np.allclose(v1, v2))


class TestModuleDataHandlerOnTestData(unittest.TestCase):

    """Test base data handler class while loading in test data
    """
    def setUp(self):

        # setup module
        config = {
            # settings for the data module
            'data_module': 'pulse_data.PulseDataModule',
            'data_settings': {
                'pulse_key': 'InIceDSTPulses',
                'dom_exclusions_key': 'BrightDOMs',
                'time_exclusions_key': None,
                'float_precision': 'float32',
                'add_charge_quantiles': False,
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
        config2 = dict(deepcopy(config))
        config2['label_settings']['label_key'] = 'wrong_key'

        # create handler object
        data_handler = ModuleDataHandler()
        data_handler.configure(config=config)
        self.data_handler = data_handler

        data_handler2 = ModuleDataHandler()
        data_handler2.configure(config=config2)
        self.data_handler2 = data_handler2

        # Test Data
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            '../../test_data/cascade_mesc_l5_nue_low.hdf5')

        self.times_618_to_627 = [11737.0, 10813.0, 10743.0, 10419.0, 10430.0,
                                 10439.0, 10481.0, 10440.0, 11055.0, 10487.0]
        self.charges_618_to_627 = [0.824999988079071, 1.024999976158142,
                                   1.774999976158142, 0.42500001192092896,
                                   1.774999976158142, 0.22499999403953552,
                                   1.1749999523162842, 0.32499998807907104,
                                   0.875, 1.4249999523162842]
        self.total_event_charge = np.array([1080.0499, 668.4249])

        self.dom_exclusions = np.ones([2, 86, 60, 1], dtype=bool)
        self.dom_exclusions[0, 25, 47] = False
        self.dom_exclusions[0, 25, 48] = False
        self.dom_exclusions[0, 25, 49] = False
        self.dom_exclusions[1, 34, 53] = False
        self.dom_exclusions[1, 34, 54] = False
        self.dom_exclusions[1, 34, 55] = False

        self.cascades_true = np.array([
            [-13.094963629354766, -197.0847391208472, -322.0192710148053,
             1.0771952265275238, 4.601483747646196, 2360.8997600199427,
             9663.551318717717],
            [-70.78487964475926, -32.47261211840669, -426.5132607462965,
             1.586083894785944, 1.5573642249002815, 924.7251046427211,
             9789.880753474426],
            ], dtype=np.float32)

    def test_get_data_from_hdf(self):
        """Test if loaded data is correct
        """

        num_events, data = self.data_handler.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertEqual(len(data), 7)
        self.assertEqual(
            data[self.data_handler.tensors.get_index('x_time_exclusions')],
            None)
        self.assertEqual(
            data[self.data_handler.tensors.get_index('x_time_exclusions_ids')],
            None)

        # check specific values for pulse times
        self.assertListEqual(list(data[3][618:628, 1]),
                             self.times_618_to_627)

        # check specific values for pulse charges
        self.assertListEqual(list(data[3][618:628, 0]),
                             self.charges_618_to_627)

        # check total event charge
        event_sum = np.sum(
            data[self.data_handler.tensors.get_index('x_dom_charge')],
            axis=(1, 2, 3))
        self.assertTrue(np.allclose(self.total_event_charge, event_sum))

        # collect all pulses of an event and accumulate charge
        pulses = data[self.data_handler.tensors.get_index('x_pulses')]
        pulses_ids = data[self.data_handler.tensors.get_index('x_pulses_ids')]
        total_charge = [np.sum(pulses[pulses_ids[:, 0] == 0][:, 0]),
                        np.sum(pulses[pulses_ids[:, 0] == 1][:, 0])]
        self.assertTrue(np.allclose(self.total_event_charge, total_charge))

        # check dom exclusions
        self.assertTrue(
            (data[self.data_handler.tensors.get_index('x_dom_exclusions')]
             == self.dom_exclusions).all())

    def test_get_data_from_hdf_missing_label_key(self):
        """Test if module properly skips a file if there is an error, such as
        that the label key is missing.
        """
        # check error message when attempting to overwrite file
        num_events, data = self.data_handler2.get_data_from_hdf(
                self.file_path)
        self.assertEqual(num_events, None)
        self.assertEqual(data, None)

    def test_get_batch_generator_test_one_epoch(self):
        """Test batch generator
        """
        iterator = self.data_handler.get_batch_generator(
                                            input_data=self.file_path,
                                            batch_size=1,
                                            sample_randomly=False,
                                            pick_random_files_forever=False,
                                            file_capacity=1,
                                            batch_capacity=1,
                                            num_jobs=1,
                                            num_add_files=0,
                                            num_repetitions=1)

        index = self.data_handler.tensors.get_index('x_parameters')
        for i, data_batch in enumerate(iterator):
            self.assertEqual(len(data_batch[0]), 1)
            self.assertTrue((self.cascades_true[i] == data_batch[index]).all())

        self.data_handler.kill()

    def test_get_batch_generator_forever(self):
        """Test batch generator
        """
        batch_size = 3
        iterator = self.data_handler.get_batch_generator(
                                            input_data=self.file_path,
                                            batch_size=batch_size,
                                            sample_randomly=True,
                                            pick_random_files_forever=True,
                                            file_capacity=1,
                                            batch_capacity=1,
                                            num_jobs=1,
                                            num_add_files=2,
                                            num_repetitions=3)
        param_index = self.data_handler.tensors.get_index('x_parameters')
        for i in range(10):
            data_batch = next(iterator)
            self.assertEqual(len(data_batch[0]), batch_size)
            self.assertTrue(
                (self.cascades_true[0] == data_batch[param_index][0]).all() or
                (self.cascades_true[1] == data_batch[param_index][0]).all())

        self.data_handler.kill()


if __name__ == '__main__':
    unittest.main()

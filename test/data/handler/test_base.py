#!/usr/local/bin/python3

import unittest
import os
import numpy as np

from egenerator.data.handler.base import BaseDataHandler
from egenerator.data.tensor import DataTensor, DataTensorList


class TestBaseDataHandler(unittest.TestCase):

    """Test base data handler class.
    """
    def setUp(self):
        # create handler object
        data_handler = BaseDataHandler()

        # fake setup
        data_handler._is_setup = True
        data_handler.config = {'setting1': 1337}
        data_handler.skip_check_keys = ['ignore_key']
        data_handler.tensors = DataTensorList([DataTensor(name='data_tensor',
                                                          shape=[None, 86, 1],
                                                          tensor_type='data',
                                                          dtype='float32')])
        self.data_handler = data_handler

    def test_object_initialization(self):
        data_handler = BaseDataHandler()
        self.assertEqual(data_handler.tensors, None)
        self.assertEqual(data_handler.config, None)
        self.assertEqual(data_handler.skip_check_keys, None)
        self.assertEqual(data_handler._mp_processes, [])
        self.assertEqual(data_handler._mp_managers, [])
        self.assertEqual(data_handler._is_setup, False)

    def test_method_check_if_setup(self):
        """Test if check if setup raises an error
        """
        data_handler = BaseDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler.check_if_setup()
        self.assertTrue('Data handler needs to be set up first!'
                        in str(context.exception))

        # if we setup the data handler this shoudl run without any errors
        data_handler._is_setup = True
        data_handler.check_if_setup()

    def test_method_configure_settings_not_setup_yet(self):
        data_handler = BaseDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler._is_setup = True
            data_handler._configure_settings([2, 3], {})
        self.assertTrue('The data handler is already set up!'
                        in str(context.exception))

    def test_method_configure_settings_wrong_tensor_type(self):
        data_handler = BaseDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler._configure_settings([2, 3], {})
        self.assertTrue('Unsupported type:' in str(context.exception))

    def test_method_configure_settings_pure_virtual_method(self):
        data_handler = BaseDataHandler()

        with self.assertRaises(NotImplementedError) as context:
            data_handler._configure_settings(DataTensorList([]), {})

    def test_method_setup_pure_virtual_method(self):
        data_handler = BaseDataHandler()

        # with no config data
        with self.assertRaises(NotImplementedError) as context:
            data_handler.setup(None)

        # with a file path string
        with self.assertRaises(NotImplementedError) as context:
            data_handler.setup(None, 'dummy_file_path')

        # with a list of file path strings
        with self.assertRaises(NotImplementedError) as context:
            data_handler.setup(None, ['dummy_file_path1', 'dummy_file_path2'])

    def test_methods_load_and_save(self):
        """Test the saving and loading of a previously created data handler obj.
        """

        # save trafo model
        file_path = os.path.join(
            os.path.dirname(__file__),
            '../../../data/temp_test_files/data_handler/data_handler.yaml')

        # remove it if it already exists
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(os.path.dirname(file_path)):
            os.removedirs(os.path.dirname(file_path))

        self.data_handler.save(file_path, overwrite=False)
        self.data_handler.save(file_path, overwrite=True)

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            self.data_handler.save(file_path, overwrite=False)
        self.assertTrue(' already exists!' in str(context.exception))

        # check error message when attempting to load file file
        data_handler_new = BaseDataHandler()
        with self.assertRaises(NotImplementedError) as context:
            data_handler_new.load(file_path)

    def test_method_check_data_structure(self):

        # create data
        correct_data = np.ones([2, 86, 1])
        wrong_rank_data = np.ones([2, 1])
        wrong_shape_data = np.ones([2, 34, 1])

        # check error message when length does not match
        with self.assertRaises(ValueError) as context:
            self.data_handler._check_data_structure(wrong_rank_data)
        self.assertTrue('Length' in str(context.exception))

        # check error message when rank does not match
        with self.assertRaises(ValueError) as context:
            self.data_handler._check_data_structure([wrong_rank_data])
        self.assertTrue('Rank' in str(context.exception))

        # check error message when shapes does not match
        with self.assertRaises(ValueError) as context:
            self.data_handler._check_data_structure([wrong_shape_data])
        self.assertTrue('Shapes' in str(context.exception))

        # check not implemented error message when trying to check
        # vector data tensors (tensor.vector_info is not None)
        with self.assertRaises(NotImplementedError) as context:
            self.data_handler._check_data_structure([wrong_shape_data], True)

    def test_not_implemented_method_get_data_from_hdf(self):
        with self.assertRaises(NotImplementedError) as context:
            self.data_handler.get_data_from_hdf('file_name_path')

    def test_not_implemented_method_get_data_from_frame(self):
        with self.assertRaises(NotImplementedError) as context:
            self.data_handler.get_data_from_frame(None)

    def test_not_implemented_method_create_data_from_frame(self):
        with self.assertRaises(NotImplementedError) as context:
            self.data_handler.create_data_from_frame(None)

    def test_not_implemented_method_write_data_to_frame(self):
        with self.assertRaises(NotImplementedError) as context:
            self.data_handler.write_data_to_frame(None, None)

    def test_method_batch_to_event_structure(self):
        """Test restructuring method which restructures a vector shape to
        a structure where the first dimension corresponds to the event id.
        """
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

        values_list, indices_list = self.data_handler.batch_to_event_structure(
            values, indices, num_events)

        for v1, v2 in zip(values_list, values_list_true):
            self.assertTrue(np.allclose(v1, v2))
        for v1, v2 in zip(indices_list, indices_list_true):
            self.assertTrue(np.allclose(v1, v2))


if __name__ == '__main__':
    unittest.main()

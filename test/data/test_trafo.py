#!/usr/local/bin/python3

import unittest
import os
import numpy as np

from egenerator.data.trafo import DataTransformer
from egenerator.data.tensor import DataTensor, DataTensorList


class DummyDataHandler(object):

    """A dummy data handler that can be used to test the DataTransformer class.
    """

    def __init__(self, is_setup=True, config={'setting1': 42},
                 skip_keys=['key_to_ignore'], trafo=True, trafo_log=None,
                 trafo_log_axis=-1, seed=1337):
        """Initialize DummyDataHandler object

        Parameters
        ----------
        is_setup : bool, optional
            Sets the member variable 'is_setup' to this value.
        config : dict, optional
            A dictionary containing settings for the data handler class.
        skip_keys : list, optional
            A list of keys that do not need to match when checking for
            correct settings.
        trafo : bool, optional
            Indicates whether or not the data tensors should be transformed.
        trafo_log : bool, optional
            Whether or not to perform logarithm on values during
            transformation.
        trafo_log_axis : int, optional
            The axis along which to perform logarithm.
        seed : int, optional
            Random number generator seed.
        """
        if isinstance(trafo_log, bool) or trafo_log is None:
            trafo_log = [trafo_log, trafo_log]
        self.trafo_log = trafo_log
        self.is_setup = is_setup
        self.config = config
        self.skip_check_keys = skip_keys

        # create a list of tensors
        tensors = [
            DataTensor(name='data_tensor',
                       shape=[None, 86, 60, 1],
                       tensor_type='data',
                       dtype='float32',
                       trafo=trafo,
                       trafo_log=trafo_log[0],
                       trafo_log_axis=trafo_log_axis),
            DataTensor(name='label_tensor',
                       shape=[None, 7],
                       tensor_type='label',
                       dtype='float32',
                       trafo=trafo,
                       trafo_log=trafo_log[1],
                       trafo_log_axis=trafo_log_axis),
        ]
        self.tensors = DataTensorList(tensors)

        self.n_batches = 5
        self.batch_size = 4
        self.random_state = np.random.RandomState(seed)
        self.data_tensor_values = self.random_state.uniform(
            size=(self.n_batches, self.batch_size, 86, 60, 1))
        self.label_tensor_values = self.random_state.uniform(
            size=(self.n_batches, self.batch_size, 7))

    def create_data_iterator(self):
        """Create a python iterator object
        """
        def iterator():
            batch_counter = 0
            for i in range(self.n_batches):
                yield (self.data_tensor_values[i], self.label_tensor_values[i])
        return iterator()


class TestDataTransformer(unittest.TestCase):

    """Test data transformer class.
    """

    def test_initialization_with_unconfigured_data_handler(self):
        """The data trafo init should raise an error if attempting to
        instantiate a DataTransformer object from an unconfigured data handler
        object.
        """
        data_handler = DummyDataHandler(is_setup=False)
        with self.assertRaises(ValueError) as context:
            DataTransformer(data_handler)

        self.assertTrue('Data Handler is not set up!'
                        in str(context.exception))

    def test_correct_initialization(self):
        """Check if member variables are correctly set when instantiating
        a DataTransformer object.
        """
        config_list = [
            {'setting1': 42},
            {},
            {'setting1': None, 'setting2': 133, None: 'another_value'},
        ]
        skip_keys_list = [
            [],
            ['key_to_ignore'],
            ['key_to_ignore', 'another_key_to_ignore'],
        ]
        for config, skip_keys in zip(config_list, skip_keys_list):
            data_handler = DummyDataHandler(config=config,
                                            skip_keys=skip_keys)
            data_trafo = DataTransformer(data_handler)

            self.assertEqual(data_trafo._setup_complete, False)
            self.assertEqual(data_trafo.trafo_model['config'], config)
            self.assertEqual(data_trafo.trafo_model['skip_keys'], skip_keys)
            self.assertTrue(
                data_trafo.trafo_model['tensors'] == data_handler.tensors)

    def test_update_online_variance_vars(self):
        """Check if computed mean and variance by online calculation are
        correct.
        """
        num_values = 100
        x = np.random.uniform(size=(num_values, 2))
        data_trafo = DataTransformer(DummyDataHandler())

        n, mean, m2 = data_trafo._update_online_variance_vars(
                                x, 0., np.zeros_like(x), np.zeros_like(x))

        self.assertEqual(n, num_values)
        self.assertTrue(np.allclose(np.mean(x, axis=0), mean))
        self.assertTrue(np.allclose(np.var(x, axis=0), m2 / n))
        self.assertTrue(np.allclose(np.std(x, axis=0), np.sqrt(m2 / n)))

    def test_trafo_model_creation_without_any_trafo_tensors(self):
        """Test the creation of the trafo model in the case that no
        transformation is created for any of the defined tensors.
        """
        data_handler = DummyDataHandler(trafo=False)
        data_trafo = DataTransformer(data_handler)

        data_iterator = data_handler.create_data_iterator()

        # create trafo model
        self.assertFalse(data_trafo._setup_complete)
        data_trafo.create_trafo_model_iteratively(
            data_iterator, data_handler.n_batches)
        self.assertTrue(data_trafo._setup_complete)

        # check member variables and trafo model
        true_trafo_model = {
            'config': data_handler.config,
            'skip_keys': data_handler.skip_check_keys,
            'tensors': data_handler.tensors,
            'float_precision': 'float64',
            'norm_constant': 1e-6,
        }
        for key, value in true_trafo_model.items():
            self.assertEqual(value, data_trafo.trafo_model[key])

        # check if trafo model has expected keys
        self.assertListEqual(sorted(data_trafo.trafo_model.keys()),
                             sorted(['creation_time']
                                    + list(true_trafo_model.keys())))

        # check that trafo model can not be setup twice
        with self.assertRaises(ValueError) as context:
            data_trafo.create_trafo_model_iteratively(data_iterator,
                                                      data_handler.n_batches)
        self.assertTrue('Trafo model is already setup!'
                        in str(context.exception))

    def check_equal(self, data_trafo, data_handler):
        """Check if a created data trafo model is correct

        Parameters
        ----------
        data_trafo : DataTransformer object
            The data transformer
        data_handler : DummyDataHandler object
            The data handler
        """
        self.assertTrue(data_trafo._setup_complete)

        # check member variables and trafo model
        data_values = data_handler.data_tensor_values
        label_values = data_handler.label_tensor_values
        if data_handler.trafo_log[0]:
            data_tensor_mean = np.mean(np.log(data_values + 1), (0, 1))
            data_tensor_std = np.std(np.log(data_values + 1), (0, 1))

        else:
            data_tensor_mean = np.mean(data_values, (0, 1))
            data_tensor_std = np.std(data_values, (0, 1))
        if data_handler.trafo_log[1]:
            label_tensor_mean = np.mean(np.log(label_values + 1), (0, 1))
            label_tensor_std = np.std(np.log(label_values + 1), (0, 1))
        else:
            label_tensor_mean = np.mean(label_values, (0, 1))
            label_tensor_std = np.std(label_values, (0, 1))

        true_trafo_model = {
            'config': data_handler.config,
            'skip_keys': data_handler.skip_check_keys,
            'tensors': data_handler.tensors,
            'float_precision': 'float64',
            'norm_constant': 1e-6,
            'data_tensor_mean': data_tensor_mean,
            'data_tensor_std': data_tensor_std,
            'label_tensor_mean': label_tensor_mean,
            'label_tensor_std': label_tensor_std,
        }
        for key, value in true_trafo_model.items():
            if '_mean' in key or '_std' in key:
                self.assertTrue(np.allclose(value,
                                            data_trafo.trafo_model[key]))
            else:
                self.assertEqual(value, data_trafo.trafo_model[key])

        # check if trafo model has expected keys
        self.assertListEqual(sorted(data_trafo.trafo_model.keys()),
                             sorted(['creation_time']
                                    + list(true_trafo_model.keys())))

    def check_correct_trafo(self, data_trafo, data_handler):
        """Check if data is transformed correctly

        Parameters
        ----------
        data_trafo : DataTransformer object
            The data transformer
        data_handler : DummyDataHandler object
            The data handler
        """
        data_values = data_handler.data_tensor_values
        label_values = data_handler.label_tensor_values
        if data_handler.trafo_log[0]:
            data_tensor_mean = np.mean(np.log(data_values + 1), (0, 1))
            data_tensor_std = np.std(np.log(data_values + 1), (0, 1))
        else:
            data_tensor_mean = np.mean(data_values, (0, 1))
            data_tensor_std = np.std(data_values, (0, 1))
        if data_handler.trafo_log[1]:
            label_tensor_mean = np.mean(np.log(label_values + 1), (0, 1))
            label_tensor_std = np.std(np.log(label_values + 1), (0, 1))
        else:
            label_tensor_mean = np.mean(label_values, (0, 1))
            label_tensor_std = np.std(label_values, (0, 1))

        # check if transformations are correct
        data = data_values[0]
        labels = label_values[0]
        data_traf = data_trafo.transform(data, 'data_tensor')
        label_trafo = data_trafo.transform(labels, 'label_tensor')
        data_traf_bias = data_trafo.transform(data, 'data_tensor', False)
        label_trafo_bias = data_trafo.transform(labels, 'label_tensor', False)

        if data_handler.trafo_log[0]:
            data_trafo_true_bias = data / (1e-6 + data_tensor_std)
            data_trafo_true = \
                (np.log(data + 1) - data_tensor_mean) / \
                (1e-6 + data_tensor_std)
        else:
            data_trafo_true_bias = data / (1e-6 + data_tensor_std)
            data_trafo_true = (data - data_tensor_mean) / \
                (1e-6 + data_tensor_std)

        if data_handler.trafo_log[1]:
            label_trafo_true_bias = labels / (1e-6 + label_tensor_std)
            label_trafo_true = \
                (np.log(labels + 1) - label_tensor_mean) / \
                (1e-6 + label_tensor_std)
        else:
            label_trafo_true_bias = labels / (1e-6 + label_tensor_std)
            label_trafo_true = (labels - label_tensor_mean) / \
                (1e-6 + label_tensor_std)

        self.assertTrue(np.allclose(label_trafo_true, label_trafo))
        self.assertTrue(np.allclose(data_trafo_true, data_traf))
        self.assertTrue(np.allclose(label_trafo_true_bias, label_trafo_bias))
        self.assertTrue(np.allclose(data_trafo_true_bias, data_traf_bias))

        # now transform back
        data_traf_inv = data_trafo.inverse_transform(data_traf, 'data_tensor')
        label_trafo_inv = data_trafo.inverse_transform(label_trafo,
                                                       'label_tensor')
        data_traf_inv_bias = data_trafo.inverse_transform(
            data_traf_bias, 'data_tensor', False)
        label_trafo_inv_bias = data_trafo.inverse_transform(
            label_trafo_bias, 'label_tensor', False)

        self.assertTrue(np.allclose(label_trafo_inv, labels))
        self.assertTrue(np.allclose(data_traf_inv, data))
        self.assertTrue(np.allclose(label_trafo_inv_bias, labels))
        self.assertTrue(np.allclose(data_traf_inv_bias, data))

    def test_trafo_model_creation_without_log(self):
        """Test the creation of the trafo model without logarithm
        """
        for trafo_log in [False, None]:
            data_handler = DummyDataHandler(trafo=True, trafo_log=trafo_log)
            data_trafo = DataTransformer(data_handler)

            data_iterator = data_handler.create_data_iterator()

            # create trafo model
            self.assertFalse(data_trafo._setup_complete)
            data_trafo.create_trafo_model_iteratively(
                data_iterator, data_handler.n_batches)

            self.check_equal(data_trafo, data_handler)
            self.check_correct_trafo(data_trafo, data_handler)

    def test_trafo_model_creation_with_log(self):
        """Test the creation of the trafo model with logarithm.
        """
        data_handler = DummyDataHandler(trafo=True, trafo_log=True)
        data_trafo = DataTransformer(data_handler)

        data_iterator = data_handler.create_data_iterator()

        # create trafo model
        self.assertFalse(data_trafo._setup_complete)
        data_trafo.create_trafo_model_iteratively(
            data_iterator, data_handler.n_batches)

        self.check_equal(data_trafo, data_handler)
        self.check_correct_trafo(data_trafo, data_handler)

    def test_trafo_model_creation_with_and_without_log(self):
        """Test the creation of the trafo model without logarithm
        """
        trafo_log = [True, False]
        data_handler = DummyDataHandler(trafo=True, trafo_log=trafo_log)
        data_trafo = DataTransformer(data_handler)

        data_iterator = data_handler.create_data_iterator()

        # create trafo model
        self.assertFalse(data_trafo._setup_complete)
        data_trafo.create_trafo_model_iteratively(
            data_iterator, data_handler.n_batches)

        self.check_equal(data_trafo, data_handler)
        self.check_correct_trafo(data_trafo, data_handler)

    def test_trafo_model_transformation_without_setup(self):
        """Test the creation of the trafo model without logarithm
        """
        trafo_log = [True, False]
        data_handler = DummyDataHandler(trafo=True, trafo_log=trafo_log)
        data_trafo = DataTransformer(data_handler)

        with self.assertRaises(ValueError) as context:
            data_trafo.transform(data_handler.data_tensor_values[0],
                                 'data_tensor')
        self.assertTrue('DataTransformer needs to create or load a trafo' in
                        str(context.exception))

    def test_trafo_model_transformation_with_wrong_tensor_name(self):
        """Test the creation of the trafo model without logarithm
        """
        trafo_log = [True, False]
        data_handler = DummyDataHandler(trafo=True, trafo_log=trafo_log)
        data_trafo = DataTransformer(data_handler)

        # create trafo model
        data_trafo.create_trafo_model_iteratively(
            data_handler.create_data_iterator(), data_handler.n_batches)

        with self.assertRaises(ValueError) as context:
            data_trafo.transform(data_handler.data_tensor_values[0],
                                 'unknown tensor')
        self.assertTrue('is unknown!' in str(context.exception))

    def test_trafo_model_transformation_with_wrong_tensor_shape(self):
        """Test the creation of the trafo model without logarithm
        """
        trafo_log = [True, False]
        data_handler = DummyDataHandler(trafo=True, trafo_log=trafo_log)
        data_trafo = DataTransformer(data_handler)

        # create trafo model
        data_trafo.create_trafo_model_iteratively(
            data_handler.create_data_iterator(), data_handler.n_batches)

        with self.assertRaises(ValueError) as context:
            data_trafo.transform(data_handler.data_tensor_values[0][0],
                                 'data_tensor')
        self.assertTrue('Shape of data' in str(context.exception))

    def test_trafo_model_transformation_with_unimplemented_trafo_axis(self):
        """Test the creation of the trafo model without logarithm
        """
        trafo_log = [True, False]
        data_handler = DummyDataHandler(trafo=True, trafo_log=trafo_log,
                                        trafo_log_axis=1)
        data_trafo = DataTransformer(data_handler)

        # create trafo model
        data_trafo.create_trafo_model_iteratively(
            data_handler.create_data_iterator(), data_handler.n_batches)

        with self.assertRaises(NotImplementedError) as context:
            data_trafo.transform(data_handler.data_tensor_values[0],
                                 'data_tensor')

        with self.assertRaises(NotImplementedError) as context:
            data_trafo.inverse_transform(data_handler.data_tensor_values[0],
                                         'data_tensor')

    def test_loading_and_saving_of_trafo_model(self):
        """Test the saving and loading of a previously created trafo model.
        """
        data_handler = DummyDataHandler(trafo=True, trafo_log=True)
        data_trafo = DataTransformer(data_handler)

        data_iterator = data_handler.create_data_iterator()

        # create trafo model
        data_trafo.create_trafo_model_iteratively(
            data_iterator, data_handler.n_batches)

        # save trafo model
        file_path = os.path.join(os.path.dirname(__file__),
                                 '../../data/temp_test_files/trafo_model.npy')
        data_trafo.save_trafo_model(file_path, overwrite=True)

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            data_trafo.save_trafo_model(file_path, overwrite=False)
        self.assertTrue('File already exists!' in str(context.exception))

        # load trafo model
        loaded_data_trafo = DataTransformer(data_handler)
        loaded_data_trafo.load_trafo_model(file_path)

        # make sure both trafo models are correct
        self.check_equal(data_trafo, data_handler)
        self.check_equal(loaded_data_trafo, data_handler)
        self.check_correct_trafo(data_trafo, data_handler)
        self.check_correct_trafo(loaded_data_trafo, data_handler)

        # check that trafo model can not be setup twice
        with self.assertRaises(ValueError) as context:
            loaded_data_trafo.load_trafo_model(file_path)
        self.assertTrue('Trafo model is already setup!'
                        in str(context.exception))

    def test_check_loading_of_mismatched_trafo_model(self):
        """Test the saving and loading of a previously created trafo model.
        """
        data_handler = DummyDataHandler(trafo=True, trafo_log=True)
        data_handler_wrong = DummyDataHandler(trafo=True, trafo_log=False)
        data_trafo = DataTransformer(data_handler)

        data_iterator = data_handler.create_data_iterator()

        # create trafo model
        data_trafo.create_trafo_model_iteratively(
            data_iterator, data_handler.n_batches)

        # save trafo model
        file_path = os.path.join(os.path.dirname(__file__),
                                 '../../data/temp_test_files/trafo_model.npy')
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(os.path.dirname(file_path)):
            os.removedirs(os.path.dirname(file_path))
        data_trafo.save_trafo_model(file_path, overwrite=True)

        # check that trafo model can not be loaded with wrong settings
        with self.assertRaises(ValueError) as context:
            loaded_data_trafo = DataTransformer(data_handler_wrong)
            loaded_data_trafo.load_trafo_model(file_path)
        self.assertTrue('does not match!' in str(context.exception))

    def test_check_loading_of_mismatched_trafo_model_additional_key(self):
        """Test the saving and loading of a previously created trafo model.
        """
        data_handler = DummyDataHandler(trafo=True, trafo_log=True)
        data_handler_wrong = DummyDataHandler(trafo=True, trafo_log=False)
        data_trafo = DataTransformer(data_handler)

        data_iterator = data_handler.create_data_iterator()

        # create trafo model
        data_trafo.create_trafo_model_iteratively(
            data_iterator, data_handler.n_batches)

        # save trafo model
        file_path = os.path.join(os.path.dirname(__file__),
                                 '../../data/temp_test_files/trafo_model.npy')
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(os.path.dirname(file_path)):
            os.removedirs(os.path.dirname(file_path))
        data_trafo.save_trafo_model(file_path, overwrite=True)

        # check if additional key exists, but not in loaded file
        with self.assertRaises(KeyError) as context:
            loaded_data_trafo = DataTransformer(data_handler_wrong)
            loaded_data_trafo.trafo_model['wrong_key'] = 'wrong'
            loaded_data_trafo.load_trafo_model(file_path)
        self.assertTrue('does not exist in' in str(context.exception))

    def test_check_loading_of_mismatched_numpy_arrays_ub_trafo_model(self):
        """Test the saving and loading of a previously created trafo model.
        """
        data_handler = DummyDataHandler(trafo=True, trafo_log=True)

        # create trafo models
        data_trafo = DataTransformer(data_handler)
        data_trafo.create_trafo_model_iteratively(
            data_handler.create_data_iterator(), data_handler.n_batches)

        data_trafo.trafo_model['numpy_array'] = np.array([1, 3, 4, 3])

        # save trafo model
        file_path = os.path.join(os.path.dirname(__file__),
                                 '../../data/temp_test_files/trafo_model.npy')
        data_trafo.save_trafo_model(file_path, overwrite=True)

        # check if numpy arrays are wrong
        with self.assertRaises(ValueError) as context:
            data_trafo_mod = DataTransformer(data_handler)
            data_trafo_mod.trafo_model['numpy_array'] = np.array([1, 3, 4, 5])
            data_trafo_mod.load_trafo_model(file_path)
        self.assertTrue('does not match' in str(context.exception))

        # check if numpy arrays are correct
        data_trafo_mod = DataTransformer(data_handler)
        data_trafo_mod.trafo_model['numpy_array'] = np.array([1, 3, 4, 3])
        data_trafo_mod.load_trafo_model(file_path)

    def test_trafo_model_transformations(self):
        """Test the transformation methods of a previously created trafo model.
        """


if __name__ == '__main__':
    unittest.main()

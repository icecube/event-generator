#!/usr/local/bin/python3

import unittest
import os
import shutil
import numpy as np
import tensorflow as tf
from copy import deepcopy

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.trafo import DataTransformer
from egenerator.data.handler.base import BaseDataHandler
from egenerator.data.tensor import DataTensor, DataTensorList


class DummyDataHandler(BaseDataHandler):

    """Create a dummy data hanlder for testing purposes
    """
    @property
    def n_batches(self):
        return self.configuration.config['num_batches']

    @property
    def batch_size(self):
        return self.configuration.config['batch_size']

    def _configure_derived_class(self, config, config_data=None,
                                 batch_size=5, num_batches=4, seed=42):
        """Setup the data handler with a test input file.
        This method needs to be implemented by derived class.

        Parameters
        ----------
        config : dict
            Configuration of the data handler.
        config_data : str or list of str, optional
            File name pattern or list of file patterns which define the paths
            to input data files. The first of the specified files will be
            read in to obtain meta data.
        batch_size : int, optional
            The number of elements in a batch.
        num_batches : int
            The number of batches to generate.
        seed : int, optional
            Random number generator seed.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            as these are automatically gathered.The dependent_sub_components
            may also be left empty. This is later filled by the base class
            from the returned sub components dict.
            Settings that need to be defined are:
                class_string:
                    misc.get_full_class_string_of_object(self)
                settings: dict
                    The settings of the component.
                mutable_settings: dict, default={}
                    The mutable settings of the component.
                check_values: dict, default={}
                    Additional check values.
        dict
            The data of the component.
            This must at least contain the data tensor list which must be
            stored under the key 'tensors'.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        # create dummy data
        random_state = np.random.RandomState(seed)
        self._untracked_data['data'] = random_state.uniform(
            size=(num_batches, batch_size, 86, 60, 1))
        self._untracked_data['labels'] = random_state.uniform(
            size=(num_batches, batch_size, 7))

        # create a list of tensors
        tensors = [
            DataTensor(name='data_tensor',
                       shape=[None, 86, 60, 1],
                       tensor_type='data',
                       dtype='float32',
                       trafo=config['trafo'][0],
                       trafo_log=config['trafo_log'][0],
                       trafo_log_axis=config['trafo_log_axis'][0]),
            DataTensor(name='label_tensor',
                       shape=[None, 7],
                       tensor_type='label',
                       dtype='float32',
                       trafo=config['trafo'][1],
                       trafo_log=config['trafo_log'][1],
                       trafo_log_axis=config['trafo_log_axis'][1]),
        ]
        data = {'tensors': DataTensorList(tensors)}

        # define configuration
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={'config': config,
                      'batch_size': batch_size,
                      'num_batches': num_batches,
                      'seed': seed,
                      })

        return configuration, data, {}

    def get_batch_generator(self):
        """Create a python iterator object
        """
        def iterator():
            batch_counter = 0
            for i in range(self.configuration.config['num_batches']):
                yield (self._untracked_data['data'][i],
                       self._untracked_data['labels'][i])
        return iterator()


class TestDataTransformer(unittest.TestCase):

    """Test data transformer class.
    """
    @property
    def config(self):
        return dict(deepcopy(self.default_config))

    def setUp(self):
        self.data_trafo = DataTransformer()
        self.default_config = {
            'trafo': [True, True],
            'trafo_log': [True, True],
            'trafo_log_axis': [-1, -1],
        }

    def get_data_handler(self, config, **kwargs):
        """Helper Function to obtain a configured dummy data handler

        Parameters
        ----------
        config : dict
            The data handler config.
        **kwargs
            Description

        Returns
        -------
        DummyDataHandler object
            The configured dummy data handler object.
        """
        data_handler = DummyDataHandler()
        data_handler.configure(config=config, **kwargs)
        return data_handler

    def get_data_trafo_and_handler(self, **kwargs):
        config = self.config
        config.update(kwargs)
        data_handler = self.get_data_handler(config)
        data_trafo = DataTransformer()

        # create trafo model
        self.assertFalse(data_trafo._is_configured)
        data_trafo.configure(
                data_handler=data_handler, data_iterator_settings={},
                num_batches=data_handler.n_batches)
        return data_trafo, data_handler

    def check_correct_configuration(self, data_trafo, data_handler):
        """Helper function to check if a created data trafo model is correct

        Parameters
        ----------
        data_trafo : DataTransformer object
            The data transformer
        data_handler : DummyDataHandler object
            The data handler
        """
        self.assertTrue(data_trafo._is_configured)

        # check member variables and trafo model
        data_values = data_handler._untracked_data['data']
        label_values = data_handler._untracked_data['labels']
        if data_handler.configuration.settings['config']['trafo_log'][0]:
            data_tensor_mean = np.mean(np.log(data_values + 1), (0, 1))
            data_tensor_std = np.std(np.log(data_values + 1), (0, 1))

        else:
            data_tensor_mean = np.mean(data_values, (0, 1))
            data_tensor_std = np.std(data_values, (0, 1))
        if data_handler.configuration.settings['config']['trafo_log'][1]:
            label_tensor_mean = np.mean(np.log(label_values + 1), (0, 1))
            label_tensor_std = np.std(np.log(label_values + 1), (0, 1))
        else:
            label_tensor_mean = np.mean(label_values, (0, 1))
            label_tensor_std = np.std(label_values, (0, 1))

        true_trafo_model = {
            'tensors': data_handler.tensors,
            'norm_constant': 1e-6,
            'np_float_dtype': np.float64,
            'tf_float_dtype': tf.float64,
        }
        true_check_values = {}
        if data_handler.configuration.settings['config']['trafo'][0]:
            true_trafo_model['data_tensor_mean'] = data_tensor_mean
            true_trafo_model['data_tensor_std'] = data_tensor_std
            true_check_values['data_tensor_mean'] = np.mean(data_tensor_mean)
            true_check_values['data_tensor_std'] = np.mean(data_tensor_std)
        if data_handler.configuration.settings['config']['trafo'][1]:
            true_trafo_model['label_tensor_mean'] = label_tensor_mean
            true_trafo_model['label_tensor_std'] = label_tensor_std
            true_check_values['label_tensor_mean'] = np.mean(label_tensor_mean)
            true_check_values['label_tensor_std'] = np.mean(label_tensor_std)

        for key, value in true_trafo_model.items():
            if '_mean' in key or '_std' in key:
                self.assertTrue(np.allclose(value,
                                            data_trafo.data[key]))
            else:
                self.assertEqual(value, data_trafo.data[key])

        # check if trafo model has expected keys
        self.assertListEqual(sorted(data_trafo.data.keys()),
                             sorted(['creation_time']
                                    + list(true_trafo_model.keys())))

        # check settings
        true_trafo_settings = {
            'float_precision': 'float64',
            'norm_constant': 1e-6,
            'num_batches': data_handler.n_batches,
            'data_iterator_settings': {},
        }
        self.assertDictEqual(data_trafo.configuration.settings,
                             true_trafo_settings)
        self.assertDictEqual(data_trafo.configuration.mutable_settings,
                             {})

        # check hash values
        for key, value in true_check_values.items():
            self.assertAlmostEqual(value,
                                   data_trafo.configuration.check_values[key])

    def check_correct_trafo(self, data_trafo, data_handler):
        """Helper function to check if data is transformed correctly

        Parameters
        ----------
        data_trafo : DataTransformer object
            The data transformer
        data_handler : DummyDataHandler object
            The data handler
        """
        data_values = data_handler._untracked_data['data']
        label_values = data_handler._untracked_data['labels']
        if data_handler.configuration.settings['config']['trafo_log'][0]:
            data_tensor_mean = np.mean(np.log(data_values + 1), (0, 1))
            data_tensor_std = np.std(np.log(data_values + 1), (0, 1))

        else:
            data_tensor_mean = np.mean(data_values, (0, 1))
            data_tensor_std = np.std(data_values, (0, 1))
        if data_handler.configuration.settings['config']['trafo_log'][1]:
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

        if data_handler.configuration.settings['config']['trafo_log'][0]:
            data_trafo_true_bias = data / (1e-6 + data_tensor_std)
            data_trafo_true = \
                (np.log(data + 1) - data_tensor_mean) / \
                (1e-6 + data_tensor_std)
        else:
            data_trafo_true_bias = data / (1e-6 + data_tensor_std)
            data_trafo_true = (data - data_tensor_mean) / \
                (1e-6 + data_tensor_std)

        if data_handler.configuration.settings['config']['trafo_log'][1]:
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

    def test_configuration_with_unconfigured_data_handler(self):
        """The data trafo configure method should raise an error if attempting
        to configure a DataTransformer object with an unconfigured
        data handler  object.
        """
        data_trafo = DataTransformer()
        data_handler = BaseDataHandler()
        with self.assertRaises(ValueError) as context:
            data_trafo.configure(data_handler=data_handler,
                                 data_iterator_settings=None,
                                 num_batches=None)
        self.assertTrue('Component' in str(context.exception) and
                        'is not configured!' in str(context.exception))

    def test_correct_initialization(self):
        """Check if member variables are correctly set when instantiating
        a DataTransformer object.
        """
        data_trafo = DataTransformer()
        self.assertEqual(data_trafo._is_configured, False)
        self.assertEqual(data_trafo.data, None)
        self.assertEqual(data_trafo.configuration, None)
        self.assertEqual(data_trafo._untracked_data, {})
        self.assertEqual(data_trafo._sub_components, {})

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
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo=[False, False])

        # check member variables and trafo model
        true_trafo_model = {
            'tensors': data_handler.tensors,
            'norm_constant': 1e-6,
            'np_float_dtype': np.float64,
            'tf_float_dtype': tf.float64,
        }
        for key, value in true_trafo_model.items():
            self.assertEqual(value, data_trafo.data[key])

        self.assertEqual(data_trafo.configuration.settings['float_precision'],
                         'float64')

        # check if trafo model has expected keys
        self.assertListEqual(sorted(data_trafo.data.keys()),
                             sorted(['creation_time']
                                    + list(true_trafo_model.keys())))

        # check that trafo model can not be setup twice
        with self.assertRaises(ValueError) as context:
            data_trafo.configure(
                data_handler=data_handler, data_iterator_settings={},
                num_batches=data_handler.n_batches)
        self.assertTrue('Component is already configured!'
                        in str(context.exception))

        # test correctness of data trafo model
        self.check_correct_configuration(data_trafo, data_handler)

    def test_trafo_model_creation_without_log(self):
        """Test the creation of the trafo model without logarithm
        """
        for trafo_log in [False, None]:
            data_trafo, data_handler = \
                self.get_data_trafo_and_handler(trafo_log=[trafo_log] * 2)

            self.check_correct_configuration(data_trafo, data_handler)
            self.check_correct_trafo(data_trafo, data_handler)

    def test_trafo_model_creation_with_log(self):
        """Test the creation of the trafo model with logarithm.
        """
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo_log=[True, True])

        self.check_correct_configuration(data_trafo, data_handler)
        self.check_correct_trafo(data_trafo, data_handler)

    def test_trafo_model_creation_with_and_without_log(self):
        """Test the creation of the trafo model with and without logarithm
        """
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo_log=[True, False])

        self.check_correct_configuration(data_trafo, data_handler)
        self.check_correct_trafo(data_trafo, data_handler)

    def test_trafo_model_transformation_without_setup(self):
        """Test the creation of the trafo model without logarithm
        """
        config = self.config
        config['trafo_log'] = [True, False]
        data_handler = self.get_data_handler(config)
        data_trafo = DataTransformer()

        with self.assertRaises(ValueError) as context:
            data_trafo.transform(data_handler._untracked_data['data'][0],
                                 'data_tensor')
        self.assertTrue('DataTransformer needs to create or load a trafo' in
                        str(context.exception))

    def test_trafo_model_transformation_with_wrong_tensor_name(self):
        """The trafo model should raise an error if a transformation is
        attempted with a non-existing tensor name
        """
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo_log=[True, False])

        with self.assertRaises(ValueError) as context:
            data_trafo.transform(data_handler._untracked_data['data'][0],
                                 'unknown tensor')
        self.assertTrue('is unknown!' in str(context.exception))

    def test_trafo_model_transformation_with_wrong_tensor_shape(self):
        """The trafo model should raise an error if a transformation is
        attempted with a wrong tensor shape.
        """
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo_log=[True, False])

        with self.assertRaises(ValueError) as context:
            data_trafo.transform(data_handler._untracked_data['data'][0][0],
                                 'data_tensor')
        self.assertTrue('Shape of data' in str(context.exception))

    def test_trafo_model_transformation_with_unimplemented_trafo_axis(self):
        """Log transformations along a different axis than -1 are currently
        not supported and should raise a NotImplementedError
        """
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo_log=[True, False],
                                            trafo_log_axis=[1, 1])

        with self.assertRaises(NotImplementedError) as context:
            data_trafo.transform(data_handler._untracked_data['data'][0],
                                 'data_tensor')

        with self.assertRaises(NotImplementedError) as context:
            data_trafo.inverse_transform(
                data_handler._untracked_data['data'][0],  'data_tensor')

    def test_loading_and_saving_of_trafo_model(self):
        """Test the saving and loading of a previously created trafo model.
        """
        data_trafo, data_handler = \
            self.get_data_trafo_and_handler(trafo=[True, True],
                                            trafo_log=[True, True])

        # save data handler
        directory = os.path.join(
            os.path.dirname(__file__),
            '../../data/temp_test_files/trafo_model')

        # remove it if it already exists
        if os.path.exists(directory):
            shutil.rmtree(directory)

        # save trafo model
        data_trafo.save(directory, overwrite=False)
        data_trafo.save(directory, overwrite=True)

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            data_trafo.save(directory, overwrite=False)
        self.assertTrue('File ' in str(context.exception) and
                        'already exists!' in str(context.exception))

        # load trafo model
        loaded_data_trafo = DataTransformer()
        loaded_data_trafo.load(directory)

        # make sure both trafo models are correct
        self.check_correct_configuration(data_trafo, data_handler)
        self.check_correct_configuration(loaded_data_trafo, data_handler)
        self.check_correct_trafo(data_trafo, data_handler)
        self.check_correct_trafo(loaded_data_trafo, data_handler)

        # make sure they are compatible
        data_handler.is_compatible(loaded_data_trafo)

        # check that trafo model can not be setup twice
        with self.assertRaises(ValueError) as context:
            loaded_data_trafo.load(directory)
        self.assertTrue('Component is already configured!'
                        in str(context.exception))

    # def test_check_loading_of_mismatched_trafo_model(self):
    #     """Trying
    #     """
    #     data_trafo, data_handler = \
    #         self.get_data_trafo_and_handler(trafo=[True, True],
    #                                         trafo_log=[True, True])
    #     data_trafo_wrong, data_handler_wrong = \
    #         self.get_data_trafo_and_handler(trafo=[True, True],
    #                                         trafo_log=[True, False])
    #     data_handler = DummyDataHandler(trafo=True, trafo_log=True)
    #     data_handler_wrong = DummyDataHandler(trafo=True, trafo_log=False)
    #     data_trafo = DataTransformer(data_handler)

    #     data_iterator = data_handler.create_data_iterator()

    #     # create trafo model
    #     data_trafo.configure(
    #         data_iterator, data_handler.n_batches)

    #     # save trafo model
    #     file_path = os.path.join(
    #         os.path.dirname(__file__),
    #         '../../data/temp_test_files/trafo_model/trafo_model.npy')
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #     if os.path.exists(os.path.dirname(file_path)):
    #         os.removedirs(os.path.dirname(file_path))
    #     data_trafo.save(file_path, overwrite=True)

    #     # check that trafo model can not be loaded with wrong settings
    #     loaded_data_trafo = DataTransformer(data_handler_wrong)
    #     with self.assertRaises(ValueError) as context:
    #         loaded_data_trafo.load(file_path)
    #     self.assertTrue('does not match' in str(context.exception))

    # def test_check_loading_of_mismatched_trafo_model_additional_key(self):
    #     """Test the saving and loading of a previously created trafo model.
    #     """
    #     data_handler = DummyDataHandler(trafo=True, trafo_log=True)
    #     data_trafo = DataTransformer(data_handler)

    #     data_iterator = data_handler.create_data_iterator()

    #     # create trafo model
    #     data_trafo.configure(
    #         data_iterator, data_handler.n_batches)

    #     # save trafo model
    #     file_path = os.path.join(
    #         os.path.dirname(__file__),
    #         '../../data/temp_test_files/trafo_model/trafo_model.npy')
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #     if os.path.exists(os.path.dirname(file_path)):
    #         os.removedirs(os.path.dirname(file_path))
    #     data_trafo.save(file_path, overwrite=True)

    #     # check if additional key exists, but not in loaded file
    #     loaded_data_trafo = DataTransformer(data_handler)
    #     loaded_data_trafo._trafo_model['wrong_key'] = 'wrong'
    #     with self.assertRaises(KeyError) as context:
    #         loaded_data_trafo.load(file_path)
    #     self.assertTrue('does not exist in' in str(context.exception))

    # def test_check_loading_of_mismatched_numpy_arrays_ub_trafo_model(self):
    #     """Test the saving and loading of a previously created trafo model.
    #     """
    #     data_handler = DummyDataHandler(trafo=True, trafo_log=True)

    #     # create trafo models
    #     config = {
    #         'data_handler': data_handler,
    #         'data_iterator_settings': {},
    #         'num_batches': data_handler.n_batches,
    #     }
    #     data_trafo = DataTransformer()
    #     data_trafo.configure(**config)

    #     data_trafo._trafo_model['numpy_array'] = np.array([1, 3, 4, 3])

    #     # save trafo model
    #     file_path = os.path.join(
    #         os.path.dirname(__file__),
    #         '../../data/temp_test_files/trafo_model/trafo_model.npy')
    #     data_trafo.save(file_path, overwrite=True)

    #     # check if numpy arrays are wrong
    #     data_trafo_mod = DataTransformer()
    #     data_trafo_mod._trafo_model['numpy_array'] = np.array([1, 3, 4, 5])
    #     with self.assertRaises(ValueError) as context:
    #         data_trafo_mod.load(file_path)
    #     self.assertTrue('does not match' in str(context.exception))

    #     # check if numpy arrays are correct
    #     data_trafo_mod = DataTransformer()
    #     data_trafo_mod._trafo_model['numpy_array'] = np.array([1, 3, 4, 3])
    #     data_trafo_mod.load(file_path)


if __name__ == '__main__':
    unittest.main()

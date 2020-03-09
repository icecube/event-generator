#!/usr/local/bin/python3

import unittest
import numpy as np
import os

from egenerator.data.modules.labels.cascades import CascadeGeneratorLabelModule
from egenerator.data.tensor import DataTensorList, DataTensor


class TestCascadeLabelsModule(unittest.TestCase):

    def setUp(self):
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            '../../../test_data/cascade_mesc_l5_nue_low.hdf5')

        self.cascades_true = np.array([
            [-13.094963629354766, -197.0847391208472, -322.0192710148053,
             1.0771952265275238, 4.601483747646196, 2360.8997600199427,
             9663.551318717717],
            [-70.78487964475926, -32.47261211840669, -426.5132607462965,
             1.586083894785944, 1.5573642249002815, 924.7251046427211,
             9789.880753474426],
            ])

    """Test cascade label module.
    """
    def test_class_initialization_parameters(self):
        """Check that initializer only tak
        """
        with self.assertRaises(TypeError) as context:
            CascadeGeneratorLabelModule(config_data=4)

        with self.assertRaises(TypeError) as context:
            CascadeGeneratorLabelModule(4, label_key='labels')

        with self.assertRaises(ValueError) as context:
            CascadeGeneratorLabelModule(4, True)
        self.assertTrue('is not a boolean value!' in str(context.exception))

        module = CascadeGeneratorLabelModule(True, False, label_key='labels')
        module = CascadeGeneratorLabelModule(True, [True, True],)

    def test_member_variables(self):
        """Test if member variables have correct values.
        """
        module = CascadeGeneratorLabelModule(True, False, label_key='labels')
        self.assertEqual(module.skip_check_keys, [])
        self.assertEqual(module.settings['shift_cascade_vertex'], True)
        self.assertEqual(module.settings['trafo_log'], False)
        self.assertEqual(module.settings['label_key'], 'labels')

    def test_configuration_check(self):
        """Check whether passed tensor list in confguration is checked and
        found to be wrong.
        """
        # config = {
        #     'config_data': 'file_path_string',
        #     'shift_cascade_vertex': True,
        #     'trafo_log': trafo_log,
        #     'float_precision': 'float64',
        #     'label_key': 'labels'
        # }
        # module = CascadeGeneratorLabelModule()
        # module.configure(**config)

        module = CascadeGeneratorLabelModule(True, False, label_key='labels')

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            module.configure(config_data=DataTensorList([]))
        self.assertTrue(' != ' in str(context.exception))

        # pasing a file path should be fine
        module.configure(config_data='file_path_string')

    def test_correct_configuration(self):
        """Check if the module correctly creates the tensors
        """
        for trafo_log in [True, False, None]:
            config = {
                'config_data': 'file_path_string',
                'shift_cascade_vertex': True,
                'trafo_log': trafo_log,
                'float_precision': 'float64',
                'label_key': 'labels'
            }
            module = CascadeGeneratorLabelModule()
            module.configure(**config)

            tensors_true = DataTensorList([DataTensor(
                                        name='cascade_labels',
                                        shape=[None, 7],
                                        tensor_type='label',
                                        dtype='float32',
                                        trafo=True,
                                        trafo_log=trafo_log)])
            self.assertTrue(module.data['label_tensors'] == tensors_true)

            # make sure the internal check also works
            config = {
                'config_data': tensors_true,
                'shift_cascade_vertex': True,
                'trafo_log': trafo_log,
                'float_precision': 'float64',
                'label_key': 'labels'
            }
            module = CascadeGeneratorLabelModule()
            module.configure(**config)

    def test_shift_to_maximum(self):
        """Check if cascade shift works correctly
        """
        config = {
            'config_data': 'dummy_data',
            'shift_cascade_vertex': True,
            'trafo_log': False,
            'float_precision': 'float64',
            'label_key': 'labels'
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        # create cascades (x, y, z, zenith, azimuth, energy)
        cascades = np.array([
                    [4.32215303e+002, -2.50316524e+001,  3.81876940e+002,
                     3.27917145e+000,  1.59322221e+000,  2.32044063e+000],
                    [4.25103158e+002, -4.50171424e+002, -1.54742065e+002,
                     1.70672932e+000,  5.17145295e+000,  1.60450154e+000],
                    [-1.93660862e+002,  1.62074241e+002, -1.47496612e+002,
                     5.83964799e+000, -9.00899446e-001,  6.42165298e+000],
                    [-5.58199663e-002, -1.49285523e+002,  2.04746672e+002,
                     1.82316036e+000,  2.67957510e+000,  3.16299454e+000],
                    [-4.55858510e+002, -4.56435700e+002, -3.92204323e+002,
                     -2.23230024e+000,  4.70895608e+000,  4.25942900e+000],
                    [1.42867949e+002,  6.24530479e+001,  8.84962221e+001,
                     1.39066482e+000, -5.13348979e+000,  4.46057810e+000],
                    [-7.08375332e+001,  1.02039617e+002,  4.80702481e+001,
                     1.56647924e+000, -1.76599099e+000,  2.06194396e+000],
                    [-4.20642358e+002, -4.56640785e+002, -3.60856445e+002,
                     -1.58925978e+000, -2.28654059e+000,  1.44055993e+000],
                    [4.54689113e+002, -3.46175155e+002, -1.02038907e+002,
                     5.76306824e-001,  2.50674561e+000,  4.86170398e+000],
                    [2.07859796e+070,  2.39798123e+163,  1.69130696e+202,
                     3.99624293e+002,  5.11274581e+001,  6.21003341e+002]])

        # shift the vertex
        x, y, z = module._shift_to_maximum(*(cascades.T))

        x_true = np.array([
            4.32212400e+02,  4.24748720e+02, -1.93308774e+02,  8.62038257e-01,
            -4.55861677e+02,  1.42390868e+02, -7.06629130e+01, -4.21145814e+02,
            4.55223625e+02,  2.07859796e+70])
        y_true = np.array([
            -2.49022079e+001, -4.49454350e+002,  1.61629734e+002,
            -1.49742583e+002, -4.57358288e+002,  6.13878826e+001,
            1.02922822e+002, -4.57219759e+002, -3.46568858e+002,
            2.39798123e+163])
        z_true = np.array([
            3.82812107e+002, -1.54632659e+002, -1.48690141e+002,
            2.05011074e+002, -3.91486068e+002,  8.82836822e+001,
            4.80663614e+001, -3.60842277e+002, -1.03060376e+002,
            1.69130696e+202])

        self.assertTrue(np.allclose(x, x_true))
        self.assertTrue(np.allclose(y, y_true))
        self.assertTrue(np.allclose(z, z_true))

    def test_get_data_from_hdf_check_configured(self):
        """Check if error is raised when not configured
        """
        module = CascadeGeneratorLabelModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.get_data_from_hdf('wrong_file_path')
        self.assertTrue('Module not configured yet!' in str(context.exception))

    def test_get_data_from_hdf_skip_file(self):
        """Check if file is skipped correctly if a label does not exist.
        """
        config = {
            'config_data': 'dummy_data',
            'shift_cascade_vertex': True,
            'trafo_log': False,
            'float_precision': 'float64',
            'label_key': 'labels'
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)

        self.assertEqual(num_events, None)
        self.assertEqual(data, None)

    def test_get_data_from_hdf_wrong_file_name(self):
        """Check if IOError is raised if file does not exist
        """
        config = {
            'config_data': 'dummy_data',
            'shift_cascade_vertex': True,
            'trafo_log': False,
            'float_precision': 'float64',
            'label_key': 'labels'
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        with self.assertRaises(IOError) as context:
            num_events, data = module.get_data_from_hdf('wrong_file_path')
        self.assertTrue('does not exist' in str(context.exception))

    def test_get_data_from_hdf(self):
        """Test if loaded data is correct
        """
        config = {
            'config_data': 'dummy_data',
            'shift_cascade_vertex': False,
            'trafo_log': False,
            'float_precision': 'float64',
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertTrue((self.cascades_true == data[0]).all())

    def test_get_data_from_hdf_with_shifted_vertex(self):
        """Test if loaded data is correct
        """
        config = {
            'config_data': 'dummy_data',
            'shift_cascade_vertex': True,
            'trafo_log': False,
            'float_precision': 'float64',
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertTrue((self.cascades_true[:, 3:] == data[0][:, 3:]).all())


if __name__ == '__main__':
    unittest.main()

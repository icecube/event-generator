import unittest
import numpy as np
import os

from egenerator.data.modules.labels.cascades import CascadeGeneratorLabelModule
from egenerator.data.tensor import DataTensorList, DataTensor


class TestCascadeLabelsModule(unittest.TestCase):

    def setUp(self):
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            "../../../test_data/cascade_mesc_l5_nue_low.hdf5",
        )

        self.cascades_true = np.array(
            [
                [
                    -13.094963629354766,
                    -197.0847391208472,
                    -322.0192710148053,
                    1.0771952265275238,
                    4.601483747646196,
                    2360.8997600199427,
                    9663.551318717717,
                ],
                [
                    -70.78487964475926,
                    -32.47261211840669,
                    -426.5132607462965,
                    1.586083894785944,
                    1.5573642249002815,
                    924.7251046427211,
                    9789.880753474426,
                ],
            ]
        )

    """Test cascade label module.
    """

    def test_member_variables(self):
        """Test if member variables have correct values."""
        config = {
            "config_data": "file_path_string",
            "shift_cascade_vertex": True,
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)
        settings = module.configuration.settings
        self.assertEqual(settings["shift_cascade_vertex"], True)
        self.assertEqual(settings["trafo_log"], False)
        self.assertEqual(settings["label_key"], "labels")
        self.assertEqual(settings["float_precision"], "float64")

    def test_configuration_check(self):
        """Check whether passed tensor list in confguration is checked and
        found to be wrong.
        """
        config = {
            "config_data": DataTensorList([]),
            "shift_cascade_vertex": True,
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            module.configure(**config)
        self.assertTrue("Tensors are wrong:" in str(context.exception))

        # pasing the correct data tensors should work
        data_tensor_list = DataTensorList(
            [
                DataTensor(
                    name="x_parameters",
                    shape=[None, 7],
                    tensor_type="label",
                    dtype="float32",
                    trafo=True,
                    trafo_log=False,
                )
            ]
        )
        config["config_data"] = data_tensor_list
        module.configure(**config)

    def test_configuration_check_boolean(self):
        """Shift cascade vertex must be a bool. A TypeError should be raised
        if the label component is being attempted to set up with a wrong type.
        """
        config = {
            "config_data": "file_path_string",
            "shift_cascade_vertex": None,
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(TypeError) as context:
            module.configure(**config)
        self.assertTrue("is not a boolean value!" in str(context.exception))

        config = {
            "config_data": "file_path_string",
            "shift_cascade_vertex": DataTensorList([]),
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(TypeError) as context:
            module.configure(**config)
        self.assertTrue("is not a boolean value!" in str(context.exception))

    def test_correct_configuration(self):
        """Check if the module correctly creates the tensors"""
        for trafo_log in [True, False, None]:
            config = {
                "config_data": "file_path_string",
                "shift_cascade_vertex": True,
                "trafo_log": trafo_log,
                "float_precision": "float64",
                "label_key": "labels",
            }
            module = CascadeGeneratorLabelModule()
            module.configure(**config)

            tensors_true = DataTensorList(
                [
                    DataTensor(
                        name="x_parameters",
                        shape=[None, 7],
                        tensor_type="label",
                        dtype=config["float_precision"],
                        trafo=True,
                        trafo_log=trafo_log,
                    )
                ]
            )
            self.assertTrue(module.data["label_tensors"] == tensors_true)

            # make sure the internal check also works
            config = {
                "config_data": tensors_true,
                "shift_cascade_vertex": True,
                "trafo_log": trafo_log,
                "float_precision": "float64",
                "label_key": "labels",
            }
            module = CascadeGeneratorLabelModule()
            module.configure(**config)

    def test_shift_to_maximum(self):
        """Check if cascade shift works correctly"""
        config = {
            "config_data": "dummy_data",
            "shift_cascade_vertex": True,
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        # create cascades (x, y, z, zenith, azimuth, energy, time)
        cascades = np.array(
            [
                [
                    4.32215303e002,
                    -2.50316524e001,
                    3.81876940e002,
                    3.27917145e000,
                    1.59322221e000,
                    2.32044063e000,
                    0,
                ],
                [
                    4.25103158e002,
                    -4.50171424e002,
                    -1.54742065e002,
                    1.70672932e000,
                    5.17145295e000,
                    1.60450154e000,
                    0,
                ],
                [
                    -1.93660862e002,
                    1.62074241e002,
                    -1.47496612e002,
                    5.83964799e000,
                    -9.00899446e-001,
                    6.42165298e000,
                    0,
                ],
                [
                    -5.58199663e-002,
                    -1.49285523e002,
                    2.04746672e002,
                    1.82316036e000,
                    2.67957510e000,
                    3.16299454e000,
                    0,
                ],
                [
                    -4.55858510e002,
                    -4.56435700e002,
                    -3.92204323e002,
                    -2.23230024e000,
                    4.70895608e000,
                    4.25942900e000,
                    0,
                ],
                [
                    1.42867949e002,
                    6.24530479e001,
                    8.84962221e001,
                    1.39066482e000,
                    -5.13348979e000,
                    4.46057810e000,
                    0,
                ],
                [
                    -7.08375332e001,
                    1.02039617e002,
                    4.80702481e001,
                    1.56647924e000,
                    -1.76599099e000,
                    2.06194396e000,
                    0,
                ],
                [
                    -4.20642358e002,
                    -4.56640785e002,
                    -3.60856445e002,
                    -1.58925978e000,
                    -2.28654059e000,
                    1.44055993e000,
                    0,
                ],
                [
                    4.54689113e002,
                    -3.46175155e002,
                    -1.02038907e002,
                    5.76306824e-001,
                    2.50674561e000,
                    4.86170398e000,
                    0,
                ],
                [
                    2.07859796e070,
                    2.39798123e163,
                    1.69130696e202,
                    3.99624293e002,
                    5.11274581e001,
                    6.21003341e002,
                    0,
                ],
            ]
        )

        # shift the vertex
        x, y, z, t = module._shift_to_maximum(*(cascades.T))

        x_true = np.array(
            [
                4.32212290e02,
                4.24737646e02,
                -1.93291463e02,
                9.00762144e-01,
                -4.55861821e02,
                1.42368941e02,
                -7.06566264e01,
                -4.21160337e02,
                4.55248631e02,
                2.07859796e70,
            ]
        )
        y_true = np.array(
            [
                -2.48972997e001,
                -4.49431946e002,
                1.61607878e002,
                -1.49761867e002,
                -4.57400269e002,
                6.13389252e001,
                1.02954618e002,
                -4.57236460e002,
                -3.46587276e002,
                2.39798123e163,
            ]
        )
        z_true = np.array(
            [
                3.82847566e002,
                -1.54629241e002,
                -1.48748823e002,
                2.05022229e002,
                -3.91453385e002,
                8.82739133e001,
                4.80662215e001,
                -3.60841869e002,
                -1.03108165e002,
                1.69130696e202,
            ]
        )
        t_true = np.array(
            [
                3.26854347,
                2.777119,
                4.62438669,
                3.68113922,
                4.07755652,
                4.13901825,
                3.11122659,
                2.63355698,
                4.25371555,
                10.71373189,
            ]
        )

        self.assertTrue(np.allclose(x, x_true))
        self.assertTrue(np.allclose(y, y_true))
        self.assertTrue(np.allclose(z, z_true))
        self.assertTrue(np.allclose(t, t_true))

    def test_get_data_from_hdf_check_configured(self):
        """Check if error is raised when not configured"""
        module = CascadeGeneratorLabelModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.get_data_from_hdf("wrong_file_path")
        self.assertTrue("Module not configured yet!" in str(context.exception))

    def test_get_data_from_hdf_skip_file(self):
        """Check if file is skipped correctly if a label does not exist."""
        config = {
            "config_data": "dummy_data",
            "shift_cascade_vertex": True,
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)

        self.assertEqual(num_events, None)
        self.assertEqual(data, None)

    def test_get_data_from_hdf_wrong_file_name(self):
        """Check if IOError is raised if file does not exist"""
        config = {
            "config_data": "dummy_data",
            "shift_cascade_vertex": True,
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        with self.assertRaises(IOError) as context:
            num_events, data = module.get_data_from_hdf("wrong_file_path")
        self.assertTrue("does not exist" in str(context.exception))

    def test_get_data_from_hdf(self):
        """Test if loaded data is correct"""
        config = {
            "config_data": "dummy_data",
            "shift_cascade_vertex": False,
            "trafo_log": False,
            "float_precision": "float64",
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertTrue((self.cascades_true == data[0]).all())

    def test_get_data_from_hdf_with_shifted_vertex(self):
        """Test if loaded data is correct"""
        config = {
            "config_data": "dummy_data",
            "shift_cascade_vertex": True,
            "trafo_log": False,
            "float_precision": "float64",
        }
        module = CascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertTrue((self.cascades_true[:, 3:6] == data[0][:, 3:6]).all())


if __name__ == "__main__":
    unittest.main()

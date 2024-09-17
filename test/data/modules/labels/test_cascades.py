import unittest
import numpy as np
import os

from egenerator.data.modules.labels.snowstorm_cascades import (
    SnowstormCascadeGeneratorLabelModule,
)
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
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = SnowstormCascadeGeneratorLabelModule()
        module.configure(**config)
        settings = module.configuration.settings
        self.assertEqual(settings["trafo_log"], False)
        self.assertEqual(settings["label_key"], "labels")
        self.assertEqual(settings["float_precision"], "float64")

    def test_configuration_check(self):
        """Check whether passed tensor list in configuration is checked and
        found to be wrong.
        """
        config = {
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        data_tensor_list_incorrect = DataTensorList(
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
        data_tensor_list_correct = DataTensorList(
            [
                DataTensor(
                    name="x_parameters",
                    shape=[None, 7],
                    tensor_type="label",
                    dtype="float64",
                    trafo=True,
                    trafo_log=False,
                )
            ]
        )
        module = SnowstormCascadeGeneratorLabelModule()

        # check if error is correctly raised when wrong data type is passed
        with self.assertRaises(ValueError) as context:
            config["config_data"] = data_tensor_list_incorrect
            module.configure(**config)
        self.assertTrue("Tensors are wrong:" in str(context.exception))

        # passing the correct data tensors should work
        config["config_data"] = data_tensor_list_correct
        module.configure(**config)

    def test_correct_configuration(self):
        """Check if the module correctly creates the tensors"""
        for trafo_log in [True, False, None]:
            config = {
                "config_data": "file_path_string",
                "trafo_log": trafo_log,
                "float_precision": "float64",
                "label_key": "labels",
            }
            module = SnowstormCascadeGeneratorLabelModule()
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
            print(module.data["label_tensors"])
            print(tensors_true)
            t = module.data["label_tensors"].list[0]
            print(
                module.data["label_tensors"]
                .list[0]
                .compare(tensors_true.list[0])
            )
            print(t.trafo_log)
            print(tensors_true.list[0].trafo_log)
            self.assertTrue(module.data["label_tensors"] == tensors_true)

            # make sure the internal check also works
            config = {
                "config_data": tensors_true,
                "trafo_log": trafo_log,
                "float_precision": "float64",
                "label_key": "labels",
            }
            module = SnowstormCascadeGeneratorLabelModule()
            module.configure(**config)

    def test_get_data_from_hdf_check_configured(self):
        """Check if error is raised when not configured"""
        module = SnowstormCascadeGeneratorLabelModule()

        with self.assertRaises(ValueError) as context:
            num_events, data = module.get_data_from_hdf("wrong_file_path")
        self.assertTrue("Module not configured yet!" in str(context.exception))

    def test_get_data_from_hdf_skip_file(self):
        """Check if file is skipped correctly if a label does not exist."""
        config = {
            "config_data": "dummy_data",
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = SnowstormCascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)

        self.assertEqual(num_events, None)
        self.assertEqual(data, None)

    def test_get_data_from_hdf_wrong_file_name(self):
        """Check if IOError is raised if file does not exist"""
        config = {
            "config_data": "dummy_data",
            "trafo_log": False,
            "float_precision": "float64",
            "label_key": "labels",
        }
        module = SnowstormCascadeGeneratorLabelModule()
        module.configure(**config)

        with self.assertRaises(IOError) as context:
            num_events, data = module.get_data_from_hdf("wrong_file_path")
        self.assertTrue("does not exist" in str(context.exception))

    def test_get_data_from_hdf(self):
        """Test if loaded data is correct"""
        config = {
            "config_data": "dummy_data",
            "trafo_log": False,
            "float_precision": "float64",
        }
        module = SnowstormCascadeGeneratorLabelModule()
        module.configure(**config)

        num_events, data = module.get_data_from_hdf(self.file_path)
        self.assertEqual(num_events, 2)
        self.assertTrue((self.cascades_true == data[0]).all())


if __name__ == "__main__":
    unittest.main()

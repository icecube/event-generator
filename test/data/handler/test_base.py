import unittest
import os
import numpy as np

from egenerator.manager.component import Configuration
from egenerator.data.handler.base import BaseDataHandler
from egenerator.data.tensor import DataTensor, DataTensorList


class DummyDataHandler(BaseDataHandler):

    def _configure(self, data, new_subcomponent=None, **kwargs):
        kwargs["data"] = data
        class_string = (
            os.path.splitext(__file__.split("event-generator/")[-1])[
                0
            ].replace("/", ".")
            + ".DummyDataHandler"
        )
        configuration = Configuration(class_string, kwargs)
        if new_subcomponent is None:
            dependent_components = None
        else:
            dependent_components = {"new_subcomponent": new_subcomponent}
        return configuration, data, dependent_components


class TestBaseDataHandler(unittest.TestCase):
    """Test base data handler class."""

    def setUp(self):
        # create handler object
        data_handler = DummyDataHandler()

        # fake setup
        tensors = DataTensorList(
            [
                DataTensor(
                    name="data_tensor",
                    shape=[None, 86, 1],
                    tensor_type="data",
                    dtype="float32",
                )
            ]
        )
        settings = dict(data={"tensors": tensors}, param=42)
        data_handler.configure(**settings)

        self.data_handler = data_handler
        self.config = {"config": "good_setting"}

    def test_object_initialization(self):
        data_handler = BaseDataHandler()
        self.assertEqual(data_handler.tensors, None)
        self.assertEqual(data_handler.configuration, None)
        self.assertEqual(data_handler._untracked_data["mp_processes"], [])
        self.assertEqual(data_handler._untracked_data["mp_managers"], [])
        self.assertEqual(data_handler._is_configured, False)

    def test_method_check_if_configured(self):
        """Test if check if configured raises an error"""
        data_handler = BaseDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler.check_if_configured()
        self.assertTrue(
            "Data handler needs to be set up first!" in str(context.exception)
        )

        # if we setup the data handler this should run without any errors
        data_handler._is_configured = True
        data_handler.check_if_configured()

    def test_method_configure_already_configured(self):
        data_handler = BaseDataHandler()

        with self.assertRaises(ValueError) as context:
            data_handler._is_configured = True
            data_handler._configure(**self.config)
        self.assertTrue(
            "The data handler is already set up!" in str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            data_handler._is_configured = True
            data_handler.configure(**self.config)
        self.assertTrue(
            "Component is already configured!" in str(context.exception)
        )

    def test_method_configure_pure_virtual_method(self):
        data_handler = BaseDataHandler()

        # with no config data
        with self.assertRaises(NotImplementedError):
            data_handler.configure(**self.config)

        # with a file path string
        with self.assertRaises(NotImplementedError):
            data_handler.configure(config=None, config_data="dummy_file_path")

        # with a list of file path strings
        with self.assertRaises(NotImplementedError):
            data_handler.configure(
                config=None,
                config_data=["dummy_file_path1", "dummy_file_path2"],
            )

    def test_methods_load_and_save(self):
        """Test the saving and loading of a previously created data handler obj."""

        # save trafo model
        directory = os.path.join(
            os.path.dirname(__file__),
            "../../../data/temp_test_files/data_handler",
        )

        # remove it if it already exists
        if os.path.exists(directory):
            os.remove(os.path.join(directory, "configuration.yaml"))
            os.remove(os.path.join(directory, "data.pickle"))
            os.removedirs(directory)

        self.data_handler.save(directory, overwrite=False)
        self.data_handler.save(directory, overwrite=True)

        # check error message when attempting to overwrite file
        with self.assertRaises(IOError) as context:
            self.data_handler.save(directory, overwrite=False)
        self.assertTrue(" already exists!" in str(context.exception))

        # check error message when attempting to load component with wrong class
        data_handler_new = BaseDataHandler()
        with self.assertRaises(TypeError) as context:
            data_handler_new.load(directory)
        self.assertTrue(
            "The object's class" in str(context.exception)
            and "not match the saved class" in str(context.exception)
        )

        # check loading of data handler
        data_handler_new = DummyDataHandler()
        data_handler_new.load(directory)
        self.assertTrue(self.data_handler.is_compatible(data_handler_new))

    def test_method_check_data_structure(self):

        # create data
        # correct_data = np.ones([2, 86, 1])
        wrong_rank_data = np.ones([2, 1])
        wrong_shape_data = np.ones([2, 34, 1])

        # check error message when length does not match
        with self.assertRaises(ValueError) as context:
            self.data_handler._check_data_structure(wrong_rank_data)
        self.assertTrue("Length" in str(context.exception))

        # check error message when rank does not match
        with self.assertRaises(ValueError) as context:
            self.data_handler._check_data_structure([wrong_rank_data])
        self.assertTrue("Rank" in str(context.exception))

        # check error message when shapes does not match
        with self.assertRaises(ValueError) as context:
            self.data_handler._check_data_structure([wrong_shape_data])
        self.assertTrue("Shapes" in str(context.exception))

        # check not implemented error message when trying to check
        # vector data tensors (tensor.vector_info is not None)
        with self.assertRaises(NotImplementedError):
            self.data_handler._check_data_structure([wrong_shape_data], True)

    def test_not_implemented_method_get_data_from_hdf(self):
        with self.assertRaises(NotImplementedError):
            self.data_handler.get_data_from_hdf("file_name_path")

    def test_not_implemented_method_get_data_from_frame(self):
        with self.assertRaises(NotImplementedError):
            self.data_handler.get_data_from_frame(None)

    def test_not_implemented_method_create_data_from_frame(self):
        with self.assertRaises(NotImplementedError):
            self.data_handler.create_data_from_frame(None)

    def test_not_implemented_method_write_data_to_frame(self):
        with self.assertRaises(NotImplementedError):
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
            values, indices, num_events
        )

        for v1, v2 in zip(values_list, values_list_true):
            self.assertTrue(np.allclose(v1, v2))
        for v1, v2 in zip(indices_list, indices_list_true):
            self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    unittest.main()

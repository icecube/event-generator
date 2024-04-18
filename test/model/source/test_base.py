import unittest
import os
import tensorflow as tf
import shutil
from copy import deepcopy
import numpy as np
import ruamel.yaml as yaml
import time
from datetime import datetime

from egenerator import misc
from egenerator.manager.component import Configuration, BaseComponent
from egenerator.model.source.base import Source, InputTensorIndexer


class DummyDataTrafo(BaseComponent):

    def _configure(self, config_data):
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={"config_data": config_data},
        )
        return configuration, {}, {}


class DummySourceModel(Source):

    def _build_architecture(self, config, name=None):
        self.assert_configured(False)

        parameter_names = [
            "x",
            "y",
            "z",
            "zenith",
            "azimuth",
            "energy",
            "time",
        ]

        self._untracked_data["dummy_var"] = tf.Variable(1.0, name="dummy_var")

        return parameter_names

    def get_tensors(self, parameters, pulses, pulses_ids):
        self.assert_configured(True)

        temp_var_reshaped = tf.reshape(
            tf.reduce_sum(parameters, axis=1), [-1, 1, 1, 1]
        )
        dom_charges = tf.ones([1, 86, 60, 1]) * temp_var_reshaped
        pulse_pdf = tf.reduce_sum(pulses, axis=1)

        tensor_dict = {
            "dom_charges": dom_charges,
            "pulse_pdf": pulse_pdf,
        }

        return tensor_dict


class DummySourceModelWrong(Source):

    def _build_architecture(self, config, name=None):
        self.assert_configured(False)

        parameter_names = [
            "x",
            "y",
            "z",
            "zenith",
            "azimuth",
            "energy",
            "time",
        ]

        self._untracked_data["dummy_var"] = tf.Variable(1.0, name="dummy_var")
        new_untracked_var = tf.Variable(  # noqa: F841
            42, name="this_should_raise_error"
        )

        return parameter_names

    def get_tensors(self, parameters, pulses, pulses_ids):
        self.assert_configured(True)

        temp_var_reshaped = tf.reshape(
            tf.reduce_sum(parameters, axis=1), [-1, 1, 1, 1]
        )
        dom_charges = tf.ones([1, 86, 60, 1]) * temp_var_reshaped
        pulse_pdf = tf.reduce_sum(pulses, axis=1)

        tensor_dict = {
            "dom_charges": dom_charges,
            "pulse_pdf": pulse_pdf,
        }

        return tensor_dict


class TestInputTensorIndexer(unittest.TestCase):

    def test_input_tensor_indexer(self):
        tensor = np.arange(10)
        var_names = ["var_{:02d}".format(i) for i in tensor]

        # check sanity check
        with self.assertRaises(ValueError) as context:
            indexer = InputTensorIndexer(tensor, var_names + ["too_many"])
        self.assertTrue("Shapes do not match up" in str(context.exception))

        indexer = InputTensorIndexer(tensor, var_names)
        for i, name in enumerate(var_names):
            self.assertEqual(indexer[name], tensor[i])
        self.assertEqual(indexer.var_00, 0)
        self.assertEqual(indexer.var_01, 1)
        self.assertEqual(indexer.var_09, 9)


class TestSourceBase(unittest.TestCase):
    """Test base class for Models"""

    def setUp(self):
        self.data_trafo = DummyDataTrafo()
        self.data_trafo.configure(config_data={"trafo_setting": 42})

        self.parameter_names = [
            "x",
            "y",
            "z",
            "zenith",
            "azimuth",
            "energy",
            "time",
        ]

        config = {"dummysourcesetting": 1337}
        self.source = self.get_dummy_source(
            config=config, data_trafo=self.data_trafo
        )

        class_string = (
            os.path.splitext(__file__.split("event-generator/")[-1])[
                0
            ].replace("/", ".")
            + ".DummySourceModel"
        )
        self.configuration = Configuration(
            class_string=class_string,
            settings=dict(config=config),
            mutable_settings=dict(name="egenerator.model.source.base"),
        )
        self.configuration.add_sub_components({"data_trafo": self.data_trafo})

    def get_dummy_source(self, **kwargs):
        model = DummySourceModel()
        model.configure(**kwargs)
        return model

    def get_directory(self, idx):
        directory = os.path.join(
            os.path.dirname(__file__),
            "../../../data/temp_test_files/source_base_{:02d}".format(idx),
        )

        # remove it if it already exists
        if os.path.exists(directory):
            shutil.rmtree(directory)
        return directory

    def check_model_checkpoint_meta_data(self, true_meta_data, directory):
        """Check if meta data of a saved checkpoint matches expected meta data.

        Parameters
        ----------
        true_meta_data : dict
            The expected meta data.
        directory : str
            The path to the saved checkpoint directory
        """

        true_meta_data = dict(deepcopy(true_meta_data))

        # load  meta data
        yaml_file = os.path.join(directory, "model_checkpoint.yaml")
        with open(yaml_file, "r") as stream:
            meta_data = yaml.safe_load(stream)

        # check approximate values of time stamps (these will naturally differ)
        for key in ["unprotected_checkpoints", "protected_checkpoints"]:
            for index in true_meta_data[key].keys():

                # make sure that time stamps are within 10 seconds
                time_stamp_true = true_meta_data[key][index].pop("time_stamp")
                time_stamp = meta_data[key][index].pop("time_stamp")
                self.assertLess(np.abs(time_stamp - time_stamp_true), 10)

                # check year, month, day and hour of date
                # Note: in very unlikely cases this may file
                date_true = true_meta_data[key][index].pop("creation_date")
                date = meta_data[key][index].pop("creation_date")
                self.assertEqual(date_true[:17], date[:17])

        self.assertDictEqual(meta_data, true_meta_data)

    def test_correct_initialization(self):
        """Check if member variables are correctly set when instantiating
        a Model object.
        """
        model = DummySourceModel()
        self.assertEqual(model._is_configured, False)
        self.assertEqual(model.data, None)
        self.assertEqual(model.configuration, None)
        self.assertEqual(model._untracked_data, {})
        self.assertEqual(model._sub_components, {})
        self.assertEqual(model.checkpoint, None)
        self.assertEqual(model.data_trafo, None)
        self.assertEqual(model.name, None)
        self.assertEqual(model.parameter_names, None)

    def test_configuration_of_abstract_class(self):
        """Model is an abstract class with a pure virtual method.
        Attempting to configure it should result in a NotImplementedError
        """
        model = Source()

        with self.assertRaises(NotImplementedError):
            model.configure(config={}, data_trafo=self.data_trafo)

    def test_correct_configuration_name(self):
        """Test if the name passed in to the configuration is properly saved"""
        model = DummySourceModel()
        model.configure(config={}, data_trafo=self.data_trafo, name="dummy")
        self.assertEqual(model.name, "dummy")

    # def test_configuration_check_for_untracked_variables(self):
    #     """All tensorflow variables should be tracked. the Source method
    #     _configure_derived_class has a safety check built in that checks for
    #     this. Test if it can find created variables
    #     """
    #     model = DummySourceModelWrong()
    #     with self.assertRaises(ValueError) as context:
    #         tf.compat.v1.disable_eager_execution()
    #         model.configure(config={}, data_trafo=self.data_trafo)
    #     self.assertTrue('Found variable that is not part of the tf.Module: '
    #                     in str(context.exception))

    def test_correct_configuration(self):
        """Check if member variables are set correctly after the dummy model
        is configured
        """
        self.assertEqual(self.source.is_configured, True)
        self.assertTrue(
            isinstance(self.source.checkpoint, tf.train.Checkpoint)
        )
        self.assertEqual(self.source.data, {})
        self.assertEqual(self.source.data_trafo, self.data_trafo)
        self.assertEqual(self.source.name, "egenerator.model.source.base")
        self.assertEqual(self.source.parameter_names, self.parameter_names)
        self.assertTrue(
            self.source.configuration.is_compatible(self.configuration)
        )

        self.assertDictEqual(
            self.source._untracked_data,
            {
                "name": "egenerator.model.source.base",
                "checkpoint": self.source.checkpoint,
                "step": self.source._untracked_data["step"],
                "dummy_var": self.source._untracked_data["dummy_var"],
                "variables": self.source._untracked_data["variables"],
                "num_parameters": 7,
                "parameter_index_dict": self.source._untracked_data[
                    "parameter_index_dict"
                ],
                "parameter_name_dict": self.source._untracked_data[
                    "parameter_name_dict"
                ],
                "parameter_names": [
                    "x",
                    "y",
                    "z",
                    "zenith",
                    "azimuth",
                    "energy",
                    "time",
                ],
            },
        )
        self.assertEqual(
            self.source._sub_components, {"data_trafo": self.data_trafo}
        )

    def test_method_assert_configured(self):
        """Check the method assert_configured()"""
        model = Source()
        model.assert_configured(False)
        with self.assertRaises(ValueError) as context:
            model.assert_configured(True)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

        self.source.assert_configured(True)
        with self.assertRaises(ValueError) as context:
            self.source.assert_configured(False)
        self.assertTrue("Model is already set up!" in str(context.exception))

        with self.assertRaises(TypeError) as context:
            model.assert_configured(None)
        self.assertTrue("Expected bool, but got" in str(context.exception))

    def test_save_weights_variables_changed(self):
        """If the model's variables changed after it was configured, this
        should raise an Error. The variables of a Model instance must be fully
        defined in the configure() call.
        """
        directory = self.get_directory(1)

        # check if normal saving works
        self.source.save_weights(directory)
        self.source.dummy_var = tf.Variable(43)

        # Now modify the architecture by adding another variable
        with self.assertRaises(ValueError) as context:
            self.source.save_weights(directory)
        self.assertTrue(
            "Model has changed since configuration call:"
            in str(context.exception)
        )

    def test_save_weights_meta_data(self):
        """Test the save weights method and make sure the saved meta data is
        corrected.
        """
        directory = self.get_directory(2)

        # add entry to meta data
        checkpoint_meta_data = {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_stamp": time.time(),
            "file_basename": "model_checkpoint_{:08d}".format(1),
            "description": None,
        }

        true_meta_data = {
            "latest_checkpoint": 1,
            "unprotected_checkpoints": {1: checkpoint_meta_data},
            "protected_checkpoints": {},
        }

        # Save model for the first time
        self.source.save_weights(directory)

        # load and check meta data
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

        # now add another checkpoint
        second_checkpoint_meta_data = {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_stamp": time.time(),
            "file_basename": "model_checkpoint_{:08d}".format(2),
            "description": None,
        }

        true_meta_data = {
            "latest_checkpoint": 2,
            "unprotected_checkpoints": {
                1: checkpoint_meta_data,
                2: second_checkpoint_meta_data,
            },
            "protected_checkpoints": {},
        }

        # Save model for the first time
        self.source.save_weights(directory)

        # load and check meta data
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

        # now add a protected checkpoint with a description
        description = "This is my model at a tagged version 1.0.1"
        third_checkpoint_meta_data = {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_stamp": time.time(),
            "file_basename": "model_checkpoint_{:08d}".format(3),
            "description": "This is my model at a tagged version 1.0.1",
        }

        true_meta_data = {
            "latest_checkpoint": 3,
            "unprotected_checkpoints": {
                1: checkpoint_meta_data,
                2: second_checkpoint_meta_data,
            },
            "protected_checkpoints": {3: third_checkpoint_meta_data},
        }

        # Save model for the first time
        self.source.save_weights(
            directory, protected=True, description=description
        )

        # load and check meta data
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

    def test_method_save(self):
        """Test the save method and make sure the saved meta data is
        corrected.
        """
        directory = self.get_directory(6)

        # add entry to meta data
        checkpoint_meta_data = {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_stamp": time.time(),
            "file_basename": "model_checkpoint_{:08d}".format(1),
            "description": None,
        }

        true_meta_data = {
            "latest_checkpoint": 1,
            "unprotected_checkpoints": {1: checkpoint_meta_data},
            "protected_checkpoints": {},
        }

        # Save model for the first time
        self.source.save(directory)

        # load and check meta data
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

        new_checkpoint_meta_data = {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_stamp": time.time(),
            "file_basename": "model_checkpoint_{:08d}".format(2),
            "description": None,
        }
        true_meta_data = {
            "latest_checkpoint": 2,
            "unprotected_checkpoints": {
                1: checkpoint_meta_data,
                2: new_checkpoint_meta_data,
            },
            "protected_checkpoints": {},
        }

        self.source.save(directory, overwrite=True)
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

    def test_method_load_weights(self):
        """Test the load weights method and make sure the loaded weights are
        correct
        """
        directory = self.get_directory(7)
        self.source._untracked_data["dummy_var"].assign(1337)
        self.assertTrue(
            self.source._untracked_data["dummy_var"].numpy() == 1337
        )

        self.source.save_weights(directory)
        self.source._untracked_data["dummy_var"].assign(37)
        self.source.save_weights(directory)
        self.source._untracked_data["dummy_var"].assign(42)

        self.assertTrue(self.source._untracked_data["dummy_var"].numpy() == 42)

        # now load the latest saved model weights which was 37
        self.source.load_weights(directory)
        self.assertTrue(self.source._untracked_data["dummy_var"].numpy() == 37)

        # now load the first saved model weights which was 1337
        self.source.load_weights(directory, checkpoint_number=1)
        self.assertTrue(
            self.source._untracked_data["dummy_var"].numpy() == 1337
        )

    def test_method_load(self):
        """Test the load method and make sure the loaded weights are correct"""
        directory = self.get_directory(8)
        self.source._untracked_data["dummy_var"].assign(1337)
        self.assertTrue(
            self.source._untracked_data["dummy_var"].numpy() == 1337
        )

        self.source.save(directory)
        self.source._untracked_data["dummy_var"].assign(37)
        self.source.save(directory, overwrite=True, protected=True)
        self.source._untracked_data["dummy_var"].assign(42)

        self.assertTrue(self.source._untracked_data["dummy_var"].numpy() == 42)

        # create a new model from scratch
        model = DummySourceModel()
        # now load the latest saved model weights which was 37
        model.load(directory)
        self.assertTrue(model._untracked_data["dummy_var"].numpy() == 37)

        # now load the first saved model weights which was 1337
        model.load_weights(directory, checkpoint_number=1)
        self.assertTrue(model._untracked_data["dummy_var"].numpy() == 1337)

    def test_method_load_weights_file_error(self):
        """Test if IOError is raised if there is no saved checkpoint meta data"""
        directory = self.get_directory(10)

        # index 1, but this index already exists, so it should raise an error
        with self.assertRaises(IOError) as context:
            self.source.load_weights(directory)
        self.assertTrue(
            "Could not find checkpoints meta data" in str(context.exception)
        )

    def test_method_get_index(self):
        for i, name in enumerate(self.parameter_names):
            self.assertEqual(self.source.get_index(name), i)
            self.assertEqual(self.source.get_name(i), name)

    def test_method_get_tensors_not_implemented_error(self):
        source = Source()
        with self.assertRaises(ValueError) as context:
            source.get_tensors(None, True)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

        with self.assertRaises(NotImplementedError) as context:
            source._is_configured = True
            source.get_tensors(None, True)

    def test_parameter_indexing(self):
        tensor = tf.constant(self.parameter_names)
        self.source.add_parameter_indexing(tensor)
        self.assertEqual(tensor.params.x, "x")


if __name__ == "__main__":
    unittest.main()

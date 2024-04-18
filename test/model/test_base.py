import unittest
import os
import tensorflow as tf
import shutil
from copy import deepcopy
import numpy as np
import ruamel.yaml as yaml
import time
import glob
from datetime import datetime

from egenerator.manager.component import Configuration
from egenerator.model.base import Model


class DummyModel(Model):

    def _configure_derived_class(self, **kwargs):
        class_string = (
            os.path.splitext(__file__.split("event-generator/")[-1])[
                0
            ].replace("/", ".")
            + ".DummyModel"
        )
        configuration = Configuration(class_string, kwargs)
        data = {}
        dependent_components = {}

        # create a variable
        self._untracked_data["var"] = tf.Variable(1)

        return configuration, data, dependent_components


WrongDummyModel_COUNTER = 0


class WrongDummyModel(Model):
    """This is a dummy class that will modify tracked data in the _configure()
    method if the global counter is greater than zero.
    """

    def _configure_derived_class(self, **kwargs):
        global WrongDummyModel_COUNTER
        class_string = (
            os.path.splitext(__file__.split("event-generator/")[-1])[
                0
            ].replace("/", ".")
            + ".WrongDummyModel"
        )
        configuration = Configuration(class_string, kwargs)
        data = {}
        dependent_components = {}

        # create a variable
        self._untracked_data["var"] = tf.Variable(1)
        if WrongDummyModel_COUNTER > 0:
            self._data["this_changes"] = WrongDummyModel_COUNTER

        return configuration, data, dependent_components


class TestModel(unittest.TestCase):
    """Test base class for Models"""

    def setUp(self):
        self.model = self.get_dummy_model()

        class_string = (
            os.path.splitext(__file__.split("event-generator/")[-1])[
                0
            ].replace("/", ".")
            + ".DummyModel"
        )
        self.configuration = Configuration(class_string, {})

    def get_dummy_model(self, **kwargs):
        model = DummyModel()
        model.configure(**kwargs)
        return model

    def get_directory(self, index):
        directory = os.path.join(
            os.path.dirname(__file__),
            "../../data/temp_test_files/trafo_model_base_{:02d}".format(index),
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
        model = Model()
        self.assertEqual(model._is_configured, False)
        self.assertEqual(model.data, None)
        self.assertEqual(model.configuration, None)
        self.assertEqual(model._untracked_data, {})
        self.assertEqual(model._sub_components, {})
        self.assertEqual(model.checkpoint, None)

    def test_configuration_of_abstract_class(self):
        """Model is an abstract class with a pure virtual method.
        Attempting to configure it should result in a NotImplementedError
        """
        model = Model()

        with self.assertRaises(NotImplementedError):
            model.configure()

    def test_correct_configuration(self):
        """Check if member variables are set correctly after the dummy model
        is configured
        """
        self.assertEqual(self.model.is_configured, True)
        self.assertTrue(isinstance(self.model.checkpoint, tf.train.Checkpoint))
        self.assertEqual(self.model.data, {})
        self.assertTrue(
            self.model.configuration.is_compatible(self.configuration)
        )

        untracked_data = {
            "checkpoint": self.model.checkpoint,
            "variables": self.model.variables,
            "step": self.model._untracked_data["step"],
            "var": self.model._untracked_data["var"],
        }
        self.assertDictEqual(self.model._untracked_data, untracked_data)
        self.assertEqual(self.model._sub_components, {})

    def test_method_assert_configured(self):
        """Check the method assert_configured()"""
        model = Model()
        model.assert_configured(False)
        with self.assertRaises(ValueError) as context:
            model.assert_configured(True)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

        self.model.assert_configured(True)
        with self.assertRaises(ValueError) as context:
            self.model.assert_configured(False)
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
        self.model.save_weights(directory)
        self.model.new_var = tf.Variable(43)

        # Now modify the architecture by adding another variable
        with self.assertRaises(ValueError) as context:
            self.model.save_weights(directory)
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
        self.model.save_weights(directory)

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
        self.model.save_weights(directory)

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
        self.model.save_weights(
            directory, protected=True, description=description
        )

        # load and check meta data
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

    def test_save_weights_file_already_exists(self):
        """Test if the save weights method raises an error if the checkpoint
        file already exists
        """

        directory = self.get_directory(3)

        # Save model for the first time
        self.model.save_weights(directory)
        self.model.save_weights(directory)

        # now lets delete the configuration file
        os.remove(os.path.join(directory, "model_checkpoint.yaml"))

        # Saving now will create a new meta data file and attempt to save to
        # index 1, but this already exists, so it should raise an error
        with self.assertRaises(IOError) as context:
            self.model.save_weights(directory)
        self.assertTrue(
            "Checkpoint file " in str(context.exception)
            and "already exists!" in str(context.exception)
        )

    def test_save_weights_meta_data_already_exists(self):
        """Test if the save weights method raises an error if the checkpoint
        meta data already exists
        """

        directory = self.get_directory(4)

        # create an entry for a checkpoint
        checkpoint_meta_data = {
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_stamp": time.time(),
            "file_basename": "model_checkpoint_{:08d}".format(1),
            "description": None,
        }

        true_meta_data = {
            "latest_checkpoint": 0,
            "unprotected_checkpoints": {1: checkpoint_meta_data},
            "protected_checkpoints": {},
        }

        # save this to the model_checkpoint.yaml file
        os.makedirs(directory)
        yaml_file = os.path.join(directory, "model_checkpoint.yaml")
        with open(yaml_file, "w") as stream:
            yaml.dump(true_meta_data, stream)

        # Saving now will load the existing meta data and attempt to save to
        # index 1, but this index already exists, so it should raise an error
        with self.assertRaises(KeyError) as context:
            self.model.save_weights(directory)
        self.assertTrue(
            "Checkpoint index " in str(context.exception)
            and "already exists in meta dat" in str(context.exception)
        )

    def test_save_weights_max_keep(self):
        """Test if the max_keep option works as expected an only keeps at most
        that many unprotected checkpoints
        """
        directory = self.get_directory(5)

        self.model.save_weights(directory)
        self.model.save_weights(directory)
        self.model.save_weights(directory)

        self.assertEqual(
            3,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*.index")
                )
            ),
        )

        # adding a fourth model will result in the first one being deleted
        self.model.save_weights(directory)
        self.assertEqual(
            0,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*01.index")
                )
            ),
        )
        self.assertEqual(
            3,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*.index")
                )
            ),
        )

        # check changing max_keep
        self.model.save_weights(directory, max_keep=5)
        self.model.save_weights(directory, max_keep=5)
        self.assertEqual(
            0,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*01.index")
                )
            ),
        )
        self.assertEqual(
            5,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*.index")
                )
            ),
        )

        # check changing back
        self.model.save_weights(directory)
        self.model.save_weights(directory)
        self.assertEqual(
            0,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*05.index")
                )
            ),
        )
        self.assertEqual(
            3,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*.index")
                )
            ),
        )

        # now test protected models: we should be able to have as many as we
        # want
        self.model.save_weights(directory, protected=True)
        self.model.save_weights(directory, protected=True)
        self.model.save_weights(directory, protected=True)
        self.model.save_weights(directory, protected=True)
        self.model.save_weights(directory, protected=True)
        self.assertEqual(
            0,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*05.index")
                )
            ),
        )
        self.assertEqual(
            1,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*06.index")
                )
            ),
        )
        self.assertEqual(
            8,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*.index")
                )
            ),
        )

        # adding a new unprotected models, should now only remove the
        # unprotected ones
        self.model.save_weights(directory)  # 14
        self.model.save_weights(directory)  # 15
        self.model.save_weights(directory)  # 16
        self.model.save_weights(directory)  # 17
        self.model.save_weights(directory)  # 18
        self.assertEqual(
            0,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*08.index")
                )
            ),
        )
        self.assertEqual(
            1,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*09.index")
                )
            ),
        )
        self.assertEqual(
            1,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*013.index")
                )
            ),
        )
        self.assertEqual(
            0,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*014.index")
                )
            ),
        )
        self.assertEqual(
            1,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*016.index")
                )
            ),
        )
        self.assertEqual(
            8,
            len(
                glob.glob(
                    os.path.join(directory, "model_checkpoint_00*.index")
                )
            ),
        )

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
        self.model.save(directory)

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

        self.model.save(directory, overwrite=True)
        self.check_model_checkpoint_meta_data(true_meta_data, directory)

    def test_method_load_weights(self):
        """Test the load weights method and make sure the loaded weights are
        correct
        """
        directory = self.get_directory(7)
        self.model._untracked_data["var"].assign(1337)
        self.assertTrue(self.model._untracked_data["var"].numpy() == 1337)

        self.model.save_weights(directory)
        self.model._untracked_data["var"].assign(37)
        self.model.save_weights(directory)
        self.model._untracked_data["var"].assign(42)

        self.assertTrue(self.model._untracked_data["var"].numpy() == 42)

        # now load the latest saved model weights which was 37
        self.model.load_weights(directory)
        self.assertTrue(self.model._untracked_data["var"].numpy() == 37)

        # now load the first saved model weights which was 1337
        self.model.load_weights(directory, checkpoint_number=1)
        self.assertTrue(self.model._untracked_data["var"].numpy() == 1337)

    def test_method_load(self):
        """Test the load method and make sure the loaded weights are correct"""
        directory = self.get_directory(8)
        self.model._untracked_data["var"].assign(1337)
        self.assertTrue(self.model._untracked_data["var"].numpy() == 1337)

        self.model.save(directory)
        self.model._untracked_data["var"].assign(37)
        self.model.save(directory, overwrite=True, protected=True)
        self.model._untracked_data["var"].assign(42)

        self.assertTrue(self.model._untracked_data["var"].numpy() == 42)

        # create a new model from scratch
        model = DummyModel()
        # now load the latest saved model weights which was 37
        model.load(directory)
        self.assertTrue(model._untracked_data["var"].numpy() == 37)

        # now load the first saved model weights which was 1337
        model.load_weights(directory, checkpoint_number=1)
        self.assertTrue(model._untracked_data["var"].numpy() == 1337)

    def test_method_load_weights_tracked_components_changed(self):
        """Test the load_weights method and check that it raises an error
        if the tracked components change in the _configure method
        """
        directory = self.get_directory(9)
        model = WrongDummyModel()
        model.configure()
        model.save(directory)

        # now update counter
        global WrongDummyModel_COUNTER
        WrongDummyModel_COUNTER += 1

        # index 1, but this index already exists, so it should raise an error
        model = WrongDummyModel()
        with self.assertRaises(ValueError) as context:
            model.load(directory)
        self.assertTrue(
            "Tracked components were changed!" in str(context.exception)
        )

    def test_method_load_weights_file_error(self):
        """Test if IOError is raised if there is no saved checkpoint meta data"""
        directory = self.get_directory(10)

        # index 1, but this index already exists, so it should raise an error
        with self.assertRaises(IOError) as context:
            self.model.load_weights(directory)
        self.assertTrue(
            "Could not find checkpoints meta data" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()

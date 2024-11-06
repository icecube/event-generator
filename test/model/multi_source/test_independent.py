import unittest
import os
import tensorflow as tf
import shutil
from copy import deepcopy
import numpy as np
import time
from datetime import datetime

from egenerator import misc
from egenerator.manager.component import Configuration, BaseComponent
from egenerator.data.tensor import DataTensor, DataTensorList
from egenerator.model.multi_source.base import MultiSource
from egenerator.model.multi_source.independent import IndependentMultiSource
from egenerator.model.source.cascade.dummy import DummyCascadeModel
from egenerator.settings.yaml import yaml_loader


class DummyDataTrafo(BaseComponent):

    def _configure(self, config_data):

        # create "data" component of the data trafo
        data = {
            "tensors": DataTensorList(
                [
                    DataTensor(
                        name="x_parameters",
                        shape=[1, 7],
                        tensor_type="data",
                        dtype="float32",
                    ),
                    DataTensor(
                        name="x_pulses",
                        shape=[7, 2],
                        tensor_type="data",
                        dtype="float32",
                    ),
                    DataTensor(
                        name="x_pulses_ids",
                        shape=[7, 4],
                        tensor_type="data",
                        dtype="int32",
                    ),
                ]
            )
        }

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={"config_data": config_data},
        )
        return configuration, data, {}


class TestIndependentMultiSource(unittest.TestCase):
    """Test base class for Models"""

    def setUp(self):
        self.data_trafo = DummyDataTrafo()
        self.data_trafo.configure(config_data={"trafo_setting": 42})

        self.parameter_names = [
            "cascade_00001_x",
            "cascade_00001_y",
            "cascade_00001_z",
            "cascade_00001_zenith",
            "cascade_00001_azimuth",
            "cascade_00001_energy",
            "cascade_00001_time",
            "cascade_00002_x",
            "cascade_00002_y",
            "cascade_00002_z",
            "cascade_00002_zenith",
            "cascade_00002_azimuth",
            "cascade_00002_energy",
            "cascade_00002_time",
        ]
        self.config_cascade = {
            "cascade_setting": 1337,
            "float_precision": "float32",
        }
        self.base_models = {
            "cascade": self.get_cascade_source(
                config=self.config_cascade, data_trafo=self.data_trafo
            )
        }
        self.sources = {
            "cascade_00001": "cascade",
            "cascade_00002": "cascade",
        }
        self.config = {
            "sources": self.sources,
            "float_precision": "float32",
        }
        self.sub_components = {"cascade": self.base_models["cascade"]}
        self.source = self.get_muon(
            config=self.config,
            base_models=self.base_models,
        )

        class_string = "egenerator.model.multi_source.independent."
        class_string += "IndependentMultiSource"
        self.configuration = Configuration(
            class_string=class_string,
            settings=dict(config=self.config),
            mutable_settings=dict(name="egenerator.model.multi_source.base"),
        )
        self.configuration.add_sub_components(self.sub_components)

        # add dependent sub components to this configuration
        self.configuration.add_dependent_sub_components(
            [component for component in self.sub_components.keys()]
        )

    def get_muon(self, **kwargs):
        model = IndependentMultiSource()
        model.configure(**kwargs)
        return model

    def get_cascade_source(self, **kwargs):
        model = DummyCascadeModel()
        model.configure(**kwargs)
        return model

    def get_directory(self, idx):
        directory = os.path.join(
            os.path.dirname(__file__),
            "../../../../data/temp_test_files/",
            "muon_multi_source_base_{:02d}".format(idx),
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
            meta_data = yaml_loader.load(stream)

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
        model = IndependentMultiSource()
        self.assertEqual(model._is_configured, False)
        self.assertEqual(model.data, None)
        self.assertEqual(model.configuration, None)
        self.assertEqual(model._untracked_data, {})
        self.assertEqual(model._sub_components, {})
        self.assertEqual(model.checkpoint, None)
        self.assertEqual(model.name, None)
        self.assertEqual(model.parameter_names, None)
        self.assertEqual(model.num_parameters, None)

    def test_configuration_of_abstract_class(self):
        """MultiSource is an abstract class with a pure virtual method.
        Attempting to configure it should result in a NotImplementedError
        """
        model = MultiSource()

        with self.assertRaises(NotImplementedError):
            model.configure(config=self.config, base_models=self.base_models)

    def test_correct_configuration_name(self):
        """Test if the name passed in to the configuration is properly saved"""
        model = IndependentMultiSource()
        model.configure(
            config=self.config, base_models=self.base_models, name="dummy"
        )
        self.assertEqual(model.name, "dummy")

    # # def test_configuration_check_for_untracked_variables(self):
    # #     """All tensorflow variables should be tracked. the Source method
    # #     _configure_derived_class has a safety check built in that checks for
    # #     this. Test if it can find created variables
    # #     """
    # #     model = DummySourceModelWrong()
    # #     with self.assertRaises(ValueError) as context:
    # #         tf.compat.v1.disable_eager_execution()
    # #         model.configure(config={}, data_trafo=self.data_trafo)
    # #     self.assertTrue('Found variable that is not part of the tf.Module: '
    # #                     in str(context.exception))

    def test_correct_configuration(self):
        """Check if member variables are set correctly after the dummy model
        is configured
        """
        self.assertEqual(self.source.is_configured, True)
        self.assertTrue(
            isinstance(self.source.checkpoint, tf.train.Checkpoint)
        )
        self.assertEqual(self.source.data, {})
        self.assertEqual(
            self.source.name, "egenerator.model.multi_source.base"
        )
        self.assertEqual(self.source.parameter_names, self.parameter_names)
        self.assertTrue(
            self.source.configuration.is_compatible(self.configuration)
        )

        self.maxDiff = None
        self.assertDictEqual(
            self.source._untracked_data,
            {
                "name": "egenerator.model.multi_source.base",
                "num_parameters": 14,
                "checkpoint": self.source.checkpoint,
                "step": self.source._untracked_data["step"],
                "variables": self.source._untracked_data["variables"],
                "parameter_index_dict": self.source._untracked_data[
                    "parameter_index_dict"
                ],
                "parameter_name_dict": self.source._untracked_data[
                    "parameter_name_dict"
                ],
                "parameter_names": self.parameter_names,
                "models_mapping": self.sources,
            },
        )
        self.assertEqual(self.source._sub_components, self.sub_components)

    def test_method_assert_configured(self):
        """Check the method assert_configured()"""
        model = IndependentMultiSource()
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
        """If the model's variables (not their values) changed after it was
        configured, this should raise an Error. The variables of a Model
        instance must be fully defined in the configure() call.
        """
        directory = self.get_directory(1)

        # check if normal saving works
        self.source.save_weights(directory)
        self.source.sub_components["cascade"].new_var = tf.Variable(43)

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
        for name, sub_component in self.source.sub_components.items():
            sub_dir_path = os.path.join(directory, name)
            self.check_model_checkpoint_meta_data(true_meta_data, sub_dir_path)

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
        for name, sub_component in self.source.sub_components.items():
            sub_dir_path = os.path.join(directory, name)
            self.check_model_checkpoint_meta_data(true_meta_data, sub_dir_path)

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
        for name, sub_component in self.source.sub_components.items():
            sub_dir_path = os.path.join(directory, name)
            self.check_model_checkpoint_meta_data(true_meta_data, sub_dir_path)

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

        # load and check meta data of cascade sub component
        self.maxDiff = None
        self.check_model_checkpoint_meta_data(
            true_meta_data, os.path.join(directory, "cascade")
        )

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
        self.check_model_checkpoint_meta_data(
            true_meta_data, os.path.join(directory, "cascade")
        )

    def test_method_load_weights(self):
        """Test the load weights method and make sure the loaded weights are
        correct
        """
        directory = self.get_directory(7)

        cascade = self.source.sub_components["cascade"]
        cascade._untracked_data["dummy_var"].assign(1337)
        self.assertTrue(cascade._untracked_data["dummy_var"].numpy() == 1337)

        self.source.save_weights(directory)
        cascade._untracked_data["dummy_var"].assign(37)
        self.source.save_weights(directory)
        cascade._untracked_data["dummy_var"].assign(42)

        self.assertTrue(cascade._untracked_data["dummy_var"].numpy() == 42)

        # now load the latest saved model weights which was 37
        self.source.load_weights(directory)
        self.assertTrue(cascade._untracked_data["dummy_var"].numpy() == 37)

        # now load the first saved model weights which was 1337
        self.source.load_weights(directory, checkpoint_number=1)
        self.assertTrue(cascade._untracked_data["dummy_var"].numpy() == 1337)

    def test_method_load(self):
        """Test the load method and make sure the loaded weights are correct"""
        directory = self.get_directory(8)

        cascade = self.source.sub_components["cascade"]
        cascade._untracked_data["dummy_var"].assign(1337)
        self.assertTrue(cascade._untracked_data["dummy_var"].numpy() == 1337)

        self.source.save(directory)
        cascade._untracked_data["dummy_var"].assign(37)
        self.source.save(directory, overwrite=True, protected=True)
        cascade._untracked_data["dummy_var"].assign(42)

        self.assertTrue(cascade._untracked_data["dummy_var"].numpy() == 42)

        # create a new model from scratch
        model = IndependentMultiSource()
        # now load the latest saved model weights which was 37
        model.load(directory)

        new_cascade = model.sub_components["cascade"]
        self.assertTrue(new_cascade._untracked_data["dummy_var"].numpy() == 37)

        # now load the first saved model weights which was 1337
        model.load_weights(directory, checkpoint_number=1)
        self.assertTrue(
            new_cascade._untracked_data["dummy_var"].numpy() == 1337
        )

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

    def test_method_get_tensors_not_configured_error(self):
        source = MultiSource()
        with self.assertRaises(ValueError) as context:
            source.get_tensors(None, True)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

    def test_parameter_indexing(self):
        tensor = tf.constant(self.parameter_names)
        self.source.add_parameter_indexing(tensor)
        self.assertEqual(tensor.params.cascade_00002_x, "cascade_00002_x")

    def test_method_get_tensors(self):
        """Check if the the get_tensors() method works

        This is only a very simple check for correct output shapes.
        More extensive tests are necessary
        """
        source = self.get_muon(
            config=self.config,
            base_models=self.base_models,
            data_trafo=self.data_trafo,
        )

        data_batch_dict = {
            "x_parameters": tf.ones([1, source.num_parameters]),
            "x_pulses": tf.ones([7, 2]),
            "x_pulses_ids": tf.zeros([7, 4], dtype=tf.int32),
        }
        result_tensors = self.source.get_tensors(data_batch_dict, True)
        self.assertTrue(result_tensors["dom_charges"].shape == [1, 86, 60, 1])
        self.assertTrue(result_tensors["pulse_pdf"].shape == [7])

    # def test_chaining_of_multi_source_objects(self):
    #     """MultiSource objects should behave like normal Source objects.
    #     Therefore, it should be possible to create a MultiSource object
    #     of multiple MultSource objects
    #     """
    #     multi_source1 = self.get_muon(config=self.config,
    #                                   base_models=self.base_models,
    #                                   name='Muon1')
    #     multi_source2 = self.get_muon(config=self.config,
    #                                   base_models=self.base_models,
    #                                   name='Muon2')

    #     model = MultiSource()
    #     model.configure(**kwargs)


if __name__ == "__main__":
    unittest.main()

import unittest
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.model.decoder.base import LatentToPDFDecoder


class DummyDecoderModel(LatentToPDFDecoder):

    def _build_architecture(self, config, name=None):
        self.assert_configured(False)

        parameter_names = [
            "mu",
            "sigma",
            "r",
            "scale",
        ]

        self._untracked_data["dummy_var"] = tf.Variable(1.0, name="dummy_var")

        return parameter_names

    def pdf(self, x, latent_vars):
        self.assert_configured(True)
        return x


class TestDecoderBase(unittest.TestCase):
    """Test base class for Decoder models"""

    def setUp(self):

        self.parameter_names = [
            "mu",
            "sigma",
            "r",
            "scale",
        ]

        config = {"dummy_setting": 1337}
        self.decoder = self.get_dummy_decoder(
            config=config,
        )

        class_string = misc.get_full_class_string_of_object(self.decoder)
        self.configuration = Configuration(
            class_string=class_string,
            settings=dict(config=config),
            mutable_settings=dict(name="egenerator.model.decoder.base"),
        )

    def get_dummy_decoder(self, **kwargs):
        model = DummyDecoderModel()
        model.configure(**kwargs)
        return model

    def test_correct_initialization(self):
        """Check if member variables are correctly set when instantiating
        a Model object.
        """
        model = DummyDecoderModel()
        self.assertEqual(model._is_configured, False)
        self.assertEqual(model.data, None)
        self.assertEqual(model.configuration, None)
        self.assertEqual(model._untracked_data, {})
        self.assertEqual(model._sub_components, {})
        self.assertEqual(model.checkpoint, None)
        self.assertEqual(model.name, None)
        self.assertEqual(model.parameter_names, None)

    def test_configuration_of_abstract_class(self):
        """Model is an abstract class with a pure virtual method.
        Attempting to configure it should result in a NotImplementedError
        """
        model = LatentToPDFDecoder()

        with self.assertRaises(NotImplementedError):
            model.configure(config={})

    def test_correct_configuration_name(self):
        """Test if the name passed in to the configuration is properly saved"""
        model = DummyDecoderModel()
        model.configure(config={}, name="dummy")
        self.assertEqual(model.name, "dummy")

    def test_correct_configuration(self):
        """Check if member variables are set correctly after the dummy model
        is configured
        """
        self.assertEqual(self.decoder.is_configured, True)
        self.assertTrue(
            isinstance(self.decoder.checkpoint, tf.train.Checkpoint)
        )
        self.assertEqual(self.decoder.data, {})
        self.assertEqual(self.decoder.name, "egenerator.model.decoder.base")
        self.assertEqual(self.decoder.parameter_names, self.parameter_names)
        self.assertTrue(
            self.decoder.configuration.is_compatible(self.configuration)
        )

        self.assertDictEqual(
            self.decoder._untracked_data,
            {
                "name": "egenerator.model.decoder.base",
                "checkpoint": self.decoder.checkpoint,
                "step": self.decoder._untracked_data["step"],
                "dummy_var": self.decoder._untracked_data["dummy_var"],
                "variables": self.decoder._untracked_data["variables"],
                "num_parameters": 4,
                "parameter_index_dict": self.decoder._untracked_data[
                    "parameter_index_dict"
                ],
                "parameter_name_dict": self.decoder._untracked_data[
                    "parameter_name_dict"
                ],
                "parameter_names": [
                    "mu",
                    "sigma",
                    "r",
                    "scale",
                ],
            },
        )
        self.assertEqual(self.decoder._sub_components, {})

    def test_method_assert_configured(self):
        """Check the method assert_configured()"""
        model = LatentToPDFDecoder()
        model.assert_configured(False)
        with self.assertRaises(ValueError) as context:
            model.assert_configured(True)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

        self.decoder.assert_configured(True)
        with self.assertRaises(ValueError) as context:
            self.decoder.assert_configured(False)
        self.assertTrue("Model is already set up!" in str(context.exception))

        with self.assertRaises(TypeError) as context:
            model.assert_configured(None)
        self.assertTrue("Expected bool, but got" in str(context.exception))

    def test_method_get_index(self):
        for i, name in enumerate(self.parameter_names):
            self.assertEqual(self.decoder.get_index(name), i)
            self.assertEqual(self.decoder.get_name(i), name)

    def test_method_pdf_not_implemented_error(self):
        decoder = LatentToPDFDecoder()
        with self.assertRaises(ValueError) as context:
            decoder.pdf(None, None)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

        with self.assertRaises(NotImplementedError) as context:
            decoder._is_configured = True
            decoder.pdf(None, None)

    def test_parameter_indexing(self):
        tensor = tf.constant(self.parameter_names)
        self.decoder.add_parameter_indexing(tensor)
        self.assertEqual(tensor.params.sigma, "sigma")


if __name__ == "__main__":
    unittest.main()

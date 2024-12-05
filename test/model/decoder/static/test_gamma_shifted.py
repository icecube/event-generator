import unittest
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.utils import basis_functions
from egenerator.manager.component import Configuration
from egenerator.model.decoder import ShiftedGammaFunctionDecoder


class TestShiftedGammaFunctionDecoder(unittest.TestCase):
    """Test ShiftedGammaFunctionDecoder class"""

    def setUp(self):

        alpha = np.array([1.0, 2.0, 3.0])
        beta = np.array([0.3, 3.0, 2.0])
        self.offset = np.array([-0.2, 0.0, 2.0])
        self.latent_vars = np.stack([alpha, beta, self.offset], axis=-1)

        self.parameter_names = [
            "alpha",
            "beta",
            "offset",
        ]

        config = {"float_precision": "float64"}
        self.decoder = self.get_decoder(config=config)

        class_string = misc.get_full_class_string_of_object(self.decoder)
        self.configuration = Configuration(
            class_string=class_string,
            settings=dict(config=config),
            mutable_settings=dict(name="egenerator.model.decoder.base"),
        )

    def get_decoder(self, **kwargs):
        model = ShiftedGammaFunctionDecoder()
        model.configure(**kwargs)
        return model

    def test_correct_pdf(self):
        """Check if the pdf method is correctly implemented"""

        x = np.array([-1.0, -0.36, 0.0, 0.56, 1.0]).reshape(-1, 1)
        pdf_np = basis_functions.gamma_pdf(
            x=x - self.latent_vars[..., 2],
            alpha=self.latent_vars[..., 0],
            beta=self.latent_vars[..., 1],
        )
        pdf = self.decoder.pdf(x, self.latent_vars).numpy()
        self.assertTrue(np.allclose(pdf, pdf_np))

    def test_correct_cdf(self):
        """Check if the cdf method is correctly implemented"""

        x = np.array([0.0, 0.56, 1.0]).reshape(-1, 1)
        cdf_np = basis_functions.gamma_cdf(
            x=x - self.latent_vars[..., 2],
            alpha=self.latent_vars[..., 0],
            beta=self.latent_vars[..., 1],
        )
        cdf = self.decoder.cdf(x, self.latent_vars).numpy()
        self.assertTrue(np.allclose(cdf, cdf_np))

    def test_correct_ppf(self):
        """Check if the ppf method is correctly implemented"""

        q = np.array([0.0, 0.56, 1.0]).reshape(-1, 1)
        cdf_np = basis_functions.gamma_ppf(
            q=q,
            alpha=self.latent_vars[..., 0],
            beta=self.latent_vars[..., 1],
        )
        cdf_np += self.latent_vars[..., 2]
        cdf = self.decoder.ppf(q, self.latent_vars).numpy()
        self.assertTrue(
            np.allclose(
                q,
                basis_functions.gamma_cdf(
                    x=cdf_np - self.offset,
                    alpha=self.latent_vars[..., 0],
                    beta=self.latent_vars[..., 1],
                ),
            )
        )
        self.assertTrue(
            np.allclose(q, self.decoder.cdf(cdf, self.latent_vars))
        )
        self.assertTrue(np.allclose(cdf, cdf_np))

    def test_correct_initialization(self):
        """Check if member variables are correctly set when instantiating
        a Model object.
        """
        model = ShiftedGammaFunctionDecoder()
        self.assertEqual(model._is_configured, False)
        self.assertEqual(model.data, None)
        self.assertEqual(model.configuration, None)
        self.assertEqual(model._untracked_data, {})
        self.assertEqual(model._sub_components, {})
        self.assertEqual(model.checkpoint, None)
        self.assertEqual(model.name, None)
        self.assertEqual(model.parameter_names, None)

    def test_correct_configuration_name(self):
        """Test if the name passed in to the configuration is properly saved"""
        model = ShiftedGammaFunctionDecoder()
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
        self.maxDiff = None

        self.assertDictEqual(
            self.decoder._untracked_data,
            {
                "name": "egenerator.model.decoder.base",
                "checkpoint": self.decoder.checkpoint,
                "variables": self.decoder.variables,
                "variables_top_level": self.decoder._untracked_data[
                    "variables_top_level"
                ],
                "step": self.decoder._untracked_data["step"],
                "n_parameters": 3,
                "parameter_index_dict": self.decoder._untracked_data[
                    "parameter_index_dict"
                ],
                "parameter_name_dict": self.decoder._untracked_data[
                    "parameter_name_dict"
                ],
                "parameter_names": [
                    "alpha",
                    "beta",
                    "offset",
                ],
                "value_range_mapping": self.decoder._untracked_data[
                    "value_range_mapping"
                ],
            },
        )
        self.assertEqual(self.decoder._sub_components, {})

    def test_method_assert_configured(self):
        """Check the method assert_configured()"""
        model = ShiftedGammaFunctionDecoder()
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

    def test_parameter_indexing(self):
        tensor = tf.constant(self.parameter_names)
        self.decoder.add_parameter_indexing(tensor)
        self.assertEqual(tensor.params.beta, "beta")


if __name__ == "__main__":
    unittest.main()
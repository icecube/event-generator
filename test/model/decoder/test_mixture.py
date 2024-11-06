import unittest
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.model.decoder.mixture import MixtureModel
from egenerator.model.decoder import AsymmetricGaussianDecoder
from egenerator.utils import basis_functions


class TestMixtureModel(unittest.TestCase):
    """Test base class for Mixture models"""

    def setUp(self):

        self.parameter_names = [
            "AssymetricGaussian_mu_000",
            "AssymetricGaussian_sigma_000",
            "AssymetricGaussian_r_000",
            "AssymetricGaussian_weight_000",
            "AssymetricGaussian_mu_001",
            "AssymetricGaussian_sigma_001",
            "AssymetricGaussian_r_001",
            "AssymetricGaussian_weight_001",
            "AssymetricGaussian_mu_002",
            "AssymetricGaussian_sigma_002",
            "AssymetricGaussian_r_002",
            "AssymetricGaussian_weight_002",
        ]
        self.config = {
            "decoder_mapping": {
                "AssymetricGaussian": ["AssymetricGaussian", 3],
            },
            "float_precision": "float64",
        }
        self.config_single = {
            "decoder_mapping": {
                "AssymetricGaussian": ["AssymetricGaussian", 1],
            },
            "float_precision": "float64",
        }
        self.ag = self.get_ag_decoder(config={"float_precision": "float64"})

        self.base_models = {
            "AssymetricGaussian": self.ag,
        }

        self.mixture = self.get_mixture(
            config=self.config,
            base_models=self.base_models,
        )
        self.mixture_single = self.get_mixture(
            config=self.config_single,
            base_models=self.base_models,
        )

        class_string = misc.get_full_class_string_of_object(self.mixture)
        self.configuration = Configuration(
            class_string=class_string,
            settings=dict(config=self.config),
            mutable_settings=dict(name="egenerator.model.decoder.mixture"),
        )
        self.configuration.add_sub_components(self.mixture.sub_components)

        # ----------------
        # Define Test Data
        # ----------------
        # shape: (5, 2)
        self.x = np.array(
            [
                [-1.0, -0.36, 0.0, 0.56, 1.0],
                [-1.0, -0.36, 0.0, 0.56, 1.0],
            ]
        ).reshape((5, 2))

        # shape: (5, 2, 1)
        self.x_np = self.x[..., np.newaxis]

        # shape: (1, 2, n_latent = n_components * num_parameters)
        self.latent_vars = np.array(
            [
                [
                    [
                        # mu, sigm, r, weight
                        1.0,
                        0.3,
                        1.0,
                        1.0,
                        0.0,
                        3.0,
                        2.0,
                        2.0,
                        3.0,
                        2.0,
                        0.3,
                        3.0,
                    ],
                    [
                        # mu, sigm, r, weight
                        4.2,
                        1.3,
                        2.0,
                        1.3,
                        0.1,
                        2.3,
                        0.4,
                        0.3,
                        -3.0,
                        2.0,
                        0.3,
                        1.0,
                    ],
                ]
            ]
        )
        self.latent_vars_mu = np.array(
            [
                [
                    [
                        # mu, sigm, r, weight
                        0.0,
                        0.3,
                        1.0,
                        1.0,
                        0.0,
                        3.0,
                        1.0,
                        2.0,
                        0.0,
                        2.0,
                        1.0,
                        3.0,
                    ],
                    [
                        # mu, sigm, r, weight
                        3.0,
                        1.3,
                        1.0,
                        1.3,
                        3.0,
                        2.3,
                        1.0,
                        0.3,
                        3.0,
                        2.0,
                        1.0,
                        1.0,
                    ],
                ]
            ]
        )

        # shape: (1, 2, n_components, num_parameters)
        self.latent_vars_np = self.latent_vars.reshape((1, 2, 3, 4))

        # normalize weights for np, mixture model does this internally
        self.latent_vars_np[..., -1] /= np.sum(
            self.latent_vars_np[..., -1] + self.mixture.epsilon,
            axis=-1,
            keepdims=True,
        )
        # ----------------

    def get_ag_decoder(self, **kwargs):
        model = AsymmetricGaussianDecoder()
        model.configure(**kwargs)
        return model

    def get_mixture(self, base_models, **kwargs):
        model = MixtureModel()
        model.configure(base_models=base_models, **kwargs)
        return model

    def test_correct_pdf_reduced(self):
        """Check if the pdf method is correctly implemented"""

        pdf_np = np.sum(
            basis_functions.asymmetric_gauss(
                x=self.x_np,
                mu=self.latent_vars_np[..., 0],
                sigma=self.latent_vars_np[..., 1],
                r=self.latent_vars_np[..., 2],
            )
            * self.latent_vars_np[..., 3],
            axis=-1,
        )
        pdf = self.mixture.pdf(self.x, self.latent_vars).numpy()

        self.assertTrue(np.allclose(pdf, pdf_np))

    def test_correct_pdf(self):
        """Check if the pdf method is correctly implemented"""

        pdf_np = (
            basis_functions.asymmetric_gauss(
                x=self.x_np,
                mu=self.latent_vars_np[..., 0],
                sigma=self.latent_vars_np[..., 1],
                r=self.latent_vars_np[..., 2],
            )
            * self.latent_vars_np[..., 3]
        )
        pdf = self.mixture.pdf(
            self.x, self.latent_vars, reduce_components=False
        ).numpy()

        self.assertTrue(np.allclose(pdf, pdf_np))

    def test_correct_cdf_reduced(self):
        """Check if the cdf method is correctly implemented"""

        cdf_np = np.sum(
            basis_functions.asymmetric_gauss_cdf(
                x=self.x_np,
                mu=self.latent_vars_np[..., 0],
                sigma=self.latent_vars_np[..., 1],
                r=self.latent_vars_np[..., 2],
            )
            * self.latent_vars_np[..., 3],
            axis=-1,
        )
        cdf = self.mixture.cdf(self.x, self.latent_vars).numpy()

        self.assertTrue(np.allclose(cdf, cdf_np))

    def test_correct_cdf(self):
        """Check if the cdf method is correctly implemented"""

        cdf_np = (
            basis_functions.asymmetric_gauss_cdf(
                x=self.x_np,
                mu=self.latent_vars_np[..., 0],
                sigma=self.latent_vars_np[..., 1],
                r=self.latent_vars_np[..., 2],
            )
            * self.latent_vars_np[..., 3]
        )
        cdf = self.mixture.cdf(
            self.x, self.latent_vars, reduce_components=False
        ).numpy()
        self.assertTrue(np.allclose(cdf, cdf_np))

    def test_sample(self):
        """Check if sampling runs without errors and if shape is ok

        Note: this does not check if the samples are correct!
        """

        random_numbers = np.arange(20).reshape(5, 2, 2) / 20.0
        samples = self.mixture.sample(random_numbers, self.latent_vars).numpy()

        self.assertEqual(samples.shape, (5, 2))

    def test_sample_mean(self):
        """Check if sampling runs without errors

        Note: this does not check if the samples are correct!
        """

        random_numbers = np.arange(20).reshape(5, 2, 2) / 20.0
        random_numbers[..., 1] = 0.5
        samples = self.mixture.sample(
            random_numbers, self.latent_vars_mu
        ).numpy()
        self.assertTrue(np.allclose(samples[..., 0], 0.0))
        self.assertTrue(np.allclose(samples[..., 1], 3.0))

    def test_correct_initialization(self):
        """Check if member variables are correctly set when instantiating
        a Model object.
        """
        model = MixtureModel()
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
        model = MixtureModel()
        model.configure(
            config=self.config, base_models=self.base_models, name="dummy"
        )
        self.assertEqual(model.name, "dummy")

    def test_correct_configuration(self):
        """Check if member variables are set correctly after the dummy model
        is configured
        """
        self.assertEqual(self.mixture.is_configured, True)
        self.assertTrue(
            isinstance(self.mixture.checkpoint, tf.train.Checkpoint)
        )
        self.assertEqual(self.mixture.data, {})
        self.assertEqual(self.mixture.name, "egenerator.model.decoder.mixture")
        self.assertEqual(self.mixture.parameter_names, self.parameter_names)
        self.assertTrue(
            self.mixture.configuration.is_compatible(self.configuration)
        )
        self.maxDiff = None

        self.assertDictEqual(
            self.mixture._untracked_data,
            {
                "name": "egenerator.model.decoder.mixture",
                "checkpoint": self.mixture.checkpoint,
                "variables": self.mixture.variables,
                "step": self.mixture._untracked_data["step"],
                "num_parameters": 12,
                "parameter_index_dict": self.mixture._untracked_data[
                    "parameter_index_dict"
                ],
                "parameter_name_dict": self.mixture._untracked_data[
                    "parameter_name_dict"
                ],
                "parameter_names": self.parameter_names,
                "models_mapping": self.mixture._untracked_data[
                    "models_mapping"
                ],
                "n_parameters_per_decoder": [4],
                "n_components_per_decoder": [3],
                "parameter_slice_per_decoder": [slice(0, 12, None)],
                "decoder_names": ["AssymetricGaussian"],
            },
        )
        self.assertEqual(self.mixture._sub_components, self.base_models)

    def test_method_assert_configured(self):
        """Check the method assert_configured()"""
        model = AsymmetricGaussianDecoder()
        model.assert_configured(False)
        with self.assertRaises(ValueError) as context:
            model.assert_configured(True)
        self.assertTrue(
            "Model needs to be set up first!" in str(context.exception)
        )

        self.mixture.assert_configured(True)
        with self.assertRaises(ValueError) as context:
            self.mixture.assert_configured(False)
        self.assertTrue("Model is already set up!" in str(context.exception))

        with self.assertRaises(TypeError) as context:
            model.assert_configured(None)
        self.assertTrue("Expected bool, but got" in str(context.exception))

    def test_method_get_index(self):
        for i, name in enumerate(self.parameter_names):
            self.assertEqual(self.mixture.get_index(name), i)
            self.assertEqual(self.mixture.get_name(i), name)

    def test_parameter_indexing(self):
        tensor = tf.constant(self.parameter_names)
        self.mixture.add_parameter_indexing(tensor)
        self.assertEqual(
            tensor.params.AssymetricGaussian_mu_001,
            "AssymetricGaussian_mu_001",
        )

    def test_fail_on_mismatched_float_precision(self):
        """Check if the model fails to configure when the float precision
        of the mixture does not match the float precision of the base models.
        """
        model = AsymmetricGaussianDecoder()
        model.configure(config={"float_precision": "float64"})

        # this should work as float_precision is not defined
        self.get_mixture(
            config=self.config,
            base_models={"AssymetricGaussian": model},
        )

        # this should fail due to mismatched float precision
        config = {
            "decoder_mapping": {
                "AssymetricGaussian": ["AssymetricGaussian", 7],
            },
            "float_precision": "float32",
        }
        with self.assertRaises(ValueError) as context:
            self.get_mixture(
                config=config,
                base_models={"AssymetricGaussian": model},
            )
        self.assertTrue("Mismatched float precision" in str(context.exception))


if __name__ == "__main__":
    unittest.main()

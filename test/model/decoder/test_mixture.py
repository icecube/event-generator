import unittest
import numpy as np
import tensorflow as tf

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.model.decoder.mixture import MixtureModel
from egenerator.model.decoder import AsymmetricGaussianDecoder
from egenerator.model.decoder import GammaFunctionDecoder
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
        self.config_big = {
            "decoder_mapping": {
                "AssymetricGaussian": ["AssymetricGaussian", 10],
                "GammaFunction": ["GammaFunction", 10],
            },
            "float_precision": "float64",
            "value_range_mapping": {
                "weight": {
                    "value_range_class": "egenerator.utils.value_range.EluValueRange",
                    "config": {
                        "scale": 10.0,
                        "offset": 1.0,
                        "min_value": 0.00001,
                    },
                },
            },
        }
        self.config = {
            "decoder_mapping": {
                "AssymetricGaussian": ["AssymetricGaussian", 3],
            },
            "float_precision": "float64",
            "value_range_mapping": {
                "weight": {
                    "value_range_class": "egenerator.utils.value_range.EluValueRange",
                    "config": {
                        "scale": 10.0,
                        "offset": 1.0,
                        "min_value": 0.00001,
                    },
                },
            },
        }
        self.ag = self.get_ag_decoder(
            config={
                "float_precision": "float64",
                "value_range_mapping": {
                    "mu": {
                        "value_range_class": "egenerator.utils.value_range.BaseValueRange",
                        "config": {
                            "scale": 0.5,
                            "offset": 0.0,
                        },
                    },
                    "sigma": {
                        "value_range_class": "egenerator.utils.value_range.EluValueRange",
                        "config": {
                            "scale": 2.0,
                            "offset": 2.0,
                            "min_value": 0.0001,
                        },
                    },
                    "r": {
                        "value_range_class": "egenerator.utils.value_range.EluValueRange",
                        "config": {
                            "scale": 1.0,
                            "offset": 1.0,
                            "min_value": 0.0001,
                        },
                    },
                },
            }
        )
        self.ag_no_mapping = self.get_ag_decoder(
            config={
                "float_precision": "float64",
            }
        )

        self.gamma_decoder = self.get_gamma_decoder(
            config={
                "float_precision": "float64",
                "value_range_mapping": {
                    "alpha": {
                        "value_range_class": "egenerator.utils.value_range.EluValueRange",
                        "config": {
                            "scale": 0.3,
                            "offset": 2.5,
                            "min_value": 0.5,
                        },
                    },
                    "beta": {
                        "value_range_class": "egenerator.utils.value_range.EluValueRange",
                        "config": {
                            "scale": 1.0,
                            "offset": 10.0,
                            "min_value": 1.0,
                        },
                    },
                },
            },
        )

        self.base_models = {
            "AssymetricGaussian": self.ag,
        }
        self.base_models_big = {
            "AssymetricGaussian": self.ag,
            "GammaFunction": self.gamma_decoder,
        }

        self.mixture = self.get_mixture(
            config=self.config,
            base_models=self.base_models,
        )
        self.mixture_big = self.get_mixture(
            config=self.config_big,
            base_models=self.base_models_big,
        )
        self.mixture_no_mapping = self.get_mixture(
            config=self.config,
            base_models={"AssymetricGaussian": self.ag_no_mapping},
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

        # shape: (1, 2, n_latent = n_components * n_parameters)
        # mu, sigm, r, weight
        self.latent_vars = np.array(
            [
                [
                    [
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

        # shape: (1, 2, n_components, n_parameters)
        self.latent_vars_np = self.latent_vars.reshape((1, 2, 3, 4))

        def elu(x):
            return np.where(x > 0, x, np.exp(x) - 1)

        self.latent_vars_np_trafo = np.array(self.latent_vars_np)
        self.latent_vars_np_trafo[..., 0] = self.latent_vars_np[..., 0] * 0.5
        self.latent_vars_np_trafo[..., 1] = (
            elu(self.latent_vars_np[..., 1] * 2.0 + 2.0) + 1.0001
        )
        self.latent_vars_np_trafo[..., 2] = (
            elu(self.latent_vars_np[..., 2] * 1.0 + 1.0) + 1.0001
        )
        self.latent_vars_np_trafo[..., 3] = (
            elu(self.latent_vars_np[..., 3] * 10.0 + 1.0) + 1.00001
        )

        # normalize weights for np, mixture model does this internally
        self.latent_vars_np = np.array(self.latent_vars_np)
        self.latent_vars_np[..., -1] /= np.sum(
            self.latent_vars_np[..., -1] + self.mixture.epsilon,
            axis=-1,
            keepdims=True,
        )
        self.latent_vars_np_trafo_norm = np.array(self.latent_vars_np_trafo)
        self.latent_vars_np_trafo_norm[..., -1] /= np.sum(
            self.latent_vars_np_trafo_norm[..., -1] + self.mixture.epsilon,
            axis=-1,
            keepdims=True,
        )
        # ----------------

    def get_gamma_decoder(self, **kwargs):
        model = GammaFunctionDecoder()
        model.configure(**kwargs)
        return model

    def get_ag_decoder(self, **kwargs):
        model = AsymmetricGaussianDecoder()
        model.configure(**kwargs)
        return model

    def get_mixture(self, base_models, **kwargs):
        model = MixtureModel()
        model.configure(base_models=base_models, **kwargs)
        return model

    def test_value_range_mapping_keys(self):
        """Test if the value range mapping contains the correct keys"""

        self.assertEqual(
            list(self.ag.value_range_mapping.keys()),
            ["mu", "sigma", "r"],
        )

        self.assertEqual(
            list(self.ag_no_mapping.value_range_mapping.keys()),
            [],
        )

        self.assertEqual(
            list(self.mixture.value_range_mapping.keys()),
            [
                "AssymetricGaussian_weight_000",
                "AssymetricGaussian_weight_001",
                "AssymetricGaussian_weight_002",
            ],
        )

    def test_correct_slices(self):
        """Test if the slices are correctly set"""

        self.assertEqual(
            self.mixture._untracked_data["parameter_slice_per_decoder"],
            [slice(0, 12, None)],
        )
        self.assertEqual(
            self.mixture_big._untracked_data["parameter_slice_per_decoder"],
            [slice(0, 40, None), slice(40, 70, None)],
        )

    def test_value_range_of_cdf(self):
        """Check if cdf values are within [0, 1]"""

        rng = np.random.default_rng(42)

        n_points = 100000
        x = rng.uniform(-10, 10, n_points)
        latent_vars = rng.uniform(
            -10, 10, (n_points, self.mixture_big.n_parameters)
        )

        cdf = self.mixture_big.cdf(x, latent_vars).numpy()
        self.assertTrue(np.all(cdf >= 0.0))
        self.assertTrue(np.all(cdf <= 1.0))

        cdf = self.mixture_big.cdf(
            x, latent_vars, reduce_components=False
        ).numpy()
        self.assertTrue(np.all(cdf >= 0.0))
        self.assertTrue(np.all(cdf <= 1.0))

    def test_correct_value_range_mapping_of_mixture(self):
        """Test if the value range mapping is correctly applied for mixture"""

        latent_vars_mapped = self.mixture._apply_value_range(
            self.latent_vars
        ).numpy()
        latent_vars_mapped = latent_vars_mapped.reshape(
            self.latent_vars_np.shape
        )

        latent_weights = latent_vars_mapped[..., -1]
        latent_weights_expected = self.latent_vars_np_trafo[..., -1]
        self.assertTrue(np.allclose(latent_weights, latent_weights_expected))

    def test_correct_value_range_mapping(self):
        """Test if the value range mapping is correctly applied"""

        # take everything but every 4th element
        latent_vars = self.latent_vars[..., :3]
        latent_vars_mapped = self.ag._apply_value_range(latent_vars).numpy()

        latent_weights_expected = self.latent_vars_np_trafo[..., 0, :3]
        self.assertTrue(
            np.allclose(latent_vars_mapped, latent_weights_expected)
        )

    def test_correct_value_range_mapping_none_applied(self):
        """Test if the value range mapping is correctly applied"""

        # take everything but every 4th element
        latent_vars = self.latent_vars[..., :3]
        latent_vars_mapped = self.ag_no_mapping._apply_value_range(
            latent_vars
        ).numpy()

        latent_weights_expected = self.latent_vars_np[..., 0, :3]
        self.assertTrue(
            np.allclose(latent_vars_mapped, latent_weights_expected)
        )

    def test_correct_pdf_reduced(self):
        """Check if the pdf method is correctly implemented"""

        pdf_np_ = np.sum(
            basis_functions.asymmetric_gauss(
                x=self.x_np,
                mu=self.latent_vars_np_trafo_norm[..., 0],
                sigma=self.latent_vars_np_trafo_norm[..., 1],
                r=self.latent_vars_np_trafo_norm[..., 2],
            )
            * self.latent_vars_np[..., 3],
            axis=-1,
        )
        pdf_np = np.sum(
            basis_functions.asymmetric_gauss(
                x=self.x_np,
                mu=self.latent_vars_np_trafo_norm[..., 0],
                sigma=self.latent_vars_np_trafo_norm[..., 1],
                r=self.latent_vars_np_trafo_norm[..., 2],
            )
            * self.latent_vars_np_trafo_norm[..., 3],
            axis=-1,
        )
        pdf = self.mixture.pdf(self.x, self.latent_vars).numpy()
        pdf_ = self.mixture._pdf(self.x, self.latent_vars).numpy()

        self.assertTrue(np.allclose(pdf, pdf_np))
        self.assertTrue(np.allclose(pdf_, pdf_np_))

    def test_correct_pdf(self):
        """Check if the pdf method is correctly implemented"""

        pdf_np_ = (
            basis_functions.asymmetric_gauss(
                x=self.x_np,
                mu=self.latent_vars_np_trafo_norm[..., 0],
                sigma=self.latent_vars_np_trafo_norm[..., 1],
                r=self.latent_vars_np_trafo_norm[..., 2],
            )
            * self.latent_vars_np[..., 3]
        )
        pdf_np = (
            basis_functions.asymmetric_gauss(
                x=self.x_np,
                mu=self.latent_vars_np_trafo_norm[..., 0],
                sigma=self.latent_vars_np_trafo_norm[..., 1],
                r=self.latent_vars_np_trafo_norm[..., 2],
            )
            * self.latent_vars_np_trafo_norm[..., 3]
        )
        pdf_ = self.mixture._pdf(
            self.x, self.latent_vars, reduce_components=False
        ).numpy()
        pdf = self.mixture.pdf(
            self.x, self.latent_vars, reduce_components=False
        ).numpy()

        self.assertTrue(np.allclose(pdf, pdf_np))
        self.assertTrue(np.allclose(pdf_, pdf_np_))

    def test_correct_cdf_reduced(self):
        """Check if the cdf method is correctly implemented"""

        cdf_np = np.sum(
            basis_functions.asymmetric_gauss_cdf(
                x=self.x_np,
                mu=self.latent_vars_np_trafo_norm[..., 0],
                sigma=self.latent_vars_np_trafo_norm[..., 1],
                r=self.latent_vars_np_trafo_norm[..., 2],
            )
            * self.latent_vars_np_trafo_norm[..., 3],
            axis=-1,
        )
        cdf = self.mixture.cdf(self.x, self.latent_vars).numpy()

        self.assertTrue(np.allclose(cdf, cdf_np))

    def test_correct_cdf(self):
        """Check if the cdf method is correctly implemented"""

        cdf_np = (
            basis_functions.asymmetric_gauss_cdf(
                x=self.x_np,
                mu=self.latent_vars_np_trafo_norm[..., 0],
                sigma=self.latent_vars_np_trafo_norm[..., 1],
                r=self.latent_vars_np_trafo_norm[..., 2],
            )
            * self.latent_vars_np_trafo_norm[..., 3]
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
        samples = self.mixture_no_mapping.sample(
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
                "variables_top_level": self.mixture._untracked_data[
                    "variables_top_level"
                ],
                "step": self.mixture._untracked_data["step"],
                "n_parameters": 12,
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
                "value_range_mapping": self.mixture._untracked_data[
                    "value_range_mapping"
                ],
                "n_components_total": 3,
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

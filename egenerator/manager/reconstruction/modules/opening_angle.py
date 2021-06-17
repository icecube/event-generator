import timeit
import numpy as np
import tensorflow as tf
from scipy.stats import chi2
from scipy import optimize

from egenerator.utils import angles, basis_functions


class CircularizedAngularUncertainty:

    def __init__(self, manager, loss_module, function_cache,
                 fit_parameter_list,
                 reco_key,
                 covariance_key=None,
                 minimize_in_trafo_space=True,
                 parameter_tensor_name='x_parameters',
                 scipy_optimizer_settings={'method': 'L-BFGS-B'},
                 num_fit_points=10,
                 random_seed=42):
        """Initialize module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        fit_parameter_list : bool or list of bool, optional
            Indicates whether a parameter is to be minimized.
            The ith element in the list specifies if the ith parameter
            is minimized.
        reco_key : str
            The name of the reconstruction module to use. The covariance
            matrix will be calculated at the best fit point of the specified
            reconstruction module.
        covariance_key : str, optional
            The name of the covariance matrix module to use. The covariance
            is used to estimate the range of delta psi from which to sample.
            If None is provided, the range will be estimated by performing
            a bisection search to find the appropriate range for delta psi.
        minimize_in_trafo_space : bool, optional
            If True, covariance is calculated in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            calculation and inversion.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
        scipy_optimizer_settings : dict, optional
            Settings that will be passed on to the scipy.optmize.minimize
            function.
        num_fit_points : int, optional
            This defines the number of delta psi (opening angles) to randomly
            sample. These points are used as a seed to the reconstruction
            and minimized. The delta log likelihood is then computed compared
            to the best fit and a Rayleigh CDF is fitted to this distribution.
            The more point, the more accurate the result, but also the longer
            it takes.
        random_seed : int, optional
            A random seed for the numpy Random State which is used to sample
            the random opening angles (delta psi).
        """

        # store settings
        self.manager = manager
        self.fit_parameter_list = fit_parameter_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.num_fit_points = num_fit_points
        self.reco_key = reco_key
        self.covariance_key = covariance_key

        # specify a random number generator for reproducibility
        self.rng = np.random.RandomState(random_seed)

        # fit parameter list
        self.mapping = self.manager.configuration.config['config'][
            'I3ParticleMapping']
        self.zenith_index = self.manager.models[0].get_index(
            self.mapping['zenith'])
        self.azimuth_index = self.manager.models[0].get_index(
            self.mapping['azimuth'])
        self.unc_fit_parameter_list = list(fit_parameter_list)
        self.unc_fit_parameter_list[self.zenith_index] = False
        self.unc_fit_parameter_list[self.azimuth_index] = False

        # parameter input signature
        self.param_dtype = getattr(tf, manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype)

        unc_param_signature = tf.TensorSpec(
            shape=[None, np.sum(self.unc_fit_parameter_list, dtype=int)],
            dtype=self.param_dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_parameter_list, dtype=int)],
            dtype=self.param_dtype)
        param_signature_full = tf.TensorSpec(
            shape=[None, len(fit_parameter_list)],
            dtype=self.param_dtype)

        # data batch input signature
        data_batch_signature = manager.data_handler.get_data_set_signature()

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------

        # get normal parameter loss function
        func_settings = dict(
            input_signature=(
                param_signature, data_batch_signature, param_signature_full),
            loss_module=loss_module,
            fit_parameter_list=fit_parameter_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        # Get parameter loss function
        self.loss_function = function_cache.get(
            'parameter_loss_function', func_settings)

        if self.loss_function is None:
            self.loss_function = manager.get_parameter_loss_function(
                **func_settings)
            function_cache.add(self.loss_function, func_settings)

        # get specific functions for uncertainty estimation
        function_settings = dict(
            input_signature=(
                unc_param_signature,
                data_batch_signature,
                param_signature_full
            ),
            loss_module=loss_module,
            fit_parameter_list=self.unc_fit_parameter_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        # get loss and gradients function
        loss_and_gradients_function = function_cache.get(
            'loss_and_gradients_function', function_settings)

        if loss_and_gradients_function is None:
            loss_and_gradients_function = \
                manager.get_loss_and_gradients_function(**function_settings)
            function_cache.add(loss_and_gradients_function, function_settings)

        # get parameter loss function
        self.unc_loss_function = function_cache.get(
            'parameter_loss_function', function_settings)

        if self.unc_loss_function is None:
            self.unc_loss_function = \
                manager.get_parameter_loss_function(**function_settings)
            function_cache.add(self.unc_loss_function, function_settings)

        def unc_reconstruction_method(data_batch, seed_tensor):
            return manager.reconstruct_events(
                data_batch, loss_module,
                loss_and_gradients_function=loss_and_gradients_function,
                fit_parameter_list=self.unc_fit_parameter_list,
                minimize_in_trafo_space=minimize_in_trafo_space,
                seed=seed_tensor,
                parameter_tensor_name=parameter_tensor_name,
                **scipy_optimizer_settings)

        self.unc_reconstruction_method = unc_reconstruction_method

    def execute(self, data_batch, results, **kwargs):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of array_like
            A batch of data consisting of a tuple of data arrays.
        results : dict
            A dictrionary with the results of previous modules.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        TYPE
            Description
        """

        # The following assumes that result is the full hypothesis
        assert np.all(self.fit_parameter_list)

        result_trafo = results[self.reco_key]['result_trafo']
        result_inv = results[self.reco_key]['result']

        # calculate delta degrees of freedom
        ddof = len(result_inv[0]) - 2

        # define reconstruction method
        def reconstruct_at_angle(zeniths, azimuths):
            data_batch_combined = [t for t in data_batch]
            seed_tensor = np.array(result_inv)
            unc_results_list = []
            unc_loss_list = []
            for zen, azi in zip(zeniths, azimuths):

                # put together seed tensor and new data batch
                seed_tensor[:, self.zenith_index] = zen
                seed_tensor[:, self.azimuth_index] = azi
                tf_seed_tensor = tf.convert_to_tensor(
                    seed_tensor, self.param_dtype)

                # reconstruct (while keeping azimuth and zenith fixed)
                unc_result, result_obj = self.unc_reconstruction_method(
                    tuple(data_batch_combined), seed_tensor)

                # get loss
                unc_loss = self.unc_loss_function(
                    parameters_trafo=unc_result,
                    data_batch=tuple(data_batch_combined),
                    seed=tf_seed_tensor).numpy()

                # append data
                unc_results_list.append(unc_result)
                unc_loss_list.append(unc_loss)

            unc_results = np.concatenate(unc_results_list, axis=0)
            unc_losses = np.array(unc_loss_list)

            return unc_results, unc_losses

        # start timer
        unc_start_t = timeit.default_timer()

        # get loss of reco best fit
        unc_loss_best = self.loss_function(
            parameters_trafo=result_trafo,
            data_batch=data_batch,
            seed=result_inv).numpy()

        # define zenith and azimuth of reconstruction result
        result_zenith = result_inv[:, self.zenith_index]
        result_azimuth = result_inv[:, self.azimuth_index]

        # ------------------------
        # get scale of uncertainty
        # ------------------------
        def bisection_step(low, high, target=0.95, ddof=5):
            center = low + (high - low) / 2.
            zen, azi = angles.get_delta_psi_vector(
                zenith=result_zenith,
                azimuth=result_azimuth,
                delta_psi=[center],
                random_service=self.rng)
            unc_results, unc_losses = reconstruct_at_angle(zen, azi)

            # calculate cdf value assuming Wilk's Theorem
            cdf_value = chi2(ddof).cdf(2*(unc_losses - unc_loss_best))

            # pack values together
            values = ([center], zen, azi, unc_results,
                      unc_losses, cdf_value)
            if cdf_value > target:
                high = center
            else:
                low = center
            return low, high, values

        if self.covariance_key is not None:
            cov_sand = results[self.covariance_key]['cov_sand']

            stds = np.sqrt(np.diag(cov_sand))
            circ_sigma = np.sqrt((
                stds[self.zenith_index]**2 +
                stds[self.azimuth_index]**2 *
                np.sin(result_zenith)**2) / 2.)[0]

            unc_upper_bound = min(89.9, 2*np.rad2deg(circ_sigma))
        else:
            num_unc_scale_steps = 4
            lower_bound = 0.
            upper_bound = 90.
            for i in range(num_unc_scale_steps):
                lower_bound, upper_bound, values = bisection_step(
                    lower_bound, upper_bound, ddof=ddof)
            unc_upper_bound = min(89.9, values[0][0])

        print('Upper bound: {} | ddof: {}'.format(unc_upper_bound, ddof))
        # ------------------------

        # generate random vectors at different opening angles delta psi
        delta_psi = self.rng.uniform(1, unc_upper_bound,
                                     size=self.num_fit_points)
        delta_psi = np.linspace(0, unc_upper_bound, self.num_fit_points)
        zen, azi = angles.get_delta_psi_vector(
            zenith=result_zenith,
            azimuth=result_azimuth,
            delta_psi=delta_psi,
            random_service=self.rng)

        # reconstruct at chosen angles
        unc_results, unc_losses = reconstruct_at_angle(zen, azi)

        # calculate delta_log_prob
        delta_loss = unc_losses - unc_loss_best

        # fit chi2 CDF and estimate circularized uncertainty
        circular_unc_deg = self.get_circularized_estimate(
            delta_psi, delta_loss, x0=5., ddof=ddof)

        # end timer
        unc_end_t = timeit.default_timer()

        print('delta_loss', delta_loss)
        print('Uncertainty estimation took: {:3.3f}s'.format(
            unc_end_t - unc_start_t))

        results = {
            'delta_psi': delta_psi,
            'delta_loss': delta_loss,
            'circular_unc': np.deg2rad(circular_unc_deg),
            'circular_unc_deg': circular_unc_deg,
        }

        return results

    def get_circularized_estimate(self, delta_psi, delta_loss, x0=5., ddof=5):
        """Estimate circularized sigma based on Wilk's Theorem.

        Parameters
        ----------
        delta_psi : array_like
            The opening angles between best fit and sample point.
        delta_loss : array_like
            The delta log likelihood values of best fit vs sample point.
        x0 : float, optional
            Initial guess for circularized uncertainty
        ddof : int, optional
            Degrees of freedom to use for chi2 fit.

        Returns
        -------
        float
            The estimated circularized uncertainty assuming Wilk's theorem.
        """
        ts = 2 * delta_loss
        cdf_values = chi2(ddof).cdf(ts)

        def loss_cdf(sigma, delta_psi, cdf_values):
            loss = (cdf_values -
                    basis_functions.rayleigh_cdf(delta_psi, sigma))**2
            return np.sum(loss)

        result_wilks = optimize.minimize(
            loss_cdf, x0=x0, args=(delta_psi, cdf_values))
        return result_wilks.x[0]

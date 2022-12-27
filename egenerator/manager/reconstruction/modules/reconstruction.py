import numpy as np
import healpy as hp
import timeit
import tensorflow as tf

from egenerator.manager.reconstruction.modules.utils import trafo
from egenerator.utils import skyscan
from egenerator.utils import angles


class Reconstruction:

    def __init__(self, manager, loss_module, function_cache,
                 fit_parameter_list,
                 seed_tensor_name,
                 seed_from_previous_module=False,
                 minimize_in_trafo_space=True,
                 randomize_seed=False,
                 parameter_tensor_name='x_parameters',
                 reco_optimizer_interface='scipy',
                 scipy_optimizer_settings={'method': 'BFGS'},
                 tf_optimizer_settings={'method': 'bfgs_minimize',
                                        'x_tolerance': 0.001},
                 verbose=True,
                 ):
        """Initialize reconstruction module and setup tensorflow functions.

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
        seed_tensor_name : str
            This defines the seed for the reconstruction. Depending on the
            value of `seed_from_previous_module` this defines one of 2 things:
            `seed_from_previous_module` == True:
                Name of the previous module from which the 'result' field will
                be used as the seed tensor.
            `seed_from_previous_module` == False:
                Name of the seed tensor which defines that the tensor from the
                input data batch that will be used as the seed.
        seed_from_previous_module : bool, optional
            This defines what `seed_tensor_name` refers to.
            `seed_from_previous_module` == True:
                Name of the previous module from which the 'result' field will
                be used as the seed tensor.
            `seed_from_previous_module` == False:
                Name of the seed tensor which defines that the tensor from the
                input data batch that will be used as the seed.
        minimize_in_trafo_space : bool, optional
            If True, minimization is performed in transformed and normalized
            parameter space. This is usually desired, because the scales of
            the parameters will all be normalized which should facilitate
            minimization.
        randomize_seed : bool, optional
            Description
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
        reco_optimizer_interface : str, optional
            The reconstruction interface to use. Options are 'scipy' and 'tfp'
            for the scipy minimizer and tensorflow probability minimizers.
        scipy_optimizer_settings : dict, optional
            Settings that will be passed on to the scipy.optmize.minimize
            function.
        tf_optimizer_settings : dict, optional
            Settings that will be passed on to the tensorflow probability
            minimizer.
        verbose : bool, optional
            If True, additional information will be printed to the console.

        Raises
        ------
        ValueError
            Description
        """

        # store settings
        self.manager = manager
        self.fit_parameter_list = fit_parameter_list
        self.minimize_in_trafo_space = minimize_in_trafo_space
        self.parameter_tensor_name = parameter_tensor_name
        self.seed_tensor_name = seed_tensor_name
        self.seed_from_previous_module = seed_from_previous_module
        self.randomize_seed = randomize_seed
        self.verbose = verbose

        if not self.seed_from_previous_module:
            self.seed_index = manager.data_handler.tensors.get_index(
                seed_tensor_name)

        # parameter input signature
        param_dtype = getattr(tf, manager.data_trafo.data['tensors'][
            parameter_tensor_name].dtype)
        param_signature = tf.TensorSpec(
            shape=[None, np.sum(fit_parameter_list, dtype=int)],
            dtype=param_dtype)
        param_signature_full = tf.TensorSpec(
            shape=[None, len(fit_parameter_list)],
            dtype=param_dtype)

        data_batch_signature = manager.data_handler.get_data_set_signature()

        # --------------------------------------------------
        # get concrete functions for reconstruction and loss
        # --------------------------------------------------

        # Get loss function
        loss_settings = dict(
            input_signature=(param_signature_full, data_batch_signature),
            loss_module=loss_module,
            fit_parameter_list=[True for f in fit_parameter_list],
            minimize_in_trafo_space=False,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )
        self.parameter_loss_function = function_cache.get(
            'parameter_loss_function', loss_settings)

        if self.parameter_loss_function is None:
            self.parameter_loss_function = manager.get_parameter_loss_function(
                **loss_settings)
            function_cache.add(self.parameter_loss_function, loss_settings)

        # Get loss and gradients function
        function_settings = dict(
            input_signature=(
                param_signature, data_batch_signature, param_signature_full),
            loss_module=loss_module,
            fit_parameter_list=fit_parameter_list,
            minimize_in_trafo_space=minimize_in_trafo_space,
            seed=None,
            parameter_tensor_name=parameter_tensor_name,
        )

        loss_and_gradients_function = function_cache.get(
            'loss_and_gradients_function', function_settings)

        if loss_and_gradients_function is None:
            loss_and_gradients_function = \
                manager.get_loss_and_gradients_function(**function_settings)
            function_cache.add(loss_and_gradients_function, function_settings)

        # choose reconstruction method depending on the optimizer interface
        if reco_optimizer_interface.lower() == 'scipy':
            def reconstruction_method(data_batch, seed_tensor):
                return manager.reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_parameter_list=fit_parameter_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor,
                    parameter_tensor_name=parameter_tensor_name,
                    **scipy_optimizer_settings)

        elif reco_optimizer_interface.lower() == 'shog':
            def reconstruction_method(data_batch, seed_tensor):
                return manager.scipy_global_reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_parameter_list=fit_parameter_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor,
                    parameter_tensor_name=parameter_tensor_name,
                    minimizer_kwargs=scipy_optimizer_settings)

        elif reco_optimizer_interface.lower() == 'tfp':
            @tf.function(
                input_signature=(data_batch_signature, param_signature_full))
            def reconstruction_method(data_batch, seed_tensor):
                return manager.tf_reconstruct_events(
                    data_batch, loss_module,
                    loss_and_gradients_function=loss_and_gradients_function,
                    fit_parameter_list=fit_parameter_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor,
                    parameter_tensor_name=parameter_tensor_name,
                    **tf_optimizer_settings)

        elif reco_optimizer_interface.lower() == 'spherical_opt':

            func_loss_settings = dict(function_settings)

            batch_size = 1
            if batch_size > 1:
                func_loss_settings['reduce_to_scalar'] = False
                func_loss_settings['sort_loss_terms'] = True

            # get concrete loss function
            loss_function = function_cache.get(
                'parameter_loss_function', func_loss_settings)

            if loss_function is None:
                loss_function = manager.get_parameter_loss_function(
                    **func_loss_settings)
                function_cache.add(loss_function, func_loss_settings)

            def reconstruction_method(data_batch, seed_tensor):
                return manager.reconstruct_events_spherical_opt(
                    data_batch, loss_module,
                    loss_function=loss_function,
                    fit_parameter_list=fit_parameter_list,
                    minimize_in_trafo_space=minimize_in_trafo_space,
                    seed=seed_tensor,
                    parameter_tensor_name=parameter_tensor_name)

        else:
            raise ValueError('Unknown interface {!r}. Options are {!r}'.format(
                reco_optimizer_interface, ['scipy', 'tfp']))

        self.reconstruction_method = reconstruction_method

    def execute(self, data_batch, results, **kwargs):
        """Execute reconstruction for a given batch of data.

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

        # get seed: either from seed tensor or from previous results
        if self.seed_from_previous_module:
            seed_tensor = results[self.seed_tensor_name]['result']
        else:
            seed_tensor = data_batch[self.seed_index]

        # -------------------
        # Hack to modify seed
        # -------------------
        if self.randomize_seed:
            shape = seed_tensor.shape
            x0 = np.array(seed_tensor)
            x_true = data_batch[self.manager.data_handler.tensors.get_index(
                self.parameter_tensor_name)].numpy()

            x0[:, :3] = np.random.normal(loc=x0[:, :3], scale=5)
            x0[:, 3] = np.random.uniform(low=0., high=np.pi)
            x0[:, 4] = np.random.uniform(low=0., high=2*np.pi)
            x0[:, 6] = np.random.normal(loc=x0[:, 6], scale=20)
            x0[:, 7] = np.random.uniform(low=0.91, high=1.09)  # Absorption
            x0[:, 8] = np.random.uniform(low=0.01, high=1.99)  # AnisotropyScale
            x0[:, 9] = np.random.uniform(low=0.91, high=1.09)  # DOMEfficiency
            x0[:, 10] = np.random.uniform(low=-0.99, high=0.99)  # HoleIceForward_Unified_00
            x0[:, 11] = np.random.uniform(low=-0.19, high=0.19)  # HoleIceForward_Unified_01
            x0[:, 12] = np.random.uniform(low=0.91, high=1.09)  # Scattering

            # # set vertex to MC truth
            # x0[:, :3] = x_true[:, :3]
            # x0[:, 6] = x_true[:, 6]
            if self.verbose:
                print('New Seed:', x0)

            seed_tensor = x0
        # -------------------

        result_trafo, result_object = self.reconstruction_method(
            data_batch, seed_tensor)

        # invert possible transformation and put full hypothesis together
        result = trafo.get_reco_result_batch(
            result_trafo=result_trafo,
            seed_tensor=seed_tensor,
            fit_parameter_list=self.fit_parameter_list,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            data_trafo=self.manager.data_trafo,
            parameter_tensor_name=self.parameter_tensor_name)

        loss_seed = self.parameter_loss_function(
            seed_tensor, data_batch).numpy()
        loss_reco = self.parameter_loss_function(result, data_batch).numpy()

        result_dict = {
            'result': result,
            'result_trafo': result_trafo,
            'result_object': result_object,
            'loss_seed': loss_seed,
            'loss_reco': loss_reco,
            'seed_tensor_name': self.seed_tensor_name,
            'seed_from_previous_module': self.seed_from_previous_module,
        }
        return result_dict


class SkyScanner:

    def __init__(self, manager, loss_module, function_cache,
                 fit_parameter_list,
                 zenith_key, azimuth_key,
                 seed_tensor_name,
                 skyscan_nside=2,
                 skyscan_focus_bounds=[5, 15, 30],
                 skyscan_focus_nsides=[32, 16, 8],
                 skyscan_focus_seeds=[],
                 verbose=True,
                 **kwargs
                 ):
        """Initialize reconstruction module and setup tensorflow functions.

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
            is minimized. Note that the entries corresponding to the defined
            zenith and azimuth keys will be set to False. In other words,
            the reconstruction at each Healpix will not fit for the direction
            as this one is fixed to the corresponding direction of the
            given Healpix.
        zenith_key : str
            The name of the key that defines the zenith direction.
            This is utilized to adjust the `fit_parameter_list` to
            ensure that the direction isn't fit for.
        azimuth_key : str
            The name of the key that defines the azimuth direction.
            This is utilized to adjust the `fit_parameter_list` to
            ensure that the direction isn't fit for.
        seed_tensor_name : str
            This defines the seed for the reconstruction.
            Must be given as the name of the seed tensor as
            loaded in the input data batch or as 'reco' to pull from
            the result dictionary of previous reconstruction modules.
            Each skyscan pixel will be seeded with this parameters, except
            for zenith and azimuth which are fixed to the corresponding
            values for the given Healpix.
        skyscan_nside : int, optional
            The default nside that will be used for the skyscan.
            This nside will be applied everywhere that is outside of the
            focus regions as defined in the `skyscan_focus_*` parameters.
        skyscan_focus_bounds : list, optional
            The skyscan will increase resolution in rings around the seeds
            provided in `skyscan_focus_seeds`. This parameter defines the
            boundaries [in degrees] at which the next nside is chosen.
            The provided list of floats must be given in ascending order
            and the corresponding nside is given in `skyscan_focus_nsides`.
        skyscan_focus_nsides : list, optional
            The skyscan will increase resolution in rings around the seeds
            provided in `skyscan_focus_seeds`. This parameter defines the
            nsides for each of these rings. See also `skyscan_focus_bounds`,
            which defines the distance of these rings.
        skyscan_focus_seeds : list of str, optional
            A list of tensor names that will be used to define which parts
            of the sky to scan in increased resolution. These tensors must
            be included in the input data batch or if set to 'reco' they
            will use the result of a previous reconstruction module.
            The parameter `skyscan_focus_bounds` and `skyscan_focus_nsides`
            define how this increased resolution scan around these seed
            directions will be performed.

        verbose : bool, optional
            If True, additional information will be printed to the console.
        **kwargs
            Description

        Deleted Parameters
        ------------------
        skyscan_nside_ranges : dict
            This defines the resolution in which the skyscan will be performed.
            If provided, rings of increased resolution will be created around
            each of the provided seeds in `skyscan_focus_seeds`.
            The resolution rings are defined in a dictionary with keys:
                'max_dist[deg]': nside
            {

            }

        """

        # store settings
        self.manager = manager
        self.fit_parameter_list = fit_parameter_list
        self.seed_tensor_name = seed_tensor_name
        self.zenith_key = zenith_key
        self.azimuth_key = azimuth_key
        self.skyscan_nside = skyscan_nside
        self.skyscan_focus_bounds = skyscan_focus_bounds
        self.skyscan_focus_nsides = skyscan_focus_nsides
        self.skyscan_focus_seeds = skyscan_focus_seeds
        self.verbose = verbose

        self.zenith_index = manager.models[0].get_index(zenith_key)
        self.azimuth_index = manager.models[0].get_index(azimuth_key)

        # sanity checks (same length and sorted bounds)
        assert len(self.skyscan_focus_bounds) == len(self.skyscan_focus_nsides)
        assert np.allclose(
            self.skyscan_focus_bounds, np.sort(self.skyscan_focus_bounds))

        # adjust reconstruction settings for skyscan
        fit_parameter_list_mod = [f for f in fit_parameter_list]
        fit_parameter_list_mod[self.zenith_index] = False
        fit_parameter_list_mod[self.azimuth_index] = False

        ignored_keys = [
            'seed_from_previous_module',
            'randomize_seed',
        ]
        for k in ignored_keys:
            if k in kwargs:
                print('SkyScanner ignoring reco setting: {}'.format(k))
                kwargs.pop(k)

        # create reconstruction module
        self.reco_module = Reconstruction(
            manager=manager,
            loss_module=loss_module,
            function_cache=function_cache,
            fit_parameter_list=fit_parameter_list_mod,
            seed_tensor_name='SkyScanSeed',
            seed_from_previous_module=True,
            verbose=verbose,
            **kwargs
        )

    def execute(self, data_batch, results, **kwargs):
        """Execute skyscan for a given batch of data.

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

        # -------------------------------------------------------------
        # Generate list of nside and ipix that need to be reconstructed
        # -------------------------------------------------------------
        focus_zeniths = []
        focus_azimuths = []
        for seed_name in self.skyscan_focus_seeds:
            if seed_name == 'reco':
                seed = np.array(results['reco']['result'])
            else:
                seed_index = self.manager.data_handler.tensors.get_index(
                    seed_name)
                seed = np.array(data_batch[seed_index])

            # currently only support single batch
            assert len(seed) == 1, seed

            # make sure zenith and azimuth are in range [0, pi), [0, 2pi)
            zenith, azimuth = angles.convert_to_range(
                zenith=seed[0][self.zenith_index],
                azimuth=seed[0][self.azimuth_index],
            )

            focus_zeniths.append(zenith)
            focus_azimuths.append(azimuth)

        scan_pixels = skyscan.get_scan_pixels(
            default_nside=self.skyscan_nside,
            focus_bounds=self.skyscan_focus_bounds,
            focus_nsides=self.skyscan_focus_nsides,
            focus_zeniths=focus_zeniths,
            focus_azimuths=focus_azimuths,
        )

        # ----------------------------------------------
        # Perform reconstruction for each nside and ipix
        # ----------------------------------------------

        # Format: {nside: {ipix: llh/res}}
        skyscan_llh = {}
        skyscan_res = {}

        # now walk through each nside/ipix pair and:
        #   1. set zenith/azimuth to match ipix
        #   2. create results_i in which this seed is saved to 'SkyScanSeed'
        #   3. pass on to reconstruction module and run reco
        #   4. extract reco result and save away fit params and llh
        scan_min_val = float('inf')
        scan_min_fit = None
        scan_min_ipix = None
        scan_min_nside = None
        for nside, ipix_list in scan_pixels.items():

            start_t = timeit.default_timer()
            if self.verbose:
                print('Scanning {} pixels of nside {} ...'.format(
                    nside, len(ipix_list)))

            skyscan_llh_i = {}
            skyscan_res_i = {}

            for ipix in ipix_list:
                theta, phi = hp.pix2ang(nside, ipix)

                # get seed either from previous reconstruction result
                # or from one loaded in the input data batch
                if self.seed_tensor_name == 'reco':
                    skyscanseed = np.array(results['reco']['result'])
                else:
                    seed_index = self.manager.data_handler.tensors.get_index(
                        self.seed_tensor_name)
                    skyscanseed = np.array(data_batch[seed_index])

                assert len(skyscanseed) == 1, skyscanseed
                skyscanseed[0][self.zenith_index] = theta
                skyscanseed[0][self.azimuth_index] = phi

                # create pseudo results dict to pass on to reconstruction
                # module which will pull the seed of the "previous"
                # reconstruction to use as seed (seed_from_previous_module)
                results_i = {'SkyScanSeed': {'result': skyscanseed}}

                # run reconstruction for this seed
                results_i = self.reco_module.execute(
                    data_batch, results_i, **kwargs)

                # extract best-fit params and llh
                skyscan_llh_i[ipix] = results_i['loss_reco']
                skyscan_res_i[ipix] = results_i['result']

                # keep track of scan minimum
                if results_i['loss_reco'] < scan_min_val:
                    scan_min_val = results_i['loss_reco']
                    scan_min_fit = results_i['result']
                    scan_min_nside = nside
                    scan_min_ipix = ipix

            skyscan_llh[nside] = skyscan_llh_i
            skyscan_res[nside] = skyscan_res_i

            end_t = timeit.default_timer()
            if self.verbose:
                print('   ... that took {:3.3}s'.format(end_t - start_t))

        result_dict = {
            'skyscan_llh': skyscan_llh,
            'skyscan_res': skyscan_res,
            'scan_min_fit': scan_min_fit,
            'scan_min_val': scan_min_val,
            'scan_min_nside': scan_min_nside,
            'scan_min_ipix': scan_min_ipix,
        }
        return result_dict


class SelectBestReconstruction:

    def __init__(
            self, manager, loss_module, function_cache, reco_names,
            verbose=True):
        """Initialize reconstruction module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
        reco_names : list of str
            A list of reco names from previous reconstruction modules. The
            best reconstruction, e.g. lowest reco loss, will be chosen from
            these reconstructions.
        verbose : bool, optional
            If True, additional information will be printed to the console.
        """

        # store settings
        self.reco_names = reco_names
        self.verbose = verbose

    def execute(self, data_batch, results, **kwargs):
        """Execute selection.

        Choose best reconstruction from a list of results

        Parameters
        ----------
        data_batch : tuple of tf.Tensors
            A data batch which consists of a tuple of tf.Tensors.
        results : dict
            A dictrionary with the results of previous modules.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        TYPE
            Description
        """
        min_loss = float('inf')
        min_results = None

        for reco_name in self.reco_names:
            reco_results = results[reco_name]

            if self.verbose:
                print('Loss: {:3.3f} | {}'.format(
                    reco_results['loss_reco'], reco_name))

            # check if it has a lower loss
            if reco_results['loss_reco'] < min_loss:
                min_loss = reco_results['loss_reco']
                min_results = reco_results

        if min_results is None:

            # something went wrong, did the fit return NaN or inf loss?
            # For now: just choose last reco_results
            print('Did not find a minimium, choosing last reco: {}!'.format(
                reco_name))
            min_results = reco_results

        min_results = dict(min_results)

        # rename runtime key of result
        min_results['reco_runtime'] = min_results.pop('runtime')

        return min_results

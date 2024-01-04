import os
import numpy as np
import tensorflow as tf
import timeit

from icecube import dataclasses, icetray
from icecube.icetray.i3logging import log_info, log_warn

from egenerator.utils.configurator import ManagerConfigurator
from egenerator.utils import basis_functions


class EventGeneratorSimulation(icetray.I3ConditionalModule):

    """Class to simulate events with Event-Generator model.

    """

    def __init__(self, context):
        """Class to simulate events with Event-Generator Model.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox('OutBox')

        # Required settings
        self.AddParameter('model_name',
                          'A model name that defines which model to '
                          'apply. The full path is given by `model_base_dir` +'
                          ' model_name if `model_base_dir` is provided. '
                          'Otherwise `model_name` must define the full '
                          'model path.')

        # Optional settings
        self.AddParameter('default_values',
                          'Not all parameters of the source hypothesis can '
                          'be extracted from the I3MCTree. For these '
                          'parameters, default values may be defined. The '
                          '`default_values` must be a dictionary of the '
                          'format `parameter_name`: `value`. `value` may '
                          'either be a double or a string. If a string is '
                          'provided, it is assumed that an I3Double exists in '
                          'the frame under the key as provided by `value`',
                          {
                            'Absorption': 1.,
                            'AnisotropyScale': 1.,
                            'DOMEfficiency': 1.,
                            'HoleIceForward_Unified_00': 0.,
                            'HoleIceForward_Unified_01': 0.,
                            'Scattering': 1.,
                          })
        self.AddParameter('model_base_dir',
                          'The base directory in which the model is located.'
                          ' The full path is given by `model_base_dir` + '
                          ' model_name if `model_base_dir` is provided. '
                          'Otherwise `model_name` must define the full '
                          'model path.',
                          '/data/user/mhuennefeld/exported_models/egenerator')
        self.AddParameter('output_key',
                          'The output base key to which pulses will be saved',
                          'EventGeneratorSimulation')
        self.AddParameter('mc_tree_name',
                          'The name of the propagated I3MCTree for which the '
                          'light yield and measured pulses will be simulated',
                          'I3MCTree')
        self.AddParameter('max_simulation_distance',
                          'Particles further away from the center of the '
                          'detector than this distance are not simulated',
                          2250)
        self.AddParameter('min_simulation_energy',
                          'Particles below this energy threshold are not '
                          'simulated.',
                          100)
        self.AddParameter('correct_sampled_charge',
                          'Correct individual charges of sampled pulses to '
                          'exactly match estimated total charge. If False '
                          'additional randomization is added since the charge '
                          'of a sampled pulse will only equal to 1PE on '
                          'average. If True, charges at DOMs tend to be '
                          'discretized to integer values for lower-charge '
                          'DOMs. False is usually preferred.',
                          False)
        self.AddParameter('num_threads',
                          'Number of threads to use for tensorflow. This will '
                          'be passed on to tensorflow where it is used to set '
                          'the "intra_op_parallelism_threads" and '
                          '"inter_op_parallelism_threads" settings. If a '
                          'value of zero (default) is provided, the system '
                          'uses an appropriate number. Note: when running '
                          'this as a job on a cluster you might want to limit '
                          '"num_threads" to the amount of allocated CPUs.',
                          0)
        self.AddParameter('random_service',
                          'The random service or seed to use. If this is an '
                          'integer, a numpy random state will be created with '
                          'the seed set to `random_service`',
                          42)
        self.AddParameter('SimulateElectronDaughterParticles',
                          'If true, look for daughter particles of an electron and '
                          'simulate their light yield rather than from the '
                          'mother electron. This is necessary to correctly '
                          'simulate the cascade longitudinal extension of '
                          'high energy cascades.',
                          False)
        self.AddParameter('MergePulses',
                          'If true, pulses in one DOM within a `PulseMergeWindow` '
                          'will be merged together.',
                          False)
        self.AddParameter('PulseMergeWindow',
                          'Timewindow within pulses are merged together.',
                          5)

    def Configure(self):
        """Configures Module and loads model from file.
        """
        self.model_name = self.GetParameter('model_name')
        self.default_values = self.GetParameter('default_values')
        self.model_base_dir = self.GetParameter('model_base_dir')
        self.output_key = self.GetParameter('output_key')
        self.mc_tree_name = self.GetParameter('mc_tree_name')
        self.max_simulation_distance = \
            self.GetParameter('max_simulation_distance')
        self.min_simulation_energy = self.GetParameter('min_simulation_energy')
        self.correct_sampled_charge = self.GetParameter(
            'correct_sampled_charge')
        self.num_threads = self.GetParameter('num_threads')
        self.random_service = self.GetParameter('random_service')
        self.simulate_electron_daughters = self.GetParameter('SimulateElectronDaughterParticles')
        self.merge_pulses = self.GetParameter('MergePulses')
        self.pulse_merge_window = self.GetParameter('PulseMergeWindow')

        if isinstance(self.random_service, int):
            self.random_service = np.random.RandomState(self.random_service)

        if self.model_base_dir is not None:
            self.model_dir = os.path.join(self.model_base_dir, self.model_name)
        else:
            self.model_dir = self.model_name

        # --------------------------------------------------
        # Build and configure SourceManager and extrac Model
        # --------------------------------------------------
        self.manager_configurator = ManagerConfigurator(
            manager_dirs=[self.model_dir],
            num_threads=self.num_threads,
        )
        self.manager = self.manager_configurator.manager

        for model in self.manager.models:
            num_vars, num_total_vars = model.num_variables
            msg = '\nNumber of Model Variables:\n'
            msg += '\tFree: {}\n'
            msg += '\tTotal: {}'
            log_info(msg.format(num_vars, num_total_vars))

        if len(self.manager.models) > 1:
            raise NotImplementedError(
                'Currently does not support model ensemble.')

        self.model = self.manager.models[0]

        # get parameter names that have to be set
        self.param_names = sorted([n for n in self.model.parameter_names
                                   if n not in self.default_values])

        # make sure that the only parameters that need to be set are provided
        included_parameters = [
                'azimuth', 'energy', 'time', 'x', 'y', 'z', 'zenith']

        # search if there is a common prefix that we can use
        prefix_list = []
        for name in included_parameters:
            if name in self.param_names:
                prefix_list.append('')
            else:
                found_match = False
                for param in self.param_names:
                    if param[-len(name):] == name:
                        # only allow prefixes ending on '_'
                        if param[-len(name)-1] == '_':
                            prefix_list.append(param[:-len(name)])
                            found_match = True
                if not found_match:
                    msg = (
                        'Did not find a parameter name match for "{}". Model '
                        'Parameter names are: {}'
                    ).format(name, self.param_names)
                    raise ValueError(msg)

        prefix_list = np.unique(prefix_list)
        if len(prefix_list) != 1:
            msg = 'Could not find common parameter prefix. Found: {}'.format(
                prefix_list)
            raise ValueError(msg)

        self._prefix = prefix_list[0]

        # double check that the prefix now works
        for name in self.param_names:
            if name[len(self._prefix):] not in included_parameters:
                raise KeyError('Unknown parameter name:', name)

        # Create concrete tensorflow function to obtain DOM expectations
        self.param_dtype = getattr(
            tf, self.manager.data_trafo.data['tensors']['x_parameters'].dtype)
        self.get_model_tensors = self.manager.get_model_tensors_function()

        # ---------------------------------------------------
        # Define which particles are simulated by which Model
        # ---------------------------------------------------

        # define allowed parent particles
        # These are particles which we can safely simulate by only looking at
        # their daughters
        self.allowed_parent_particles = [
            dataclasses.I3Particle.NuE,
            dataclasses.I3Particle.NuMu,
            dataclasses.I3Particle.NuTau,
            dataclasses.I3Particle.NuEBar,
            dataclasses.I3Particle.NuMuBar,
            dataclasses.I3Particle.NuTauBar,
            dataclasses.I3Particle.Hadrons,
        ]
        if self.simulate_electron_daughters:
            self.allowed_parent_particles += [
                dataclasses.I3Particle.EMinus,
                dataclasses.I3Particle.EPlus,
            ]

        # Define type of particles that can be simulated as EM cascades
        self.em_cascades = [
            dataclasses.I3Particle.Pi0,
            dataclasses.I3Particle.Gamma,
            dataclasses.I3Particle.PairProd,
            dataclasses.I3Particle.Brems,
            dataclasses.I3Particle.DeltaE,
            dataclasses.I3Particle.EMinus,
            dataclasses.I3Particle.EPlus,
        ]

        # Define type of particles that can be simulated as tracks
        self.tracks = [
        ]

        # define particles that do not deposit light, in other words
        # if we end up with a final state particle of this type it is ok
        # not to simulate the light yield for these particles.
        self.dark_particles = [
            dataclasses.I3Particle.NuE,
            dataclasses.I3Particle.NuMu,
            dataclasses.I3Particle.NuTau,
            dataclasses.I3Particle.NuEBar,
            dataclasses.I3Particle.NuMuBar,
            dataclasses.I3Particle.NuTauBar,
        ]

    def DAQ(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Q-Frame.
        """

        # start timer
        t_0 = timeit.default_timer()

        # get sources
        cascades, tracks = self.get_light_sources(frame[self.mc_tree_name])

        if len(tracks) > 0:
            raise NotImplementedError('Tracks not yet supported')

        # convert cascade to source hypotheses
        cascade_sources = self.convert_cascades_to_tensor(cascades, frame)

        # timer after source collection
        t_1 = timeit.default_timer()

        # get result tensors from model
        result_tensors = self.get_model_tensors(cascade_sources)

        # timer after NN evaluation
        t_2 = timeit.default_timer()

        # sample pulses
        pulses = self.sample_pulses(result_tensors, cascade_sources)

        # timer after Sampling
        t_3 = timeit.default_timer()

        log_info('Simulation took: {:3.3f}ms'.format((t_3 - t_0) * 1000))
        log_info('\t Gathering Sources: {:3.3f}ms'.format((t_1 - t_0) * 1000))
        log_info('\t Evaluating NN model: {:3.3f}ms'.format((t_2 - t_1) * 1000))
        log_info('\t Sampling Pulses: {:3.3f}ms'.format((t_3 - t_2) * 1000))

        # write to frame
        frame[self.output_key] = pulses

        # push frame to next modules
        self.PushFrame(frame)

    def sample_pulses(self, result_tensors, cascade_sources):
        """Sample pulses from PDF and create a I3RecoPulseSeriesMap

        Parameters
        ----------
        result_tensors : dict
            A dictionary with the result tensors.
        cascade_sources : tf.Tensor
            The source hypothesis paramter tensor.

        Returns
        -------
        I3RecoPulseSeriesMap
            The sampled pulse series map.
        """

        # draw total charge per DOM and cascade
        dom_charges = basis_functions.sample_from_negative_binomial(
            rng=self.random_service,
            mu=result_tensors['dom_charges'].numpy(),
            alpha_or_var=result_tensors['dom_charges_variance'].numpy(),
            param_is_alpha=False,
        )
        # dom_charges = self.random_service.poisson(
        #     result_tensors['dom_charges'].numpy())
        dom_charges_total = np.sum(dom_charges, axis=0)
        num_cascades = dom_charges.shape[0]

        cascade_times = cascade_sources.numpy()[
            :, self.model.get_index(self._prefix + 'time')]

        if self._prefix != '':

            # allow for max 1-depth of nested results for mixture model comp.
            if 'latent_var_scale' not in result_tensors:
                result_tensors = result_tensors['nested_results'][
                    self._prefix[:-1]]
            log_warn(
                'Using nested result tensors, this is potentially wrong, '
                'since this is not the complete time PDF!'
            )

        cum_scale = np.cumsum(
            result_tensors['latent_var_scale'].numpy(), axis=-1)

        latent_var_mu = result_tensors['latent_var_mu'].numpy()
        latent_var_sigma = result_tensors['latent_var_sigma'].numpy()
        latent_var_r = result_tensors['latent_var_r'].numpy()

        # for numerical stability:
        cum_scale[..., -1] = 1.00000001

        pulse_series_map = dataclasses.I3RecoPulseSeriesMap()

        # walk through DOMs
        for string in range(86):
            for om in range(60):

                # shortcut
                if dom_charges_total[string, om, 0] <= 0:
                    continue

                pulse_times_list = []
                pulse_charges_list = []
                for i in range(num_cascades):

                    num_pe = dom_charges[i, string, om, 0]
                    if num_pe <= 0:
                        continue

                    # we will uniformly choose the charge and then (optionally)
                    # correct again to obtain correct total charge
                    # ToDo: figure out actual chage distribution of pulses!
                    pulse_charges = self.random_service.uniform(
                        0.25, 1.75, size=num_pe)

                    if self.correct_sampled_charge:
                        pulse_charges *= num_pe / np.sum(pulse_charges)

                    # for each pulse, draw 2 random numbers which we will need
                    # to figure out which mixtue model component to choose
                    # and at what time the pulse gets injected
                    rngs = self.random_service.uniform(size=(num_pe, 2))

                    idx = np.searchsorted(cum_scale[i, string, om], rngs[:, 0])

                    # get parameters for chosen asymmetric gaussian
                    pulse_mu = latent_var_mu[i, string, om, idx]
                    pulse_sigma = latent_var_sigma[i, string, om, idx]
                    pulse_r = latent_var_r[i, string, om, idx]

                    # caclulate time of pulse
                    pulse_times = basis_functions.asymmetric_gauss_ppf(
                        rngs[:, 1], mu=pulse_mu, sigma=pulse_sigma, r=pulse_r)

                    # fix scale
                    pulse_times *= self.model.time_unit_in_ns

                    # fix offset
                    pulse_times += cascade_times[i]

                    pulse_times_list.append(pulse_times)
                    pulse_charges_list.append(pulse_charges)

                pulse_charges = np.concatenate(pulse_charges_list)
                pulse_times = np.concatenate(pulse_times_list)

                # sort pulses in time
                sorted_indices = np.argsort(pulse_times)
                pulse_times = pulse_times[sorted_indices]
                pulse_charges = pulse_charges[sorted_indices]

                if self.merge_pulses:
                    if len(pulse_times) > 1:
                        time_bins = np.arange(pulse_times[0], pulse_times[-1] + self.pulse_merge_window, self.pulse_merge_window) # ns
                        n, bin_edges = np.histogram(pulse_times, time_bins, weights=pulse_charges)

                        pulse_times = bin_edges[:-1] # take left bin edge as time
                        pulse_charges = n

                # create pulse series
                pulse_series = dataclasses.I3RecoPulseSeries()
                for time, charge in zip(pulse_times, pulse_charges):
                    if charge <= 0:
                        continue
                    pulse = dataclasses.I3RecoPulse()
                    pulse.time = time
                    pulse.charge = charge
                    pulse_series.append(pulse)

                # add to pulse series map
                om_key = icetray.OMKey(string+1, om+1)
                pulse_series_map[om_key] = pulse_series

        return pulse_series_map

    def convert_cascades_to_tensor(self, cascades, frame):
        """Convert a list of cascades to a source parameter tensor.

        Parameters
        ----------
        cascades : list of I3Particles
            The list of cascade I3Particles.

        frame : I3Frame
            Necessary to get additional parameter from frame.
        
        Returns
        -------
        tf.Tensor
            The source hypothesis paramter tensor.
        """
        # create parameter array
        parameters = np.empty([len(cascades), self.model.num_parameters])

        # Fill default values
        for name, value in self.default_values.items():

            # assume that a I3Double exists in frame
            if isinstance(value, str):
                value = frame[value].value

            # assign values
            parameters[:, self.model.get_index(name)] = value

        # now fill cascade parameters (x, y, z, zenith, azimuth, E, t)
        for i, cascade in enumerate(cascades):
            for name in self.param_names:

                # remove prefix
                name = name[len(self._prefix):]

                if name in ['x', 'y', 'z']:
                    value = getattr(cascade.pos, name)
                elif name in ['azimuth', 'zenith']:
                    value = getattr(cascade.dir, name)
                else:
                    value = getattr(cascade, name)

                index = self.model.get_index(self._prefix + name)
                parameters[i, index] = value

        return tf.convert_to_tensor(parameters, dtype=self.param_dtype)

    def _get_light_sources(self, mc_tree, parent):
        """Get a list of cascade and track light sources from an I3MCTree

        Recursive helper function for the method `get_light_sources`.

        Parameters
        ----------
        mc_tree : I3MCTree
            The I3MCTree from which to get the light sources

        Returns
        -------
        array_like
            The list of cascades.
        array_like
            The list of tracks.
        """

        # Stopping condition for recursion: either found a track or cascade
        if (parent.energy < self.min_simulation_energy or
                parent.pos.magnitude - parent.length
                > self.max_simulation_distance):
            return [], []

        if parent.type in self.em_cascades:
            if self.simulate_electron_daughters and len(mc_tree.get_daughters(parent)):
                pass # get light sourced from daughters
            else:
                return [parent], []

        elif parent.type in self.tracks:
            return [], [parent]

        cascades = []
        tracks = []

        # need further recursion
        daughters = mc_tree.get_daughters(parent)

        # check if we have a branch that we can't simulate
        if len(daughters) == 0 and parent.type not in self.dark_particles:
            log_warn(f"Particle: {parent}")
            log_warn(f"Tree: {mc_tree}")
            raise NotImplementedError(
                'Particle can not be simulated: ', parent.type)

        if len(daughters) > 0:
            if parent.type not in self.allowed_parent_particles:
                raise ValueError('Unknown parent type:', parent.type)

        for daughter in daughters:

            # get list of cascades and tracks
            cascades_i, tracks_i = self._get_light_sources(mc_tree, daughter)

            # extend lists
            cascades.extend(cascades_i)
            tracks.extend(tracks_i)

        return cascades, tracks

    def get_light_sources(self, mc_tree):
        """Get a list of cascade and track light sources from an I3MCTree

        This method recursively walks through the I3MCTree by looking at
        daughter particles. All daughter particles are collected, that need
        to be simulated.

        Parameters
        ----------
        mc_tree : I3MCTree
            The I3MCTree from which to get the light sources

        Returns
        -------
        array_like
            The list of cascades.
        array_like
            The list of tracks.
        """

        cascades = []
        tracks = []
        for primary in mc_tree.get_primaries():

            # get list of cascades and tracks
            cascades_i, tracks_i = self._get_light_sources(mc_tree, primary)

            # extend lists
            cascades.extend(cascades_i)
            tracks.extend(tracks_i)

        return cascades, tracks

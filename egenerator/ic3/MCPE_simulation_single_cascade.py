import os
import numpy as np
import tensorflow as tf
import timeit

from icecube import dataclasses, icetray, simclasses
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
        #self.AddOutBox('OutBox')

        print("constructor")

        # Required settings
        self.AddParameter('model_name',
                          'A model name that defines which model to '
                          'apply. The full path is given by `model_base_dir` +'
                          ' model_name if `model_base_dir` is provided. '
                          'Otherwise `model_name` must define the full '
                          'model path.',
                          'lom_mcpe_june_2023')

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
                          })
        self.AddParameter('model_base_dir',
                          'The base directory in which the model is located.'
                          ' The full path is given by `model_base_dir` + '
                          ' model_name if `model_base_dir` is provided. '
                          'Otherwise `model_name` must define the full '
                          'model path.',
                          '/data/user/jvara/exported_models/event-generator/lom')
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
                          22500)#the original value is 10 times smaller Gen2?
        self.AddParameter('min_simulation_energy',
                          'Particles below this energy threshold are not '
                          'simulated.',
                          100)
        self.AddParameter('num_threads',
                          'Number of threads to use for tensorflow. This will '
                          'be passed on to tensorflow where it is used to set '
                          'the "intra_op_parallelism_threads" and '
                          '"inter_op_parallelism_threads" settings. If a '
                          'value of zero (default) is provided, the system '
                          'uses an appropriate number. Note: when running '
                          'this as a job on a cluster you might want to limit '
                          '"num_threads" to the amount of allocated CPUs.',
                          4)
        self.AddParameter('random_service',
                          'The random service or seed to use. If this is an '
                          'integer, a numpy random state will be created with '
                          'the seed set to `random_service`',
                          42)

    def Configure(self):
        """Configures Module and loads model from file.
        """

        self.count = 0

        self.model_name = self.GetParameter('model_name')
        self.default_values = self.GetParameter('default_values')
        self.model_base_dir = self.GetParameter('model_base_dir')
        self.output_key = self.GetParameter('output_key')
        self.mc_tree_name = self.GetParameter('mc_tree_name')
        self.max_simulation_distance = \
            self.GetParameter('max_simulation_distance')
        self.min_simulation_energy = self.GetParameter('min_simulation_energy')
        self.num_threads = self.GetParameter('num_threads')
        self.random_service = self.GetParameter('random_service')

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

    def DAQ(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Q-Frame.
        """

        self.count +=1

        print(f"Running frame {self.count}")

        # start timer
        t_0 = timeit.default_timer()

        # convert cascade to source hypotheses
        cascades = [frame['LabelsDeepLearning']]

        cascade_sources = self.convert_cascades_to_tensor(cascades)

        # timer after source collection
        t_1 = timeit.default_timer()

        # get result tensors from model
        result_tensors = self.get_model_tensors(cascade_sources)

        # timer after NN evaluation
        t_2 = timeit.default_timer()

        # sample pulses
        pulses = self.sample_mcpes(result_tensors, cascade_sources)

        # timer after Sampling
        t_3 = timeit.default_timer()

        print('Simulation took: {:3.3f}ms'.format((t_3 - t_0) * 1000))
        print('\t Gathering Sources: {:3.3f}ms'.format((t_1 - t_0) * 1000))
        print('\t Evaluating NN model: {:3.3f}ms'.format((t_2 - t_1) * 1000))
        print('\t Sampling Pulses: {:3.3f}ms'.format((t_3 - t_2) * 1000))


        log_info('Simulation took: {:3.3f}ms'.format((t_3 - t_0) * 1000))
        log_info('\t Gathering Sources: {:3.3f}ms'.format((t_1 - t_0) * 1000))
        log_info('\t Evaluating NN model: {:3.3f}ms'.format((t_2 - t_1) * 1000))
        log_info('\t Sampling Pulses: {:3.3f}ms'.format((t_3 - t_2) * 1000))

        # write to frame
        frame[self.output_key] = pulses

        # push frame to next modules
        self.PushFrame(frame)
    

    def sample_mcpes(self, result_tensors, cascade_sources):
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
        #dom_charges = basis_functions.sample_from_negative_binomial(
        #    rng=self.random_service,
        #    mu=result_tensors['dom_charges'].numpy(),
        #    alpha_or_var=result_tensors['dom_charges_variance'].numpy(),
        #    param_is_alpha=False,
        #)
        t_a = timeit.default_timer()

        dom_charges = self.random_service.poisson(
             result_tensors['dom_charges'].numpy())
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

        mcpe_series_map = simclasses.I3MCPESeriesMap()

        # walk through DOMs

        #Future: obtain these values from .pickle
        num_strings = 120
        num_om_string = 80
        num_pmt_om = 16

        t_b = timeit.default_timer()

        for string in range(num_strings):
            for om in range(num_om_string):
                for pmt in range(num_pmt_om):

                    # shortcut
                    if dom_charges_total[string, om*num_pmt_om + pmt, 0] <= 0:
                        continue

                    pulse_times_list = []
                    pulse_charges_list = []
                    for i in range(num_cascades):

                        num_pe = dom_charges[i, string, om*num_pmt_om + pmt, 0]
                        if num_pe <= 0:
                            continue
                        # we will uniformly choose the charge and then correct
                        # again to obtain correct total charge
                        # ToDo: figure out actual chage distribution of pulses!
                        pulse_charges = np.ones(int(num_pe))

                        # for each pulse, draw 2 random numbers which we will need
                        # to figure out which mixtue model component to choose
                        # and at what time the pulse gets injected
                        rngs = self.random_service.uniform(size=(num_pe, 2))

                        idx = np.searchsorted(cum_scale[i, string, om*num_pmt_om + pmt], rngs[:, 0])

                        # get parameters for chosen asymmetric gaussian
                        pulse_mu = latent_var_mu[i, string, om*num_pmt_om + pmt, idx]
                        pulse_sigma = latent_var_sigma[i, string, om*num_pmt_om + pmt, idx]
                        pulse_r = latent_var_r[i, string, om*num_pmt_om + pmt, idx]

                        # caclulate time of pulse
                        pulse_times = basis_functions.asymmetric_gauss_ppf(
                            rngs[:, 1], mu=pulse_mu, sigma=pulse_sigma, r=pulse_r)

                        # fix scale
                        pulse_times *= 1000.

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

                     # create mcpe series map
                    mcpe_series = simclasses.I3MCPESeries()
                    for time, npe in zip(pulse_times, pulse_charges):
                        mcpe = simclasses.I3MCPE()
                        mcpe.time = time
                        mcpe.npe = 1
                        mcpe_series.append(mcpe)

                     # add to pulse series map
                    om_key = icetray.OMKey(string+1001, om+1, pmt)
                    mcpe_series_map[om_key] = mcpe_series
        t_c = timeit.default_timer()
        print('Before for loop {:3.3f}ms'.format((t_b- t_a) * 1000))
        print('During for loop {:3.3f}ms'.format((t_c - t_b) * 1000))
        return mcpe_series_map





    def convert_cascades_to_tensor(self, cascades):
        """Convert a list of cascades to a source parameter tensor.

        Parameters
        ----------
        cascades : list of I3Particles
            The list of cascade I3Particles.

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
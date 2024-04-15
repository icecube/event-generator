import os
import numpy as np
import tensorflow as tf
import timeit
import math


import matplotlib.pyplot as plt

from icecube import dataclasses, icetray, photonics_service
from icecube.icetray.i3logging import log_info, log_warn

from egenerator.utils.configurator import ManagerConfigurator
from egenerator.utils import basis_functions


def calculate_loss(pred, true):
    assert true % 1 == 0, "Not a int value"
    return np.exp(-1 * pred) * pred ** true / math.factorial(true)


class CalculateLikelihood(icetray.I3ConditionalModule):

    """Class to evaluate predictions of MCPE with Event-Generator model.

    """

    def __init__(self, context):
        """Class to evaluate predictions of MCPE with Event-Generator model.

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
                          'EventGeneratorPrediction')
        self.AddParameter('mc_tree_name',
                          'The name of the propagated I3MCTree for which the '
                          'light yield and measured pulses will be simulated',
                          'I3MCTree')
        self.AddParameter('max_simulation_distance',
                          'Particles further away from the center of the '
                          'detector than this distance are not predicted',
                          2250)
        self.AddParameter('min_simulation_energy',
                          'Particles below this energy threshold are not '
                          'predicted.',
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
        self.AddParameter('AddDarkDOMs', '', True)

        self.AddParameter('CascadeService',
                          '',
                          None)


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
        self.num_threads = self.GetParameter('num_threads')
        self.random_service = self.GetParameter('random_service')
        self.simulate_electron_daughters = self.GetParameter('SimulateElectronDaughterParticles')
        self.add_dark_doms = self.GetParameter("AddDarkDOMs")

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
            dataclasses.I3Particle.MuMinus,
            dataclasses.I3Particle.MuPlus,
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

        self.cascade_service = self.GetParameter('CascadeService')

        if self.cascade_service is not None:
            self.photonics = Photonics(self.cascade_service)

        self._counter = 0

    def DAQ(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Q-Frame.
        """

        self._counter += 1

        # if self._counter >= 5:
        #     return

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

        # sample pdf
        self.sample_from_pdf(frame, result_tensors, validate_photonics=self.cascade_service is not None)

        if self._counter % 10 == 0:
            self.plot_pdfs(frame, result_tensors, validate_photonics=self.cascade_service is not None)


        # timer after Sampling
        t_3 = timeit.default_timer()

        log_info('Simulation took: {:3.3f}ms'.format((t_3 - t_0) * 1000))
        log_info('\t Gathering Sources: {:3.3f}ms'.format((t_1 - t_0) * 1000))
        log_info('\t Evaluating NN model: {:3.3f}ms'.format((t_2 - t_1) * 1000))
        log_info('\t Sampling Pulses: {:3.3f}ms'.format((t_3 - t_2) * 1000))

        # push frame to next modules
        self.PushFrame(frame)


    def sample_from_pdf(self, frame, result_tensors, validate_photonics=False):
        total_charge = result_tensors['dom_charges'].numpy()
        eps = 1e-7
        mcpe_series_map = frame["I3MCPESeriesMap"]

        loss_event_generator = dataclasses.I3MapKeyDouble()
        loss_event_generator_total = 0

        if validate_photonics:
            loss_photonics = dataclasses.I3MapKeyDouble()
            loss_photonics_total = 0
            geo = frame["I3Geometry"]
            particle = frame["I3MCTree"].get_head()

        for string in range(86):
            for om in range(60):
                omkey = icetray.OMKey(string + 1, om + 1)


                if omkey in mcpe_series_map:
                    pulse_series = mcpe_series_map[omkey]

                    dom_charges_true = np.array([pulse.npe for pulse in pulse_series])
                    pulse_log_pdf_values = np.log([self.model.pdf_per_dom(pulse.time, result_tensors=result_tensors,
                                                    string=string, dom=om) + eps for pulse in pulse_series])

                    dom_charges_pred = total_charge[0, string, om, 0]

                    time_log_likelihood = -dom_charges_true * pulse_log_pdf_values
                    llh_poisson = dom_charges_pred - np.sum(dom_charges_true) * np.log(dom_charges_pred + eps)

                    loss = np.sum(time_log_likelihood) + np.sum(llh_poisson)

                    loss_event_generator_total += loss
                    loss_event_generator[omkey] = loss

                    if validate_photonics:
                        position = geo.omgeo[omkey].position
                        # for pulse in pulse_series:
                        table_pdf, dom_charges_pred = self.photonics.get_pdf(particle, position, [pulse.time for pulse in pulse_series])

                        time_log_likelihood = -dom_charges_true * np.log(np.array(table_pdf) + eps)
                        llh_poisson = dom_charges_pred - np.sum(dom_charges_true) * np.log(dom_charges_pred + eps)

                        loss = np.sum(time_log_likelihood) + np.sum(llh_poisson)

                        loss_photonics_total += loss
                        loss_photonics[omkey] = loss

                elif self.add_dark_doms:
                    expected_charge = total_charge[0, string, om, 0]

                    loss = expected_charge - 0 * np.log(expected_charge + 1e-7)
                    loss_event_generator_total += loss
                    loss_event_generator[omkey] = loss

                    if validate_photonics:
                        position = geo.omgeo[omkey].position
                        expected_charge, _ = self.photonics.get_PE(particle, position)

                        loss = expected_charge - 0 * np.log(expected_charge + 1e-7)
                        loss_photonics_total += loss
                        loss_photonics[omkey] = loss

                else:
                    pass

        frame.Put("EventGeneratorLoss", loss_event_generator)
        frame.Put("EventGeneratorLossTotal", dataclasses.I3Double(loss_event_generator_total))

        if validate_photonics:
            frame.Put("PhotonicsLoss", loss_photonics)
            frame.Put("PhotonicsLossTotal", dataclasses.I3Double(loss_photonics_total))

        # t0 = timeit.default_timer()
        # pdf_values = self.model.pdf(x_times, result_tensors=result_tensors)
        # t1 = timeit.default_timer()
        # print(f"Predict PDF EG: {t1 - t0:.2f}s")

        # assert pdf_values.shape[0] == 1, "PDF result has unexpected shape"
        # pdf_values = pdf_values[0]
        # dt = x_times[1] - x_times[0]
        # pdf_values = pdf_values * dt  # For normalisation

        # pulse_series_map = dataclasses.I3RecoPulseSeriesMap()

        # if self.cascade_service is not None:
        #     pulse_series_map_photonics = dataclasses.I3RecoPulseSeriesMap()

        # total_charge = result_tensors['dom_charges'].numpy()
        # assert total_charge.shape[0] == 1, "Total charge has unexpected shape"
        # total_charge = np.squeeze(total_charge[0])

        # geo = frame["I3Geometry"]

        # # walk through DOMs
        # for string in range(86):
        #     for om in range(60):
        #         om_key = icetray.OMKey(string+1, om+1)
        #         position = geo.omgeo[om_key].position

        #         if self.cascade_service is not None:

        #             table_pdf, table_charge = self.photonics.get_pdf(frame["I3MCTree"].get_head(), position, x_times)
        #             if table_charge > 0.1:
        #                 pulse_series_pht = dataclasses.I3RecoPulseSeries()
        #                 for t, charge in zip(x_times, table_pdf * table_charge):
        #                     pulse = dataclasses.I3RecoPulse()
        #                     if charge <= 0:
        #                         continue
        #                     pulse.time = t
        #                     pulse.charge = charge
        #                     pulse_series_pht.append(pulse)

        #                 pulse_series_map_photonics[om_key] = pulse_series_pht

        #         if total_charge[string, om] > 0.1:
        #             # create pulse series
        #             pulse_series = dataclasses.I3RecoPulseSeries()
        #             charge_distribution = pdf_values[string, om] * total_charge[string, om]
        #             for t, charge in zip(x_times, charge_distribution):
        #                 if charge <= 0:
        #                     continue
        #                 pulse = dataclasses.I3RecoPulse()
        #                 pulse.time = t
        #                 pulse.charge = charge
        #                 pulse_series.append(pulse)

        #             # add to pulse series map
        #             pulse_series_map[om_key] = pulse_series

        # if self.cascade_service is not None:
        #     frame["Photonics_Prediction"] = pulse_series_map_photonics


    def plot_pdfs(self, frame, result_tensors, validate_photonics=False):

        total_charge = result_tensors['dom_charges'].numpy()
        mcpe_series_map = frame["I3MCPESeriesMap"]

        if validate_photonics:
            geo = frame["I3Geometry"]
            particle = frame["I3MCTree"].get_head()

        hom = None
        max_charge = 0

        for omkey, pulses in mcpe_series_map:
            tot_charge = np.sum(p.npe for p in pulses)

            if tot_charge > max_charge:
                max_charge = tot_charge
                hom = omkey

        if hom is None:
            return

        string = hom.string - 1
        om = hom.om - 1

        times = np.arange(0, 1000, 5)
        dt = np.diff(times)[0]
        ctime = times[:-1] + dt / 2

        pulse_log_pdf_values = np.array([self.model.pdf_per_dom(time, result_tensors=result_tensors,
                                         string=string, dom=om) for time in ctime]) * dt
        tot_charge = total_charge[0, string, om, 0]
        fig, ax = plt.subplots()
        ax.hist([p.time for p in mcpe_series_map[hom]], times, weights=[p.npe for p in mcpe_series_map[hom]], alpha=0.5, color="k", label="MC")

        ax.plot(ctime, pulse_log_pdf_values * tot_charge, label="EventGenerator", lw=1)

        if validate_photonics:
            geo = frame["I3Geometry"]
            particle = frame["I3MCTree"].get_head()
            position = geo.omgeo[hom].position
            table_pdf, dom_charges_pred = self.photonics._get_pdf2(particle, position, times)

            ax.plot(ctime, np.array(table_pdf) * dom_charges_pred, label="Photonics", lw=1, ls=":")


        ax.set_xlabel("time / ns")
        ax.set_ylabel("npe")
        ax.legend()
        ax.grid()

        plt.tight_layout()
        plt.savefig(f"pdf_{self._counter}.png")



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



def convert_params_to_particle(params):
    """
    Converts a list of parameters defining a source into a I3Particle

    Parameters
    ----------

    params: list
        List of parameters defining a source: [x, y, z, zenith, azimuth, energy, time]

    Returns
    -------

    cascade: icecube.dataclasses.I3Particle
        Particle describing a cascade
    """
    cascade = dataclasses.I3Particle()

    cascade.pos = dataclasses.I3Position(params[0], params[1], params[2])
    cascade.dir = dataclasses.I3Direction(params[3], params[4])

    cascade.fit_status == dataclasses.I3Particle.NotSet
    cascade.location_type = dataclasses.I3Particle.InIce
    cascade.time = 0

    # set particle type
    cascade.type = dataclasses.I3Particle.ParticleType.EMinus

    # set energy for each particle
    cascade.energy = params[5] * I3Units.GeV

    return cascade


class Photonics(object):
    """
    Little wrapper around photonics_service
    """

    def __init__(self, cascadePhotonicsService=None):

        if cascadePhotonicsService is None:

            if os.environ.get('I3_DATA') is None or os.environ.get('I3_DATA') == "":
                raise ValueError("The environment variable \"I3_DATA\" "
                                 "is not set. Are you in an IceTray environment?")

            table_base = os.path.expandvars('$I3_DATA/photon-tables/splines/ems_spice1_z20_a10.%s.fits')
            cascadePhotonicsService = photonics_service.I3PhotoSplineService(
                table_base % 'abs', table_base % 'prob', timingSigma=0)

        self.cascade_pxs = cascadePhotonicsService


    def _get_PE(self, source, position):
        """ Returns the number of photons and geometrical distance for a source & reciever pair """

        if not isinstance(source, dataclasses.I3Particle):
            source = convert_params_to_particle(source)

        if source.energy == 0:
            return 0, 0

        self.cascade_pxs.SelectModuleCoordinates(*position)
        pes, dist, _ = self.cascade_pxs.SelectSource(photonics_service.PhotonicsSource(source))
        if pes < 0:
            pes = 0

        return pes, dist


    def get_PE(self, sources, position):
        """ Wrapper around self._get_PE to allow providing a list of sources """

        if isinstance(sources, dataclasses.I3Particle):
            return self._get_PE(sources, position)

        elif isinstance(sources, list) and isinstance(sources[0], float):
            return self._get_PE(sources, position)

        elif isinstance(sources, list):
            output = np.array([self._get_PE(source, position) for source in sources])
            pe_sum = np.sum(output[:, 0])
            mask = output[:, 0] > 0
            if not np.any(mask):
                return 0, 0
            return pe_sum, np.amin(output[mask, 1])
        else:
            raise TypeError(f"The type of sources \"{type(sources)}\" is not supported.")


    def _get_pdf(self, source, position, times):

        if not isinstance(source, dataclasses.I3Particle):
            source = convert_params_to_particle(source)

        if source.energy == 0:
            return np.zeros_like(times), 0

        self.cascade_pxs.SelectModuleCoordinates(*position)
        pes, _, t0 = self.cascade_pxs.SelectSource(photonics_service.PhotonicsSource(source))
        if pes <= 0:
            return np.zeros_like(times), 0

        pdf = [self.cascade_pxs.GetProbabilityDensity(t0 + t) for t in times]

        # quantiles = self.cascade_pxs.GetProbabilityQuantiles(times, t0, 0)  # Reading charge distribution (normalized to 1)
        # print(pdf, quantiles)

        return pdf, pes


    def _get_pdf2(self, source, position, times):

        if not isinstance(source, dataclasses.I3Particle):
            source = convert_params_to_particle(source)

        if source.energy == 0:
            return np.zeros_like(times), 0

        self.cascade_pxs.SelectModuleCoordinates(*position)
        pes, _, t0 = self.cascade_pxs.SelectSource(photonics_service.PhotonicsSource(source))
        if pes <= 0:
            return np.zeros_like(times), 0

        quantiles = self.cascade_pxs.GetProbabilityQuantiles(times, t0, 0)  # Reading charge distribution (normalized to 1)

        return quantiles, pes


    def get_pdf(self, sources, position, t):
        if isinstance(sources, dataclasses.I3Particle):
            return self._get_pdf(sources, position, t)

        elif isinstance(sources, list) and isinstance(sources[0], float):
            return self._get_pdf(sources, position, t)

        else:
            pdf_tot = np.zeros_like(t)
            pdfs = []
            nps_tot = 0
            for source in sources:
                pdf, nps = self._self._get_pdf(source, position, t)
                pdfs.append(pdf)
                pdf_tot += pdf * nps
                nps_tot += nps

            pdfs.append(pdf_tot / nps_tot)
            return pdfs
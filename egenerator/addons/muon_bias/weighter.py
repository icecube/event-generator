import os
import numpy as np
import timeit

from icecube import dataclasses, icetray, MuonGun

from egenerator.utils.configurator import ManagerConfigurator

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils import detector
from ic3_labels.labels.utils import muon as mu_utils


class BiasedMuonWeighter(icetray.I3ConditionalModule):
    """Class to bias muon simulation"""

    def __init__(self, context):
        """Class to bias MuonGun simulation.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox("OutBox")

        # Required settings
        self.AddParameter(
            "model_name",
            "A model name that defines which model to "
            "apply. The full path is given by `model_base_dir` +"
            " model_name if `model_base_dir` is provided. "
            "Otherwise `model_name` must define the full "
            "model path.",
        )
        self.AddParameter(
            "bias_function",
            "The bias function. This function computes the "
            "probability that the given event is kept. "
            "This probability should always be in (0, 1]."
            "The input to this function is a dictionary with "
            "the expected charges of losses in each layer "
            "as well as geometrical information.",
        )

        # Optional settings
        self.AddParameter(
            "keep_all_events",
            "If True, all events are kept and the bias results "
            "are only written to the frame",
            False,
        )
        self.AddParameter(
            "verbose_output",
            "If True, additional bias info is written to the " "output key.",
            True,
        )
        self.AddParameter(
            "icecube_hull",
            "The convex hull around IceCube.",
            detector.icecube_hull,
        )
        self.AddParameter(
            "outer_veto_hull",
            "The convex hull around the outer veto layer. This "
            " hull must be completely within the `icecube_hull`",
            detector.icecube_veto_hull_m1,
        )
        self.AddParameter(
            "inner_veto_hull",
            "The convex hull around the inner veto layer. This "
            " hull must be completely in the `outer_veto_layer`",
            detector.icecube_veto_hull_m2,
        )
        self.AddParameter(
            "track_bin_size",
            "The (approximate) size of the track bin along the "
            "muon track. The energy depositions in one such bin "
            " are accumulated. A cascade of that energy is "
            "placed at the middle of the track bin. A smaller "
            "bin size increases spatial resolution, but will "
            "result in longer runtimes and larger memory "
            "consumption. Also note that Event-Generator models "
            "might not be very accurate for very small cascade "
            "energies on the GeV level. It is therefore not "
            "recommended to below 10m.",
            30,
        )
        self.AddParameter(
            "bad_doms_key",
            "The frame key which lists the DOMs which are not "
            " simulated, i.e. because they are bad DOMs.",
            "BadDomsList",
        )
        self.AddParameter(
            "default_values",
            "Not all parameters of the source hypothesis can "
            "be extracted from the I3MCTree. For these "
            "parameters, default values may be defined. The "
            "`default_values` must be a dictionary of the "
            "format `parameter_name`: `value`. `value` may "
            "either be a double or a string. If a string is "
            "provided, it is assumed that an I3Double exists in "
            "the frame under the key as provided by `value`",
            {},
        )
        self.AddParameter(
            "model_base_dir",
            "The base directory in which the model is located."
            " The full path is given by `model_base_dir` + "
            " model_name if `model_base_dir` is provided. "
            "Otherwise `model_name` must define the full "
            "model path.",
            "/data/user/mhuennefeld/exported_models/egenerator",
        )
        self.AddParameter(
            "output_key",
            "The output base key to which bias weights will be " "saved.",
            "BiasedMuonWeighter",
        )
        self.AddParameter(
            "mc_tree_name",
            "The name of the propagated I3MCTree for which the "
            "light yield and measured pulses will be simulated",
            "I3MCTree",
        )
        self.AddParameter(
            "mmc_track_list_name",
            "The name of the MMCTrackList which contains the "
            "MuonGun primary muon.",
            "MMCTrackList",
        )
        self.AddParameter(
            "save_debug_info",
            "If True, the muon intersections with the convex "
            "hulls as well as the expected charge per DOM for "
            "each convex hull will be written to the frame. "
            "This can, for instance, be used to visually debug "
            "the results of the biasing module in steamshovel.",
            False,
        )
        self.AddParameter(
            "num_threads",
            "Number of threads to use for tensorflow. This will "
            "be passed on to tensorflow where it is used to set "
            'the "intra_op_parallelism_threads" and '
            '"inter_op_parallelism_threads" settings. If a '
            "value of zero (default) is provided, the system "
            "uses an appropriate number. Note: when running "
            "this as a job on a cluster you might want to limit "
            '"num_threads" to the amount of allocated CPUs.',
            0,
        )
        self.AddParameter(
            "random_service",
            "The random service or seed to use. If this is an "
            "integer, a numpy random state will be created with "
            "the seed set to `random_service`",
            42,
        )
        self.AddParameter(
            "verbose", "If True, more verbose output is provided.", False
        )

    def Configure(self):
        """Configures Module and loads model from file."""
        self.bias_function = self.GetParameter("bias_function")
        self.model_name = self.GetParameter("model_name")
        self.keep_all_events = self.GetParameter("keep_all_events")
        self.verbose_output = self.GetParameter("verbose_output")
        self.icecube_hull = self.GetParameter("icecube_hull")
        self.outer_veto_hull = self.GetParameter("outer_veto_hull")
        self.inner_veto_hull = self.GetParameter("inner_veto_hull")
        self.track_bin_size = self.GetParameter("track_bin_size")
        self.bad_doms_key = self.GetParameter("bad_doms_key")
        self.default_values = self.GetParameter("default_values")
        self.model_base_dir = self.GetParameter("model_base_dir")
        self.output_key = self.GetParameter("output_key")
        self.mc_tree_name = self.GetParameter("mc_tree_name")
        self.mmc_track_list_name = self.GetParameter("mmc_track_list_name")
        self.save_debug_info = self.GetParameter("save_debug_info")
        self.num_threads = self.GetParameter("num_threads")
        self.random_service = self.GetParameter("random_service")
        self.verbose = self.GetParameter("verbose")

        if isinstance(self.random_service, int):
            self.random_service = np.random.RandomState(self.random_service)

        if self.model_base_dir is not None:
            self.model_dir = os.path.join(self.model_base_dir, self.model_name)
        else:
            self.model_dir = self.model_name

        # ---------------------------------------------------
        # Build and configure SourceManager and extract Model
        # ---------------------------------------------------
        self.manager_configurator = ManagerConfigurator(
            manager_dirs=[self.model_dir],
            num_threads=self.num_threads,
        )
        self.manager = self.manager_configurator.manager

        for model in self.manager.models:
            num_vars, num_total_vars = model.num_variables
            msg = "\nNumber of Model Variables:\n"
            msg += "\tFree: {}\n"
            msg += "\tTotal: {}"
            print(msg.format(num_vars, num_total_vars))

        if len(self.manager.models) > 1:
            raise NotImplementedError(
                "Currently does not support model ensemble."
            )

        self.model = self.manager.models[0]

        # get parameter names that have to be set
        self.param_names = sorted(
            [
                n
                for n in self.model.parameter_names
                if n not in self.default_values
            ]
        )

        # make sure that the only parameters that need to be set are provided
        included_parameters = [
            "azimuth",
            "zenith",
            "energy",
            "time",
            "x",
            "y",
            "z",
            "cascade_azimuth",
            "cascade_energy",
            "cascade_time",
            "cascade_x",
            "cascade_y",
            "cascade_z",
            "cascade_zenith",
        ]
        for name in self.param_names:
            if name == "energy" or name == "cascade_energy":
                self.energy_index = self.model.get_index(name)
            if name not in included_parameters:
                raise KeyError("Unknown parameter name:", name)

        # Create concrete tensorflow function to obtain DOM expectations
        self.param_dtype = getattr(
            np, self.manager.data_trafo.data["tensors"]["x_parameters"].dtype
        )
        self.get_model_tensors = self.manager.get_model_tensors_function()

        # cache zero values
        self.zero_charge_values = np.zeros((86, 60), dtype=self.param_dtype)

    def DetectorStatus(self, frame):
        """Detector Frame Function: build DOM mask.

        Parameters
        ----------
        frame : I3Frame
            The DetectorStatus Frame.
        """
        print("Creating DOM Mask..")
        self.dom_mask = self.get_dom_exclusion_mask(frame)
        self.PushFrame(frame)

    def DAQ(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Q-Frame.
        """

        # start timer
        t_0 = timeit.default_timer()

        # --------------
        # Get input data
        # --------------

        # get muon and track
        muon = self.get_muongun_muon(frame)
        track = self.get_muongun_track(frame, muon)
        assert track is not None, "Could not find MuonGun track!"

        # get distances along the muon track for each layer
        layer_distances = self.get_muon_layer_distances(muon)

        # get cascade input parameter tensor for Event-Generator cascade model
        parameter_tensors = self.get_parameter_tensors(layer_distances, track)

        # timer after source collection
        t_1 = timeit.default_timer()

        # -----------------------------
        # get result tensors from model
        # -----------------------------
        layer_dom_charges = []
        for parameter_tensor in parameter_tensors:
            if parameter_tensor is None:
                layer_dom_charges.append(self.zero_charge_values)
            else:
                result_tensors = self.get_model_tensors(parameter_tensor)
                dom_charges = np.squeeze(
                    np.sum(result_tensors["dom_charges"], axis=0)
                )

                # zero out excluded DOMs
                dom_charges *= self.dom_mask

                layer_dom_charges.append(dom_charges)

        # timer after NN evaluation
        t_2 = timeit.default_timer()

        # ----------------------
        # apply biasing function
        # ----------------------
        track_lengths = []
        layer_energies = []
        entry_x = []
        entry_y = []
        entry_z = []
        exit_z = []
        for i, layer_distance in enumerate(layer_distances):
            if layer_distance is None:
                track_lengths.append(0.0)
                entry_x.append(float("nan"))
                entry_y.append(float("nan"))
                entry_z.append(float("nan"))
                exit_z.append(float("nan"))
                layer_energies.append(float("nan"))
            else:
                track_lengths.append(layer_distance[1] - layer_distance[0])
                entry_pos = muon.pos + layer_distance[0] * muon.dir
                entry_x.append(entry_pos.x)
                entry_y.append(entry_pos.y)
                entry_z.append(entry_pos.z)
                exit_z.append((muon.pos + layer_distance[1] * muon.dir).z)
                layer_energies.append(
                    float(np.sum(parameter_tensors[i][:, self.energy_index]))
                )

        bias_data = {
            "entry_x": entry_x,
            "entry_y": entry_y,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "track_lengths": track_lengths,
            "energy": muon.energy,
            "zenith": muon.dir.zenith,
            "azimuth": muon.dir.azimuth,
            "layer_energies": layer_energies,
            "layer_dom_charges": layer_dom_charges,
        }
        keep_prob = float(self.bias_function(bias_data))
        passed = self.random_service.uniform(0.0, 1.0) <= keep_prob
        assert keep_prob > 0.0 and keep_prob <= 1.0, keep_prob

        bias_weights = dataclasses.I3MapStringDouble(
            {
                "keep_prob": keep_prob,
                "weight_multiplier": 1.0 / keep_prob,
                "passed": float(passed),
            }
        )

        # add more info [flatten out bias_data]
        if self.verbose_output:
            bias_weights["energy"] = muon.energy
            bias_weights["zenith"] = muon.dir.zenith
            bias_weights["azimuth"] = muon.dir.azimuth
            for i in range(len(entry_z)):
                ending = "_{:02d}".format(i)
                bias_weights["entry_x" + ending] = entry_x[i]
                bias_weights["entry_y" + ending] = entry_y[i]
                bias_weights["entry_z" + ending] = entry_z[i]
                bias_weights["exit_z" + ending] = exit_z[i]
                bias_weights["track_length" + ending] = track_lengths[i]
                bias_weights["layer_energy" + ending] = layer_energies[i]
                bias_weights["layer_charge" + ending] = float(
                    np.sum(layer_dom_charges[i])
                )

        # timer after biasing function
        t_3 = timeit.default_timer()

        # -------------------------
        # write Debug Data to frame
        # -------------------------
        if self.save_debug_info:

            for i, distance in enumerate(layer_distances):
                if distance is not None:

                    # Layer Entry
                    pos = dataclasses.I3Position(
                        muon.pos + muon.dir * distance[0]
                    )
                    frame[self.output_key + "_entry_{:03d}".format(i)] = pos

                    # Layer Exit
                    pos = dataclasses.I3Position(
                        muon.pos + muon.dir * distance[1]
                    )
                    frame[self.output_key + "_exit_{:03d}".format(i)] = pos

            for i, dom_charges in enumerate(layer_dom_charges):

                pulse_map = dataclasses.I3RecoPulseSeriesMap()
                for s in range(86):
                    for d in range(60):
                        if dom_charges[s, d] > 0.1:
                            om_key = icetray.OMKey(s + 1, d + 1)
                            pulse = dataclasses.I3RecoPulse()
                            pulse.charge = float(dom_charges[s, d])
                            pulse.time = 0.0
                            pulse_map[om_key] = dataclasses.I3RecoPulseSeries(
                                [pulse]
                            )

                frame[self.output_key + "_layer_{:03d}".format(i)] = pulse_map

        # timer after writing to frame
        t_4 = timeit.default_timer()

        if self.verbose:
            print("Biasing took: {:3.3f}ms".format((t_4 - t_0) * 1000))
            print("\t Gathering Sources: {:3.3f}ms".format((t_1 - t_0) * 1000))
            print(
                "\t Evaluating NN model: {:3.3f}ms".format((t_2 - t_1) * 1000)
            )
            print("\t Applying Bias: {:3.3f}ms".format((t_3 - t_2) * 1000))
            print("\t Writing Results: {:3.3f}ms".format((t_4 - t_3) * 1000))

        # add runtimes
        if self.verbose_output:
            bias_weights["runtime_sources"] = t_1 - t_0
            bias_weights["runtime_nn_model"] = t_2 - t_1
            bias_weights["runtime_bias_func"] = t_3 - t_2
            bias_weights["runtime_total"] = t_4 - t_0
        frame[self.output_key] = bias_weights

        # push frame to next modules
        if self.keep_all_events:
            self.PushFrame(frame)
        else:
            if passed:
                self.PushFrame(frame)

    def get_dom_exclusion_mask(self, frame):
        """Get DOM exclusion mask.

        Parameters
        ----------
        frame : I3Frame
            The current I3Frame.

        Returns
        -------
        array_like
            A mask for the valid DOMs (values: 1.)
            Shape: (86, 60)
        """
        mask = np.ones((86, 60))

        if self.bad_doms_key is not None:
            for omkey in frame[self.bad_doms_key]:
                if omkey.om <= 60:
                    mask[omkey.string - 1, omkey.om - 1] = 0.0

        return mask

    def get_muongun_muon(self, frame):
        """Get primary muon of MuonGun Simulation

        Parameters
        ----------
        frame : I3Frame
            The current I3Frame.

        Returns
        -------
        I3Particle
            The primary muon from the MuonGun simulation.
        """
        primaries = frame[self.mc_tree_name].primaries
        assert (
            len(primaries) == 1
        ), "Expected only one primary particle, got: {}".format(primaries)

        primary = primaries[0]

        if mu_utils.is_muon(primary):
            muon = primary

        else:
            daughters = frame[self.mc_tree_name].get_daughters(primary)
            muon = daughters[0]

            # Perform some safety checks to make sure that this is MuonGun
            assert (
                len(daughters) == 1
            ), "Expected only 1 daughter for MuonGun, but got {}".format(
                daughters
            )
            assert mu_utils.is_muon(muon), "Expected muon but got {}".format(
                muon
            )
        return muon

    def get_muongun_track(self, frame, muon):
        """Get MuonGun track of muon

        Parameters
        ----------
        frame : I3Frame
            The current I3Frame.

        Returns
        -------
        I3MMCTrack
            The MuonGun track of the muon.
        """
        for track in MuonGun.Track.harvest(
            frame[self.mc_tree_name], frame[self.mmc_track_list_name]
        ):
            if track.id == muon.id:
                return track
        return None

    def get_muon_layer_distances(self, muon):
        """Get the distances along the muon track to each of the layers.

        Parameters
        ----------
        muon : I3Particle
            The muon for which to calculate the layer distances.

        Returns
        -------
        array_like
            A list of tuples with the entry and exit distance to each layer.
            If the muon does not enter the layer, this will instead be None.
            Length: n_layer
        """

        # make sure muon starts outside of the outer convex hull
        assert not geometry.point_is_inside(
            self.icecube_hull, muon.pos
        ), "Expected muon vertex outside of outer hull: {}".format(muon.pos)

        # get entry/exit points and muon distances to the convex hulls
        # hulls are sorted from inside to out
        # Note: the following logic requires that these hulls are always
        # completely within the next hull, but there is no check performed
        # to verify this.
        hulls = [self.inner_veto_hull, self.outer_veto_hull, self.icecube_hull]
        entries = [
            mu_utils.get_muon_initial_point_inside(muon, hull)
            for hull in hulls
        ]
        exits = [mu_utils.get_muon_exit_point(muon, hull) for hull in hulls]

        layer_distances = []
        last_dist_entry = None

        for entry, exit in zip(entries, exits):
            if entry is None:
                # muon did not enter this convex hull
                layer_distances.append(None)

            else:

                assert exit is not None, "What enters needs to exit!"

                exit_dist = (muon.pos - exit).magnitude
                entry_dist = (muon.pos - entry).magnitude

                # we only want to collect the distance until the next layer,
                # so we need to correct the exit point, if the muon had hit
                # another layer before
                if last_dist_entry is not None:
                    exit_dist = last_dist_entry

                # set the new last entry point
                last_dist_entry = entry_dist

                layer_distances.append((entry_dist, exit_dist))

        # now append distances from track outside
        if last_dist_entry is None:

            # track never entered the detector, so we will add everything
            # up to the closest approach distance.
            # The idea is that in order for a muon to look like a cascade,
            # the first energy loss
            distance = max(
                0.0,
                mu_utils.get_distance_along_track_to_point(
                    muon.pos, muon.dir, dataclasses.I3Position(0.0, 0.0, 0.0)
                ),
            )
            layer_distances.append((0.0, distance))
        else:
            layer_distances.append((0.0, last_dist_entry))

        return layer_distances

    def get_parameter_tensors(self, layer_distances, track):
        """Get Input Parameters for Event-Generator Model

        Parameters
        ----------
        layer_distances : array_like
            The layer distances from `get_muon_layer_distances`.
        track : I3MMTrack
            The MuonGun track.

        Returns
        -------
        list of array_like
            The input parameter tensors for each layer.
        """
        parameter_tensors = []
        for layer_distance in layer_distances:
            if layer_distance is None:
                parameter_tensors.append(None)

            else:
                # bin distances in layer according to defined resolution
                n_bins = (
                    max(int(np.diff(layer_distance) / self.track_bin_size), 1)
                    + 1
                )
                distances = np.linspace(
                    layer_distance[0], layer_distance[1], n_bins
                )
                energies = [track.get_energy(d) for d in distances[::-1]]
                # energies = [d for d in distances]

                cascade_energies = np.diff(energies)
                cascade_distances = distances[:-1] + 0.5 * np.diff(distances)

                parameters = np.empty(
                    [len(cascade_energies), self.model.num_parameters],
                    dtype=self.param_dtype,
                )

                # Fill default values
                for name, value in self.default_values.items():

                    # assume that a I3Double exists in frame
                    if isinstance(value, str):
                        value = frame[value].value

                    # assign values
                    parameters[:, self.model.get_index(name)] = value

                # now fill cascade parameters (x, y, z, zenith, azimuth, E, t)
                for i in range(len(cascade_distances)):

                    # compute position of cascade
                    pos = track.pos + cascade_distances[i] * track.dir

                    for name in self.param_names:
                        name_s = name.replace("cascade_", "")
                        if name_s in ["x", "y", "z"]:
                            value = getattr(pos, name_s)
                        elif name_s in ["azimuth", "zenith"]:
                            value = getattr(track.dir, name_s)
                        elif name_s == "energy":
                            value = cascade_energies[i]
                        else:
                            value = getattr(track, name_s)

                        index = self.model.get_index(name)
                        parameters[i, index] = value

                parameter_tensors.append(parameters)

        return parameter_tensors

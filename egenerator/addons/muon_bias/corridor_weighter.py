import numpy as np
import timeit

from icecube import dataclasses, icetray

from egenerator.utils.detector import x_coords

from ic3_labels.labels.utils import geometry
from ic3_labels.labels.utils import detector
from ic3_labels.labels.utils import muon as mu_utils


class BiasedMuonCorridorWeighter(icetray.I3ConditionalModule):
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

        self.AddParameter(
            "lower_probability_bound",
            "A lower bound of this value is applied to the computed keep "
            "probability.",
            1e-6,
        )
        self.AddParameter(
            "sigmoid_scaling",
            "A sigmoid function is applied on the minimal distances to active "
            "DOMs to determine the keep probability. The sigmoid can be scaled"
            " and shifted. The result is: sigmoid((x - b) / s) where x is the "
            "distance to the nearest DOM, b is `sigmoid_offset` and s is "
            "`sigmoid_scaling`",
            5,
        )
        self.AddParameter(
            "sigmoid_offset",
            "A sigmoid function is applied on the minimal distances to active "
            "DOMs to determine the keep probability. The sigmoid can be scaled"
            " and shifted. The result is: sigmoid((x - b) / s) where x is the "
            "distance to the nearest DOM, b is `sigmoid_offset` and s is "
            "`sigmoid_scaling`",
            50,
        )
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
            "veto_hull",
            "The convex hull around the veto layer. This "
            " hull must be completely within the `icecube_hull`",
            detector.icecube_veto_hull_m1,
        )
        self.AddParameter(
            "ignore_doms_hull",
            "The convex hull around inner IceCube volume. DOMs located inside "
            "this hull are excluded from the distance calculation and not "
            "considered.",
            detector.icecube_veto_hull_m2,
        )
        self.AddParameter(
            "track_step_size",
            "To compute the minimal distances, points are placed equidistant "
            "along the track segment between the two intersections of the "
            "`icecube_hull` and `veto_hull`. The `track_step_size` defines "
            " the distance between these points.`",
            10,
        )
        self.AddParameter(
            "bad_doms_key",
            "The frame key which lists the DOMs which are not "
            " simulated, i.e. because they are bad DOMs.",
            "BadDomsList",
        )
        self.AddParameter(
            "output_key",
            "The output base key to which bias weights will be saved.",
            "BiasedMuonCorridorWeighter",
        )
        self.AddParameter(
            "mc_tree_name",
            "The name of the propagated I3MCTree for which the "
            "light yield and measured pulses will be simulated",
            "I3MCTree",
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
        self.lower_probability_bound = self.GetParameter(
            "lower_probability_bound"
        )
        self.sigmoid_scaling = self.GetParameter("sigmoid_scaling")
        self.sigmoid_offset = self.GetParameter("sigmoid_offset")
        self.keep_all_events = self.GetParameter("keep_all_events")
        self.verbose_output = self.GetParameter("verbose_output")
        self.icecube_hull = self.GetParameter("icecube_hull")
        self.veto_hull = self.GetParameter("veto_hull")
        self.ignore_doms_hull = self.GetParameter("ignore_doms_hull")
        self.track_step_size = self.GetParameter("track_step_size")
        self.bad_doms_key = self.GetParameter("bad_doms_key")
        self.output_key = self.GetParameter("output_key")
        self.mc_tree_name = self.GetParameter("mc_tree_name")
        self.random_service = self.GetParameter("random_service")
        self.verbose = self.GetParameter("verbose")

        if isinstance(self.random_service, int):
            self.random_service = np.random.RandomState(self.random_service)

        # shape: [5160, 1, 3]
        self.dom_coords = np.expand_dims(
            np.reshape(x_coords, [86 * 60, 3]), axis=1
        )

    def DetectorStatus(self, frame):
        """Detector Frame Function: build DOM mask.

        Parameters
        ----------
        frame : I3Frame
            The DetectorStatus Frame.
        """
        print("Creating DOM Mask..")

        # shape: [5160, 1]
        self.dom_mask = np.reshape(
            self.get_dom_exclusion_mask(frame), [86 * 60]
        )
        self.dom_coords_masked = self.dom_coords[self.dom_mask]
        print("\t DOMs remaining: {}".format(np.sum(self.dom_mask)))
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

        # get muon and track
        muon = self.get_muongun_muon(frame)

        # compute keep probability
        keep_prob, min_outer, min_inner = self.compute_probability(muon)

        keep_prob = float(keep_prob)
        passed = self.random_service.uniform(0.0, 1.0) <= keep_prob
        assert keep_prob > 0.0 and keep_prob <= 1.0, keep_prob

        bias_weights = dataclasses.I3MapStringDouble(
            {
                "keep_prob": keep_prob,
                "weight_multiplier": 1.0 / keep_prob,
                "passed": float(passed),
            }
        )

        # stop timer
        t_1 = timeit.default_timer()

        # add verbose output if desired
        if self.verbose_output:
            bias_weights["min_inner_distance"] = min_inner
            bias_weights["min_outer_distance"] = min_outer
            bias_weights["runtime_total"] = t_1 - t_0

        if self.verbose:
            print("Biasing took: {:3.3f}ms".format((t_1 - t_0) * 1000))

        frame[self.output_key] = bias_weights

        # push frame to next modules
        if self.keep_all_events:
            self.PushFrame(frame)
        else:
            if passed:
                self.PushFrame(frame)

    def compute_probability(self, muon):
        """Compute the keep probability for a muon

        Parameters
        ----------
        muon : I3Particle
            The muon for which to compute the keep probability. This is
            based on the clostest active DOM distance of the entry point
            of the icecube convex hull and of the veto hull.

        Returns
        -------
        float
            The keep probability.
        float
            The minimal distance to an active DOM to the icecube entry point.
        float
            The minimal distance to an active DOM the veto entry point.
        """
        # compute entry points
        entry_veto = mu_utils.get_muon_initial_point_inside(
            muon, self.veto_hull
        )

        if entry_veto is None:

            # muon did not enter veto hull, in this case we reject the muon
            return self.lower_probability_bound, float("nan"), float("nan")

        # we assume that the icecube hull completely covers the veto hull
        # and that the muons are injected outside of the icecube hull.
        # Therefore, if a muon enters the veto hull, it must also have entered
        # the icecube hull
        entry_icecube = mu_utils.get_muon_initial_point_inside(
            muon, self.icecube_hull
        )

        assert entry_icecube is not None, ("Muon should have entered!", muon)

        # now we have both entry points and can compute the DOM distances

        # shape: [1, 3]
        dir_vec = np.array([[muon.dir.x, muon.dir.y, muon.dir.z]])
        pos_inner = np.array([[entry_veto.x, entry_veto.y, entry_veto.z]])
        pos_outer = np.array(
            [[entry_icecube.x, entry_icecube.y, entry_icecube.z]]
        )

        # sample points in between:
        track_length = (entry_icecube - entry_veto).magnitude
        track_distances = np.expand_dims(
            np.arange(0.0, track_length, self.track_step_size), axis=-1
        )

        # go backwards from inner point, to make sure the exact intersection
        # point is included
        # shape: [N, 3] = [1, 3] + [N, 1] * [1, 3]
        track_pos = pos_inner - track_distances * dir_vec

        # Compute DOM distances
        # [N_DOMs, N] = sum([N_DOMs, N, 3]) = sum([N_DOMs, 1, 3] - [N, 3])
        dom_track_distances = np.sqrt(
            np.sum((self.dom_coords_masked - track_pos) ** 2, axis=2)
        )

        # [N_DOMs, 1] = sum([N_DOMs, 1, 3]) = sum([N_DOMs, 1, 3] - [1, 3])
        icecube_distances = np.sqrt(
            np.sum((self.dom_coords_masked - pos_outer) ** 2, axis=2)
        )

        min_veto_dist = np.min(dom_track_distances)
        min_icecube_dist = np.min(icecube_distances)

        prob_inner = self.sigmoid(
            min_veto_dist, s=self.sigmoid_scaling, b=self.sigmoid_offset
        )
        prob_outer = self.sigmoid(
            min_icecube_dist, s=self.sigmoid_scaling, b=self.sigmoid_offset
        )

        keep_prob = np.clip(
            prob_inner * prob_outer, self.lower_probability_bound, 1.0
        )

        return keep_prob, min_icecube_dist, min_veto_dist

    def sigmoid(self, x, s=1.0, b=0.0):
        """Compute Sigmoid Function

        Parameters
        ----------
        x : array_like
            The input data.
        s : float, optional
            The scale parameter of the sigmoid.
        b : float, optional
            The offset parameter of the sigmoid.

        Returns
        -------
        array_like
            Sigmoid results
        """
        xt = (x - b) / s
        return 1.0 / (1 + np.exp(-xt))

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
        mask = np.ones((86, 60), dtype=bool)

        if self.bad_doms_key is not None:
            for omkey in frame[self.bad_doms_key]:
                if omkey.om <= 60:
                    mask[omkey.string - 1, omkey.om - 1] = False

        # mask out DOMs not in relevant veto layer
        if self.ignore_doms_hull is not None:
            for s in range(86):
                for d in range(60):

                    # scale DOM pos slightly inwards towards center to ensure
                    # we are including DOMs which define the convex hull.
                    dom_pos = x_coords[s, d] * 0.99
                    dom_is_inside = geometry.point_is_inside(
                        self.ignore_doms_hull, dom_pos
                    )
                    if dom_is_inside:
                        mask[s, d] = False
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

import os
import numpy as np
import timeit

from icecube import dataclasses, icetray


from ic3_labels.labels.utils import muon as mu_utils


class BiasedMESCHotspotWeighter(icetray.I3ConditionalModule):

    """Class to bias muon simulation
    """

    def __init__(self, context):
        """Class to bias MuonGun simulation.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox('OutBox')

        self.AddParameter(
            'hotspots',
            'A list of hotspot x-y coordinates. These will be used to bias '
            'muon simulation. Muons will be biased to enter the detector '
            'close to these hotspots, but anywhere in z from -500, 500',
            [[470., -140.]])
        self.AddParameter(
            'lower_probability_bound',
            'A lower bound of this value is applied to the computed keep '
            'probability.',
            1e-6)
        self.AddParameter(
            'sigmoid_scaling',
            'A sigmoid function is applied on the minimal distances to active '
            'DOMs to determine the keep probability. The sigmoid can be scaled'
            ' and shifted. The result is: 1 - sigmoid((x - b) / s) where x is '
            'the distance to the nearest DOM, b is `sigmoid_offset` and s is '
            '`sigmoid_scaling`',
            3)
        self.AddParameter(
            'sigmoid_offset',
            'A sigmoid function is applied on the minimal distances to active '
            'DOMs to determine the keep probability. The sigmoid can be scaled'
            ' and shifted. The result is: 1 - sigmoid((x - b) / s) where x is '
            'the distance to the nearest DOM, b is `sigmoid_offset` and s is '
            '`sigmoid_scaling`',
            50)
        self.AddParameter(
            'keep_all_events',
            'If True, all events are kept and the bias results '
            'are only written to the frame',
            False)
        self.AddParameter(
            'verbose_output',
            'If True, additional bias info is written to the '
            'output key.',
            True)
        self.AddParameter(
            'output_key',
            'The output base key to which bias weights will be saved.',
            'BiasedMESCHotspotWeighter')
        self.AddParameter(
            'mc_tree_name',
            'The name of the propagated I3MCTree for which the '
            'light yield and measured pulses will be simulated',
            'I3MCTree')
        self.AddParameter(
            'random_service',
            'The random service or seed to use. If this is an '
            'integer, a numpy random state will be created with '
            'the seed set to `random_service`',
            42)
        self.AddParameter(
            'verbose', 'If True, more verbose output is provided.', False)

    def Configure(self):
        """Configures Module and loads model from file.
        """
        self.hotspots = self.GetParameter('hotspots')
        self.lower_probability_bound = self.GetParameter(
            'lower_probability_bound')
        self.sigmoid_scaling = self.GetParameter('sigmoid_scaling')
        self.sigmoid_offset = self.GetParameter('sigmoid_offset')
        self.keep_all_events = self.GetParameter('keep_all_events')
        self.verbose_output = self.GetParameter('verbose_output')
        self.output_key = self.GetParameter('output_key')
        self.mc_tree_name = self.GetParameter('mc_tree_name')
        self.random_service = self.GetParameter('random_service')
        self.verbose = self.GetParameter('verbose')

        if isinstance(self.random_service, int):
            self.random_service = np.random.RandomState(self.random_service)

        # Hotspot points in x-y
        # Shape: [N, 2]
        self.hotspots = np.array(self.hotspots)

        self.nan_distances = [float('nan') for h in self.hotspots]

    def DAQ(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current Q-Frame.
        """

        # start timer
        t_0 = timeit.default_timer()

        # get muon
        muon = self.get_muongun_muon(frame)

        # compute keep probability
        keep_prob, distances = self.compute_probability(muon)

        keep_prob = float(keep_prob)
        passed = self.random_service.uniform() <= keep_prob
        assert keep_prob > 0. and keep_prob <= 1., keep_prob

        bias_weights = dataclasses.I3MapStringDouble({
            'keep_prob': keep_prob,
            'weight_multiplier': 1. / keep_prob,
            'passed': float(passed),
        })

        # stop timer
        t_1 = timeit.default_timer()

        # add verbose output if desired
        if self.verbose_output:
            for i, distance in enumerate(distances):
                bias_weights['distance_{:06d}'.format(i)] = distance
            bias_weights['runtime_total'] = t_1 - t_0

        if self.verbose:
            print('Biasing took: {:3.3f}ms'.format((t_1 - t_0) * 1000))

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
        if np.abs(muon.pos.z) > 500:
            return self.lower_probability_bound, self.nan_distances

        # shape: [1, 2]
        muon_pos = np.asarray([[muon.pos.x, muon.pos.y]])
        dir_vec = np.asarray([[muon.dir.x, muon.dir.y]])

        # shape: [N, 2] = [N, 2] - [1, 2]
        delta_vec = self.hotspots - muon_pos

        # shape: [N, 1]
        closest_approach_track_length = np.sum(
            delta_vec * dir_vec, axis=1, keepdims=True)

        # discard events that traversed the detector first
        if np.min(closest_approach_track_length) > 500:
            return self.lower_probability_bound, self.nan_distances

        # shape: [N, 2] = [N, 1] * [N, 2] + [1, 2]
        closest_approach = closest_approach_track_length * dir_vec + muon_pos

        # shape: [N]
        distances = np.sqrt(
            np.sum((closest_approach - self.hotspots)**2, axis=1))

        min_dist = np.min(distances)

        keep_prob = 1 - self.sigmoid(
            min_dist, s=self.sigmoid_scaling, b=self.sigmoid_offset)

        keep_prob = np.clip(keep_prob, self.lower_probability_bound, 1.0)

        return keep_prob, distances

    def sigmoid(self, x, s=1., b=0.):
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
        return 1./(1 + np.exp(-xt))

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
        assert len(primaries) == 1, \
            'Expected only one primary particle, got: {}'.format(primaries)

        primary = primaries[0]

        if mu_utils.is_muon(primary):
            muon = primary

        else:
            daughters = frame[self.mc_tree_name].get_daughters(primary)
            muon = daughters[0]

            # Perform some safety checks to make sure that this is MuonGun
            assert len(daughters) == 1, \
                'Expected only 1 daughter for MuonGun, but got {}'.format(
                    daughters)
            assert mu_utils.is_muon(muon), \
                'Expected muon but got {}'.format(muon)
        return muon

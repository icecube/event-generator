import numpy as np
from scipy import ndimage

from icecube import icetray, dataclasses

from egenerator.utils import detector


class CascadeClusterSearchModule(icetray.I3ConditionalModule):
    """Class to create cascade clusters for an event.

    Clusters of charge will be computed within the detector based on the
    highest charge DOMs including their neighbours. The clusters will be
    created such that they must be separated by at least the specified
    minimum distance.
    """

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "pulse_key",
            "The name of the pulse series to use.",
            "InIceDSTPulses",
        )
        self.AddParameter(
            "output_key",
            "The output frame key to which the clusters will be written.",
            "CascadeClusters",
        )
        self.AddParameter(
            "n_clusters", "The maximum number of clusters to compute.", 10
        )
        self.AddParameter(
            "min_dist",
            "The minimum distance [m] that two clusters must be separated by.",
            200,
        )
        self.AddParameter(
            "min_cluster_charge", "The minimum cluster charge [PE].", 3
        )
        self.AddParameter(
            "min_hit_doms",
            "The minimum number of hit DOMs a cluster must have.",
            3,
        )
        self.AddParameter(
            "initial_clusters_particles",
            "A list of I3Particles keys. The vertices of these particles will "
            "be used as initial clusters.",
            None,
        )
        self.AddParameter(
            "add_to_frame",
            "If True, the clusters will be added to the I3Frame as "
            "I3Positions.",
            False,
        )

    def Configure(self):
        """Configure"""
        self._pulse_key = self.GetParameter("pulse_key")
        self._output_key = self.GetParameter("output_key")
        self._n_clusters = self.GetParameter("n_clusters")
        self._min_dist = self.GetParameter("min_dist")
        self._min_cluster_charge = self.GetParameter("min_cluster_charge")
        self._min_hit_doms = self.GetParameter("min_hit_doms")
        self._add_to_frame = self.GetParameter("add_to_frame")
        self._initial_clusters_particles = self.GetParameter(
            "initial_clusters_particles"
        )

    def Physics(self, frame):
        """Compute Clusters

        Parameters
        ----------
        frame : I3Frame
            The current I3frame.
        """

        # get pulses
        pulse_series = frame["InIceDSTPulses"].apply(frame)

        # get initial clusters
        if self._initial_clusters_particles is None:
            initial_clusters = None
        else:
            initial_clusters = []
            for key in self._initial_clusters_particles:
                p = frame[key]
                initial_clusters.append([p.pos.x, p.pos.y, p.pos.z, p.time])

        # compute clusters
        found_clusters, clusters = self.calculate_clusters(
            pulse_series=pulse_series,
            n_clusters=self._n_clusters,
            min_dist=self._min_dist,
            min_cluster_charge=self._min_cluster_charge,
            min_hit_doms=self._min_hit_doms,
            initial_clusters=initial_clusters,
        )

        # compute zenith/azimuth if these all came from one track via PCA
        if found_clusters > 1:
            times = clusters[:found_clusters, 3]
            positions = clusters[:found_clusters, :3]
            mean_pos = np.mean(positions, axis=0)
            centered_pos = positions - mean_pos
            # calculate covariance matrix of centered matrix
            V = np.cov(centered_pos.T)
            # eigendecomposition of covariance matrix
            values, vectors = np.linalg.eig(V)
            if np.iscomplex(vectors).any():
                vectors = vectors.real
            if np.iscomplex(values).any():
                values = values.real

            # choose biggest eigenvector
            dir_vec = vectors[np.argsort(values)[-1]]

            # guess sign of main component vector
            projection = dir_vec.T.dot(centered_pos.T)
            sorted_cluster_times = times[np.argsort(projection)]
            diff_sum = np.sum(np.diff(sorted_cluster_times))
            if diff_sum < 0:
                dir_vec *= -1

            direction = dataclasses.I3Direction(*dir_vec)
            zenith = direction.zenith
            azimuth = direction.azimuth
        else:
            azimuth = float("nan")
            zenith = float("nan")

        output = dataclasses.I3MapStringDouble()
        output["found_clusters"] = found_clusters
        output["azimuth"] = azimuth
        output["zenith"] = zenith

        # add clusters
        for i, cluster in enumerate(clusters):
            prefix = "cluster_{:06d}".format(i)
            output[prefix + "_x"] = cluster[0]
            output[prefix + "_y"] = cluster[1]
            output[prefix + "_z"] = cluster[2]
            output[prefix + "_t"] = cluster[3]

            if self._add_to_frame and np.isfinite(cluster).all():
                pos = dataclasses.I3Position(*cluster[:3])
                frame[self._output_key + "_" + prefix] = pos

        frame[self._output_key] = output

        self.PushFrame(frame)

    @classmethod
    def get_dom_charge_and_times(self, pulse_series):
        """Get DOM total charge and first time

        Parameters
        ----------
        pulse_series : I3RecoPulseSeries
            The pulse series map.

        Returns
        -------
        array_like
            The DOM total charge in hex-array shape.
            Shape: [10, 10, 60, 1]
        array_like
            The DOM average pulse time in hex-array shape.
            Shape: [10, 10, 60, 1]
        """
        times = np.zeros((10, 10, 60, 1))
        charges = np.zeros((10, 10, 60, 1))

        for omkey, pulses in pulse_series.items():

            # skip DeepCore DOMs
            if omkey.string > 78:
                continue

            hex_x, hex_y = detector.get_matrix_indices(omkey.string)

            dom_charges = np.array([p.charge for p in pulses])

            charges[hex_x, hex_y, omkey.om - 1] += np.sum(dom_charges)
            times[hex_x, hex_y, omkey.om - 1] = pulses[0].time

        return charges, times

    @classmethod
    def calculate_clusters(
        self,
        pulse_series,
        n_clusters=10,
        min_dist=50,
        min_cluster_charge=3,
        min_hit_doms=3,
        initial_clusters=None,
    ):
        """Compute charge clusters

        Parameters
        ----------
        pulse_series : I3RecoPulseSeries
            The pulse series map.
        n_clusters : int, optional
            The maximum number of clusters to compute.
        min_dist : int, optional
            The minimum separation distance (in meter) that two clusters have
            to be separated by.
        min_cluster_charge : float, optional
            The minimum charge allowed in a cluster.
        min_hit_doms : int, optional
            The minimum number of hit DOMs allowed in a cluster.
        initial_clusters : array_like, optional
            Initial clusters may be provided. In this case, only additional
            clusters that meet the requirements will be added.
            The coordinates and times of the provided initial clusters.
            Shape: [n_clusters, 4]

        Returns
        -------
        int
            The number of found clusters.
        array_like
            The coordinates and average first times of the computed clusters.
            Shape: [n_clusters, 4]
        """
        eps = 1e-16

        # Get DOM charges and average time: [10, 10, 60, 1]
        charges, times = self.get_dom_charge_and_times(pulse_series)
        hits = (charges > 0).astype(int)

        # sum up neighbouring strings and DOMs within approx 125m
        # but only 3 DOMs up and down, so roughly 50m
        conv_kernel = np.ones((3, 3, 7, 1))
        conv_kernel[0, 0] = 0.0
        conv_kernel[2, 2] = 0.0

        # Convolve to sum up charge in neighbourhood
        charges_conv = ndimage.convolve(
            input=charges,
            weights=conv_kernel,
            mode="constant",
            cval=0.0,
        )

        # Convolve to compute number of hit DOMs per cluster
        hits_conv = ndimage.convolve(
            input=hits,
            weights=conv_kernel,
            mode="constant",
            cval=0.0,
        )

        # shape: [10, 10, 60, 3]
        coords = detector.X_IC79_coords
        coords_time = np.concatenate((coords, times), axis=-1)

        # compute average coordinates and times for potential clusters
        avg_coords_time = ndimage.convolve(
            input=coords_time * charges,
            weights=conv_kernel,
            mode="constant",
            cval=0.0,
        ) / (charges_conv + eps)

        # flatten spatial dimensions
        charges_conv_flat = np.reshape(charges_conv, [10 * 10 * 60])
        hits_conv_flat = np.reshape(hits_conv, [10 * 10 * 60])
        avg_coords_time_flat = np.reshape(avg_coords_time, [10 * 10 * 60, 4])

        # Find n_clusters clusters with a separation of min_dist
        clusters = np.ones((n_clusters, 4)) * np.inf
        sorted_indices = np.argsort(charges_conv_flat)[::-1]

        # add initial clusters if provided
        found_clusters = 0
        if initial_clusters is not None:
            for cluster in initial_clusters:
                assert len(cluster) == 4, cluster
                clusters[found_clusters] = cluster
                found_clusters += 1

        current_charge = np.inf
        index = 0
        while (
            found_clusters < n_clusters and current_charge > min_cluster_charge
        ):

            current_charge = charges_conv_flat[sorted_indices[index]]
            current_n_hits = hits_conv_flat[sorted_indices[index]]
            current_cluster = avg_coords_time_flat[sorted_indices[index]]

            # compute distances to existing clusters
            distances = np.sqrt(
                np.sum((clusters - current_cluster)[:, :3] ** 2, axis=1)
            )
            distances[~np.isfinite(distances)] = np.inf

            if (
                np.min(distances) > min_dist
                and current_charge > min_cluster_charge
                and current_n_hits > min_hit_doms
            ):
                clusters[found_clusters] = current_cluster
                found_clusters += 1
            index += 1

        return found_clusters, clusters

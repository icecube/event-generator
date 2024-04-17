import healpy as hp
import numpy as np
import math


def get_scan_pixels(
    default_nside=2,
    focus_bounds=[5, 15, 30],
    focus_nsides=[32, 16, 8],
    focus_zeniths=[],
    focus_azimuths=[],
):
    """Create a dictionary of pixels for a skyscan

    This will create a dictionary of healpix pixels with focus regions
    that contain higher resolution.

    Parameters
    ----------
    default_nside : int, optional
        The default nside that will be used for the skyscan.
        This nside will be applied everywhere that is outside of the
        focus regions as defined in the `focus_*` parameters.
    focus_bounds : list, optional
        The skyscan will increase resolution in rings around the directions
        provided in `focus_zeniths` and `focus_azimuths`. This parameter
        defines the boundaries [in degrees] at which the next nside is chosen.
        The provided list of floats must be given in ascending order
        and the corresponding nside is given in `focus_nsides`.
    focus_nsides : list, optional
        The skyscan will increase resolution in rings around the directions
        provided in `focus_zeniths` and `focus_azimuths`. This parameter
        defines the nsides for each of these rings. See also `focus_bounds`,
        which defines the distance of these rings.
    focus_zeniths : list, optiona
        A list of zenith values in radians for each of the focus regions.
        Must have same order as `focus_azimuths`.
    focus_azimuths : list, optional
        A list of azimuth values in radians for each of the focus regions.
        Must have same order as `focus_zeniths`.

    Returns
    -------
    dict
        The scan healpix pixel ids provided as a dictionary with the
        format: {nside: [list of ipix]}
    """

    # sanity checks
    assert len(focus_bounds) == len(focus_nsides)
    assert len(focus_azimuths) == len(focus_zeniths)
    assert np.allclose(focus_bounds, np.sort(focus_bounds))

    # This dictionary will hold all required scan pixels.
    # These are given in: {nside: [list of ipix]}
    scan_pixels = {}

    # let's start with the default nside value
    scan_pixels[default_nside] = range(hp.nside2npix(default_nside))

    # now walk through each of the focus regions
    for i, (bound, nside) in enumerate(zip(focus_bounds, focus_nsides)):

        # create an empty set
        ipix = set()

        for zenith, azimuth in zip(focus_zeniths, focus_azimuths):
            ipix.update(
                hp.query_disc(
                    nside=nside,
                    vec=hp.ang2vec(theta=zenith, phi=azimuth),
                    radius=np.deg2rad(bound),
                )
            )

            if i != 0:
                # remove pixels from inner ring that utilizes higher nside
                ipix.difference_update(
                    hp.query_disc(
                        nside=nside,
                        vec=hp.ang2vec(theta=zenith, phi=azimuth),
                        radius=np.deg2rad(focus_bounds[i - 1]),
                    )
                )

        scan_pixels[nside] = list(ipix)

    return scan_pixels


def sparse_to_dense_skymap(nside, indices, values):
    """Fill a skymap from reduced representation of indices and values

    Parameters
    ----------
    nside : int
        The nside of the skymap.
    indices : array_like
        The healpix pixel indices corresponding to each of the
        provided `values`-
    values : array_like
        The values for each healpix pixel corresponding to the
        index as provided in `indices`.

    Returns
    -------
    array_like
        The filled skymap. Missing values are set to 'NaN'.
        Shape: [npix(nside)]
    """
    assert len(indices) == len(values)
    indices = np.asarray(indices)
    values = np.asarray(values)

    skymap = np.empty(hp.nside2npix(nside))
    skymap[:] = float("nan")

    skymap[indices] = values
    return skymap


def combine_skymaps(*skymaps):
    """Combine multi-resolution skymaps

    Parameters
    ----------
    *skymaps
        The healpix skymaps to be combined.
        These must be provided as an array_like object with values
        corresponding to each healpix pixel.

    Returns
    -------
    array_like
        The combined skymap. This skymap will have an nside equal to
        the highest nside of any of the provided skymaps.
    """
    skymaps = np.asarray(skymaps)

    # find out highest nside and sort maps
    nsides = []
    max_nside = 1
    for skymap in skymaps:
        nside = hp.get_nside(skymap)
        nsides.append(nside)
        if nside > max_nside:
            max_nside = nside

    # sort maps
    skymaps_sorted = skymaps[np.argsort(nsides)]

    # now up-scale all skymaps to highest nside
    skymaps_upscaled = []
    for skymap in skymaps_sorted:
        skymaps_upscaled.append(hp.ud_grade(skymap, nside_out=max_nside))

    # now combine everything, overwriting with higher resolution map
    # if values are finite
    combined_map = skymaps_upscaled[0]
    for skymap in skymaps_upscaled[1:]:
        mask_finite = np.isfinite(skymap)
        combined_map[mask_finite] = skymap[mask_finite]

    return combined_map


class SkymapSampler:

    def __init__(self, log_pdf_map, seed=42, replace_nan=None):
        """Class for sampling from skymap PDF

        Parameters
        ----------
        log_pdf_map : array_like
            The skymap given as the logarithm of the PDF at each healpix.
        seed : int, optional
            Random number generator seed.
        replace_nan : None, optional
            If provided, non-finite vlaues in the `log_pdf_map` will
            be replaced with this value.
        """
        self.offset = np.nanmax(log_pdf_map)
        self.log_pdf_map = np.array(log_pdf_map) - self.offset
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self.nside = hp.get_nside(self.log_pdf_map)

        if replace_nan is not None:
            mask = ~np.isfinite(self.log_pdf_map)
            self.log_pdf_map[mask] = replace_nan
        assert np.isfinite(self.log_pdf_map).all(), self.log_pdf_map

        # compute pdf for each pixel
        self._n_order = self._nside2norder()
        self.npix = hp.nside2npix(self.nside)
        self.dir_x_s, self.dir_y_s, self.dir_z_s = hp.pix2vec(
            self.nside, range(self.npix)
        )

        self.neg_llh_values = -self.log_pdf_dir(
            self.dir_x_s, self.dir_y_s, self.dir_z_s
        )

        # sort directions according to neg llh
        sorted_indices = np.argsort(self.neg_llh_values)
        self.dir_x_s = self.dir_x_s[sorted_indices]
        self.dir_y_s = self.dir_y_s[sorted_indices]
        self.dir_z_s = self.dir_z_s[sorted_indices]
        self.neg_llh_values = self.neg_llh_values[sorted_indices]
        self.ipix_list = sorted_indices
        self.ipix_list_list = sorted_indices.tolist()

        # get normalized probabilities and cdf
        prob = np.exp(-self.neg_llh_values)
        assert np.isfinite(prob).all(), (prob, self.neg_llh_values)

        self.prob_values = prob / math.fsum(prob)
        self.cdf_values = np.cumsum(self.prob_values)

        assert np.isfinite(self.cdf_values).all(), self.cdf_values

    def log_pdf_dir(self, dir_x, dir_y, dir_z):
        """Return the log prob for the given direction

        Parameters
        ----------
        dir_x : array_like
            The x-coordinate of the unit direction vector.
        dir_y : array_like
            The y-coordinate of the unit direction vector.
        dir_z : array_like
            The z-coordinate of the unit direction vector.

        Returns
        -------
        array_like
            The log pdf values for each provied direction vector.
        """
        return np.array(
            self.log_pdf_map[
                hp.vec2pix(nside=self.nside, x=dir_x, y=dir_y, z=dir_z)
            ]
        )

    def cdf(self, zenith, azimuth, *args, **kwargs):
        """Computes the CDF.

        Parameters
        ----------
        zenith : array_like
            The zenith angle in radians.
        azimuth : array_like
            The azimuth angle in radians.
        *args
            Additional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array_like
            The CDF evaluated at the provided positions on the sphere.
        """

        # get ipix corresponding to specified directions
        ipix = np.atleast_1d(
            hp.ang2pix(nside=self.nside, theta=zenith, phi=azimuth)
        )

        # figure out where these ipix were sorted to
        sorted_indices = [self.ipix_list_list.index(v) for v in ipix]

        # get cdf values
        return self.cdf_values[sorted_indices]

    def _nside2norder(self):
        """
        Give the HEALpix order for the given HEALpix nside parameter.

        Credit goes to:
            https://git.rwth-aachen.de/astro/astrotools/blob/master/
            astrotools/healpytools.py

        Returns
        -------
        int
            norder of the healpy pixelization

        Raises
        ------
        ValueError
            If nside is not 2**norder.
        """
        norder = np.log2(self.nside)
        if not (norder.is_integer()):
            raise ValueError("Wrong nside number (it is not 2**norder)")
        return int(norder)

    def _sample_from_ipix(
        self, ipix, nest=False, rng=None, pix_converter=hp.pix2vec
    ):
        """
        Sample vectors from a uniform distribution within a HEALpixel.

        Credit goes to
        https://git.rwth-aachen.de/astro/astrotools/blob/master/
        astrotools/healpytools.py

        :param ipix: pixel number(s)
        :param nest: set True in case you work with healpy's nested scheme
        :return: vectors containing events from the pixel(s) specified in ipix

        Parameters
        ----------
        ipix : int, list of int
            Healpy pixels.
        nest : bool, optional
            Set to True in case healpy's nested scheme is used.
        rng : None, optional
            A random number generator. If None is provided the internal
            generator will be used.
        pix_converter : callable, optional
            The ipix to direction or angle converter to use.
            Examples: hp.pix2vec or hp.pix2ang

        Returns
        -------
        np.array, np.array (, np.array)
            The sampled direction vector components if pix_converter is
            set to hp.pix2vec.
            Zenith and azimuth angle in radians if pix_converter is set
            to hp.pix2ang.
        """
        if rng is None:
            rng = self._random_state

        if not nest:
            ipix = hp.ring2nest(self.nside, ipix=ipix)

        n_up = 29 - self._n_order
        i_up = ipix * 4**n_up
        i_up += rng.randint(0, 4**n_up, size=np.size(ipix))
        return pix_converter(nside=2**29, ipix=i_up, nest=True)

    def sample_angles(self, n, rng=None):
        """Sample angles from the distribution

        Parameters
        ----------
        n : int
            Number of samples to generate.
        rng : None, optional
            A random number generator. If None is provided the internal
            generator will be used.

        Returns
        -------
        np.array, np.array
            The sampled zenith and azimuth angles in radians.
        """
        return self._sample_points(n=n, pix_converter=hp.pix2ang, rng=rng)

    def sample_dir(self, n, rng=None):
        """Sample direction vectors from the distribution

        Parameters
        ----------
        n : int
            Number of samples to generate.
        rng : None, optional
            A random number generator. If None is provided the internal
            generator will be used.

        Returns
        -------
        np.array, np.array, np.array
            The sampled direction vector components.
        """
        return self._sample_points(n=n, pix_converter=hp.pix2vec, rng=rng)

    def _sample_points(self, n, pix_converter, rng=None):
        """Sample direction vectors from the distribution

        Parameters
        ----------
        n : int
            Number of samples to generate.
        pix_converter : callable, optional
            The ipix to direction or angle converter to use.
            Examples: hp.pix2vec or hp.pix2ang
        rng : None, optional
            A random number generator. If None is provided the internal
            generator will be used.

        Returns
        -------
        np.array, np.array, np.array
            The sampled direction vector components.
        """
        if rng is None:
            rng = self._random_state

        # sample random healpy pixels given their probability
        indices = np.searchsorted(self.cdf_values, rng.rand(n))
        indices[indices > self.npix - 1] = self.npix - 1

        # get the healpy pixels
        ipix = self.ipix_list[indices]

        # sample points within these pixels
        return self._sample_from_ipix(
            ipix, pix_converter=pix_converter, rng=rng
        )

import healpy as hp
import numpy as np


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
        A list of zenith values for each of the focus regions.
        Must have same order as `focus_azimuths`.
    focus_azimuths : list, optional
        A list of azimuth values for each of the focus regions.
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
            ipix.update(hp.query_disc(
                nside=nside,
                vec=hp.ang2vec(theta=zenith, phi=azimuth),
                radius=np.deg2rad(bound),
            ))

            if i != 0:
                # remove pixels from inner ring that utilizes higher nside
                ipix.difference_update(hp.query_disc(
                    nside=nside,
                    vec=hp.ang2vec(theta=zenith, phi=azimuth),
                    radius=np.deg2rad(focus_bounds[i - 1]),
                ))

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
    skymap[:] = float('nan')

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

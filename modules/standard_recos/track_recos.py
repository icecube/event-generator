from __future__ import print_function, division
import os

from icecube import dataclasses
from icecube import spline_reco
from icecube.icetray.i3logging import log_info, log_warn


def apply_spline_mpe(tray, cfg, name='ApplySplineMPE'):
    """Apply SplineMPE.

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray segment.
    """
    # --------------------------------------------------
    # Apply SplineMPE
    # --------------------------------------------------
    if 'SplineMPE_apply_reco' in cfg and cfg['SplineMPE_apply_reco']:

        settings = cfg['SplineMPE_settings']

        if 'SplineDir' in settings:
            SplineDir = settings.pop('SplineDir')
        else:
            SplineDir = "/cvmfs/icecube.opensciencegrid.org/data/"
            SplineDir += "photon-tables/splines/"
        BareMuAmplitudeSpline = os.path.join(
                                SplineDir, 'InfBareMu_mie_abs_z20a10_V2.fits')
        BareMuTimingSpline = os.path.join(
                                SplineDir, 'InfBareMu_mie_prob_z20a10_V2.fits')

        if 'BareMuAmplitudeSpline' not in settings:
            settings['BareMuAmplitudeSpline'] = BareMuAmplitudeSpline
        if 'BareMuTimingSpline' not in settings:
            settings['BareMuTimingSpline'] = BareMuTimingSpline

        # SplineMPE expects seed to have a certain shape and fit status
        def add_track_seed(frame, seed):
            if name+'_'+seed not in frame:
                particle = dataclasses.I3Particle(frame[seed])
                particle.shape = dataclasses.I3Particle.InfiniteTrack
                particle.fit_status = dataclasses.I3Particle.FitStatus.OK
                frame[name+'_'+seed] = particle

        track_seed_list = []
        for seed in settings.pop('TrackSeedList'):
            tray.Add(add_track_seed, name+'_add_track_seed', seed=seed)
            track_seed_list.append(name+'_'+seed)

        tray.AddSegment(spline_reco.SplineMPE, name,
                        TrackSeedList=track_seed_list,
                        **settings)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Helper functions for icecube specific labels.
'''
import numpy as np
from icecube import dataclasses, MuonGun, simclasses
from icecube.phys_services import I3Calculator

from . import geometry


def is_muon(particle):
    '''Checks if particle is a Muon: MuPlus or MuMinus

    Parameters
    ----------
    particle : I3Particle or I3MMCTrack
        Particle to be checked.

    Returns
    -------
    is_muon : bool
        True if particle is a muon, else false.
    '''
    if particle is None:
        return False
    if isinstance(particle, simclasses.I3MMCTrack):
        particle = particle.particle
    return particle.pdg_encoding in (-13, 13)


def get_muon_time_at_distance(frame, muon, distance):
    '''Function to get the time of a muon at a certain
        distance from the muon vertex.
        Assumes speed = c

    Parameters
    ----------
    frame : I3Frame
        Current frame.
    muon : I3Particle
        Muon.

    distance : float
        Distance.

    Returns
    -------
    time : float
        Time.
    '''
    c = dataclasses.I3Constants.c  # m / nano s
    dtime = distance / c  # in ns
    return muon.time + dtime


def get_muon_time_at_position(frame, muon, position):
    '''Function to get the time of a muon at a certain
        position.

    Parameters
    ----------
    frame : I3Frame
        Current frame.
    muon : I3Particle
        Muon.

    position : I3Position
        Position.

    Returns
    -------
    time : float
        Time.
        If position is before muon vertex or if position is
        not on line defined by the track, this will
        return nan.
        If position is along the track, but after the end
        point of the muon, this will return the time
        the muon would have been at that position.
    '''
    distance = get_distance_along_track_to_point(muon.pos, muon.dir, position)
    if distance < 0 or np.isnan(distance):
        return float('nan')
    return get_muon_time_at_distance(frame, muon, distance)


def get_MuonGun_track(frame, particle_id):
    '''Function to get the MuonGun track corresponding
        to the particle with the id particle_id

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    particle_id : I3ParticleID
        Id of the particle of which the MuonGun
        track should be retrieved from

    Returns
    -------
    track : MuonGun.Track
            Returns None if no corresponding track
            exists
    '''
    for track in MuonGun.Track.harvest(frame['I3MCTree'],
                                       frame['MMCTrackList']):
        if track.id == particle_id:
            return track
    return None


def get_muon_energy_at_position(frame, muon, position):
    '''Function to get the energy of a muon at a certain position.

    Parameters
    ----------
    frame : I3Frame
        Current frame.
    muon : I3Particle
        Muon.

    position : I3Position
        Position.

    Returns
    -------
    energy : float
        Energy.
        If position is before muon vertex or if position is
        not on line defined by the track, this will
        return nan.
        If no corresponding MuonGun.Track can be found to the
        muon, then this will return nan.
        If position is along the track, but after the end
        point of the muon, this will return 0
    '''
    track = get_MuonGun_track(frame, muon.id)
    if track is None:
        # no track exists [BUG?]
        # That means that muon is not in the frame['MMCTrackList']
        # or that it is not correctly harvested from MuonGun
        # Assuming that this is only the case, when the muon
        # is either outside of the inice-volume or that the muon
        # is too low energetic to be listed in the frame['MMCTrackList']
        # Need to fix this ----------------BUG

        # if muon.location_type != muon.InIce: # -------------------------- DEBUG
        #     print 'No track: Muon is not in InIce-Volume:',muon# -------------------------- DEBUG
        # else:# -------------------------- DEBUG
        #     print 'No track: uknown reason', muon# -------------------------- DEBUG

        # if position is close to the vertex, we can just assume
        # that the energy at position is the initial energy
        distance = np.linalg.norm(muon.pos - position)
        if distance < 60:
            # accept if less than 60m difference
            # print 'Assuming energy at',position,'is initial energy' # -------------------------- DEBUG
            return muon.energy
        else:
            if muon.energy < 20 or muon.length < distance:
                # print 'Assuming energy at distance of',distance,'is zero' # -------------------------- DEBUG
                return 0.0
        # print 'Assuming energy at',position,'is nan' # -------------------------- DEBUG
        return float('nan')
    distance = get_distance_along_track_to_point(muon.pos, muon.dir, position)
    if distance < 0 or np.isnan(distance):
        return float('nan')
    return track.get_energy(distance)


def get_muon_energy_at_distance(frame, muon, distance):
    '''Function to get the energy of a muon at a certain
        distance from the muon vertex

    Parameters
    ----------
    frame : I3Frame
        Current frame.
    muon : I3Particle
        Muon.

    distance : float
        Distance.

    Returns
    -------
    energy : float
        Energy.
    '''
    track = get_MuonGun_track(frame, muon.id)
    if track is None:
        # no track exists [BUG?]
        # That means that muon is not in the frame['MMCTrackList']
        # or that it is not correctly harvested from MuonGun
        # Assuming that this is only the case, when the muon
        # is either outside of the inice-volume or that the muon
        # is too low energetic to be listed in the frame['MMCTrackList']
        # Need to fix this ----------------BUG
        # if muon.location_type != muon.InIce: # -------------------------- DEBUG
        #     print 'No track: Muon is not in InIce-Volume:',muon# -------------------------- DEBUG
        # else:# -------------------------- DEBUG
        #     print 'No track: uknown reason', muon# -------------------------- DEBUG

        # if position is close to the vertex, we can just assume
        # that the energy at position is the initial energy
        if distance < 60:
            # accept if less than 60m difference
            # print 'Assuming energy at distance of',distance,'is initial energy' # -------------------------- DEBUG
            return muon.energy
        else:
            if muon.energy < 20 or muon.length < distance:
                # print 'Assuming energy at distance of',distance,'is zero' # -------------------------- DEBUG
                return 0.0
        # print 'Assuming energy at distance of',distance,'is nan' # -------------------------- DEBUG
        return float('nan')
    return track.get_energy(distance)


def get_inf_muon_binned_energy_losses(
                                frame,
                                convex_hull,
                                muon,
                                bin_width=10,
                                extend_boundary=150,
                                include_under_over_flow=False,
                                ):
    '''Function to get binned energy losses along an infinite track.
    The direction and vertex of the given muon is used to create an
    infinite track. This infinte track is then binned in bins of width
    bin_width along the track. The first bin will start at the point where
    the infinite track intersects the extended convex hull. The convex hull
    is defined by convex_hull and can optionally be extended by
    extend_boundary meters.
    The I3MCTree is traversed and the energy losses of the muon are
    accumulated in the corresponding bins along the infinite track.

    If the muon does not hit the convex hull (without extension)
    an empty list is returned.

    Parameters
    ----------
    frame : current frame
        needed to retrieve I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muon : I3Particle
        Muon

    bin_width : float.
        defines width of bins [in meters]
        Energy losses in I3MCtree are binned along the
        track in bins of this width

    extend_boundary : float.
        Extend boundary of convex_hull by this distance [in meters].
        The first bin will be at convex_hull + extend_boundary

    include_under_over_flow : bool.
        If True, an underflow and overflow bin is added for energy
        losses outside of convex_hull + extend_boundary

    Returns
    -------
    binnned_energy_losses : list of float
        Returns a list of energy losses for each bin

    Deleted Parameters
    ------------------
    particle : I3Particle
        primary particle


    Raises
    ------
    ValueError
        Description
    '''

    if muon.pdg_encoding not in (13, -13):  # CC [Muon +/-]
        raise ValueError('Expected muon but got:', muon)

    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    if len(intersection_ts) == 1:
        # vertex is possible exactly on edge of convex hull
        # move vertex slightly by eps
        eps = 1e-4
        muon_pos_shifted = muon.pos + eps * muon.dir
        v_pos = (muon_pos_shifted.x, muon_pos_shifted.y, muon_pos_shifted.z)
        intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    # muon didn't hit convex_hull
    if intersection_ts.size == 0:
        return []

    # muon hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)

    bin_start = muon.pos + min_ts * muon.dir - extend_boundary * muon.dir
    bin_end = muon.pos + max_ts * muon.dir + extend_boundary * muon.dir

    total_length = (bin_end - bin_start).magnitude

    bin_edges = np.arange(0, total_length + bin_width, bin_width)

    # include overflow bin
    bin_edges = np.append(bin_edges, float('inf'))

    # get distance and energy of each loss
    distances = []
    energies = []
    for daughter in frame['I3MCTree'].get_daughters(muon):
        distances.append((daughter.pos - bin_start).magnitude)
        energies.append(daughter.energy)

    # bin energy losses in bins along track
    binnned_energy_losses, _ = np.histogram(distances,
                                            weights=energies,
                                            bins=bin_edges)

    if not include_under_over_flow:
        # remove under and over flow bin
        binnned_energy_losses = binnned_energy_losses[1:-1]

    return binnned_energy_losses


def get_muon_energy_deposited(frame, convex_hull, muon):
    '''Function to get the total energy a muon deposited in the
    volume defined by the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muon : I3Particle
        muon.

    Returns
    -------
    energy : float
        Deposited Energy.
    '''
    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    # muon didn't hit convex_hull
    if intersection_ts.size == 0:
        return 0.0

    # muon hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting track
        return muon.energy - get_muon_energy_at_distance(frame, muon, max_ts)
    if max_ts < 0:
        # muon created after the convex hull
        return 0.0
    return get_muon_energy_at_distance(frame, muon, min_ts) - \
        get_muon_energy_at_distance(frame, muon, max_ts)


def get_cascade_energy_deposited(frame, convex_hull, cascade):
    '''Function to get the total energy a cascade deposited
        in the volume defined by the convex hull. Assumes
        that Cascades lose all of their energy in the convex
        hull if their vertex is in the hull. Otherwise the enrgy
        deposited by a cascade will be 0.
        (naive: There is possibly a better solution to this)

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    cascade : I3Particle
        Cascade.

    Returns
    -------
    energy : float
        Deposited Energy.
    '''
    if geometry.point_is_inside(
            convex_hull,
            (cascade.pos.x, cascade.pos.y, cascade.pos.z)
            ):
        # if inside convex hull: add all of the energy
        return cascade.energy
    else:
        return 0.0


def get_energy_deposited(frame, convex_hull, particle):
    '''Function to get the total energy a particle deposited in the
    volume defined by the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    particle : I3Particle
        Particle.
        (Particle can be of any type: Muon, Cascade, Neutrino...)

    Returns
    -------
    energy : float
        Deposited Energy.
    '''

    raise NotImplementedError


def get_energy_deposited_including_daughters(frame,
                                             convex_hull,
                                             particle,
                                             muongun_primary_neutrino_id=None,
                                             ):
    '''Function to get the total energy a particle or any of its
    daughters deposited in the volume defined by the convex hull.
    Assumes that Cascades lose all of their energy in the convex
    hull if their vetex is in the hull. Otherwise the enrgy
    deposited by a cascade will be 0.
    (naive: There is possibly a better solution to this)

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    particle : I3Particle
        Particle.
        (Particle can be of any type: Muon, Cascade, Neutrino...)

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    energy : float
        Accumulated deposited Energy of the mother particle and
        all of the daughters.
    '''
    energy_loss = 0
    # Calculate EnergyLoss of current particle
    if particle.is_cascade:
        # cascade
        energy_loss = get_cascade_energy_deposited(frame, convex_hull, particle)
    elif particle.is_neutrino \
                or particle.id == muongun_primary_neutrino_id:  # MuonGunFix
        # neutrinos
        for daughter in frame['I3MCTree'].get_daughters(particle):
            energy_loss += get_energy_deposited_including_daughters(frame,
                                                    convex_hull, daughter)
    elif particle.pdg_encoding in (13, -13):  # CC [Muon +/-]
        energy_loss = get_muon_energy_deposited(frame, convex_hull, particle)

    # sanity Checks
    else:
        raise ValueError('Particle of type {} was not handled.'.format(particle.type))
    assert energy_loss >= 0, 'Energy deposited is negativ'
    assert energy_loss <= particle.energy + 1e-8 or particle.id == muongun_primary_neutrino_id, \
            'Deposited E is higher than total E'  # MuonGunFix
    return energy_loss


def get_muon_initial_point_inside(frame, muon, convex_hull):
    ''' Get initial point of the muon inside
        the convex hull. This is either the
        vertex for a starting muon or the
        first intersection of the muon
        and the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    intial_point : I3Position
        Returns None if muon doesn't hit
        convex hull.
    '''
    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    # muon didn't hit convex_hull
    if intersection_ts.size == 0:
        return None

    # muon hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting track
        return muon.pos
    if max_ts < 0:
        # muon created after the convex hull
        return None
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return None
    return muon.pos + min_ts*muon.dir


def get_muon_exit_point(muon, convex_hull):
    ''' Get point of the muon when it exits
        the convex hull. This is either the
        stopping point for a stopping muon or the
        second intersection of the muon
        and the convex hull.

    Parameters
    ----------
    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    intial_point : I3Position
        Returns None if muon doesn't hit
        convex hull.
    '''
    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    # muon didn't hit convex_hull
    if intersection_ts.size == 0:
        return None

    # muon hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return None
    if max_ts < 0:
        # muon created after the convex hull
        return None
    if min_ts > muon.length + 1e-8 and max_ts > muon.length + 1e-8:
        # stopping track
        return muon.pos + muon.length*muon.dir

    return muon.pos + max_ts*muon.dir


def get_mmc_particle(frame, muon):
    ''' Get corresponding I3MMCTrack
        object to the I3Particle muon

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    Returns
    -------
    muon : I3MMCTrack
        I3MMCTrack object of the I3Particle muon
        returns None if no corresponding I3MMCTrack
        object can be found
    '''
    for p in frame['MMCTrackList']:
            if p.particle.id == muon.id:
                return p
    return None


def floats_are_equal(a, b, eps=1e-6):
    ''' Check whether two float are equal
        within precision eps

    Parameters
    ----------
    a : float

    b: float

    eps : float
        Small number to indicate precision, to which
        two floats should be checked for equality

    Returns
    -------
    equal : bool
        True if equal within precision eps,
        otherwise False
    '''
    return abs(a - b) < eps


def get_distance_along_track_to_point(vertex, direction, point):
    ''' Get (signed) distance along a track (defined by position
        and direction) to a point. Negativ distance means the
        point is before the vertex.
        Assumes that point is on the infinite track.

    Parameters
    ----------
    vertex : I3Position
        Vertex (starting point) of the track

    direction : I3Direction
        Direction of the track

    point : I3Position
        Point of which to calculate distance to

    Returns
    -------
    distance : float
        Distance along track to get to point starting
        from the vertex. Negative value indicates
        the point is before the vertex.
        Returns nan if point is not on track
    '''
    distanceX = (point.x - vertex.x) / direction.x
    distanceY = (point.y - vertex.y) / direction.y
    if not floats_are_equal(distanceX, distanceY):
        return float('nan')
    distanceZ = (point.z - vertex.z) / direction.z
    if not floats_are_equal(distanceX, distanceZ):
        return float('nan')
    else:
        return distanceX


def get_particle_closest_approach_to_position(particle,
                                              position):
    ''' Get closest aproach to an I3Position position
        given a particle.

    Parameters
    ----------
    particle : I3Particle.
             Particle of which to compute the
             closest approach to position

    position : I3Position.
             Position to which the closest approach
             of the particle is to be calculated.

    Returns
    -------
    closest_position : I3Position
        I3Position of the point on the track
        that is closest to the position
    '''
    closest_position = I3Calculator.closest_approach_position(
                                                particle, position)
    distance_to_position = get_distance_along_track_to_point(particle.pos,
                                                             particle.dir,
                                                             closest_position
                                                             )
    if distance_to_position < 0:
        # closest_position is before vertex, so set
        # closest_position to vertex
        closest_position = particle.pos
    elif distance_to_position > particle.length:
        # closest_position is after end point of track,
        # so set closest_position to the endpoint
        closest_position = particle.pos + particle.dir * particle.length

    return closest_position


def get_mmc_closest_approach_to_center(frame, mmc_track):
    ''' Get closest aproach to center (0,0,0)
        of a an I3MMCTrack. Uses MMCTrackList center
        position, but checks, if particle is still on
        track. Assumes that the MMCTrackList center
        is somewhere on the line of the track.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    mmc_track : I3MMCTrack

    Returns
    -------
    center : I3Position
        I3Position of the point on the track
        that is closest to the center (0,0,0)
    '''
    center = dataclasses.I3Position(mmc_track.xc, mmc_track.yc, mmc_track.zc)
    distance_to_center = get_distance_along_track_to_point(
                                                        mmc_track.particle.pos,
                                                        mmc_track.particle.dir,
                                                        center)
    if distance_to_center < 0:
        # mmc center is before vertex, so set center to vertex
        center = mmc_track.particle.pos
    elif distance_to_center > mmc_track.particle.length:
        # mmc center is after end point of track,
        # so set center to the endpoint
        center = mmc_track.particle.pos + mmc_track.particle.dir * \
                                          mmc_track.particle.length

    return center


def get_muon_closest_approach_to_center(frame, muon):
    ''' Get closest aproach to center (0,0,0)
        of a muon. Uses MMCTrackList center position,
        but checks, if particle is still on track.
        Assumes that the MMCTrackList center
        is somewhere on the line of the track.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    Returns
    -------
    center : I3Position
        I3Position of the point on the track
        that is closest to the center (0,0,0)
    '''
    if not is_muon(muon):
        raise ValueError('Particle:\n{}\nis not a muon.'.format(muon))

    if 'MMCTrackList' in frame:
        mmc_muon = get_mmc_particle(frame, muon)
    else:
        mmc_muon = None

    if mmc_muon is None:
        # no mmc_muon exists [BUG?]
        # That means that muon is not in the frame['MMCTrackList']
        # Assuming that this is only the case, when the muon
        # is either outside of the inice-volume or that the muon
        # is too low energetic to be listed in the frame['MMCTrackList']

        # if muon.location_type != muon.InIce: # -------------------------- DEBUG
        #     print 'No MMC_Muon: Muon is not in InIce-Volume:',muon# -------------------------- DEBUG
        # else:# -------------------------- DEBUG
        #     print 'No MMC_Muon: uknown reason', muon# -------------------------- DEBUG

        # calculate center pos from muon directly
        length_to_center = -muon.pos*muon.dir
        if length_to_center < 0.0:
            # closest point is before vertex, return vertex
            return muon.pos
        elif length_to_center > muon.length:
            # closest point is after muon end point, return end point
            return muon.pos + muon.length*muon.dir
        else:
            return muon.pos + length_to_center*muon.dir

    assert is_muon(mmc_muon), 'mmc_muon should be a muon'

    return get_mmc_closest_approach_to_center(frame, mmc_muon)


def is_mmc_particle_inside(frame, mmc_particle, convex_hull):
    ''' Find out if mmc particle is inside volume
        defined by the convex hull

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    mmc_particle : I3MMCTrack

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    isInside : bool
        True if mmc muon is inside covex hull
        Returns False, if mmc_particle doesn't exist.
    '''
    if mmc_particle is None:
        return False
    point = get_mmc_closest_approach_to_center(frame, mmc_particle)
    return geometry.point_is_inside(convex_hull,
                                    (point.x, point.y, point.z))


def is_muon_inside(frame, muon, convex_hull):
    ''' Find out if muon is insice volume
        defined by the convex hull

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    muon : I3Particle
        Muon.
    '''
    if not is_muon(muon):
        raise ValueError('Particle:\n{}\nis not a muon.'.format(muon))

    mmc_muon = get_mmc_particle(frame, muon)

    assert is_muon(mmc_muon), 'mmc_muon should be a muon'

    return is_mmc_particle_inside(frame, mmc_muon, convex_hull)


def get_mmc_particles_inside(frame, convex_hull):
    '''Get mmc particles entering the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    particles : list of I3MMCTrack
        Particle mmcTracks that are inside
    '''
    mmc_particles_inside = [m for m in frame['MMCTrackList'] if
                            is_mmc_particle_inside(frame, m, convex_hull)]
    return mmc_particles_inside


def get_muons_inside(frame, convex_hull):
    '''Get muons entering the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    muons : list of I3Particle
        Muons.
    '''
    muons_inside = [m.particle for m in frame['MMCTrackList'] if
                    is_mmc_particle_inside(frame, m, convex_hull)
                    and is_muon(m.particle)]
    return muons_inside


def get_most_energetic_muon_inside(frame, convex_hull,
                                   muons_inside=None):
    '''Get most energetic Muon that is within
    the convex hull. To decide which muon is
    the most energetic, the energy at the initial
    point in the volume is compared. This is either
    the muon vertex or the entry point.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muons_inside : list of I3Particle
        Muons inside the convex hull

    Returns
    -------
    most_energetic_muon : I3Particle
        Returns most energetic muon inside convex hull.
        Returns None, if no muon exists in convex hull.
    '''
    if muons_inside is None:
        muons_inside = get_muons_inside(frame, convex_hull)

    most_energetic_muon = None
    most_energetic_muon_energy = 0

    for m in muons_inside:
        initial_point = get_muon_initial_point_inside(frame, m, convex_hull)
        intial_energy = get_muon_energy_at_position(frame, m, initial_point)
        if intial_energy > most_energetic_muon_energy:
            most_energetic_muon = m
            most_energetic_muon_energy = intial_energy

    return most_energetic_muon


def get_highest_deposit_muon_inside(frame, convex_hull,
                                    muons_inside=None):
    '''Get Muon with the most deposited energy
        that is inside or hits the convex_hull

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muons_inside : list of I3Particle
        Muons inside the convex hull

    Returns
    -------
    muon : I3Particle
        Muon.
    '''
    if muons_inside is None:
        muons_inside = get_muons_inside(frame, convex_hull)

    highest_deposit_muon = None
    highest_deposit = 0

    for m in muons_inside:
        deposit = get_muon_energy_deposited(frame, convex_hull, m)
        if deposit > highest_deposit:
            highest_deposit_muon = m
            highest_deposit = deposit

    return highest_deposit_muon


def get_most_visible_muon_inside(frame, convex_hull,
                                 pulse_map_string='InIcePulses',
                                 max_time_dif=100,
                                 method='noOfPulses'):
    '''Get Muon with the most deposited charge
        inside the detector, e.g. the most visible

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for

    method : string 'charge','noOfPulses'
        'charge' : select muon that deposits the
                    highest sum of charges
        'noOfPulses' : select muon that has the
                    most no of pulses

    Returns
    -------
    muon : I3Particle
        Muon.
    '''

    # get all muons
    muons = [m.particle for m in frame['MMCTrackList'] if is_muon(m.particle)]
    # muons = frame['I3MCTree'].get_filter(
    #                 lambda p: p.pdg_encoding in [13, -13])

    most_visible_muon = None
    if muons:
        if len(muons) == 1:
            # stop here if only one muon is inside
            return muons[0]
        ids = [m.id for m in muons]

        assert (0, 0) not in ids, 'Muon with id (0,0) should not exist'

        # I3ParticleID can't be used as a dict key in older icecube software versions
        # [works in: Version combo.trunk     r152630 (with pyhton 2.7.6)]
        # possible_ids = {m.id : set(get_ids_of_particle_and_daughters(frame,m,[]))
        #                                                          for m in muons}
        # # possible_ids = {(m.id.majorID,m.id.minorID) :
        # #                 { (i.majorID,i.minorID) for i in get_ids_of_particle_and_daughters(frame,m,[]) }
        # #                                         for m in muons}

        # create a dictionary that holds all daughter ids of each muon
        possible_ids = {}
        for m in muons:
            # get a set of daughter ids for muon m
            temp_id_set = { (i.majorID,i.minorID) for i in get_ids_of_particle_and_daughters(frame,m,[]) }

            # fill dictionary
            possible_ids[(m.id.majorID,m.id.minorID)] = temp_id_set

            # sanity check
            assert (0, 0) not in temp_id_set, 'Daughter particle with id (0,0) should not exist'

        counter = {(i.majorID, i.minorID): 0. for i in ids}

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # get candidate keys
        valid_keys = set(frame['I3MCPESeriesMap'].keys())

        # find all pulses resulting from particle or daughters of particle
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}

        for key in shared_keys:
            # mc_pulses = [ p for p in frame['I3MCPESeriesMap'][key]
            #                      if p.ID in ids_set]
            mc_pulses = frame['I3MCPESeriesMap'][key]
            pulses = in_ice_pulses[key]
            if mc_pulses:
                # speed things up:
                # pulses are sorted in time. Therefore we
                # can start from the last match
                last_index = 0
                for pulse in pulses:
                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, p in enumerate(mc_pulses[last_index:]):
                        if abs(pulse.time - p.time) < max_time_dif:
                            last_index = last_index + i
                            for ID in ids:
                                if (p.ID.majorID, p.ID.minorID) in possible_ids[(ID.majorID,ID.minorID)]:
                                    if method == 'charge':
                                        counter[(ID.majorID,ID.minorID)] += pulse.charge
                                    elif method == 'noOfPulses':
                                        counter[(ID.majorID,ID.minorID)] += 1
                            break

        most_visible_muon = muons[0]
        for k, i in enumerate(ids):
            if counter[(i.majorID, i.minorID)] > counter[
                    (most_visible_muon.id.majorID,
                     most_visible_muon.id.minorID)]:
                most_visible_muon = muons[k]
    return most_visible_muon


def get_next_muon_daughter_of_nu(frame, particle,
                                 muongun_primary_neutrino_id=None):
    '''Get the next muon daughter of a neutrino.
        Goes along I3MCTree to find the first
        muon daughter.
        Returns None if none can be found.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    particle : I3Particle

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    muon : I3Particle
        Muon.
        Returns None if no muon daughter
        can be found.
    '''
    if particle.pdg_encoding == 14 or particle.pdg_encoding == -14 \
            or particle.id == muongun_primary_neutrino_id:  # nu # MuonGunFix
        daughters = frame['I3MCTree'].get_daughters(particle)
        if len(daughters) == 0:
            return None
        codes = [p.pdg_encoding for p in daughters]
        if -13 in codes or 13 in codes:  # muon
            # CC Interaction: nu + N -> mu + hadrons
            muons = [p for p in daughters if p.pdg_encoding in (13, -13)]
            assert len(muons) == 1, 'Found more or less than one expected muon.'
            return muons[0]
        elif -14 in codes or 14 in codes:
            # NC Interaction: nu + N -> nu + hadrons
            neutrinos = [p for p in daughters if p.pdg_encoding in (14, -14)]
            assert len(neutrinos) == 1, 'Found more or less than one expected neutrino.'
            return get_next_muon_daughter_of_nu(frame, neutrinos[0])
    else:
        return None


def get_muon_track_length_inside(frame, muon, convex_hull):
    ''' Get the track length of the muon
        inside the convex hull.
        Returns 0 if muon doesn't hit hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree

    muon : I3Particle

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    track_length : float
        Returns 0 if muon doesn't hit
        convex hull.
    '''
    v_pos = (muon.pos.x, muon.pos.y, muon.pos.z)
    v_dir = (muon.dir.x, muon.dir.y, muon.dir.z)
    intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    # muon didn't hit convex_hull
    if intersection_ts.size == 0:
        return 0

    # muon hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)
    if min_ts <= 0 and max_ts >= 0:
        # starting track
        return max_ts
    if max_ts < 0:
        # muon created after the convex hull
        return 0
    if min_ts > muon.length + 1e-8:
        # muon stops before convex hull
        return 0
    return max_ts - min_ts


def get_ids_of_particle_and_daughters(frame, particle, ids):
    '''Get particle ids of particle and all its daughters.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...

    particle : I3Particle
        Any particle type.

    ids : list,
        List in which to save all ids.

    Returns
    -------
    ids: list
        List of all particle ids
    '''
    if particle is None:
        return ids
    ids.append(particle.id)
    daughters = frame['I3MCTree'].get_daughters(particle)
    for daughter in daughters:
        get_ids_of_particle_and_daughters(frame, daughter, ids)
    return ids


def get_pulse_map(frame, particle,
                  pulse_map_string='InIcePulses',
                  max_time_dif=100):
    '''Get map of pulses induced by a specific particle.
       Pulses to be used can be specified through
       pulse_map_string.
        [This is only a guess on which reco Pulses
         could be originated from the particle.
         Naively calculated by looking at time diffs.]

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...

    particle : I3Particle
        Any particle type.

    pulse_map_string : key of pulse map in frame,
        of which the pulses should be computed for

    Returns
    -------
    pulse_map : I3RecoPulseSeriesMap or I3MCPulseSeriesMap
        Map of pulses.

    ----- Better if done over I3RecoPulseSeriesMapMask ----

    '''
    if particle.id.majorID == 0 and particle.id.minorID == 0:
        raise ValueError('Can not get pulse map for particle\
                            with id == (0,0)\n{}'.format(particle))

    particle_pulse_series_map = {}
    if pulse_map_string in frame:
        # make a list of all ids
        ids = get_ids_of_particle_and_daughters(frame, particle, [])
        # older versions of icecube dont have correct hash for I3ParticleID
        # Therefore need tuple of major and minor ID
        # [works directly with I3ParticleID in  Version combo.trunk r152630]
        ids = {(i.majorID, i.minorID) for i in ids}

        assert (0, 0) not in ids, 'Daughter particle with id (0,0) should not exist'

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # get candidate keys
        valid_keys = set(frame['I3MCPESeriesMap'].keys())

        # find all pulses resulting from particle or daughters of particle
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}
        for key in shared_keys:
            mc_pulse_times = [p.time for p in frame['I3MCPESeriesMap'][key]
                              if (p.ID.majorID, p.ID.minorID) in ids]
            particle_in_ice_pulses = []
            if mc_pulse_times:
                # speed things up:
                # pulses are sorted in time. Therefore we
                # can start from the last match
                last_index = 0
                for pulse in in_ice_pulses[key]:
                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, t in enumerate(mc_pulse_times[last_index:]):
                        if abs(pulse.time - t) < max_time_dif:
                            last_index = last_index + i
                            particle_in_ice_pulses.append(pulse)
                            break
            if particle_in_ice_pulses:
                particle_pulse_series_map[key] = particle_in_ice_pulses
    return dataclasses.I3RecoPulseSeriesMap(particle_pulse_series_map)


def get_noise_pulse_map(frame,
                        pulse_map_string='InIcePulses',
                        max_time_dif=100):
    '''Get map of pulses induced by noise.
        [This is only a guess on which reco Pulses
         could be originated from noise.]

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList, I3MCTree, I3MCPE...

    pulse_map_string : key of pulse map in frame,
        of which the mask should be computed for

    Returns
    -------
    pulse_map : I3RecoPulseSeriesMap
        Map of pulses.

    ----- Better if done over I3RecoPulseSeriesMapMask ----

    '''

    noise_pulse_series_map = {}
    if pulse_map_string in frame:
        # pulses with no particle ID are likely from noise
        empty_id = dataclasses.I3ParticleID()

        # get candidate keys
        valid_keys = set(frame['I3MCPESeriesMap'].keys())

        # get pulses defined by pulse_map_string
        in_ice_pulses = frame[pulse_map_string]
        if isinstance(in_ice_pulses, dataclasses.I3RecoPulseSeriesMapMask):
            in_ice_pulses = in_ice_pulses.apply(frame)

        # find all pulses resulting from noise
        shared_keys = {key for key in in_ice_pulses.keys()
                       if key in valid_keys}
        for key in shared_keys:
            mc_pulse_times = [p.time for p in frame['I3MCPESeriesMap'][key]
                              if p.ID == empty_id]
            noise_in_ice_pulses = []
            if mc_pulse_times:
                # speed things up:
                # pulses are sorted in time. Therefore we
                # can start from the last match
                last_index = 0
                for pulse in in_ice_pulses[key]:
                    # accept a pulse if it's within a
                    # max_time_dif-Window of an actual MCPE
                    for i, t in enumerate(mc_pulse_times[last_index:]):
                        if abs(pulse.time - t) < max_time_dif:
                            last_index = last_index + i
                            noise_in_ice_pulses.append(pulse)
                            break
            if noise_in_ice_pulses:
                noise_pulse_series_map[key] = noise_in_ice_pulses
    return dataclasses.I3RecoPulseSeriesMap(noise_pulse_series_map)


def get_muon_information(frame, muon, dom_pos_dict,
                         convex_hull, pulse_map_string='InIcePulses'):
    '''Function to get labels for a muon

    Parameters
    ----------
    muon : I3Particle
        Muon.

    dom_pos_dict : dict
        Dictionary with key of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    '''

    # check if muon exists
    if muon is None:
        # create and return nan Values
        zero = dataclasses.I3Position(0, 0, 0)
        zero_dist_icecube = geometry.distance_to_icecube_hull(zero)
        zero_dist_deepcore = geometry.distance_to_deepcore_hull(zero)

        zero_dict = {
            'NoOfHitDOMs': 0,
            'NoOfPulses': 0,
            'TotalCharge': 0.,

            'COGDistanceToBorder': zero_dist_icecube,
            'COGDistanceToDeepCore': zero_dist_deepcore,
            'COGx': zero.x,
            'COGy': zero.y,
            'COGz': zero.z,

            'EntryDistanceToDeepCore': zero_dist_deepcore,
            'TimeAtEntry': 0.,
            'Entryx': zero.x,
            'Entryy': zero.y,
            'Entryz': zero.z,
            'EnergyEntry': 0.,

            'CenterDistanceToBorder': zero_dist_icecube,
            'CenterDistanceToDeepCore': zero_dist_deepcore,
            'TimeAtCenter': 0.,
            'Centerx': zero.x,
            'Centery': zero.y,
            'Centerz': zero.z,
            'EnergyCenter': 0.,

            'InDetectorTrackLength': 0.,
            'InDetectorEnergyLoss': 0.,

            'Azimuth': 0.,
            'Zenith': 0.,
            'Energy': 0.,
            'TotalTrackLength': 0.,
            'Vertexx': zero.x,
            'Vertexy': zero.y,
            'Vertexz': zero.z,
            'VertexDistanceToBorder': zero_dist_icecube,
            'VertexDistanceToDeepCore': zero_dist_deepcore,
        }
        return zero_dict

    # create empty information dictionary
    info_dict = {}

    # get labels depending on pulse map
    pulse_map = get_pulse_map(frame, muon,
                              pulse_map_string=pulse_map_string)

    NoOfHitDOMs = len(pulse_map.keys())
    NoOfPulses = 0
    TotalCharge = 0.
    COG = np.array([0., 0., 0.])

    if NoOfHitDOMs > 0:
        for key in pulse_map.keys():
            for pulse in pulse_map[key]:
                NoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                COG += pos*pulse.charge
        COG = COG / TotalCharge
    COG = dataclasses.I3Position(*COG)

    COGDistanceToBorder = geometry.distance_to_icecube_hull(COG)
    COGDistanceToDeepCore = geometry.distance_to_deepcore_hull(COG)

    # get entry point labels
    Entry = get_muon_initial_point_inside(frame, muon, convex_hull)
    if Entry:
        TimeAtEntry = get_muon_time_at_position(frame, muon, Entry)
        EntryDistanceToDeepCore = geometry.distance_to_deepcore_hull(Entry)
        EnergyEntry = get_muon_energy_at_position(frame, muon, Entry)
    else:
        # handle missing values
        Entry = dataclasses.I3Position(0, 0, 0)
        TimeAtEntry = 0
        EntryDistanceToDeepCore = 0
        EnergyEntry = 0

    # get center point labels
    Center = get_muon_closest_approach_to_center(frame, muon)
    TimeAtCenter = get_muon_time_at_position(frame, muon, Center)
    CenterDistanceToBorder = geometry.distance_to_icecube_hull(Center)
    CenterDistanceToDeepCore = geometry.distance_to_deepcore_hull(Center)
    EnergyCenter = get_muon_energy_at_position(frame, muon, Center)

    # other labels
    InDetectorTrackLength = get_muon_track_length_inside(frame, muon, convex_hull)
    InDetectorEnergyLoss = get_muon_energy_deposited(frame, convex_hull, muon)

    # add labels to info_dict
    info_dict['NoOfHitDOMs'] = NoOfHitDOMs
    info_dict['NoOfPulses'] = NoOfPulses
    info_dict['TotalCharge'] = TotalCharge

    info_dict['COGDistanceToBorder'] = COGDistanceToBorder
    info_dict['COGDistanceToDeepCore'] = COGDistanceToDeepCore
    info_dict['COGx'] = COG.x
    info_dict['COGy'] = COG.y
    info_dict['COGz'] = COG.z

    info_dict['EntryDistanceToDeepCore'] = EntryDistanceToDeepCore
    info_dict['TimeAtEntry'] = TimeAtEntry
    info_dict['Entryx'] = Entry.x
    info_dict['Entryy'] = Entry.y
    info_dict['Entryz'] = Entry.z
    info_dict['EnergyEntry'] = EnergyEntry

    info_dict['CenterDistanceToBorder'] = CenterDistanceToBorder
    info_dict['CenterDistanceToDeepCore'] = CenterDistanceToDeepCore
    info_dict['TimeAtCenter'] = TimeAtCenter
    info_dict['Centerx'] = Center.x
    info_dict['Centery'] = Center.y
    info_dict['Centerz'] = Center.z
    info_dict['EnergyCenter'] = EnergyCenter

    info_dict['InDetectorTrackLength'] = InDetectorTrackLength
    info_dict['InDetectorEnergyLoss'] = InDetectorEnergyLoss

    info_dict['Azimuth'] = muon.dir.azimuth
    info_dict['Zenith'] = muon.dir.zenith
    info_dict['Energy'] = muon.energy
    info_dict['TotalTrackLength'] = muon.length
    info_dict['Vertexx'] = muon.pos.x
    info_dict['Vertexy'] = muon.pos.y
    info_dict['Vertexz'] = muon.pos.z
    info_dict['VertexDistanceToBorder'] = geometry.distance_to_icecube_hull(
                                                                    muon.pos)
    info_dict['VertexDistanceToDeepCore'] = geometry.distance_to_deepcore_hull(
                                                                    muon.pos)

    return info_dict


def get_primary_information(frame, primary,
                            dom_pos_dict, convex_hull,
                            pulse_map_string='InIcePulses',
                            muongun_primary_neutrino_id=None):
    '''Function to get labels for the primary

    Parameters
    ----------
    frame : frame

    primary : I3Particle
        Primary particle

    dom_pos_dict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    muongun_primary_neutrino_id : I3ParticleID
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along.

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    '''
    info_dict = {}

    # get labels depending on pulse map
    pulse_map = get_pulse_map(frame, primary,
                              pulse_map_string=pulse_map_string)

    NoOfHitDOMs = len(pulse_map.keys())
    NoOfPulses = 0
    TotalCharge = 0.
    COG = np.array([0., 0., 0.])

    if NoOfHitDOMs > 0:
        for key in pulse_map.keys():
            for pulse in pulse_map[key]:
                NoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                COG += pos*pulse.charge
        COG = COG / TotalCharge
    COG = dataclasses.I3Position(*COG)

    COGDistanceToBorder = geometry.distance_to_icecube_hull(COG)
    COGDistanceToDeepCore = geometry.distance_to_deepcore_hull(COG)

    # other labels
    daughters = frame['I3MCTree'].get_daughters(primary)
    codes = [p.pdg_encoding for p in daughters]
    if -13 in codes or 13 in codes:
        # CC Interaction: nu + N -> mu + hadrons
        IsCCInteraction = True
    else:
        # NC Interaction: nu + N -> nu + hadrons
        IsCCInteraction = False

    if geometry.is_in_detector_bounds(daughters[0].pos):
        # Interaction of Primary is in Detector
        IsStartingTrack = True
    else:
        # Interaction outside of Detector
        IsStartingTrack = False
    InDetectorEnergyLoss = get_energy_deposited_including_daughters(
                    frame, convex_hull, primary,
                    muongun_primary_neutrino_id=muongun_primary_neutrino_id)

    # add labels to info_dict
    info_dict['NoOfHitDOMs'] = NoOfHitDOMs
    info_dict['NoOfPulses'] = NoOfPulses
    info_dict['TotalCharge'] = TotalCharge

    info_dict['COGDistanceToBorder'] = COGDistanceToBorder
    info_dict['COGDistanceToDeepCore'] = COGDistanceToDeepCore
    info_dict['COGx'] = COG.x
    info_dict['COGy'] = COG.y
    info_dict['COGz'] = COG.z

    info_dict['Azimuth'] = primary.dir.azimuth
    info_dict['Zenith'] = primary.dir.zenith
    info_dict['Energy'] = primary.energy
    info_dict['InDetectorEnergyLoss'] = InDetectorEnergyLoss
    info_dict['IsCCInteraction'] = IsCCInteraction
    info_dict['IsStartingTrack'] = IsStartingTrack

    return info_dict


def get_misc_information(frame,
                         dom_pos_dict, convex_hull,
                         pulse_map_string='InIcePulses'):
    '''Function to misc labels

    Parameters
    ----------
    frame : frame

    pulse_map_string : key of pulse map in frame,
        of which the mask should be computed for

    dom_pos_dict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    Returns
    -------
    info_dict : dictionary
        Dictionary with all labels
    '''
    info_dict = {}
    in_ice_pulses = frame[pulse_map_string].apply(frame)

    TotalNoOfHitDOMs = len(in_ice_pulses.keys())
    TotalNoOfPulses = 0
    TotalCharge = 0.
    TotalCOG = np.array([0., 0., 0.])
    noise_pulses = []

    if TotalNoOfHitDOMs > 0:
        for key in in_ice_pulses.keys():
            for pulse in in_ice_pulses[key]:
                TotalNoOfPulses += 1
                TotalCharge += pulse.charge
                pos = np.array(dom_pos_dict[(key.string, key.om)])
                TotalCOG += pos*pulse.charge
        TotalCOG = TotalCOG / TotalCharge
    TotalCOG = dataclasses.I3Position(*TotalCOG)

    noise_pulses = get_noise_pulse_map(frame,
                                       pulse_map_string=pulse_map_string)
    NoiseNoOfHitDOMs = len(noise_pulses.keys())
    NoiseNoOfPulses = 0
    NoiseTotalCharge = 0
    for key in noise_pulses.keys():
        for pulse in noise_pulses[key]:
            NoiseNoOfPulses += 1
            NoiseTotalCharge += pulse.charge

    info_dict['TotalNoOfHitDOMs'] = TotalNoOfHitDOMs
    info_dict['TotalNoOfPulses'] = TotalNoOfPulses
    info_dict['TotalCharge'] = TotalCharge
    info_dict['TotalCOGx'] = TotalCOG.x
    info_dict['TotalCOGy'] = TotalCOG.y
    info_dict['TotalCOGz'] = TotalCOG.z

    info_dict['NoiseNoOfHitDOMs'] = NoiseNoOfHitDOMs
    info_dict['NoiseNoOfPulses'] = NoiseNoOfPulses
    info_dict['NoiseTotalCharge'] = NoiseTotalCharge

    info_dict['NoOfPrimaries'] = len(frame['I3MCTree'].primaries)

    return info_dict


def get_labels(frame, convex_hull,
               domPosDict, primary,
               pulse_map_string='InIcePulses',
               is_muongun=False):
    '''Function to get labels for deep learning

    Parameters
    ----------
    frame : frame

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    domPosDict : dict
        Dictionary of form (string,key) : (x,y,z)
        for all DOMs.
        string and key are of type int

    primary : I3Particle
        Primary particle

    pulse_map_string : key of pulse map in frame,
        of which the mask should be computed for

    is_muongun : bool
        In case of a MuonGun dataset, the primary neutrino has
        an unknown type and a pdg_encoding of 0.
        Therefore, the I3ParticleID of the primary needs to
        be passed along to sub-functions.
        Technically, this could be done implicity, by setting
        the primary id. However, this will loosen up sanity
        checks. Therefore, an explicit decision to use MuonGun
        is prefered.

    Returns
    -------
    labels : I3MapStringDouble
        Dictionary with all labels
    '''

    if primary is None:
        raise ValueError('Primary does not exist!')

    assert primary.id is not None, 'MuonGunFix will not work if this is not true'

    # Check if MuonGun dataset
    if is_muongun:
        # This loosens up sanity checks, therefore
        # better to use only if it is really a
        # MuonGun set.
        # Should work for all datasets though,
        # as long as a primary exists

        # make sure it is a MuonGun dataset
        assert primary.type_string == 'unknown', 'Expected unknown, got {}'.format(primary.type_string)
        assert primary.pdg_encoding == 0, 'Expected 0,got {}'.format(primary.pdg_encoding)

        # set primary particle id
        muongun_primary_neutrino_id = primary.id
    else:
        muongun_primary_neutrino_id = None

    # create empty labelDict
    labels = dataclasses.I3MapStringDouble()

    # get misc info
    misc_info = get_misc_information(frame, domPosDict, convex_hull,
                                     pulse_map_string=pulse_map_string)
    labels.update(misc_info)

    muons_inside = get_muons_inside(frame, convex_hull)
    labels['NoOfMuonsInside'] = len(muons_inside)

    # get muons
    mostEnergeticMuon = get_most_energetic_muon_inside(
                                                frame, convex_hull,
                                                muons_inside=muons_inside)
    highestEDepositMuon = get_highest_deposit_muon_inside(
                                                frame, convex_hull,
                                                muons_inside=muons_inside)
    mostVisibleMuon = get_most_visible_muon_inside(
                                            frame, convex_hull,
                                            pulse_map_string=pulse_map_string)
    primaryMuon = get_next_muon_daughter_of_nu(
                    frame, primary,
                    muongun_primary_neutrino_id=muongun_primary_neutrino_id)

    labels['PrimaryMuonExists'] = not (primaryMuon is None)
    labels['VisibleStartingTrack'] = False
    for m in [mostEnergeticMuon, highestEDepositMuon, mostVisibleMuon,
              primaryMuon]:
        if m:
            if geometry.is_in_detector_bounds(m.pos, extend_boundary=60):
                labels['VisibleStartingTrack'] = True

    # get labels for most energetic muon
    mostEnergeticMuon_info = get_muon_information(
                            frame, mostEnergeticMuon, domPosDict, convex_hull,
                            pulse_map_string=pulse_map_string)
    for key in mostEnergeticMuon_info.keys():
        labels['MostEnergeticMuon'+key] = mostEnergeticMuon_info[key]

    # # get labels for highest deposit muon
    # if highestEDepositMuon == mostEnergeticMuon:
    #     highestEDepositMuon_info = mostEnergeticMuon_info
    # else:
    #     highestEDepositMuon_info = get_muon_information(frame,
    #             highestEDepositMuon, domPosDict, convex_hull,
    #             pulse_map_string=pulse_map_string)
    # for key in highestEDepositMuon_info.keys():
    #     labels['HighestEDepositMuon'+key] = highestEDepositMuon_info[key]

    # get labels for most visible muon
    if mostVisibleMuon == mostEnergeticMuon:
        mostVisibleMuon_info = mostEnergeticMuon_info
    else:
        mostVisibleMuon_info = get_muon_information(
                            frame, mostVisibleMuon, domPosDict, convex_hull,
                            pulse_map_string=pulse_map_string)
    for key in mostVisibleMuon_info.keys():
        labels['MostVisibleMuon'+key] = mostVisibleMuon_info[key]

    # get labels for muon from primary
    if primaryMuon == mostEnergeticMuon:
        primaryMuon_info = mostEnergeticMuon_info
    elif primaryMuon == mostVisibleMuon:
        primaryMuon_info = mostVisibleMuon_info
    else:
        primaryMuon_info = get_muon_information(
                                frame, primaryMuon, domPosDict, convex_hull,
                                pulse_map_string=pulse_map_string)
    for key in primaryMuon_info.keys():
        labels['PrimaryMuon'+key] = primaryMuon_info[key]

    # get labels for primary particle
    primary_info = get_primary_information(
                    frame, primary, domPosDict, convex_hull,
                    pulse_map_string=pulse_map_string,
                    muongun_primary_neutrino_id=muongun_primary_neutrino_id)
    for key in primary_info.keys():
        labels['Primary'+key] = primary_info[key]

    return labels


def get_tau_energy_deposited(frame, convex_hull,
                             tau, first_cascade, second_cascade):
    '''Function to get the total energy a tau deposited in the
    volume defined by the convex hull.

    Parameters
    ----------
    frame : current frame
        needed to retrieve MMCTrackList and I3MCTree

    convex_hull : scipy.spatial.ConvexHull
        defining the desired convex volume

    tau : I3Particle
        tau.

    first_cascade : I3Particle
        hadrons from the first tau interaction

    second_cascade : I3Particle
        hadrons from the second tau interaction
    Returns
    -------
    energy : float
        Deposited Energy.
    '''
    if tau is None or first_cascade is None or second_cascade is None:
        return np.nan
    v_pos = (tau.pos.x, tau.pos.y, tau.pos.z)
    v_dir = (tau.dir.x, tau.dir.y, tau.dir.z)
    intersection_ts = geometry.get_intersections(convex_hull, v_pos, v_dir)

    # tau didn't hit convex_hull
    if intersection_ts.size == 0:
        return 0.0

    # tau hit convex_hull:
    #   Expecting two intersections
    #   What happens if track is exactly along edge of hull?
    #   If only one ts: track exactly hit a corner of hull?
    assert len(intersection_ts) == 2, 'Expected exactly 2 intersections'

    min_ts = min(intersection_ts)
    max_ts = max(intersection_ts)

    if min_ts <= 0 and max_ts >= 0:
        # starting track
        dep_en = first_cascade.energy
        # If the tau decays before exiting:
        # - Add the hadronic energy from the second cscd
        #   and the energy lost by the tau in the detector
        if max_ts >= tau.length:
            dep_en += tau.energy - get_muon_energy_at_distance(
                frame, tau, tau.length - 1e-6)
            dep_en += second_cascade.energy

        # If the tau exits the detector before decaying:
        # - Add the energy lost in the detector
        else:
            dep_en += tau.energy - get_muon_energy_at_distance(
                frame, tau, max_ts)

    if max_ts < 0:
        # tau created after the convex hull
        return 0.0

    if min_ts > 0 and max_ts > 0:
        # Incoming Track
        # Dont count the first cascade

        # If the tau decays before exiting
        # Add the second cascade energy
        if max_ts >= tau.length:
            dep_en = get_muon_energy_at_distance(frame, tau, min_ts) - \
                get_muon_energy_at_distance(frame, tau, tau.length - 1e-6)
            dep_en += second_cascade.energy
        # Otherwise just take the energy lost from the tau
        else:
            return get_muon_energy_at_distance(frame, tau, min_ts) - \
                get_muon_energy_at_distance(frame, tau, max_ts)


def get_nutau_interactions(frame):
    mctree = frame['I3MCTree']
    # Find all neutrinos InIce
    in_ice_neutrinos = []
    for part in mctree:
        if part.is_neutrino and part.location_type_string == 'InIce':
            in_ice_neutrinos.append(part)
    # The first one is the primary neutrino
    primary_nu = in_ice_neutrinos[0]

    daughters = mctree.get_daughters(primary_nu)

    tau = None
    first_cascade = None
    second_cascade = None
    for daughter in daughters:
        if daughter.type_string == 'TauMinus' or \
                daughter.type_string == 'TauPlus':
            tau = daughter
        if daughter.type_string == 'Hadrons':
            first_cascade = daughter

    try:
        tau_daughters = mctree.get_daughters(tau)
    except Exception as e:
        return primary_nu, tau, first_cascade, second_cascade
    else:
        for daughter in tau_daughters:
            if daughter.type_string == 'Hadrons':
                second_cascade = daughter
        return primary_nu, tau, first_cascade, second_cascade


def get_tau_labels(frame, convex_hull):
    labels = dataclasses.I3MapStringDouble()

    primary_nu, tau, first_cascade, second_cascade = get_nutau_interactions(
        frame)
    labels['MC_PrimaryInDetectorEnergyLoss'] = get_tau_energy_deposited(
        frame, convex_hull, tau, first_cascade, second_cascade)
    labels['MC_PrimaryEnergy'] = primary_nu.energy

    return labels


def get_interaction_neutrino(frame, primary,
                             convex_hull=None,
                             extend_boundary=0,
                             sanity_check=False):
    """Get the first neutrino daughter of a primary neutrino, that interacted
    inside the convex hull.

    The I3MCTree is traversed to find the first interaction inside the convex
    hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        If None, the IceCube detector volume is assumed.
    extend_boundary : float, optional
        Extend boundary of IceCube detector by this distance [in meters].
        This option is only used if convex_hull is None, e.g. if the IceCube
        detector is used.
    sanity_check : bool, optional
        If true, the neutrino is obtained by two different methods and cross
        checked to see if results match.

    Returns
    -------
    I3Particle, None
        Returns None if no interaction exists inside the convex hull
        Returns the found neutrino as an I3Particle.

    Raises
    ------
    ValueError
        Description
    """

    mctree = frame['I3MCTree']

    # get first in ice neutrino
    nu_in_ice = None
    for p in mctree:
        if p.is_neutrino and p.location_type_string == 'InIce':
            nu_in_ice = p
            break

    if nu_in_ice is not None:

        # check if nu_in_ice has interaction inside convex hull
        daughters = mctree.get_daughters(nu_in_ice)
        assert len(daughters) > 0, 'Expected at least one daughter!'

        # check if point is inside
        if convex_hull is None:
            point_inside = geometry.is_in_detector_bounds(
                                daughters[0].pos,
                                extend_boundary=extend_boundary)
        else:
            point_inside = geometry.point_is_inside(convex_hull,
                                                    (daughters[0].pos.x,
                                                     daughters[0].pos.y,
                                                     daughters[0].pos.z))
        if not point_inside:
            nu_in_ice = None

    # ---------------
    # Sanity Check
    # ---------------
    if sanity_check:
        nu_in_ice_rec = get_interaction_neutrino_rec(
                                    frame=frame,
                                    primary=primary,
                                    convex_hull=convex_hull,
                                    extend_boundary=extend_boundary)

        if nu_in_ice_rec != nu_in_ice:
            if (nu_in_ice_rec is None or nu_in_ice is None or
                    nu_in_ice_rec.id != nu_in_ice.id or
                    nu_in_ice_rec.minor_id != nu_in_ice.minor_id):
                raise ValueError('{} != {}'.format(nu_in_ice_rec, nu_in_ice))
    # ---------------

    return nu_in_ice


def get_interaction_neutrino_rec(frame, primary,
                                 convex_hull=None,
                                 extend_boundary=0):
    """Get the first neutrino daughter of a primary neutrino, that interacted
    inside the convex hull.

    The I3MCTree is traversed to find the first interaction inside the convex
    hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        If None, the IceCube detector volume is assumed.
    extend_boundary : float, optional
        Extend boundary of IceCube detector by this distance [in meters].
        This option is only used if convex_hull is None, e.g. if the IceCube
        detector is used.

    Returns
    -------
    I3Particle, None
        Returns None if no interaction exists inside the convex hull
        Returns the found neutrino as an I3Particle.
    """
    if primary is None:
        return None

    mctree = frame['I3MCTree']

    # traverse I3MCTree until first interaction inside the convex hull is found
    daughters = mctree.get_daughters(primary)

    # No daughters found, so no interaction
    if len(daughters) is 0:
        return None

    # check if interaction point is inside
    if convex_hull is None:
        point_inside = geometry.is_in_detector_bounds(
                            daughters[0].pos, extend_boundary=extend_boundary)
    else:
        point_inside = geometry.point_is_inside(convex_hull,
                                                (daughters[0].pos.x,
                                                 daughters[0].pos.y,
                                                 daughters[0].pos.z))

    if point_inside:
        # interaction is inside the convex hull: neutrino found!
        if primary.is_neutrino:
            return primary
        else:
            return None

    else:
        # daughters are not inside convex hull.
        # Either one of these daughters has secondary partcles which has an
        # interaction inside, or there is no interaction within the convex hull

        interaction_neutrinos = []
        for n in daughters:
            # check if this neutrino has interaction inside the convex hull
            neutrino = get_interaction_neutrino_rec(frame, n,
                                                    convex_hull,
                                                    extend_boundary)
            if neutrino is not None:
                interaction_neutrinos.append(neutrino)

        if len(interaction_neutrinos) is 0:
            # No neutrinos interacting in the convex hull could be found.
            return None

        if len(interaction_neutrinos) > 1:
            print(interaction_neutrinos)
            raise ValueError('Expected only one neutrino to interact!')

        # Found a neutrino that had an interaction inside the convex hull
        return interaction_neutrinos[0]


def get_cascade_of_primary_nu(frame, primary,
                              convex_hull=None,
                              extend_boundary=200):
    """Get cascade of a primary particle.

    The I3MCTree is traversed to find the first interaction inside the convex
    hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        If None, the IceCube detector volume is assumed.
    extend_boundary : float, optional
        Extend boundary of IceCube detector by this distance [in meters].
        This option is only used if convex_hull is None, e.g. if the IceCube
        detector is used.

    Returns
    -------
    I3Particle, None
        Returns None if no cascade interaction exists inside the convex hull
        Returns the found cascade as an I3Particle.
        The returned I3Particle will have the vertex, direction and total
        visible energy of the cascade and the type of the neutrino that
        interacted in the detector. The deoposited energy is defined here
        as the sum of the energies of the daugther particles, unless these are
        neutrinos.
        (Does not account for energy carried away by neutrinos of tau decay)
    """
    neutrino = get_interaction_neutrino(frame, primary,
                                        convex_hull=convex_hull,
                                        extend_boundary=extend_boundary,
                                        sanity_check=True)

    if neutrino is None or not neutrino.is_neutrino:
        return None

    mctree = frame['I3MCTree']

    # traverse I3MCTree until first interaction inside the convex hull is found
    daughters = mctree.get_daughters(neutrino)

    # -----------------------
    # Sanity Checks
    # -----------------------
    assert len(daughters) > 0, 'Expected at least one daughter!'

    # check if point is inside
    if convex_hull is None:
        point_inside = geometry.is_in_detector_bounds(
                            daughters[0].pos, extend_boundary=extend_boundary)
    else:
        point_inside = geometry.point_is_inside(convex_hull,
                                                (daughters[0].pos.x,
                                                 daughters[0].pos.y,
                                                 daughters[0].pos.z))
    assert point_inside, 'Expected interaction to be inside defined volume!'
    # -----------------------

    # interaction is inside the convex hull: cascade found!

    # get cascade
    cascade = dataclasses.I3Particle(neutrino)
    cascade.dir = dataclasses.I3Direction(primary.dir)
    cascade.pos = dataclasses.I3Position(daughters[0].pos)
    cascade.time = daughters[0].time

    # sum up energies for daughters if not neutrinos
    # tau can immediately decay in neutrinos which carry away energy
    # that would not be visible, this is currently not accounted for
    deposited_energy = 0.
    for d in daughters:
        if d.is_neutrino:
            # skip neutrino: the energy is not visible
            continue
        deposited_energy += d.energy

    cascade.energy = deposited_energy
    return cascade


def get_muon_entry_info(frame, muon, convex_hull):
    """Helper function for 'get_cascade_labels'.

    Get muon information for point of entry, or closest approach point,
    if muon does not enter the volume defined by the convex_hull.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    muon : I3Particle
        Muon I3Particle for which to get the entry information.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.

    Returns
    -------
    I3Position, double, double
        Entry Point (or closest approach point)
        Time of entry point (or closest approach point)
        Energy at entry point (or closest approach point)
        Warning: If 'I3MCTree' does not exist in frame, this
                 will instead return the muon energy
    """
    entry = get_muon_initial_point_inside(frame, muon, convex_hull)
    if entry is None:
        # get closest approach point as entry approximation
        entry = get_muon_closest_approach_to_center(frame, muon)
    time = get_muon_time_at_position(frame, muon, entry)

    # Nancy's MuonGun simulation datasets do not have I3MCTree or MMCTrackList
    # included: use muon energy instead
    # This might be an ok approximation, since MuonGun muons are often injected
    # not too far out of detector volume
    if 'I3MCTree' not in frame:
        energy = muon.energy
    else:
        energy = get_muon_energy_at_position(frame, muon, entry)
    return entry, time, energy


def get_cascade_labels(frame, primary, convex_hull, extend_boundary=0):
    """Get cascade labels.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        Will be used to compute muon entry point for an entering muon.
    extend_boundary : float, optional
        Extend boundary of convex_hull by this distance [in meters].

    Returns
    -------
    I3MapStringDouble
        Labels for cascade of primary neutrino.
    """
    labels = dataclasses.I3MapStringDouble()

    labels['PrimaryEnergy'] = primary.energy
    labels['PrimaryAzimuth'] = primary.dir.azimuth
    labels['PrimaryZenith'] = primary.dir.zenith
    labels['PrimaryDirectionX'] = primary.dir.x
    labels['PrimaryDirectionY'] = primary.dir.y
    labels['PrimaryDirectionZ'] = primary.dir.z

    # set pid variables to false per default
    labels['p_starting'] = 0
    labels['p_starting_300m'] = 0
    labels['p_starting_glashow'] = 0
    labels['p_starting_nc'] = 0
    labels['p_starting_cc'] = 0
    labels['p_starting_cc_e'] = 0
    labels['p_starting_cc_mu'] = 0
    labels['p_starting_cc_tau'] = 0
    labels['p_starting_cc_tau_muon_decay'] = 0
    labels['p_starting_cc_tau_double_bang'] = 0

    labels['p_entering'] = 0
    labels['p_entering_muon_single'] = 0
    labels['p_entering_muon_bundle'] = 0

    labels['p_outside_cascade'] = 0

    if primary.is_neutrino:
        # --------------------
        # NuGen dataset
        # --------------------
        mctree = frame['I3MCTree']
        cascade = get_cascade_of_primary_nu(frame, primary,
                                            convex_hull=None,
                                            extend_boundary=extend_boundary)

        # ---------------------------
        # 300m detector boundary test
        # ---------------------------
        cascade_300 = get_cascade_of_primary_nu(frame, primary,
                                                convex_hull=None,
                                                extend_boundary=300)
        if cascade_300 is not None:
            labels['p_starting_300m'] = 1
        # ---------------------------

        if cascade is None:
            # --------------------
            # not a starting event
            # --------------------
            muon = get_next_muon_daughter_of_nu(frame, primary)

            if muon is None:
                # --------------------
                # Cascade interaction outside of defined volume
                # --------------------
                # get first in ice neutrino
                nu_in_ice = None
                for p in mctree:
                    if p.is_neutrino and p.location_type_string == 'InIce':
                        nu_in_ice = p
                        break

                assert nu_in_ice is not None, 'Expected at least one in ice nu'

                daughters = mctree.get_daughters(nu_in_ice)
                visible_energy = 0.
                for d in daughters:
                    if d.is_neutrino:
                        # skip neutrino: the energy is not visible
                        continue
                    visible_energy += d.energy
                assert len(daughters) > 0, 'Expected at least one daughter!'

                labels['p_outside_cascade'] = 1
                labels['VertexX'] = daughters[0].pos.x
                labels['VertexY'] = daughters[0].pos.y
                labels['VertexZ'] = daughters[0].pos.z
                labels['VertexTime'] = daughters[0].time
                labels['EnergyVisible'] = visible_energy
            else:
                # ------------------------------
                # NuMu CC Muon entering detector
                # ------------------------------
                entry, time, energy = get_muon_entry_info(frame, muon,
                                                          convex_hull)
                labels['p_entering'] = 1
                labels['p_entering_muon_single'] = 1
                labels['VertexX'] = entry.x
                labels['VertexY'] = entry.y
                labels['VertexZ'] = entry.z
                labels['VertexTime'] = time
                labels['EnergyVisible'] = energy

        else:
            # --------------------
            # starting NuGen event
            # --------------------
            labels['VertexX'] = cascade.pos.x
            labels['VertexY'] = cascade.pos.y
            labels['VertexZ'] = cascade.pos.z
            labels['VertexTime'] = cascade.time
            labels['EnergyVisible'] = cascade.energy

            labels['p_starting'] = 1

            if frame['I3MCWeightDict']['InteractionType'] == 1:
                    # charged current
                    labels['p_starting_cc'] = 1

                    if cascade.type_string[:3] == 'NuE':
                        # cc NuE
                        labels['p_starting_cc_e'] = 1

                    elif cascade.type_string[:4] == 'NuMu':
                        # cc NuMu
                        labels['p_starting_cc_mu'] = 1

                    elif cascade.type_string[:5] == 'NuTau':
                        # cc Tau
                        labels['p_starting_cc_tau'] = 1

                        nu_tau = get_interaction_neutrino(
                                            frame, primary,
                                            convex_hull=None,
                                            extend_boundary=extend_boundary)
                        tau = [t for t in mctree.get_daughters(nu_tau)
                               if t.type_string in ['TauMinus', 'TauPlus']]

                        assert len(tau) == 1, 'Expected exactly 1 tau!'

                        mu = [m for m in mctree.get_daughters(tau[0])
                              if m.type_string in ['MuMinus', 'MuPlus']]

                        if len(mu) > 0:
                            # tau decays into muon: No Double bang signature!
                            labels['p_starting_cc_tau_muon_decay'] = 1
                        else:
                            # Double bang signature
                            labels['p_starting_cc_tau_double_bang'] = 1

                    else:
                        raise ValueError('Unexpected type: {!r}'.format(
                                                    cascade.type_string))

            elif frame['I3MCWeightDict']['InteractionType'] == 2:
                # neutral current (2)
                labels['p_starting_nc'] = 1

            elif frame['I3MCWeightDict']['InteractionType'] == 3:
                # glashow resonance (3)
                labels['p_starting_glashow'] = 1

            else:
                #  GN -- Genie
                print('InteractionType: {!r}'.format(
                                frame['I3MCWeightDict']['InteractionType']))

    elif is_muon(primary):
        # -----------------------------
        # muon primary: MuonGun dataset
        # -----------------------------
        entry, time, energy = get_muon_entry_info(frame, primary, convex_hull)
        labels['p_entering'] = 1
        labels['p_entering_muon_single'] = 1
        labels['VertexX'] = entry.x
        labels['VertexY'] = entry.y
        labels['VertexZ'] = entry.z
        labels['VertexTime'] = time
        labels['EnergyVisible'] = energy

    else:
        # ---------------------------------------------
        # No neutrino or muon primary: Corsika dataset?
        # ---------------------------------------------
        '''
        if single muon:
            entry, time, energy = get_muon_entry_info(frame, muon, convex_hull)
            labels['p_entering'] = 1
            labels['p_entering_muon_single'] = 1
            labels['VertexX'] = entry.pos.x
            labels['VertexY'] = entry.pos.y
            labels['VertexZ'] = entry.pos.z
            labels['VertexTime'] = time
            labels['EnergyVisible'] = energy
        elif muon bundle:
            muon = get_leading_muon()
            entry, time, energy = get_muon_entry_info(frame, muon, convex_hull)
            labels['p_entering'] = 1
            labels['p_entering_muon_bundle'] = 1
            labels['VertexX'] = entry.pos.x
            labels['VertexY'] = entry.pos.y
            labels['VertexZ'] = entry.pos.z
            labels['VertexTime'] = time
            labels['EnergyVisible'] = energy
        '''
        raise NotImplementedError('Primary type {!r} is not supported'.format(
                                                            primary.type))
    return labels


def get_cascade_parameters(frame, primary, convex_hull, extend_boundary=200):
    """Get cascade parameters.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame needed to retrieve I3MCTree
    primary : I3Particle
        Primary Nu Particle for which the cascade interaction is returned.
    convex_hull : scipy.spatial.ConvexHull, optional
        Defines the desired convex volume.
        Will be used to compute muon entry point for an entering muon.
    extend_boundary : float, optional
        Extend boundary of convex_hull by this distance [in meters].

    Returns
    -------
    I3MapStringDouble
        Cascade parameters of primary neutrino: x, y, z, t, azimuth, zenith, E
    """
    labels = dataclasses.I3MapStringDouble()
    cascade = get_cascade_of_primary_nu(frame, primary,
                                        convex_hull=None,
                                        extend_boundary=extend_boundary)
    if cascade is None:
        # --------------------
        # not a starting event
        # --------------------
        muon = get_next_muon_daughter_of_nu(frame, primary)

        if muon is None:
            # --------------------
            # Cascade interaction outside of defined volume
            # --------------------
            mctree = frame['I3MCTree']
            # get first in ice neutrino
            nu_in_ice = None
            for p in mctree:
                if p.is_neutrino and p.location_type_string == 'InIce':
                    nu_in_ice = p
                    break

            assert nu_in_ice is not None, 'Expected at least one in ice nu'

            daughters = mctree.get_daughters(nu_in_ice)
            visible_energy = 0.
            for d in daughters:
                if d.is_neutrino:
                    # skip neutrino: the energy is not visible
                    continue
                visible_energy += d.energy
            assert len(daughters) > 0, 'Expected at least one daughter!'

            cascade = dataclasses.I3Particle()
            cascade.pos.x = daughters[0].pos.x
            cascade.pos.y = daughters[0].pos.y
            cascade.pos.z = daughters[0].pos.z
            cascade.time = daughters[0].time
            cascade.energy = visible_energy
            cascade.dir = dataclasses.I3Direction(nu_in_ice.dir)
        else:
            # ------------------------------
            # NuMu CC Muon entering detector
            # ------------------------------
            # set cascade parameters to muon entry information
            entry, time, energy = get_muon_entry_info(frame, muon,
                                                      convex_hull)
            cascade = dataclasses.I3Particle()
            cascade.pos.x = entry.x
            cascade.pos.y = entry.y
            cascade.pos.z = entry.z
            cascade.time = time
            cascade.energy = energy
            cascade.dir = dataclasses.I3Direction(muon.dir)

    frame['MCCascade'] = cascade

    labels['cascade_x'] = cascade.pos.x
    labels['cascade_y'] = cascade.pos.y
    labels['cascade_z'] = cascade.pos.z
    labels['cascade_t'] = cascade.time
    labels['cascade_energy'] = cascade.energy
    labels['cascade_azimuth'] = cascade.dir.azimuth
    labels['cascade_zenith'] = cascade.dir.zenith

    return labels

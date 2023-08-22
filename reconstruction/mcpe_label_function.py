import numpy as np
from icecube.icetray import I3Units
from icecube import dataclasses
from icecube.icetray.i3logging import log_warn
from icecube import icetray
from I3Tray import *
from icecube import dataclasses
from os.path import expandvars
import numpy as np
from icecube import dataclasses, icetray
from icecube import simclasses
import os
import sys

from icecube import icetray, dataio

try:
    from icecube.sim_services import I3SimConstants
    ShowerParameters = I3SimConstants.ShowerParameters

except (ImportError, AttributeError) as e:
    print("Can not include 'ShowerParameters' from icecube.sim_services")
    print('Using custom python module instead.')
    from shower_parameters import ShowerParameters

'''
Since MCPEs are on the Q frame I am experiencing some problems 
to create their labels. Here I put the final module and some
other functions that might be needed. Everything or almost everything
is taken from ic3_labels
'''

EMTypes = [
    dataclasses.I3Particle.ParticleType.EMinus,
    dataclasses.I3Particle.ParticleType.EPlus,
    dataclasses.I3Particle.ParticleType.Brems,
    dataclasses.I3Particle.ParticleType.DeltaE,
    dataclasses.I3Particle.ParticleType.PairProd,
    dataclasses.I3Particle.ParticleType.Gamma,
    # Pi0 decays to 2 gammas and produce EM showers
    dataclasses.I3Particle.ParticleType.Pi0,
    dataclasses.I3Particle.ParticleType.EMinus,
    dataclasses.I3Particle.ParticleType.EMinus,
]

HadronTypes = [
    dataclasses.I3Particle.ParticleType.Hadrons,
    dataclasses.I3Particle.ParticleType.Neutron,
    dataclasses.I3Particle.ParticleType.PiPlus,
    dataclasses.I3Particle.ParticleType.PiMinus,
    dataclasses.I3Particle.ParticleType.K0_Long,
    dataclasses.I3Particle.ParticleType.KPlus,
    dataclasses.I3Particle.ParticleType.KMinus,
    dataclasses.I3Particle.ParticleType.PPlus,
    dataclasses.I3Particle.ParticleType.PMinus,
    dataclasses.I3Particle.ParticleType.K0_Short,

    dataclasses.I3Particle.ParticleType.Eta,
    dataclasses.I3Particle.ParticleType.Lambda,
    dataclasses.I3Particle.ParticleType.SigmaPlus,
    dataclasses.I3Particle.ParticleType.Sigma0,
    dataclasses.I3Particle.ParticleType.SigmaMinus,
    dataclasses.I3Particle.ParticleType.Xi0,
    dataclasses.I3Particle.ParticleType.XiMinus,
    dataclasses.I3Particle.ParticleType.OmegaMinus,
    dataclasses.I3Particle.ParticleType.NeutronBar,
    dataclasses.I3Particle.ParticleType.LambdaBar,
    dataclasses.I3Particle.ParticleType.SigmaMinusBar,
    dataclasses.I3Particle.ParticleType.Sigma0Bar,
    dataclasses.I3Particle.ParticleType.SigmaPlusBar,
    dataclasses.I3Particle.ParticleType.Xi0Bar,
    dataclasses.I3Particle.ParticleType.XiPlusBar,
    dataclasses.I3Particle.ParticleType.OmegaPlusBar,
    dataclasses.I3Particle.ParticleType.DPlus,
    dataclasses.I3Particle.ParticleType.DMinus,
    dataclasses.I3Particle.ParticleType.D0,
    dataclasses.I3Particle.ParticleType.D0Bar,
    dataclasses.I3Particle.ParticleType.DsPlus,
    dataclasses.I3Particle.ParticleType.DsMinusBar,
    dataclasses.I3Particle.ParticleType.LambdacPlus,
    dataclasses.I3Particle.ParticleType.WPlus,
    dataclasses.I3Particle.ParticleType.WMinus,
    dataclasses.I3Particle.ParticleType.Z0,
    dataclasses.I3Particle.ParticleType.NuclInt,
]





def convert_to_em_equivalent(cascade):
    """Get electro-magnetic (EM) equivalent energy of a given cascade.
    Note: this is only an average expected EM equivalent. Possible existing
    daughter particles in the I3MCTree are not taken into account!
    Parameters
    ----------
    cascade : I3Particle
        The cascade.
    Returns
    -------
    float
        The EM equivalent energy of the given cascade.
    """
    # scale energy of cascade to EM equivalent
    em_scale = ShowerParameters(cascade.type, cascade.energy).emScale
    return cascade.energy * em_scale

def get_cascade_em_equivalent(mctree, cascade_primary):
    """Get electro-magnetic (EM) equivalent energy of a given cascade.
    Recursively walks through daughters of a provided cascade primary and
    collects EM equivalent energy.
    Note: muons and taus are added completely as EM equivalent energy!
    This disregards the fact that a tau can for instance decay and the neutrino
    may carry away a big portion of energy
    Parameters
    ----------
    mctree : I3MCTree
        The current I3MCTree
    cascade_primary : I3Particle
        The cascade primary particle.
    Returns
    -------
    float
        The total EM equivalent energy of the given cascade.
    float
        The total EM equivalent energy of the EM cascade.
    float
        The total EM equivalent energy of the hadronic cascade.
    float
        The total EM equivalent energy in muons and taus (tracks).
    """

    daughters = mctree.get_daughters(cascade_primary)

    # ---------------------------------
    # stopping conditions for recursion
    # ---------------------------------
    if (cascade_primary.location_type !=
            dataclasses.I3Particle.LocationType.InIce):
        # skip particles that are way outside of the detector volume
        return 0., 0., 0., 0.

    # check if we have a muon or tau
    if cascade_primary.type in [
            dataclasses.I3Particle.ParticleType.MuMinus,
            dataclasses.I3Particle.ParticleType.MuPlus,
            dataclasses.I3Particle.ParticleType.TauMinus,
            dataclasses.I3Particle.ParticleType.TauPlus,
            ]:
        # For simplicity we will assume that all energy is deposited.
        # Note: this is wrong for instance for taus that decay where the
        # neutrino will carry away a large fraction of the energy
        return cascade_primary.energy, 0., 0., cascade_primary.energy

    if len(daughters) == 0:
        if cascade_primary.is_neutrino:
            # skip neutrino: the energy is not visible
            return 0., 0., 0., 0.

        else:

            # get EM equivalent energy
            energy = convert_to_em_equivalent(cascade_primary)

            # EM energy
            if cascade_primary.type in EMTypes:
                return energy, energy, 0., 0.

            # Hadronic energy
            elif cascade_primary.type in HadronTypes:
                return energy, 0., energy, 0.

            else:
                log_warn('Unknown particle type: {}. Assuming hadron!'.format(
                    cascade_primary.type))
                return energy, 0., energy, 0.

    # ---------------------------------

    # collect energy from hadronic, em, and tracks
    energy_total = 0.
    energy_em = 0.
    energy_hadron = 0.
    energy_track = 0.

    # recursively walk through daughters and accumulate energy
    for daugther in daughters:

        # get energy depositions of particle and its daughters
        e_total, e_em, e_hadron, e_track = get_cascade_em_equivalent(
            mctree, daugther)

        # CMC splits up hadronic cascades to segments of electrons
        # In other words: if the cascade primary is a hadron, the daughter
        # particles need to contribute to the hadronic component of the shower
        if cascade_primary.type in HadronTypes:
            e_hadron += e_em
            e_em = 0

        # accumulate energies
        energy_total += e_total
        energy_em += e_em
        energy_hadron += e_hadron
        energy_track += e_track

    return energy_total, energy_em, energy_hadron, energy_track

def get_interaction_neutrino(frame):

    mctree = frame['I3MCTree']
    primary = frame['MCPrimary'] # I think it should be the same as NuGPrimary, but just in case

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

    # ---------------
    
    neutrino = nu_in_ice
    
    cascade = dataclasses.I3Particle(neutrino)
    cascade.shape = dataclasses.I3Particle.ParticleShape.Cascade
    cascade.dir = dataclasses.I3Direction(primary.dir)
    cascade.pos = dataclasses.I3Position(daughters[0].pos)
    cascade.time = daughters[0].time
    e_total, e_em, e_hadron, e_track = get_cascade_em_equivalent(
        mctree, neutrino)
    cascade.energy = e_total
    
    
    labels = dataclasses.I3MapStringDouble()
    
    labels['cascade_x'] = cascade.pos.x
    labels['cascade_y'] = cascade.pos.y
    labels['cascade_z'] = cascade.pos.z
    labels['cascade_t'] = cascade.time
    labels['cascade_energy'] = cascade.energy
    labels['cascade_azimuth'] = cascade.dir.azimuth
    labels['cascade_zenith'] = cascade.dir.zenith
    
    
    return cascade


class mcpe_label_module(icetray.I3Module):
    '''
     This class is to push the LabelsDeepLearning key to the frame
     so that it can be written into the hdf5 files
    '''
    def __init__(self, context):

        icetray.I3Module.__init__(self, context)

    def Physics(self,frame):

        cascade = get_interaction_neutrino(frame=frame)
        frame['LabelsDeepLearning'] = cascade
        self.PushFrame(frame)

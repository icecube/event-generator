import os
from icecube.weighting.weighting import from_simprod
from icecube.weighting import fluxes, SimprodNormalizations

from I3Tray import *
from icecube import dataio, tableio, portia
from icecube import icetray, dataclasses, simclasses, dataio, phys_services
from icecube import DomTools, linefit
from icecube import gulliver, paraboloid, lilliput
from icecube import photonics_service

import numpy as np

from icecube.icetray import I3Units

load("libtruncated_energy")
load("libgulliver")
load("liblilliput")
load("libtoprec")
load('libdataio')
load('libtopeventcleaning')
load("spline-reco")

load("libmue")
load("libbayesian-priors")
load("libDomTools")
load("libdouble-muon")


@icetray.traysegment
def NERSC_GNN_CUTS(tray, name, dataset, ftype,
                   min_qtot=4000,
                   fast=False,
                   detector_year='IC86-2',
                   newpulsesHLC='InIcePulsesHLC',
                   newpulsesSLC='InIcePulsesSLC',
                   newpulsesNoDC='InIcePulsesHLC_NoDC',
                   outputpulses='InIcePulsesHLC_NoDC_TW6000',
                   pulses='InIcePulses',
                   splitpulses='SplitInIcePulses'):

    if fast:
        truncated_seed = 'SPEFit2'
        zenith_cut_particle = 'SPEFit2'
    else:
        truncated_seed = 'SPEFit4'
        zenith_cut_particle = 'SplineMPE'

    # --------------
    # Define Modules
    # --------------

    # Remove DeepCore DOMs
    def removeDC(frame):
        # List of DeepCore DOMs
        DeepCoreStrings = [79, 80, 81, 82, 83, 84, 85, 86]
        frame[newpulsesNoDC] = dataclasses.I3RecoPulseSeriesMapMask(
            frame, newpulsesHLC,
            lambda omkey, pulse, idx: omkey.string not in DeepCoreStrings)

    def hese_info(frame):
        frame['HESE'] = dataclasses.I3MapStringDouble()
        if frame.Has('VHESelfVeto'):
            if not frame['VHESelfVeto'].value:
                # 0 == False, yes - we have a HESE event
                frame['HESE']['flag'] = 0
            else:
                # 1 == True, no - we don't have a HESE event
                frame['HESE']['flag'] = 1
        else:
            # no - we don't have a HESE event
            frame['HESE']['flag'] = 1

    @icetray.traysegment
    def DoSplineReco(tray, Name, Pulses, Seed, LLH, Suffix, spline,
                     If=lambda frame: True):
        tray.AddService("I3BasicSeedServiceFactory",
                        "SplineSeed%s" % (LLH)+Suffix,
                        FirstGuesses=[Seed])

        tray.AddService("I3SplineRecoLikelihoodFactory",
                        "LLHSpline%s" % (LLH)+Suffix,
                        PhotonicsService=spline,
                        Pulses=Pulses,
                        Likelihood=LLH,
                        NoiseRate=10*I3Units.hertz)

        tray.AddModule("I3SimpleFitter", "Spline%s" % (LLH)+Suffix,
                       SeedService="SplineSeed%s" % (LLH)+Suffix,
                       Parametrization="SimpleTrack",
                       LogLikelihood="LLHSpline%s" % (LLH)+Suffix,
                       Minimizer="Minuit",
                       If=If)
    # --------------

    # I3 Pulse Cleaning
    tray.AddModule('I3LCPulseCleaning', 'LC_Cleaning',
                   Input=splitpulses,
                   OutputHLC=newpulsesHLC,
                   OutputSLC=newpulsesSLC
                   )

    tray.AddModule(removeDC, 'RemoveDeepCore')

    # Clean up InIce with a time window of 6000 ns
    tray.AddModule("I3TimeWindowCleaning<I3RecoPulse>", "TWCleaning1",
                   InputResponse=newpulsesNoDC,
                   OutputResponse=outputpulses,
                   TimeWindow=6000,
                   )

    icetray.load('VHESelfVeto')

    tray.AddModule('VHESelfVeto', 'VHE_selfveto',
                   Pulses=newpulsesHLC,
                   # Tomasz didn't specify which pulses to use, he seems to favor InIcePulsesHLC
                   #            Pulses=pulses,
                   OutputBool='VHESelfVeto',
                   OutputVertexTime='VHESelfVetoVertexTime',
                   OutputVertexPos='VHESelfVetoVertexPos')

    tray.AddModule(hese_info, "hese_info")

    tray.AddModule('HomogenizedQTot', 'qtot_total',
                   Pulses=outputpulses, Output='QTot')

    # Charge Cut - Qtot > min_qtot
    tray.AddModule(lambda frame: (frame.Has('QTot') and
                                  frame['QTot'].value > min_qtot),
                   'simple_qtot_cut')

    # -----------------
    # Reconstruction on cleaned pulses
    # -----------------
    if zenith_cut_particle == 'SplineMPE':
        # New Linefit
        tray.AddModule('I3LineFit', "linefit",
                       InputRecoPulses=outputpulses,
                       LeadingEdge='FLE',
                       MinHits=2,
                       Name='NewLineFit'
                       )

        # Services to do Gulliver reconstruction
        tray.AddService("I3SimpleParametrizationFactory", "SimpleTrack")(
            ("StepX", 20*I3Units.m),                                     # ! 20m step size
            ("StepY", 20*I3Units.m),                                     # ! 20m step size
            ("StepZ", 20*I3Units.m),                                     # ! 20m step size
            ("StepZenith", 0.1 * I3Units.radian),                        # ! 0.1 radian step size in zenith
            ("StepAzimuth", 0.2 * I3Units.radian),                       # ! 0.2 radian step size in azimuth
            ("StepT", 0.),                                             # Default
            ("StepLinE", 0.),                                          # Default
            ("StepLogE", 0.),                                          # Default
            ("BoundsX", [-2000 * I3Units.m, 2000 * I3Units.m]),       # ! Set bounds to +-2000m
            ("BoundsY", [-2000 * I3Units.m, 2000 * I3Units.m]),        # ! Set bounds to +-2000m
            ("BoundsZ", [-2000 * I3Units.m, 2000 * I3Units.m]),       # ! Set bounds to +-2000m
            ("BoundsZenith", [0., 0.]),                              # Default
            ("BoundsAzimuth", [0., 0.]),                             # Default
            ("BoundsT", [0., 0.]),                                   # Default
            )

        # Define the gulliver minimization sevice to use
        tray.AddService("I3GulliverMinuitFactory", "Minuit")(
            ("Algorithm", "SIMPLEX"),                                  # Default
            ("Tolerance", 0.01),                                       # ! change to 0.01
            ("MaxIterations", 10000),                                  # ! change to 10000
            ("MinuitPrintLevel", -2),                                  # Default
            ("MinuitStrategy", 2),                                     # Default
            ("FlatnessCheck", True),                                   # Default
            )

        # Use convoluted pandel as the PDF for the likelihood
        tray.AddService("I3GulliverIPDFPandelFactory", "Pandel")(
            ("InputReadout",  outputpulses),                          # ! Name of pulses to use
            ("Likelihood", "SPE1st"),                                  # Default
            ("PEProb", "GaussConvoluted"),                             # Default
            ("IceModel", 2),                                           # Default
            ("IceFile", ""),                                           # Default
            ("AbsorptionLength", 98.0 * I3Units.m),                    # Default
            ("JitterTime", 15.0 * I3Units.ns),                         # Default
            ("NoiseProbability", 1.0*I3Units.hertz * 10.0*I3Units.ns)  # ! Added a little noise term
            )

        # linefit seed service
        tray.AddService("I3BasicSeedServiceFactory", "Seed")(
            ("FirstGuess", "NewLineFit"),                              # ! Use SPEFit2
            ("InputReadout", outputpulses),                            # ! Use pulses for vertex correction
            ("TimeShiftType", "TFirst"),                               # ! Use TFirst for vertex correction
            ("SpeedPolice", True),                                     # Default
            ("MaxMeanTimeResidual", 1000.0 * I3Units.ns),              # Default
            )

        # track fit
        tray.AddModule("I3IterativeFitter", "SPEFit4")(
            ("RandomService", "SOBOL"),                            # Default
            ("NIterations",  4),                                   # ! Nunmber of iterations 15+1 iterations total
            ("SeedService", "Seed"),                           # ! Name of seed service
            ("Parametrization", "SimpleTrack"),                    # ! Name of track parametrization service
            ("LogLikelihood", "Pandel"),                            # ! Name of likelihood service
            ("CosZenithRange", [-1, 1]),                         # Default
            ("Minimizer", "Minuit")                                # ! Name of minimizer service
            )

        # Add Spline MPE Fit needed for the Jakob's cuts
        tray.AddModule("muex", "muex_angular4",
                       Pulses=outputpulses,
                       rectrk="",
                       result="MuEXAngular4",
                       lcspan=0,
                       repeat=4,
                       usempe=True,
                       detail=False,
                       energy=False,
                       icedir=os.path.expandvars("$I3_BUILD/mue/resources/ice/mie")
                       )

        spline = photonics_service.I3PhotoSplineService()
        llh = "MPE"
        photosplinedir = \
            "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/"
        spline = photonics_service.I3PhotoSplineService(
                    photosplinedir + '/InfBareMu_mie_abs_z20a10_V2.fits',
                    photosplinedir + '/InfBareMu_mie_prob_z20a10_V2.fits',
                    0)

        tray.AddSegment(DoSplineReco, "spline%s" % (llh),
                        Pulses=outputpulses,
                        Seed="MuEXAngular4",
                        LLH=llh,
                        Suffix="",
                        spline=spline)

    # Truncated energy
    @icetray.traysegment
    def Truncated(tray, Name, Pulses="", Seed="", Suffix="",
                  If=lambda f: True,
                  PhotonicsService="", Model=""):
        # ! base result Name to put into frame
        TruncatedName = Seed+"TruncatedEnergy"+Suffix+Model
        if detector_year == "IC86-1":
            tray.AddModule("I3TruncatedEnergy",
                           RecoPulsesName=Pulses,  # ! Name of Pulses
                           RecoParticleName=Seed,
                           ResultParticleName=TruncatedName,  # ! Name of result Particle
                           I3PhotonicsServiceName=PhotonicsService,  # ! Name of photonics service to use
                           UseRDE=False,
                           If=If)
        else:
            tray.AddModule("I3TruncatedEnergy",
                           RecoPulsesName=Pulses,  # ! Name of Pulses
                           RecoParticleName=Seed,
                           ResultParticleName=TruncatedName,  # ! Name of result Particle
                           I3PhotonicsServiceName=PhotonicsService,  # ! Name of photonics service to use
                           UseRDE=True,  # ! Correct for HQE DOMs !!! MUST BE TRUE USUALLY, BUT GIVES A BUG IN TRUNCATED FOR IC86-1 SIMULATIONS
                           If=If)
    photonics_base = \
        "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/"
    tray.AddService("I3PhotonicsServiceFactory", "PhotonicsServiceMu_SpiceMie",
                    PhotonicsTopLevelDirectory=photonics_base,
                    DriverFileDirectory=photonics_base + "driverfiles",
                    PhotonicsLevel2DriverFile="mu_photorec.list",
                    PhotonicsTableSelection=2,
                    ServiceName="PhotonicsServiceMu_SpiceMie")

    tray.AddSegment(Truncated,
                    Pulses=outputpulses,
                    Seed=truncated_seed,
                    Suffix="",
                    PhotonicsService="PhotonicsServiceMu_SpiceMie",
                    Model="_SPICEMie")

    def reco_cut(frame):
        if frame.Has(truncated_seed + 'TruncatedEnergy_SPICEMie_BINS_Muon'):
            return True
        else:
            return False

    tray.AddModule(reco_cut, 'reco_cut')
    # -------------------

    #    New Line Fit used to select down-going events
    def only_downgoing(frame):
        track = frame[zenith_cut_particle]
        if(track.dir.zenith < np.radians(90)):
            return True
        else:
            return False

    tray.AddModule(only_downgoing, 'only_downgoing')

    # requires modified Truncated project
    def dEdx_fit(frame):
        frame['Collection'] = dataclasses.I3MapStringDouble()
        dEdxVector = frame[truncated_seed +
                           'TruncatedEnergy_SPICEMie_BINS_dEdxVector']
        # dEdxVector = frame[truncated_seed +
        #                    'TruncatedEnergy_SPICEMie_BINS_dEdX']
        if len(dEdxVector) > 3:
            return True
        else:
            return False

    # tray.AddModule(dEdx_fit, 'dEdx_fit')

    def get_nersc_gnn_weight(frame):
        if (frame.Has("I3EventHeader") and frame.Has("SRTInIcePulses") and
                frame.Has("MPEFitMuEX")):
            is_booked = True
        else:
            is_booked = False

        mctree = frame['I3MCTree']
        if ftype.lower() == "corsika":
            energy = frame['CorsikaWeightMap']['PrimaryEnergy']
            ptype = frame['CorsikaWeightMap']['PrimaryType']
            # nevents = frame['CorsikaWeightMap']["NEvents"]
            prim = dataclasses.get_most_energetic_primary(mctree)
            generator = from_simprod(dataset)
            gen = generator(energy, ptype)
            fluxGaisserH4a = fluxes.GaisserH4a()
            ffluxGaisserH4a = float(fluxGaisserH4a(energy, ptype))
            weightsfluxGaisserH4a = fluxGaisserH4a(energy, ptype)/generator(energy, ptype)
            weight = weightsfluxGaisserH4a
            is_signal = False
        elif ftype.lower() == "nugen":
            prim = dataclasses.get_most_energetic_neutrino(mctree)
            I3MCWeightDict = frame["I3MCWeightDict"]
            energy = I3MCWeightDict["PrimaryNeutrinoEnergy"]
            oneweight = I3MCWeightDict["OneWeight"]
            nevents = I3MCWeightDict["NEvents"]
            # weight = 1e-8*pow(energy, -2) * oneweight / nevents
            generator_ = from_simprod(dataset)
            gen_ = generator_(energy, prim.type,np.cos(prim.dir.zenith))
            p_int_ = frame['I3MCWeightDict']['TotalInteractionProbabilityWeight']
            unit = I3Units.cm2/I3Units.m2
            genweight = ((6.7e-18)/6) * pow( energy/1e5, -2 ) * p_int_/gen_/unit
            # print "energy",energy,"genweight",((6.7e-18)/6) * pow( energy/1e5, -2 ) * p_int_/gen_/unit
            # weight = ((6.7e-18)/6) * pow( energy/1e5, -2 ) * oneweight / nevents
            weight = genweight
            is_signal = True
        else:
            raise ValueError('Unkown data type: {!r}'.format(ftype))

        frame['nersc_gnn_info'] = dataclasses.I3MapStringDouble({
            'weight': weight,
            'booked': is_booked,
            'is_signal': is_signal,
            'qtot': frame['QTot'].value,
            'HESE': frame['HESE']['flag'],
            })

    tray.Add(get_nersc_gnn_weight, streams=[icetray.I3Frame.Physics])
    tray.Add("I3OrphanQDropper")  # drop q frames not followed by p frames

from __future__ import print_function, division
from icecube.icetray import traysegment, I3Module

class identitydict(dict):
    def __getitem__(self, k):
        if not k in self:
            return k
        else:
            return dict.__getitem__(self, k)

def define_keys(year=2010):
    keys = identitydict()
    if year == 2010:
        keys["SPEFit"] = "SPEFit4"
        keys["SPEFitFitParams"] = "SPEFit4FitParams"
        keys["CascadeVertex"] = keys["MonopodFit"]
        keys["SubEventStream"] = "nullsplit"
    elif year == 2011:
        keys["SPEFit"] = "SPEFit2"
        keys["SPEFitFitParams"] = "SPEFit2FitParams"
        keys["MonopodFit"] = "CscdL3_Monopod_SpiceMie"
        keys["MonopodFitFitParams"] = "CscdL3_Monopod_SpiceMieFitParams"
        keys["CredoFit"] = "CscdL3_Credo_SpiceMie"
        keys["CredoFitParams"] = "CscdL3_Credo_SpiceMieFitParams"
        keys["CascadeVertex"] = keys["CredoFit"]
        keys["SubEventStream"] = "in_ice"
    return keys

@traysegment
def CleanCruft(tray, name, year=2010):
    """
    Remove bloat accumulated through the various processing levels.
    """
    from icecube import icetray, dataclasses

    if year == 2010:
        bloat = [
            'ATWDPortiaPulse',
            'AtmCscdEnergyReco',
            'AtmCscdEnergyRecoParams',
            'CascadeL3_Containment',
            'CascadeL3_HighEnergyCut',
            'CascadeL3_LLHRatioCut',
            'CascadeLast',
            'CascadeLastParams',
            'CascadeLast_DC',
            'CascadeLast_DCParams',
            'CascadeLinefit',
            'CascadeLinefitParams',
            'CascadeLlhVertexFit',
            'CascadeLlhVertexFitParams',
            'ClusterCleaningExcludedStations',
            'CoreRemovalPulses_0',
            'CoreRemovalPulses_1',
            'CoreRemoval_CascadeLlhVertexFit_0',
            'CoreRemoval_CascadeLlhVertexFit_0Params',
            'CoreRemoval_CascadeLlhVertexFit_0_CLastSeed',
            'CoreRemoval_CascadeLlhVertexFit_0_CLastSeedParams',
            'CoreRemoval_CascadeLlhVertexFit_1',
            'CoreRemoval_CascadeLlhVertexFit_1Params',
            'CoreRemoval_CascadeLlhVertexFit_1_CLastSeed',
            'CoreRemoval_CascadeLlhVertexFit_1_CLastSeedParams',
            'CoreRemoval_DipoleFit_0',
            'CoreRemoval_DipoleFit_0Params',
            'CoreRemoval_DipoleFit_1',
            'CoreRemoval_DipoleFit_1Params',
            'CoreRemoval_SPEFit_0',
            'CoreRemoval_SPEFit_0FitParams',
            'CoreRemoval_SPEFit_0_LinefitSeed',
            'CoreRemoval_SPEFit_0_LinefitSeedParams',
            'CoreRemoval_SPEFit_1',
            'CoreRemoval_SPEFit_1FitParams',
            'CoreRemoval_SPEFit_1_LinefitSeed',
            'CoreRemoval_SPEFit_1_LinefitSeedParams',
            'CoreRemoval_SPERadius',
            'CramerRao_PoleL2IpdfGConvolute_multitParams',
            'CramerRao_PoleL2MPEFitParams',
            'CredoFitContained',
            'CredoFitFillRatio',
            'CredoFit_ForgedSeed',
            'DipoleFit_DC',
            'DipoleFit_DCParams',
            'EHESummaryPulseInfo',
            'EheStarFirstDoms',
            'EheStarLateralFg',
            'ExcludedHLCStations',
            'FADCPortiaPulse',
            'FiniteRecoCuts',
            'FiniteRecoFit',
            'FiniteRecoLlh',
            'LineFit',
            'LineFitParams',
            'LineFit_DC',
            'LineFit_DCParams',
            'LineFit_SLC',
            'LineFit_SLCParams',
            'MPEFit',
            'MPEFitCramerRaoParams',
            'MPEFitCramerRao_SLCParams',
            'MPEFitFitParams',
            'MPEFitMuE',
            'MPEFitMuE_SLC',
            'MPEFitPhotorecEnergy',
            'MPEFitPhotorecEnergy_SLC',
            'MPEFitPhotorecEnergy_SLC_Spice1',
            'MPEFitPhotorecEnergy_SLC_Spice1_dEdX',
            'MPEFitPhotorecEnergy_SLC_dEdX',
            'MPEFitPhotorecEnergy_Spice1',
            'MPEFitPhotorecEnergy_Spice1_dEdX',
            'MPEFitPhotorecEnergy_dEdX',
            'MPEFit_SLC',
            'MPEFit_SLCFitParams',
            'OldAtmCscdEnergyReco',
            'OldAtmCscdEnergyRecoParams',
            'OldMPEFit',
            'OldMPEFitCramerRaoParams',
            'OldMPEFitCramerRao_SLCParams',
            'OldMPEFitFitParams',
            'OldMPEFitMuE',
            'OldMPEFitMuE_SLC',
            'OldMPEFitPhotorecEnergy',
            'OldMPEFitPhotorecEnergy_SLC',
            'OldMPEFitPhotorecEnergy_SLC_Spice1',
            'OldMPEFitPhotorecEnergy_SLC_Spice1_dEdX',
            'OldMPEFitPhotorecEnergy_SLC_dEdX',
            'OldMPEFitPhotorecEnergy_Spice1',
            'OldMPEFitPhotorecEnergy_Spice1_dEdX',
            'OldMPEFitPhotorecEnergy_dEdX',
            'OldMPEFit_SLC',
            'OldMPEFit_SLCFitParams',
            'OpheliaFGParticleBTW',
            'OpheliaFirstGuess',
            'OpheliaFirstGuessBTW',
            'PoleCascadeLinefit',
            'PoleCascadeLinefitParams',
            'PoleEHESummaryPulseInfo',
            'PoleL2BayesianFitFitParams',
            'PoleL2IpdfGConvolute_multit',
            'PoleL2IpdfGConvolute_multitCuts',
            'PoleL2MPEFit',
            'PoleL2MPEFitCuts',
            'PoleL2MPEFitFitParams',
            'PoleL2MPEFitMuE',
            'PoleMuonLinefit',
            'PoleMuonLinefitParams',
            'PoleMuonLlhFit',
            'PoleMuonLlhFitFitParams',
            'PoleMuonLlhFitMuE',
            'PoleToI',
            'PoleToIParams',
            'Pole_SLC_HLCLinefit',
            'Pole_SLC_HLCLinefitParams',
            'Pole_SLC_HLCLlhFit',
            'Pole_SLC_HLCLlhFitFitParams',
            'SPEFitSingle',
            'SPEFitSingleFitParams',
            'SPEFitSingle_DC',
            'SPEFitSingle_DCFitParams',
            'SPEFitSingle_SLC',
            'SPEFitSingle_SLCFitParams',
            'SRTCoreRemovalPulses_0',
            'SRTCoreRemovalPulses_1',
            'SRTTWOfflinePulses_DC',
            'SRTTimeSplitPulses_0',
            'SRTTimeSplitPulses_1',
            'TWOfflinePulsesHLCMaxQRing',
            'TWOfflinePulses_DC',
            'TWSRTCLast',
            'TWSRTCLastParams',
            'TWSRTDipoleFit',
            'TWSRTDipoleFitParams',
            'TWSRTLineFit',
            'TWSRTLineFitParams',
            'TimeSplitPulses_0',
            'TimeSplitPulses_1',
            'TimeSplit_CascadeLlhVertexFit_0',
            'TimeSplit_CascadeLlhVertexFit_0Params',
            'TimeSplit_CascadeLlhVertexFit_0_CLastSeed',
            'TimeSplit_CascadeLlhVertexFit_0_CLastSeedParams',
            'TimeSplit_CascadeLlhVertexFit_1',
            'TimeSplit_CascadeLlhVertexFit_1Params',
            'TimeSplit_CascadeLlhVertexFit_1_CLastSeed',
            'TimeSplit_CascadeLlhVertexFit_1_CLastSeedParams',
            'TimeSplit_DipoleFit_0',
            'TimeSplit_DipoleFit_0Params',
            'TimeSplit_DipoleFit_1',
            'TimeSplit_DipoleFit_1Params',
            'TimeSplit_SPEFit_0',
            'TimeSplit_SPEFit_0FitParams',
            'TimeSplit_SPEFit_0_LinefitSeed',
            'TimeSplit_SPEFit_0_LinefitSeedParams',
            'TimeSplit_SPEFit_1',
            'TimeSplit_SPEFit_1FitParams',
            'TimeSplit_SPEFit_1_LinefitSeed',
            'TimeSplit_SPEFit_1_LinefitSeedParams',
            'ToI_DC',
            'ToI_DCParams']
    elif year == 2011:
        bloat = [
            'ATWDPortiaPulse',
            'AtmCscdEnergyReco',
            'AtmCscdEnergyRecoParams',
            'BadDomsListSLCSaturated',
            'CascadeDipoleFit',
            'CascadeDipoleFitParams',
            'CascadeFillRatio',
            'CascadeFilt_CscdLlh',
            'CascadeFilt_LFVel',
            'CascadeFilt_ToiVal',
            'CascadeImprovedLineFit',
            'CascadeImprovedLineFitParams',
            'CascadeLast',
            'CascadeLastParams',
            'CascadeLast_DC',
            'CascadeLast_DCParams',
            'CascadeLineFit',
            'CascadeLineFitParams',
            'CascadeLineFitSplit1',
            'CascadeLineFitSplit1Params',
            'CascadeLineFitSplit2',
            'CascadeLineFitSplit2Params',
            'CascadeLlhVertexFit_IC',
            'CascadeLlhVertexFit_ICParams',
            'CascadeLlhVertexFit_IC_CLastSeed',
            'CascadeLlhVertexFit_IC_CLastSeedParams',
            'CascadeLlhVertexFit_IC_Coincidence0',
            'CascadeLlhVertexFit_IC_Coincidence0Params',
            'CascadeLlhVertexFit_IC_Coincidence0_CLastSeed',
            'CascadeLlhVertexFit_IC_Coincidence0_CLastSeedParams',
            'CascadeLlhVertexFit_IC_Coincidence1',
            'CascadeLlhVertexFit_IC_Coincidence1Params',
            'CascadeLlhVertexFit_IC_Coincidence1_CLastSeed',
            'CascadeLlhVertexFit_IC_Coincidence1_CLastSeedParams',
            'CascadeSeed',
            'CascadeSplitPulses1',
            'CascadeSplitPulses2',
            'CascadeToISplit1',
            'CascadeToISplit1Params',
            'CascadeToISplit2',
            'CascadeToISplit2Params',
            'CleanTriggerHierarchy_IT',
            'ClusterCleaningExcludedStations',
            'CoreRemovalPulses_0',
            'CoreRemovalPulses_1',
            'CoreRemoval_CascadeLlhVertexFit_0',
            'CoreRemoval_CascadeLlhVertexFit_0Params',
            'CoreRemoval_CascadeLlhVertexFit_0_CLastSeed',
            'CoreRemoval_CascadeLlhVertexFit_0_CLastSeedParams',
            'CoreRemoval_CascadeLlhVertexFit_1',
            'CoreRemoval_CascadeLlhVertexFit_1Params',
            'CoreRemoval_CascadeLlhVertexFit_1_CLastSeed',
            'CoreRemoval_CascadeLlhVertexFit_1_CLastSeedParams',
            'CoreRemoval_DipoleFit_0',
            'CoreRemoval_DipoleFit_0Params',
            'CoreRemoval_DipoleFit_1',
            'CoreRemoval_DipoleFit_1Params',
            'CoreRemoval_SPEFit_0',
            'CoreRemoval_SPEFit_0FitParams',
            'CoreRemoval_SPEFit_1',
            'CoreRemoval_SPEFit_1FitParams',
            'CramerRaoPoleL2IpdfGConvolute_2itParams',
            'CramerRaoPoleL2MPEFitParams',
            'CscdL2',
            'CscdL2_Topo_HLC0',
            'CscdL2_Topo_HLC0_DCOnly',
            'CscdL2_Topo_HLC0_noDC',
            'CscdL2_Topo_HLC1',
            'CscdL2_Topo_HLC1_DCOnly',
            'CscdL2_Topo_HLC1_noDC',
            'CscdL2_Topo_HLCSplitCount',
            'CscdL3',
            'CscdL3_Bayesian16',
            'CscdL3_Bayesian16FitParams',
            'CscdL3_CascadeLlhVertexFit',
            'CscdL3_CascadeLlhVertexFitParams',
            'CscdL3_Cont_Tag',
            'CscdL3_SPEFit16',
            'CscdL3_SPEFit16FitParams',
            'DipoleFit_DC',
            'DipoleFit_DCParams',
            'EHESummaryPulseInfo',
            'FADCPortiaPulse',
            'FiniteRecoCuts',
            'FiniteRecoFit',
            'FiniteRecoLlh',
            'HowManySaturDOMs',
            'HowManySaturStrings',
            'I3DST11',
            'I3DST11Header',
            'LineFit',
            'LineFitParams',
            'LineFit_DC',
            'LineFit_DCParams',
            'MPEFit',
            'MPEFitCramerRaoParams',
            'MPEFitFitParams',
            'MPEFitMuE',
            'MPEFitMuEX',
            'MPEFitTruncatedEnergy_SPICE1_AllBINS_MuEres',
            'MPEFitTruncatedEnergy_SPICE1_AllBINS_Muon',
            'MPEFitTruncatedEnergy_SPICE1_AllBINS_Neutrino',
            'MPEFitTruncatedEnergy_SPICE1_AllDOMS_MuEres',
            'MPEFitTruncatedEnergy_SPICE1_AllDOMS_Muon',
            'MPEFitTruncatedEnergy_SPICE1_AllDOMS_Neutrino',
            'MPEFitTruncatedEnergy_SPICE1_BINS_MuEres',
            'MPEFitTruncatedEnergy_SPICE1_BINS_Muon',
            'MPEFitTruncatedEnergy_SPICE1_BINS_Neutrino',
            'MPEFitTruncatedEnergy_SPICE1_DOMS_MuEres',
            'MPEFitTruncatedEnergy_SPICE1_DOMS_Muon',
            'MPEFitTruncatedEnergy_SPICE1_DOMS_Neutrino',
            'MPEFitTruncatedEnergy_SPICE1_ORIG_Muon',
            'MPEFitTruncatedEnergy_SPICE1_ORIG_Neutrino',
            'MPEFitTruncatedEnergy_SPICE1_ORIG_dEdX',
            'MPEFitTruncatedEnergy_SPICEMie_AllBINS_MuEres',
            'MPEFitTruncatedEnergy_SPICEMie_AllBINS_Muon',
            'MPEFitTruncatedEnergy_SPICEMie_AllBINS_Neutrino',
            'MPEFitTruncatedEnergy_SPICEMie_AllDOMS_MuEres',
            'MPEFitTruncatedEnergy_SPICEMie_AllDOMS_Muon',
            'MPEFitTruncatedEnergy_SPICEMie_AllDOMS_Neutrino',
            'MPEFitTruncatedEnergy_SPICEMie_BINS_MuEres',
            'MPEFitTruncatedEnergy_SPICEMie_BINS_Muon',
            'MPEFitTruncatedEnergy_SPICEMie_BINS_Neutrino',
            'MPEFitTruncatedEnergy_SPICEMie_DOMS_MuEres',
            'MPEFitTruncatedEnergy_SPICEMie_DOMS_Muon',
            'MPEFitTruncatedEnergy_SPICEMie_DOMS_Neutrino',
            'MPEFitTruncatedEnergy_SPICEMie_ORIG_Muon',
            'MPEFitTruncatedEnergy_SPICEMie_ORIG_Neutrino',
            'MPEFitTruncatedEnergy_SPICEMie_ORIG_dEdX',
            'NCh_CscdL2_Topo_HLC0',
            'NCh_CscdL2_Topo_HLC0_DCOnly',
            'NCh_CscdL2_Topo_HLC0_noDC',
            'NCh_CscdL2_Topo_HLC1',
            'NCh_CscdL2_Topo_HLC1_DCOnly',
            'NCh_CscdL2_Topo_HLC1_noDC',
            'NCh_OfflinePulses',
            'NCh_OfflinePulsesHLC',
            'NCh_OfflinePulsesHLC_DCOnly',
            'NCh_OfflinePulsesHLC_noDC',
            'NCh_OfflinePulses_DCOnly',
            'NCh_OfflinePulses_noDC',
            'NCh_SRTOfflinePulses',
            'NCh_SRTOfflinePulses_DCOnly',
            'NCh_SRTOfflinePulses_noDC',
            'NString_CscdL2_Topo_HLC0',
            'NString_CscdL2_Topo_HLC0_DCOnly',
            'NString_CscdL2_Topo_HLC0_noDC',
            'NString_CscdL2_Topo_HLC1',
            'NString_CscdL2_Topo_HLC1_DCOnly',
            'NString_CscdL2_Topo_HLC1_noDC',
            'NString_OfflinePulses',
            'NString_OfflinePulsesHLC',
            'NString_OfflinePulsesHLC_DCOnly',
            'NString_OfflinePulsesHLC_noDC',
            'NString_OfflinePulses_DCOnly',
            'NString_OfflinePulses_noDC',
            'NString_SRTOfflinePulses',
            'NString_SRTOfflinePulses_DCOnly',
            'NString_SRTOfflinePulses_noDC',
            'OfflineIceTopHLCPulseInfo',
            'OfflineIceTopHLCTankPulses',
            'OfflineIceTopHLCVEMPulses',
            'OfflineIceTopSLCVEMPulses',
            'OfflineInIceCalibrationErrata',
            'OpheliaFGParticleBTW',
            'OpheliaFirstGuess',
            'OpheliaFirstGuessBTW',
            'PoleL2BayesianFit',
            'PoleL2BayesianFitFitParams',
            'PoleL2IpdfGConvolute_2it',
            'PoleL2IpdfGConvolute_2itFitParams',
            'PoleL2MPEFit',
            'PoleL2MPEFitCuts',
            'PoleL2MPEFitFitParams',
            'PoleL2MPEFitMuE',
            'PoleL2SPE2it_TimeSplit1',
            'PoleL2SPE2it_TimeSplit2',
            'PoleL2SPEFit2it_GeoSplit1',
            'PoleL2SPEFit2it_GeoSplit2',
            'PoleMuonLlhFit',
            'PoleMuonLlhFitCutsFirstPulseCuts',
            'PoleMuonLlhFitFitParams',
            'RTTWOfflinePulsesFR',
            'SPEFit2',
            'SPEFit2CramerRaoParams',
            'SPEFit2CramerRao_DCParams',
            'SPEFit2FitParams',
            'SPEFit2MuE',
            'SPEFit2_DC',
            'SPEFit2_DCFitParams',
            'SPEFitSingle',
            'SPEFitSingleFitParams',
            'SPEFitSingle_DC',
            'SPEFitSingle_DCFitParams',
            'SPERadius',
            'SRTCoreRemovalPulses_0',
            'SRTCoreRemovalPulses_1',
            'SRTOfflinePulsesTimeRange',
            'SRTOfflinePulses_DCOnly',
            'SRTOfflinePulses_noDC',
            'SRTTWOfflinePulsesDC',
            'SRTTimeSplitPulses_0',
            'SRTTimeSplitPulses_1',
            'TWOfflinePulsesDC',
            'TWOfflinePulsesFR',
            'TankPulseMergerExcludedStations',
            'TimeSplitPulses_0',
            'TimeSplitPulses_1',
            'TimeSplit_CascadeLlhVertexFit_0',
            'TimeSplit_CascadeLlhVertexFit_0Params',
            'TimeSplit_CascadeLlhVertexFit_0_CLastSeed',
            'TimeSplit_CascadeLlhVertexFit_0_CLastSeedParams',
            'TimeSplit_CascadeLlhVertexFit_1',
            'TimeSplit_CascadeLlhVertexFit_1Params',
            'TimeSplit_CascadeLlhVertexFit_1_CLastSeed',
            'TimeSplit_CascadeLlhVertexFit_1_CLastSeedParams',
            'TimeSplit_DipoleFit_0',
            'TimeSplit_DipoleFit_0Params',
            'TimeSplit_DipoleFit_1',
            'TimeSplit_DipoleFit_1Params',
            'TimeSplit_SPEFit_0',
            'TimeSplit_SPEFit_0FitParams',
            'TimeSplit_SPEFit_1',
            'TimeSplit_SPEFit_1FitParams',
            'ToI_DC',
            'ToI_DCParams',
            'Veto_CscdL2_Topo_HLC0',
            'Veto_CscdL2_Topo_HLC1',
            'Veto_SRTOfflinePulses']

    tray.AddModule('Delete', name + 'KillBoat', Keys=bloat)

    def linearize(frame):
        if not 'I3MCTree' in frame:
            return
        mct = dataclasses.I3LinearizedMCTree(frame['I3MCTree'])
        del frame['I3MCTree']
        frame['I3MCTree'] = mct
    tray.AddModule(linearize, name + 'LinearizeMCTree', Streams=[icetray.I3Frame.DAQ])

import copy
from icecube.icetray import I3Bool
from icecube.dataclasses import I3MapStringBool, I3Double

class PrescaledFilter(I3Module):
    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("Condition", "", None)
        self.AddParameter("Prescale", "Name of prescale bool in the frame", "L4Prescale")
        self.AddParameter("CutMap", "Name of cut map in the frame", "L4Cuts")
    def Configure(self):
        self.condition = self.GetParameter("Condition")
        self.prescale_name = self.GetParameter("Prescale")
        self.cutmap_name = self.GetParameter("CutMap")
    def Physics(self, frame):
        if self.prescale_name in frame:
            prescale = frame[self.prescale_name].value
        else:
            prescale = False
        passed = self.condition(frame)
        # Drop frame
        if not (passed or prescale):
            return

        # If
        if self.cutmap_name in frame:
            cutmap = copy.copy(frame[self.cutmap_name])
            del frame[self.cutmap_name]
        else:
            cutmap = I3MapStringBool()
        cutmap[self.name] = passed
        frame[self.cutmap_name] = cutmap
        self.PushFrame(frame)

class PrescaleCalculator(I3Module):
    """
    Randomly select events in a way that creates a flat distribution in QTot for experimental data
    """
    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddOutBox("OutBox")
        self.AddParameter("Prescale", "Name of prescale bool in the frame", "L4Prescale")
        self.AddParameter("Year", "Data taking year", 2010)
        self.AddParameter("TargetEvents", "Desired number of events in final sample", 1e4)
    def Configure(self):
        self.prescale_name = self.GetParameter("Prescale")
        year = self.GetParameter("Year")
        nevents = self.GetParameter("TargetEvents")

        from numexpr import NumExpr
        # dN/dQtot at Level 3
        if year == 2010:
            l3 = 'where(x<351.206499361, 13098188.2389 * (-(1+-1.74838791031)/55.0123698793)*(x/55.0123698793)**-1.74838791031, where(x<1588.90255416, 1218852.60519 * (-(1+-2.18177126127)/550.123698793)*(x/550.123698793)**-2.18177126127, 73876.2195483 * (-(1+-3.49765218946)/2190.08189181)*(x/2190.08189181)**-3.49765218946))'
        elif year == 2011:
            l3 = 'where(x<408.691206546, 6779067.70317 * (-(1+-1.56839961256)/55.0123698793)*(x/55.0123698793)**-1.56839961256, where(x<2820.96019967, 331621.950107 * (-(1+-3.03187598102)/550.123698793)*(x/550.123698793)**-3.03187598102, 17827.956902 * (-(1+-3.70790920185)/2190.08189181)*(x/2190.08189181)**-3.70790920185))'
        # target distribution: flat between bounds with the given integral
        final = '(({nevents}/log(1e5/1e1))/x)'.format(**locals())
        self.prescale = NumExpr('where({l3}/{final} > 1, {l3}/{final}, 1)'.format(**locals()))
        self.rng = self.context['I3RandomService']
    def Physics(self, frame):
        qtot = frame['HomogenizedQTot'].value
        if qtot > 1:
            prescale = float(self.prescale(qtot))
            passed = self.rng.uniform(prescale) < 1
        else:
            prescale = 0.
            passed = False
        frame[self.prescale_name] = I3Bool(self.rng.uniform(prescale) < 1)
        frame[self.prescale_name+"Weight"] = I3Double(prescale)

        self.PushFrame(frame)

@traysegment
def Prefilter(tray, name, year=2010, trackveto=True, prescale=False, L3Containment=True):

    from icecube import icetray, dataclasses

    to_book = []
    keys = define_keys(year)

    tray.AddModule('Delete', Keys=['HomogenizedQTot', 'L4TopologicalSplitCount', 'OfflinePulsesHLC_NoDC', 'L4VetoLayerQTot', 'L4VetoLayer0', 'L4VetoLayer1'])

    from icecube import VHESelfVeto
    tray.AddModule('HomogenizedQTot', name+'qtot', Output='HomogenizedQTot', Pulses='OfflinePulsesHLC')
    to_book += ['HomogenizedQTot']

    if prescale:
        tray.Add(PrescaleCalculator, Year=year)

    if L3Containment:
        if year == 2010:
            condition = lambda frame: frame['CascadeL3_Containment'].value or frame['HomogenizedQTot'].value > 6e3
        elif year == 2011:
            condition = lambda frame: frame["CscdL3_Cont_Tag"].value==1 or frame['HomogenizedQTot'].value > 6e3
        tray.Add(PrescaledFilter, "L3Containment", Condition=condition)

    from icecube.CascadeL3_IC79.level4.ttrigger import TopologicalCounter
    from icecube.CascadeL3_IC79.level4.veto import ACausalHitSelector, TrackVeto, LayerVeto, VetoMarginCalculator

    # Be extremely twitchy about splitting
    from icecube.icetray import I3Units
    tray.AddSegment(TopologicalCounter, 'L4Topological', XYDist=150*I3Units.m, TimeCone=0.5*I3Units.microsecond, TimeWindow=1*I3Units.microsecond)
    # Clean out strings with the wrong spacing
    def CleanDeepCore(frame):
        pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'OfflinePulsesHLC')
        mask = dataclasses.I3RecoPulseSeriesMapMask(frame, 'OfflinePulsesHLC', lambda om, idx, pulse: om.string <= 78)
        frame['OfflinePulsesHLC_NoDC'] = mask
    tray.AddModule(CleanDeepCore, 'nodc')
    # Veto events that start on the outer layers
    tray.AddSegment(LayerVeto, "L4VetoLayer", Pulses='OfflinePulsesHLC_NoDC')

    # Cut out small, non-starting, and coincident events
    def precut(frame):
        import math
        nstring = len(set(om.string for om in dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'OfflinePulsesHLC_NoDC').keys()))
        layer_veto_charge = frame['L4VetoLayer0'].value + frame['L4VetoLayer1'].value
        # z = frame[keys["CascadeVertex"]].pos.z
        nsplits = frame['L4TopologicalCount'].value
        return nstring>3 and layer_veto_charge==0 and (nsplits==1 or frame['HomogenizedQTot'].value > 6e3)

    tray.Add(PrescaledFilter, "OuterLayerVeto", Condition=precut)

    to_book += ['L4TopologicalCount', 'L4VetoLayer0', 'L4VetoLayer1']

    if trackveto:
        cleaned = tray.AddSegment(ACausalHitSelector, 'ACausalOfflinePulses', pulses='OfflinePulses', vertex=keys["CascadeVertex"])
        tray.AddSegment(TrackVeto, 'L4VetoTrack', TimeWindow=(-15, 1000), Pulses=cleaned, Radius=100,
            Direction='down', TrackType=dataclasses.I3Particle.StoppingTrack, Vertex=keys["CascadeVertex"])
        to_book += ['L4VetoTrack', 'L4VetoTrackVetoCharge', 'L4VetoTrackVetoChannels']
        tray.AddSegment(TrackVeto, 'L4StartingTrack', TimeWindow=(-30, 500), Pulses=cleaned, Radius=100,
            Direction='up', TrackType=dataclasses.I3Particle.StartingTrack, NSide=8, Vertex=keys["CascadeVertex"])
        to_book += ['L4StartingTrack', 'L4StartingTrackVetoCharge', 'L4StartingTrackVetoChannels']

        tray.AddModule(VetoMarginCalculator, 'L4VetoTrackMargin', Vertex=keys["CascadeVertex"])
        to_book += ['L4VetoTrackMarginTop', 'L4VetoTrackMarginSide']

    return to_book

@traysegment
def TrackVetoes(tray, name, vertex="L4MonopodFit", pulses="OfflinePulses",
                geometry="I3Geometry", dry_run=False):

    from icecube import dataclasses
    from icecube.CascadeL3_IC79.level4.veto import (
        ACausalHitSelector, TrackVeto, VetoMarginCalculator
    )

    to_book = []
    if not dry_run:
        cleaned = tray.AddSegment(ACausalHitSelector, name+'ACausal'+pulses, Pulses=pulses, vertex=vertex)
        tray.AddSegment(TrackVeto, name+'L4VetoTrack', TimeWindow=(-15, 1000), Pulses=cleaned, Radius=100,
            Direction='down', TrackType=dataclasses.I3Particle.StoppingTrack, Vertex=vertex)
    to_book += ['L4VetoTrack', 'L4VetoTrackVetoCharge', 'L4VetoTrackVetoChannels']
    if not dry_run:
        tray.AddSegment(TrackVeto, name+'L4UpgoingTrack', TimeWindow=(-30, 500), Pulses=cleaned, Radius=100,
            Direction='up', TrackType=dataclasses.I3Particle.StartingTrack, NSide=8, Vertex=vertex)
    to_book += ['L4UpgoingTrack', 'L4UpgoingTrackVetoCharge', 'L4UpgoingTrackVetoChannels']

    if not dry_run:
        cleaned = tray.AddSegment(ACausalHitSelector, name+'ACausal'+pulses+'HLC', Pulses=pulses+'HLC', vertex=vertex)
        tray.AddSegment(TrackVeto, name+'L4StartingTrackHLC', TimeWindow=(-30, 500), Pulses=cleaned, Radius=100,
            Direction='all', TrackType=dataclasses.I3Particle.StartingTrack, NSide=8, Vertex=vertex)
    to_book += ['L4StartingTrackHLC', 'L4StartingTrackHLCVetoCharge', 'L4StartingTrackHLCVetoChannels']
    if not dry_run:
        tray.AddSegment(TrackVeto, name+'L4UpgoingTrackHLC', TimeWindow=(-30, 500), Pulses=cleaned, Radius=100,
            Direction='up', TrackType=dataclasses.I3Particle.StartingTrack, NSide=8, Vertex=vertex)
    to_book += ['L4UpgoingTrackHLC', 'L4UpgoingTrackHLCVetoCharge', 'L4UpgoingTrackHLCVetoChannels']

    if not dry_run:
        tray.AddModule(VetoMarginCalculator, name+'L4VetoTrackMargin', Vertex=vertex, Geometry=geometry)
    to_book += ['L4VetoTrackMarginTop', 'L4VetoTrackMarginSide']

    return to_book

@traysegment
def TrackVetoes_noDC(tray, name, vertex="L4MonopodFit", pulses="OfflinePulsesHLC_noDC", geometry="I3Geometry", dry_run=False):
        to_book = []
        from icecube import dataclasses
        from icecube.CascadeL3_IC79.level4.veto import ACausalHitSelector, TrackVeto, LayerVeto, VetoMarginCalculator
        if not dry_run:
                cleaned = tray.AddSegment(ACausalHitSelector, 'ACausal'+pulses+'_'+vertex, pulses=pulses, vertex=vertex)
                tray.AddSegment(TrackVeto, 'cscdSBU_L4StartingTrackHLC_'+vertex+'_'+pulses, TimeWindow=(-30, 500), Pulses=cleaned, Radius=100,
                   Direction='all', TrackType=dataclasses.I3Particle.StartingTrack, NSide=8, Vertex=vertex)
        to_book += ['cscdSBU_L4StartingTrackHLC_'+vertex+'_'+pulses, 'cscdSBU_L4StartingTrackHLC_'+vertex+'_'+pulses+'VetoCharge', 'cscdSBU_L4StartingTrackHLC_'+vertex+'_'+pulses+'VetoChannels']
        return to_book



@traysegment
def IC79FilterObservables(tray, name, dry_run=False):
    """
    [Re]calculate observables used in the online and L3 filters for IC79
    """
    from icecube import clast, linefit
    if not dry_run:
        tray.AddModule('I3LineFit',
            InputRecoPulses='TWOfflinePulsesHLC', LeadingEdge='FLE',
            Name='CascadeLineFit')
        tray.AddModule('I3CLastModule',
            InputReadout='TWOfflinePulsesHLC',
            Name='CascadeLast')

        from icecube.CascadeL3_IC79.level3.cutparams import FillRatio
        tray.AddSegment(FillRatio,
            Pulses="OfflinePulses", Vertex="L4MonopodFit", Output="L4MonopodFitFillRatio",
            Scale=0.3, If=lambda frame: True)

    return ['CascadeLineFit', 'CascadeLastParams', 'L4MonopodFitFillRatio']

def validate_input_files(fnames, logfile=None):
    """
    Check input files for corruption, optionally copying them to local scratch space first.
    """
    import os
    from subprocess import call, check_call
    valid = []
    corrupt = []
    scratch = os.environ.get('_CONDOR_SCRATCH_DIR', None)
    for fn in fnames:
        if scratch:
            newfn = os.path.join(scratch, os.path.basename(fn))
            check_call(['cp', fn, newfn])
            fn = newfn

        if fn.endswith('.bz2'):
            bad = call(['bzip2', '-t', fn]) != 0
        elif fn.endswith('.gz'):
            bad = call(['gzip', '-t', fn]) != 0
        else:
            bad = False

        if bad:
            corrupt.append(fn)
        else:
            valid.append(fn)

    if len(corrupt) and logfile:
        if not os.path.isdir(os.path.dirname(logfile)):
            check_call(['mkdir', '-p', os.path.dirname(logfile)])
        with open(logfile, 'a') as fh:
            for fn in corrupt:
                fh.write(fn+'\n')
    return valid

def shift_to_maximum(shower, ref_energy):
    """
    PPC does its own cascade extension, leaving the showers at the
    production vertex. Reapply the parametrization to find the
    position of the shower maximum, which is also the best approximate
    position for a point cascade.
    """
    import numpy
    from icecube import dataclasses
    from icecube.icetray import I3Units
    a = 2.03 + 0.604 * numpy.log(ref_energy/I3Units.GeV)
    b = 0.633
    lrad = (35.8*I3Units.cm/0.9216)
    lengthToMaximum = ((a-1.)/b)*lrad
    p = dataclasses.I3Particle(shower)
    p.energy = ref_energy
    p.fit_status = p.OK
    p.pos.x = shower.pos.x + p.dir.x*lengthToMaximum
    p.pos.y = shower.pos.y + p.dir.y*lengthToMaximum
    p.pos.z = shower.pos.z + p.dir.z*lengthToMaximum
    return p

def get_losses(frame):
    from icecube import dataclasses
    from icecube.icetray import I3Units
    import copy

    mctree = frame['I3MCTree']
    cascade = copy.copy(mctree.most_energetic_cascade)
    if cascade is None:
        return
    losses = 0
    emlosses = 0
    hadlosses = 0
    for p in mctree:
        if not p.is_cascade: continue
        if not p.location_type == p.InIce: continue
        # catch a bug in simprod set 9250 where CMC cascades have non-null shapes
        # if p.shape == p.Dark or (p.shape == p.Null and p.type != p.EMinus): continue
        if p.shape == p.Dark: continue
        if p.type in [p.Hadrons, p.PiPlus, p.PiMinus, p.NuclInt]:
            hadlosses += p.energy
            if p.energy < 1*I3Units.GeV:
                losses += 0.8*p.energy
            else:
                energyScalingFactor = 1.0 + ((p.energy/I3Units.GeV/0.399)**-0.130)*(0.467 - 1)
                losses += energyScalingFactor*p.energy
        else:
            emlosses += p.energy
            losses += p.energy
    if losses == 0:
        print(frame['I3MCTree'])
        print([str(p.shape) for p in frame['I3MCTree']])
    frame['refcascade'] = shift_to_maximum(cascade, losses)





from icecube import DomTools
from optparse import OptionParser
from icecube import icetray, phys_services, simclasses, VHESelfVeto
from icecube import dataclasses, dataio
from I3Tray import *


@icetray.traysegment
def hese_filter(tray, name,
                pulses='InIcePulses',
                pulses_split='SplitInIcePulses',
                only_InIceSplit=True):

    def InIceSplitOnly(frame):
        if "I3EventHeader" not in frame:
            return False
        h = frame["I3EventHeader"]
        if h.sub_event_stream != "InIceSplit":
            return False
        return True

    if only_InIceSplit:
        tray.AddModule(InIceSplitOnly, "InIceSplitOnly",
                       Streams=[icetray.I3Frame.Physics])

    def EnsureInIcePulses(frame):
        # if pulses_split not in frame: return False
        if pulses not in frame:
            print("Pulses", pulses, "missing in frame:",
                  frame["I3EventHeader"])
            return False
        return True

    tray.AddModule(EnsureInIcePulses, "EnsureInIcePulses",
                   Streams=[icetray.I3Frame.Physics])

    tray.AddModule('HomogenizedQTot', 'qtot_total',
                   Pulses=pulses_split, Output="QTot")

    def qtotcut(fr):
        qtot = fr["QTot"].value
        if qtot < 1500.:
            return False
        else:
            return True

    tray.AddModule(qtotcut, 'qtotcut')

    tray.AddModule('I3LCPulseCleaning', 'cleaning',
                   OutputHLC='HLCPulses', OutputSLC='', Input=pulses_split)
    tray.AddModule('VHESelfVeto', 'selfveto', Pulses='HLCPulses')
    tray.AddModule('HomogenizedQTot', 'qtot_causal', Pulses=pulses_split,
                   Output='CausalQTot', VertexTime='VHESelfVetoVertexTime')

    # muon tagger
    tray.AddModule('VHESelfVeto', Pulses='HLCPulses', TopBoundaryWidth=60,
                   BottomBoundaryWidth=10, DustLayer=-10000,
                   OutputBool='MuonTag', OutputVertexTime='MuonTagTime',
                   OutputVertexPos='MuonTagPos', Geometry="I3Geometry")

    # shrink the detector (i.e. remove the muon tagging region)
    tray.AddModule('DetectorShrinker', Pulses=pulses_split,
                   OutPulses=pulses_split+'Trimmed', TopBoundaryWidth=90,
                   BottomBoundaryWidth=10, OutGeometry='I3ShrunkenGeometry',
                   InGeometry="I3Geometry")
    tray.AddModule('I3LCPulseCleaning', OutputHLC='HLCPulsesTrimmed',
                   OutputSLC="", Input=pulses_split+'Trimmed')

    tray.AddModule('VHESelfVeto', Pulses='HLCPulsesTrimmed',
                   Geometry='I3ShrunkenGeometry',
                   OutputBool='VHEInnerSelfVeto',
                   OutputVertexTime='VHEInnerSelfVetoVertexTime',
                   OutputVertexPos='VHEInnerSelfVetoVertexPos')
    tray.AddModule('HomogenizedQTot', Pulses=pulses_split,
                   Output='MuonTagCausalQTot', VertexTime='MuonTagTime')
    tray.AddModule('HomogenizedQTot', Pulses=pulses_split+'Trimmed',
                   Output='InteriorCausalQTot',
                   VertexTime='VHEInnerSelfVetoVertexTime')

    def hese_filter(fr):
        # if fr["CausalQTot"].value<6000:
        #    return False
        print("SelfVeto: ", fr["VHESelfVeto"].value, ", muon: ",
              fr["MuonTag"].value, ", qtot: ", fr["CausalQTot"].value,
              fr["I3EventHeader"].run_id, fr["I3EventHeader"].event_id)
        if "VHESelfVeto" not in fr or "MuonTag" not in fr:
            return False
        # first let all true HESE events pass
        if fr["VHESelfVeto"].value is False:
            print("HESE event!!!")
            return True
        if fr["VHESelfVeto"].value is True and fr["MuonTag"].value is False:
            print("Muon Tag event!")
            return True
        return False

    tray.Add(hese_filter)

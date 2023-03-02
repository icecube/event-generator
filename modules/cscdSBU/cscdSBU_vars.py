from icecube import icetray, dataclasses, dataio
from icecube.icetray import traysegment
from I3Tray import *

from .geometry import distance_to_icecube_hull
from .geometry import distance_to_deepcore_hull


@icetray.traysegment
def addvars(tray, name, pulses='OfflinePulses', vertex='cscdSBU_MonopodFit4'):

    def get_current_keys(frame):
        current_keys = set(frame.keys())
        frame['current_keys'] = dataclasses.I3VectorString(current_keys)
    tray.AddModule(get_current_keys, 'get_current_keys_'+vertex)

    # Add HLC pulses if not there yet
    icetray.load("DomTools", False)
    tray.AddModule("I3LCPulseCleaning", name+"_I3LCPulseCleaning",
                   Input=pulses,
                   OutputHLC=pulses+'HLC',
                   OutputSLC="",
                   If=lambda frame: pulses+'HLC' not in frame)

    # add Qtot/MaxQtotRatio calculations
    from .cscdSBU_misc import misc
    tray.AddSegment(misc, 'misc_'+vertex, pulses=pulses)

    # add outer layer veto related variables
    from icecube import CascadeVariables
    from icecube import phys_services
    load("CascadeVariables")
    tray.AddModule("I3VetoModule", "Veto_Common_Off_"+vertex,
                   HitmapName=pulses,
                   OutputName='cscdSBU_Veto_' + pulses,
                   DetectorGeometry=86,
                   useAMANDA=False,
                   FullOutput=True,
                   )

    # -----------------------------------------------------------------------
    # Added by Mirco: sometimes Veto_SRT* does not exist? Add it in this case
    # -----------------------------------------------------------------------
    tray.AddModule("I3VetoModule", "Veto_Common_SRTOFF_"+vertex,
                   HitmapName='SRTOfflinePulses',
                   OutputName="Veto_SRTOfflinePulses",
                   DetectorGeometry=86,
                   useAMANDA=False,
                   FullOutput=True,
                   If=lambda f: 'SRTOfflinePulses' in f and (
                    'Veto_SRTOfflinePulses' not in f and
                    'Veto_SRTInIcePulses' not in f),
                   )
    tray.AddModule("I3VetoModule", "Veto_Common_SRTICE_"+vertex,
                   HitmapName='SRTInIcePulses',
                   OutputName="Veto_SRTInIcePulses",
                   DetectorGeometry=86,
                   useAMANDA=False,
                   FullOutput=True,
                   If=lambda f: 'SRTInIcePulses' in f and (
                    'Veto_SRTOfflinePulses' not in f and
                    'Veto_SRTInIcePulses' not in f),
                   )
    tray.AddModule("I3VetoModule", "Veto_Common_SRTICEDST_"+vertex,
                   HitmapName='SRTInIceDSTPulses',
                   OutputName="Veto_SRTInIceDSTPulses",
                   DetectorGeometry=86,
                   useAMANDA=False,
                   FullOutput=True,
                   If=lambda f: 'SRTInIceDSTPulses' in f and (
                    'Veto_SRTInIceDSTPulses' not in f and
                    'Veto_SRTOfflinePulses' not in f and
                    'Veto_SRTInIcePulses' not in f),
                   )
    # -----------------------------------------------------------------------

    def veto(frame):
        if frame.Has('Veto_SRTOfflinePulses'):
            veto = frame['Veto_SRTOfflinePulses']
        elif frame.Has('Veto_SRTInIceDSTPulses'):
            veto = frame['Veto_SRTInIceDSTPulses']
        elif frame.Has('Veto_SRTInIcePulses'):
            veto = frame['Veto_SRTInIcePulses']
        else:
            veto = None
            print('WARNING did not find Veto_SRT object in frame!')
            print(frame)
            raise ValueError

        if veto is None:
            frame['cscdSBU_VetoDepthFirstHit'] = dataclasses.I3Double(float('nan'))
            frame['cscdSBU_VetoEarliestLayer'] = dataclasses.I3Double(float('nan'))

        else:
            frame['cscdSBU_VetoDepthFirstHit'] = dataclasses.I3Double(veto.depthFirstHit)
            frame['cscdSBU_VetoEarliestLayer'] = dataclasses.I3Double(veto.earliestLayer)

        vetoOff = frame['cscdSBU_Veto_' + pulses]
        frame['cscdSBU_VetoMaxDomChargeOM'] = dataclasses.I3Double(vetoOff.maxDomChargeOM)
        frame['cscdSBU_VetoMaxDomChargeString'] = dataclasses.I3Double(vetoOff.maxDomChargeString)

        return True

    tray.AddModule(veto, "veto_"+vertex)

    # add Achim's I3Scale Variable
    from .cscdSBU_I3Scale import I3Scale
    tray.AddModule(I3Scale,"icecubescale_"+vertex,
                   vertex        = vertex,
                   geometry      = "I3Geometry",
                   ic_config     = "IC86",
                   outputname    = "cscdSBU_I3XYScale"
                   )

    #tray.AddModule(I3Scale,"icecubescale2",
    #               vertex        = "cscdSBU_MonopodFit4_noDC",
    #               geometry      = "I3Geometry",
    #               ic_config     = "IC86",
    #               outputname    = "cscdSBU_I3XYScale_noDC"
    #               )

    # add Jakobs Track Charges
    #sys.path.append(os.path.expandvars('$I3_BUILD/CascadeL3_IC79/resources/level4'))
    from . import common2 as common
    tray.AddSegment(common.TrackVetoes, "pulses_all"+vertex, vertex=vertex,
                    pulses=pulses)
    '''
    tray.AddSegment(common.TrackVetoes_noDC, "pulses_noDC", vertex='cscdSBU_MonopodFit4')
    tray.AddSegment(common.TrackVetoes, "pulses_all_monopod_noDC", vertex='cscdSBU_MonopodFit4_noDC')
    '''

    # add Mariola variables
    from .cscdSBU_polygon import ContainmentCut
    tray.AddSegment(ContainmentCut, 'Containment_'+vertex, Vertex=vertex,
                    pulses=pulses)
    #from cscdSBU_polygon import ContainmentCut
    #tray.AddSegment(ContainmentCut, 'Containment_Monopod4_noDC', Vertex="cscdSBU_MonopodFit4_noDC")


    from .mlb_DelayTime_noNoise import calc_dt_nearly_ice
    # calculate delay time with new monopod
    tray.AddModule(calc_dt_nearly_ice,'delaytime_'+vertex,
                   name='cscdSBU_' + vertex,
                   reconame=vertex,
                   # pulsemapname='OfflinePulsesHLC_noSaturDOMs',
                   pulsemapname=pulses,
                   )
    #tray.AddModule(calc_dt_nearly_ice,'delaytime_monopod_noDC',name='cscdSBU_MonopodFit4_noDC',
    #               reconame='cscdSBU_MonopodFit4_noDC',pulsemapname='OfflinePulsesHLC_noSaturDOMs')

    # -----------------------------------------------------------
    # Calculate Vertex distance
    # -----------------------------------------------------------
    def add_distances(frame, particle_key):
        pos = frame[particle_key].pos
        frame['distance_icecube'] = dataclasses.I3Double(
                                  distance_to_icecube_hull(pos))
        frame['distance_deepcore'] = dataclasses.I3Double(
                                  distance_to_deepcore_hull(pos))
    tray.AddModule(add_distances, 'add_distances'+vertex, particle_key=vertex)

    # -----------------------------------------------------------
    # clean up: gather all newly created values and save in one
    # I3MapDouble named 'cscdSBU_labels_' + vertex
    # -----------------------------------------------------------
    def combine_and_clean_up(frame):
        cscdSBU_labels_double = {}
        cscdSBU_labels_int = {}
        cscdSBU_labels_bool = {}
        new_keys = set(frame.keys()) - set(frame['current_keys'])
        for k in new_keys:
            if isinstance(frame[k], dataclasses.I3Double):
                cscdSBU_labels_double[k] = frame[k].value

            elif isinstance(frame[k], icetray.I3Int):
                cscdSBU_labels_int[k] = frame[k].value

            elif isinstance(frame[k], icetray.I3Bool):
                cscdSBU_labels_bool[k] = frame[k].value

            del frame[k]

        frame['cscdSBU_I3Double_'+vertex] = dataclasses.I3MapStringDouble(
                                                        cscdSBU_labels_double)
        frame['cscdSBU_I3Int_'+vertex] = dataclasses.I3MapStringDouble(
                                                        cscdSBU_labels_int)
        frame['cscdSBU_I3bool_'+vertex] = dataclasses.I3MapStringDouble(
                                                        cscdSBU_labels_bool)
    tray.AddModule(combine_and_clean_up, 'combine_and_clean_up_'+vertex)

from icecube import icetray,dataclasses
from icecube.icetray import traysegment

@icetray.traysegment
def misc(tray, name, pulses='OfflinePulses'):

    def removeSaturatedDOMs(frame,pulses):

        mask = dataclasses.I3RecoPulseSeriesMapMask(frame, pulses)
        pmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses)

        saturDOMs=[]
        if frame.Has('SaturationWindows'):
            for x in frame['SaturationWindows'].keys():
                saturDOMs.append(x)

        for om in saturDOMs:
                if om in pmap:

                    mask.set(om, False)

        frame['%s_noSaturDOMs' %(pulses)] = mask

    tray.AddModule(removeSaturatedDOMs, name+'cleanOffHLC',
                   pulses=pulses+'HLC')

    def qtotCalculation(frame,pulses):
        dc_strings = range(79, 87)
        qtot_out_IC = 0.0
        qtot_out = 0.0
        qt = 0.0
        qtot_out = 0.0
        maxCharge = 0.0
        pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses)
        for om, thesepulses in pulsemap:
            qt = sum([pulse.charge for pulse in thesepulses])
            qtot_out += qt
            if not om.string in dc_strings:
                qtot_out_IC += qt
            if qt > maxCharge:
                maxCharge = qt
        if qtot_out>0:
            MaxQTotRatio = maxCharge/qtot_out
        else:
            MaxQTotRatio = 0

        if 'HLC' in pulses:
            frame['cscdSBU_MaxQtotRatio_HLC'] = dataclasses.I3Double(MaxQTotRatio)
            frame['cscdSBU_Qtot_HLC'] = dataclasses.I3Double(qtot_out)
            frame['cscdSBU_Qtot_HLC_IC'] = dataclasses.I3Double(qtot_out_IC)
        else:
            frame['cscdSBU_MaxQtotRatio_%s' %(pulses)] = dataclasses.I3Double(MaxQTotRatio)
            frame['cscdSBU_Qtot_%s' %(pulses)] = dataclasses.I3Double(qtot_out)
            frame['cscdSBU_Qtot_%s_IC' %(pulses)] = dataclasses.I3Double(qtot_out_IC)
        return True

    tray.AddModule(qtotCalculation, name+'qtotal', pulses=pulses)
    tray.AddModule(qtotCalculation, name+'qtotalHLC', pulses=pulses+'HLC')



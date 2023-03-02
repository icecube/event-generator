from __future__ import print_function, division
from icecube import icetray,dataclasses
from icecube.icetray import traysegment

@icetray.traysegment
def select_L3SC(tray, name):
    print("... selecting cascade L3 single, contained branch.")
    def selection(frame):
            # remove events that are uncontained or coincident
            if frame.Has('CscdL3_Cont_Tag') and frame['CscdL3_Cont_Tag']==1:
                    return True
            return False

    tray.AddModule(selection, 'containment')

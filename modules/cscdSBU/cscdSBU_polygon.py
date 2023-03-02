def ContainmentCut(tray, name, Pulses="TWOfflinePulsesHLC", Vertex="CscdL3_Credo_SpiceMie"):
    """
    Tag reasonably contained events. Not actually a cut.
    """
    from icecube import icetray, dataclasses
    from .mlb_CutParams import ContainmentCut, RingClassifier

    vertex_tag = Vertex+"Contained"
    pulse_tag = Pulses+"MaxQRing"
    final_tag = "cscdSBU_PolygonContTag_"+Vertex

    tray.AddModule(ContainmentCut, name+'_PolygonCut',
                       Vertex=Vertex, Output=vertex_tag)


    tray.AddModule(RingClassifier, name+"_RingClassifier",
                       Pulses=Pulses, Output=pulse_tag)

    def define_containment(frame):
        vertex = frame[vertex_tag]
        ring = frame[pulse_tag]
        inside = vertex.value and ring.value < 3
        frame[final_tag] = icetray.I3Bool(inside)
        return True

    tray.AddModule(define_containment, name+"_Combiner")



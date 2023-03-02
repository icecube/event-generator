from icecube import icetray, dataclasses

def FillRatio(tray, name, Vertex='CscdL3_Credo_SpiceMie', Pulses='OfflinePulses', Output='CredoFitFillRatio',
    BadDOMList='BadDomsListSLC', BadOMs=[], NoDeepCore=False, UseCharge=False, Scan=False,
    Lite=True, Scale=0.3, If=lambda: True):
    """
    Run FillRatio on the given pulse series and vertex, optionally scanning over
    a set of mean distance scales.

    :param Vertex: name of an I3Particle to use as a vertex
    :param Pulses: name of the pulses to use to find the mean vertex-hit distance
    :param BadDOMList: name of the BadDOMList in the DetectorStatus frame
    :param BadOMs: static list of DOMs to exclude from the calculation, to be
                   merged with the contents of the dynamic BadDOMList
    :param NoDeepCore: exclude DeepCore DOMs from the calculation
    :param UseCharge: weight DOMs by charge in the mean distance calculation
    :param Scan: scan over a range of distance scales to find the best value
    :param Lite: use I3FillRatioLite rather than I3FillRatioModule. I3FillRatioLite
                 only calculates the fill ratio from the mean distance (rather than
             mean, stddev, mean + stddev, nch, and energy), but is 10--30 times
             faster (depending on the size of the BadDOMList).
    """

    from icecube import fill_ratio

    output = []

    skipOMs = list(BadOMs)
    # Instruct FillRatioModule to ignore DeepCore DOMs, and mask the
    # corresponding pulses out of the input
    PulseName = Pulses
    if NoDeepCore:
#        for string in xrange(81, 87):
        for string in xrange(79, 87):
            for om in xrange(1, 61):
                skipOMs.append(icetray.OMKey(string, om))
        MaskedPulses = name+'_'+Pulses
        def DeepCoreStripper(frame):
            try:
                mask = dataclasses.I3RecoPulseSeriesMapMask(frame, Pulses)
            except:
                return
            rpsm = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, Pulses)
            for k in rpsm.iterkeys():
#                if (k.string) > 80:
                if (k.string) > 78:
                    mask.set(k, False);
            frame[MaskedPulses] = mask
        tray.AddModule(DeepCoreStripper, name+'_DeepCoreStripper', If=If)
        PulseName = MaskedPulses

    ampweight = 0.
    if UseCharge:
        ampweight = 1.

    if Scan:
        step = 0.01
        alpha = 0.2
        while alpha <= 0.4 + step:
            out = "%s_%.3f" % (name, alpha)
            tag = out
            if Lite:
                tray.AddModule("I3FillRatioLite", tag,
                    ExcludeDOMs=skipOMs,
                    BadDOMListName=BadDOMList,
                    Pulses=PulseName,
                    Output=out,
                    Vertex=Vertex,
                    Scale=alpha,
                    AmplitudeWeightingPower=ampweight,
                    If=If,
                )
            else:
                tray.AddModule("I3FillRatioModule", tag,
                    BadOMs=skipOMs,
                    BadDOMListName=BadDOMList,
                    DetectorConfig=1, #IceCube, as opposed to IC+AMANDA
                    RecoPulseName=PulseName,
                    ResultName=out,
                    VertexName=Vertex,
                    SphericalRadiusMean=alpha,
                    SphericalRadiusRMS=alpha,
                    SphericalRadiusMeanPlusRMS=alpha,
                    SphericalRadiusNCh=alpha,
                    AmplitudeWeightingPower=ampweight,
                    If=If,
                )
            output.append(out)
            alpha += step
    else:
        mean, rms, mean_plus_rms, nch = Scale, Scale, Scale, Scale

        #tag = Vertex + "FillRatio"
        tag = Output
        output = [tag]

        if Lite:
            tray.AddModule("I3FillRatioLite", name + "_" + tag,
                ExcludeDOMs=skipOMs,
                BadDOMListName=BadDOMList,
                Pulses=PulseName,
                Output=tag,
                Vertex=Vertex,
                Scale=mean,
                AmplitudeWeightingPower=ampweight,
                If=If,
            )
        else:
            tray.AddModule("I3FillRatioModule", tag,
                BadOMs=skipOMs,
                BadDOMList=BadDOMList,
                DetectorConfig=1, #IceCube, as opposed to IC+AMANDA
                RecoPulseName=PulseName,
                ResultName=tag,
                VertexName=Vertex,
                SphericalRadiusMean=mean,
                SphericalRadiusRMS=rms,
                SphericalRadiusMeanPlusRMS=mean_plus_rms,
                SphericalRadiusNCh=nch,
                AmplitudeWeightingPower=ampweight,
                If=If,
            )

    if NoDeepCore:
        tray.AddModule('Delete', name+'_DeleteDeepCoreLessPulses', Keys=[MaskedPulses], If=If)
    return output

def charge_sort(frame, pulsename):
    """
    Sort collected charge into two buckets: DeepCore fiducial volume and everything else.
    """
    if not pulsename in frame:
        return True
    qtot = lambda pulses: sum([q.charge for q in pulses])
    pulsemap = frame[pulsename]
    qt_ic = 0
    qt_dc = 0
    ic_strings = set([26, 27, 35, 36, 37, 45, 46])
    for om, pulses in pulsemap.iteritems():
        if (om.string >= 79 and om.om >= 11) or (om.string in ic_strings and om.om >= 37):
            qt_dc += qtot(pulses)
        else:
            qt_ic += qtot(pulses)
    out = dataclasses.I3MapStringDouble()
    out['qtot_bulk'] = qt_ic
    out['qtot_deepcore'] = qt_dc
    frame['%s_chargeclass' % pulsename] = out
    return True

def ic_dc_classify(tray, pulsename):

    tray.AddModule(charge_sort, '%s_chargeclass' % pulsename, pulsename=pulsename)
    return ['%s_chargeclass' % pulsename]

class RingFinder(object):
    def __init__(self):
        ###### IC86 ####
        self.ring3 = set([1, 2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 73, 72, 78, 77, 76, 75, 68, 60, 51, 41, 31, 22, 14, 7])
        self.ring2 = set([10, 11, 12, 20, 29, 39, 49, 58, 66, 71, 70, 64, 65, 69, 61, 52, 42, 32, 23, 15, 8, 9])
        self.ring1 = set([17, 18, 19, 28, 38, 48, 55, 56, 57, 63, 62, 53, 43, 34, 25, 44, 54, 47, 33, 24, 16])
        self.core = set([26, 27, 35, 36, 37, 45, 46, 81, 82, 83, 84, 85, 86, 80, 79]) # already part of DC fiducial
        self.rings = (self.core, self.ring1, self.ring2, self.ring3)
        ###### IC79 ####
#        self.ring3 = set([2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 73, 72, 78, 77, 76, 75, 68, 60, 51, 41, 32, 23, 15, 8])
#        self.ring2 = set([9, 10, 11, 12, 20, 29, 39, 49, 58, 66, 71, 70, 64, 65, 69, 61, 52, 42, 33, 24, 16])
#        self.ring1 = set([17, 18, 19, 28, 38, 48, 55, 56, 57, 63, 62, 53, 43, 34, 25, 44, 54, 47])
#        self.core = set([26, 27, 35, 36, 37, 45, 46, 81, 82, 83, 84, 85, 86]) # already part of DC fiducial
#        self.rings = (self.core, self.ring1, self.ring2, self.ring3)

    def plot_geometry(self, gcd):
        from icecube import icetray, dataclasses, dataio
        import pylab
        f = dataio.I3File(gcd)
        fr = None
        while f.more():
            fr = f.pop_frame()
            if fr.Stop is icetray.I3Frame.Geometry:
                break
        omgeo = fr['I3Geometry'].omgeo
        strings = set([om.string for om in omgeo.iterkeys()])
        labels = ['Core'] + ['Ring %d' % i for i in xrange(1,4)]
        colors = iter(['k', 'b', 'r', 'g'])
        syms = iter(['x','d','s','o'])

        fig = pylab.figure(figsize=(6,6))
        size = 1
        offset = 5
        for ring, label in zip(self.rings, labels):
            x = [omgeo[icetray.OMKey(string,1)].position.x for string in ring]
            y = [omgeo[icetray.OMKey(string,1)].position.y for string in ring]
            pylab.scatter(x, y, s=size, label=label, c=colors.next(), marker=syms.next())
            for x_, y_, string in zip(x, y, ring):
                pylab.text(x_+offset, y_+offset, string, fontsize=10)
            size = 100
            offset = 20

        pylab.xlabel('Grid x [m]')
        pylab.ylabel('Grid y [m]')
        pylab.title('IC79 Cascade L3 ring scheme')
        pylab.legend(loc='best', prop=dict(size='small'))
        pylab.gca().set_aspect('equal', 'datalim')

    def max_charge(self, pulsemap):
        qtot = lambda pulses: sum([q.charge for q in pulses])
        maxcharge = 0
        maxom = icetray.OMKey(0,0)
        for om, pulses in pulsemap.iteritems():
            qt = qtot(pulses)
            if qt > maxcharge:
                maxcharge = qt
                maxom = om
        return maxom

    def earliest_pulse(self, pulsemap):
        time = lambda pulses: min([q.time for q in pulses])
        mintime = float('inf')
        maxom = icetray.OMKey(0,0)
        for om, pulses in pulsemap.iteritems():
            t = time(pulses)
            if t < mintime:
                mintime = t
                maxom = om
        return maxom

    def __call__(self, pulsemap):

        maxom = self.max_charge(pulsemap)
        #maxom = self.earliest_pulse(pulsemap)

        maxring = -1
        for i, ring in enumerate(self.rings):
            if maxom.string in ring:
                maxring = i
                break
        return maxring

def RingClassifier(frame, Pulses, Output):
    if frame.Has(Output):
        return
    if not Pulses in frame:
        return

    pulses = frame[Pulses]
    if isinstance(pulses, dataclasses.I3RecoPulseSeriesMapMask):
        pulses = pulses.apply(frame)

    ring = RingClassifier.ringer(pulses)
    frame[Output] = icetray.I3Int(ring)
RingClassifier.ringer = RingFinder()

class ContainmentCut(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.AddOutBox("OutBox")
        self.AddParameter("Vertex", "Vertex to test", None)
        self.AddParameter("Output", "Decision name", None)

        from .polygon import point_in_polygon
        self.point_in_polygon = point_in_polygon

    def Configure(self):
        self.vertex_name = self.GetParameter("Vertex")
        self.output_name = self.GetParameter("Output")

    def Geometry(self, frame):
        import numpy
        geo = frame['I3Geometry'].omgeo
        xe = list(); ye = list()
        for string in RingFinder().ring3:
            key = icetray.OMKey(string, 1)
            g = geo[key]
            xe.append(g.position.x)
            ye.append(g.position.y)

        xe = numpy.array(xe)
        ye = numpy.array(ye)
        order = numpy.argsort(numpy.arctan2(ye, xe))
        self.edge_x = xe[order]
        self.edge_y = ye[order]

        self.PushFrame(frame)

    def Physics(self, frame):
        vertex = frame[self.vertex_name]
        contained = self.point_in_polygon(vertex.pos.x, vertex.pos.y, self.edge_x, self.edge_y)
        frame[self.output_name] = icetray.I3Bool(bool(contained))
        self.PushFrame(frame)


def on_edge(frame, pulsename):
###=============== IC86
    ring3 = set([1, 2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 73, 72, 78, 77, 76, 75, 68, 60, 51, 41, 31, 22, 14, 7])
###=============== IC79
#    ring3 = set([2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 78, 77, 76, 75, 68, 60, 51, 41, 32, 23, 15, 8])
    ring2 = set([9, 10, 11, 12, 20, 29, 39, 49, 58, 66, 73, 72, 71, 70, 69, 61, 52, 42, 33, 24, 16])
    ring1 = set([17, 18, 19, 28, 38, 48, 57, 65, 64, 63, 62, 53, 43, 34, 25])
    ring0 = set([44, 54, 55, 56, 47])
    deepcore = set([26, 27, 35, 36, 37, 45, 46]) # already part of DC fiducial

    if not pulsename in frame:
        return True

    qtot = lambda pulses: sum([q.charge for q in pulses])
    maxcharge = 0
    maxom = icetray.OMKey(0,0)
    for om, pulses in frame[pulsename].iteritems():
        qt = qtot(pulses)
        if qt > maxcharge:
            maxcharge = qt
            maxom = om

    # The event is edge-dominated if the largest charge is seen in the top 3 DOMs
    # on a (non-DeepCore) string or on one of the edge strings
    on_edge = (maxom.om < 3 and om.string > 78) or (maxom.string in ring3)

    frame['%sEdgeVeto' % pulsename] = icetray.I3Bool(on_edge)

    return True

def edge_veto(tray, pulsename):
    tag = '%sEdgeVeto' % pulsename
    tray.AddModule(on_edge, tag, pulsename=pulsename)
    return [tag]

def mask_first_pulse(frame, Pulses):
    mask = dataclasses.I3RecoPulseSeriesMapMask(frame, Pulses)
    mask.set_none()
    for om in frame[Pulses].iterkeys():
        mask.set(om, 0, True)
    frame['First'+Pulses] = mask

def track_reco_cuts(tray, name, Track='MPEFit', Pulses='OfflinePulses'):
    """
    Calculate NDirC, LDirC properly for a given track reco.
    """

    FirstPulses = 'First'+Pulses

    tray.AddModule(mask_first_pulse, name+'FirstPulses',
        Pulses=Pulses,
    )
    tray.AddModule('I3CutsModule', name+'TrackCuts',
        ParticleNames=Track,
        PulsesName=FirstPulses,
        NameTag=name,
    )
    tray.AddModule('Delete', name+'_delete_intermediates',
        Keys=[FirstPulses],
    )

def travel_speeds(frame, Pulses='OfflinePulses', Vertex='CascadeLlhVertexFit'):
    import numpy
    from icecube.dataclasses import I3Constants, I3VectorDouble
    geo = frame['I3Geometry'].omgeo
    pmap = frame[Pulses]
    vertex = frame[Vertex]

    label = 'TravelSpeed_%s_%s' % (Pulses, Vertex)

    speeds  = []
    charges = []
    for om, pulses in pmap:
        d = vertex.pos.calc_distance(geo[om].position)
        for pulse in pulses:
            speeds.append(d/((vertex.time-pulse.time)*I3Constants.c))
            charges.append(pulse.charge)
    edges = numpy.linspace(-5, 5, 101)
    hist, edges = numpy.histogram(speeds, bins=edges, weights=charges)
    dubs = I3VectorDouble(hist)

    frame[label] = dubs

def LLHRatios(tray, name):
    from .reco import SPELLHCalculator
    from icecube.dataclasses import I3MapStringDouble

    tray.AddSegment(SPELLHCalculator, 'SPEFit4_SLC_HLCFitParams',
        Pulses='TWNFEMergedPulsesHLC', Track='SPEFit4_SLC')

    def calc_ratios(frame):
        mappy = I3MapStringDouble()
        try:
            mappy['SPE4_CscdLLH'] = frame['CascadeLlhVertexFitParams'].NegLlh - frame['SPEFit4FitParams'].logl
            mappy['SPE4_SLC_CscdLLH_SLC'] = frame['CascadeLlhVertexSLCFitParams'].NegLlh - frame['SPEFit4_SLCFitParams'].logl
            mappy['SPE4_SLC_CscdLLH'] = frame['CascadeLlhVertexFitParams'].NegLlh - frame['SPEFit4_SLC_HLCFitParams'].logl
            frame[name] = mappy
        except KeyError:
            return
    tray.AddModule(calc_ratios, name)

    return [name]

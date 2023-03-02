#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' I3Module to add Labels for deep Learning
'''
import numpy as np
from icecube import dataclasses, icetray
from scipy.spatial import ConvexHull

from . import geometry
from . import icecube_labels


class MCLabelsDeepLearning(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("PulseMapString", "Name of pulse map to use.",
                          'InIcePulses')
        self.AddParameter("PrimaryKey", "Name of the primary.", 'MCPrimary')
        self.AddParameter("OutputKey", "Save labels to this frame key.",
                          'LabelsDeepLearning')
        self.AddParameter("IsMuonGun",
                          "Indicate whether this is a MuonGun dataset.", False)

    def Configure(self):
        self._pulse_map_string = self.GetParameter("PulseMapString")
        self._primary_key = self.GetParameter("PrimaryKey")
        self._output_key = self.GetParameter("OutputKey")
        self._is_muongun = self.GetParameter("IsMuonGun")

    def Geometry(self, frame):
        geoMap = frame['I3Geometry'].omgeo
        domPosDict = {(i[0][0], i[0][1]): (i[1].position.x,
                                           i[1].position.y,
                                           i[1].position.z)
                      for i in geoMap if i[1].omtype.name == 'IceCube'}
        points = [
            domPosDict[(31, 1)], domPosDict[(1, 1)],
            domPosDict[(6, 1)], domPosDict[(50, 1)],
            domPosDict[(74, 1)], domPosDict[(72, 1)],
            domPosDict[(78, 1)], domPosDict[(75, 1)],

            domPosDict[(31, 60)], domPosDict[(1, 60)],
            domPosDict[(6, 60)], domPosDict[(50, 60)],
            domPosDict[(74, 60)], domPosDict[(72, 60)],
            domPosDict[(78, 60)], domPosDict[(75, 60)]
            ]
        self._convex_hull = ConvexHull(points)
        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)

    def Physics(self, frame):
        labels = icecube_labels.get_labels(
                    frame=frame,
                    convex_hull=self._convex_hull,
                    domPosDict=self._dom_pos_dict,
                    primary=frame[self._primary_key],
                    pulse_map_string=self._pulse_map_string,
                    is_muongun=self._is_muongun
                    )

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsTau(MCLabelsDeepLearning):
    def Physics(self, frame):
        labels = icecube_labels.get_tau_labels(
                    frame=frame,
                    convex_hull=self._convex_hull)

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsCascadeParameters(MCLabelsDeepLearning):
    def Physics(self, frame):
        labels = icecube_labels.get_cascade_parameters(
                                            frame=frame,
                                            primary=frame[self._primary_key],
                                            convex_hull=self._convex_hull,
                                            extend_boundary=500,
                                            )

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsCascades(MCLabelsDeepLearning):
    def Physics(self, frame):
        labels = icecube_labels.get_cascade_labels(
                                            frame=frame,
                                            primary=frame[self._primary_key],
                                            convex_hull=self._convex_hull,
                                            )

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)


class MCLabelsCorsikaAzimuthExcess(MCLabelsDeepLearning):
    def Physics(self, frame):
        # create empty labelDict
        labels = dataclasses.I3MapStringDouble()

        muons_inside = icecube_labels.get_muons_inside(frame,
                                                       self._convex_hull)
        labels['NoOfMuonsInside'] = len(muons_inside)

        # get muons
        mostEnergeticMuon = icecube_labels.get_most_energetic_muon_inside(
                                                frame, self._convex_hull,
                                                muons_inside=muons_inside)
        if mostEnergeticMuon is None:
            labels['Muon_energy'] = np.nan
            labels['Muon_vertexX'] = np.nan
            labels['Muon_vertexY'] = np.nan
            labels['Muon_vertexZ'] = np.nan
            labels['Muon_vertexTime'] = np.nan
            labels['Muon_azimuth'] = np.nan
            labels['Muon_zenith'] = np.nan
        else:
            labels['Muon_energy'] = mostEnergeticMuon.energy
            labels['Muon_vertexX'] = mostEnergeticMuon.pos.x
            labels['Muon_vertexY'] = mostEnergeticMuon.pos.y
            labels['Muon_vertexZ'] = mostEnergeticMuon.pos.z
            labels['Muon_vertexTime'] = mostEnergeticMuon.time
            labels['Muon_azimuth'] = mostEnergeticMuon.dir.azimuth
            labels['Muon_zenith'] = mostEnergeticMuon.dir.zenith

        # write to frame
        frame.Put(self._output_key, labels)

        self.PushFrame(frame)

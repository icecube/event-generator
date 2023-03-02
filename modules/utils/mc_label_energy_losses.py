#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' I3Module to add Labels for deep Learning
'''
import numpy as np
from icecube import dataclasses, icetray
import geometry
import icecube_labels
from scipy.spatial import ConvexHull



class MCLabelsMuonEnergyLosses(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("MuonKey","Name of the muon.",'MCPrimary')
        self.AddParameter("OutputKey",
                          "Save labels to this frame key.",
                          'LabelsDeepLearning')
        self.AddParameter("BinWidth","Bin width [in meters].",10)
        self.AddParameter("ExtendBoundary",
                          "Extend boundary of convex hull [in meters].",
                          150)
        self.AddParameter("IncludeUnderOverFlow",
                          "Include over and under flow bins.",
                          False)
        self.AddParameter("ForceNumBins",
                          "Force number of bins to be this value."
                          "Will append zeros or remove last bins."
                          ,None)

    def Configure(self):
        self._muon_key = self.GetParameter("MuonKey")
        self._output_key = self.GetParameter("OutputKey")
        self._bin_width = self.GetParameter("BinWidth")
        self._extend_boundary = self.GetParameter("ExtendBoundary")
        self._include_under_over_flow = self.GetParameter("IncludeUnderOverFlow")
        self._force_num_bins = self.GetParameter("ForceNumBins")
        

    def Geometry(self, frame):
        geoMap = frame['I3Geometry'].omgeo
        domPosDict = {
                        (i[0][0],i[0][1]):( i[1].position.x,
                                            i[1].position.y,
                                            i[1].position.z) 
                        for i in geoMap if i[1].omtype.name == 'IceCube'
                     }
        points = [
            domPosDict[(31,1)],domPosDict[(1,1)],
            domPosDict[(6,1)],domPosDict[(50,1)],
            domPosDict[(74,1)],domPosDict[(72,1)],
            domPosDict[(78,1)],domPosDict[(75,1)],
            
            domPosDict[(31,60)],domPosDict[(1,60)],
            domPosDict[(6,60)],domPosDict[(50,60)],
            domPosDict[(74,60)],domPosDict[(72,60)],
            domPosDict[(78,60)],domPosDict[(75,60)]
            ]
        self._convex_hull = ConvexHull(points)
        self._dom_pos_dict = domPosDict
        self.PushFrame(frame)

    def Physics(self, frame):

        labels = dataclasses.I3MapStringDouble()

        binnned_energy_losses = icecube_labels.get_inf_muon_binned_energy_losses(
                                    frame = frame,
                                    convex_hull = self._convex_hull,
                                    muon = frame[self._muon_key],
                                    bin_width = self._bin_width,
                                    extend_boundary = self._extend_boundary,
                                    include_under_over_flow=self._include_under_over_flow,
                                    )

        # force the number of bins to match ForceNumBins 
        if not self._force_num_bins is None:

            num_bins = len(binnned_energy_losses)

            # too many bins: remove last bins
            if num_bins > self._force_num_bins:
                binnned_energy_losses = binnned_energy_losses[:self._force_num_bins]

            # too few bins: append zeros
            elif num_bins < self._force_num_bins:
                num_bins_to_add = self._force_num_bins - num_bins
                # print('Appending {} zeros'.format(num_bins_to_add))
                binnned_energy_losses = np.concatenate((binnned_energy_losses,
                                                        np.zeros(num_bins_to_add)))

        # write to frame
        for i, energy_i in enumerate(binnned_energy_losses):
            labels['EnergyLoss_{:04d}'.format(i)] = energy_i

        frame.Put(self._output_key,labels)

        self.PushFrame(frame)

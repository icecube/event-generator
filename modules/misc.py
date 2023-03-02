#!/usr/bin/env python
# -*- coding: utf-8 -*
from __future__ import print_function, division

from icecube import dataclasses, icetray


def create_cascade_classification_base_cascades(
        tray, cfg, name='add_cascade_base'):
    """Add cascade classification model base cascades to frame

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray module.
    """
    # --------------------------------------------------
    # Add cascade classification model base cascade
    # --------------------------------------------------
    if 'create_cascade_classification_base_cascades' in cfg:

        def add_cascade_base(frame, config, output_key=None):
            if config['I3ParticleBase'] in frame:
                particle = frame[config['I3ParticleBase']]
                labels = dataclasses.I3MapStringDouble()
                labels['VertexX'] = particle.pos.x
                labels['VertexY'] = particle.pos.y
                labels['VertexZ'] = particle.pos.z
                labels['VertexTime'] = particle.time
                labels['VertexX_unc'] = config['VertexX_unc']
                labels['VertexY_unc'] = config['VertexY_unc']
                labels['VertexZ_unc'] = config['VertexZ_unc']
                labels['VertexTime_unc'] = config['VertexTime_unc']
                if output_key is None:
                    output_key = \
                        'cscd_classification_base_'+config['I3ParticleBase']
                frame[output_key] = labels

        cscd_base_configs = cfg['create_cascade_classification_base_cascades']
        if isinstance(cscd_base_configs, dict):
            cscd_base_configs = [cscd_base_configs]

        for i, cscd_base_config in enumerate(cscd_base_configs):
            tray.AddModule(
                add_cascade_base, name + '_{:03d}'.format(i),
                config=cscd_base_config)


class AddPseudePhysicsFrames(icetray.I3ConditionalModule):
    def __init__(self, context):
        """Class to add pseudo physics frames

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox('OutBox')
        self.AddParameter("RunID", "RunID for I3EventHeader")

    def Configure(self):
        self.run_id = self.GetParameter("RunID")
        self.event_id = 0

    def DAQ(self, frame):
        """Inject casacdes into I3MCtree.

        Parameters
        ----------
        frame : icetray.I3Frame.DAQ
            An I3 q-frame.
        """
        self.PushFrame(frame)
        pseudo_frame = icetray.I3Frame()
        pseudo_frame.Stop = icetray.I3Frame.Physics

        # create pseudo event header
        event_header = dataclasses.I3EventHeader()
        event_header.run_id = self.run_id
        event_header.event_id = self.event_id
        event_header.sub_event_id = 0
        event_header.sub_run_id = 0

        pseudo_frame['I3EventHeader'] = event_header
        self.event_id += 1

        self.PushFrame(pseudo_frame)

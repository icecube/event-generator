from __future__ import print_function, division
from icecube import icetray,dataclasses
from icecube.icetray import traysegment

@icetray.traysegment
def nugen_weights(tray, name):

         # add atmospheric neutrino weights
         from icecube import NewNuFlux
         from icecube.weighting import fluxes, get_weighted_primary
         import math

         honda = NewNuFlux.makeFlux('honda2006')
         honda.knee_reweighting_model = 'gaisserH3a_elbert'
         h3a = fluxes.GaisserH3a()
         ers = NewNuFlux.makeFlux('sarcevic_std')
         ers.knee_reweighting_model = 'gaisserH3a_elbert'
         berss = NewNuFlux.makeFlux('BERSS_H3a_central')

         def saveAtmWeight(frame):
              if(not frame.Has("I3MCTree")):
                  print('---NO MCTree in EvID = ',
                        frame['I3EventHeader'].event_id)
                  return

              get_weighted_primary(frame,"cscdSBU_MCPrimary")
              p = frame['cscdSBU_MCPrimary']
              conv = honda.getFlux(p.type, p.energy, math.cos(p.dir.zenith))
              prompt = ers.getFlux(p.type, p.energy, math.cos(p.dir.zenith))
              prompt_berss = berss.getFlux(p.type, p.energy, math.cos(p.dir.zenith))

              ##  passing rate calulation
              from icecube.AtmosphericSelfVeto import AnalyticPassingFraction
              particleType = p.pdg_encoding
              conventional_veto = AnalyticPassingFraction('conventional', veto_threshold=100)
              prompt_veto = AnalyticPassingFraction('charm', veto_threshold=100)

              ##  add penetrating depth dependence to the self veto probability calculation
              from icecube import MuonGun, simclasses
              surface = MuonGun.Cylinder(1000, 500)
              d = surface.intersection(p.pos, p.dir)
              getDepth=p.pos + d.first*p.dir
              impactDepth = MuonGun.depth((p.pos + d.first*p.dir).z)*1.e3
              conv_passing_fraction = conventional_veto(particleType, enu=p.energy, ct=math.cos(p.dir.zenith), depth=impactDepth)
              prompt_passing_fraction = prompt_veto(particleType, enu=p.energy, ct=math.cos(p.dir.zenith), depth=impactDepth)
              frame["cscdSBU_AtmWeight_Conv"]=dataclasses.I3Double(conv)
              frame["cscdSBU_AtmWeight_Prompt"]=dataclasses.I3Double(prompt)
              frame["cscdSBU_AtmWeight_Prompt_berss"]=dataclasses.I3Double(prompt_berss)
              frame["cscdSBU_AtmWeight_Conv_PassRate"]=dataclasses.I3Double(conv_passing_fraction.item(0))
              frame["cscdSBU_AtmWeight_Prompt_PassRate"]=dataclasses.I3Double(prompt_passing_fraction.item(0))


         tray.AddModule(saveAtmWeight,'saveAFlux')

@icetray.traysegment
def nugen_truth(tray, name):

     # adds "visible truth" of cascade
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

     def GetLosses(frame):
         from icecube import dataclasses
         from icecube.icetray import I3Units
         import numpy as n

         tree = frame['I3MCTree']
         neutrinos = []
         vertices = []

         for p in tree:
                     if p.is_neutrino == True and p.location_type == dataclasses.I3Particle.LocationType.InIce and not n.isnan(p.length):
                             neutrinos.append(p)

         for p in neutrinos:
                     secondary = tree.get_daughters(p)[0]
                     pos = secondary.pos
                     distance = n.sqrt(n.power(pos.x,2)+n.power(pos.y,2)+n.power(pos.z,2))
                     vertices.append(distance)

         cascade = neutrinos[int(n.argmin(vertices))]
         for q in tree.get_daughters(cascade):
             if q.type == dataclasses.I3Particle.MuPlus or q.type == dataclasses.I3Particle.MuMinus:
		 if not frame.Has('cscdSBU_MCMuon'):
                      frame['cscdSBU_MCMuon']=q
             else:
                 if not frame.Has('cscdSBU_MCMuon'):
                      frame['cscdSBU_MCMuon']=dataclasses.I3Particle()

         cascade.pos = tree.get_daughters(cascade)[0].pos


         losses = 0
         emlosses = 0
         hadlosses = 0
         for p in tree:
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

         frame['cscdSBU_MCTruth'] = shift_to_maximum(cascade, losses)

     tray.AddModule(GetLosses, 'cascade_energy')

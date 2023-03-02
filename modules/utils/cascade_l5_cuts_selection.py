'''
Quick and dirty copy of cascade_l5_cuts (script from Nancy) to add key to frame
denoting whether event is part of MESE selection without actually cutting
it away.
'''
from icecube import icetray, phys_services, simclasses
from icecube import dataclasses, dataio, hdfwriter
from I3Tray import *
import math
import numpy


def passed_final_track_cut(frame):
    # get TrackFit_zenith
    TrackFit_zenith = frame['TrackFit'].dir.zenith

    # get energy_millipede
    energy_millipede = frame['MillipedeDepositedEnergy'].value

    is_cascade_reco = frame['IsCascade_reco'].value

    if 'MESE_selection_precut' not in frame:
        # this means that none of the return statements got triggered
        # in Nancy's cascade_l5_cut scripts.
        # not returning True or False means that the event is kept, e.g.
        # it passes the filter. We will just assume this here and assign
        # that these events passed the MESE selection
        frame['MESE_selection_precut'] = icetray.I3Bool(True)

    precut = frame['MESE_selection_precut'].value

    # mask events
    track_mask = is_cascade_reco | ~((numpy.cos(TrackFit_zenith) > 0.3) & (energy_millipede < 10e3))

    frame['MESE_selection'] = icetray.I3Bool(bool(track_mask) & precut)
    return True


def update_mese_selection_precut(frame, value):
    if 'MESE_selection_precut' in frame:
        previous_value = bool(frame['MESE_selection_precut'].value)
        del frame['MESE_selection_precut']
        frame['MESE_selection_precut'] = icetray.I3Bool(previous_value & value)
    else:
        frame['MESE_selection_precut'] = icetray.I3Bool(value)
    return True

#---------------------------------------------------------------
# Scripts for MuonGun
#---------------------------------------------------------------
@icetray.traysegment
def cascade_l5_cuts_muongun(tray, name, infiles):

    from icecube import MuonGun

    if 'high' in infiles[1]:
        sframe_file = '/data/user/nwandkowsky/utils/sframes/muongun_highe_sframe.i3.bz2'
    elif 'med' in infiles[1]:
        sframe_file = '/data/user/nwandkowsky/utils/sframes/muongun_mede_sframe.i3.bz2'
    elif 'low' in infiles[1]:
        sframe_file = '/data/user/nwandkowsky/utils/sframes/muongun_lowe_sframe.i3.bz2'
    print('\033[93mWarning: Inserting {} to infiles.\033[0m'.format(sframe_file))
    print('\033[93m         Make sure this is correct... \033[0m')
    infiles.insert(0, sframe_file)

    tray.Add('Rename',Keys=["I3MCTree_preMuonProp","I3MCTree"])

    def harvest_generators(sinfiles):

        from icecube.icetray.i3logging import log_info as log
        generator = None
        f = dataio.I3File(sinfiles[0])
        while True:
            try:
                fr = f.pop_frame(icetray.I3Frame.Stream('S'))
            except RuntimeError as e:
                log('Caught the following exception:', str(e))
                fr = None
            if fr is None:
                break
            for k in fr.keys():
                v = fr[k]
                if isinstance(v, MuonGun.GenerationProbability):
                    log('%s: found "%s" (%s)' % (sinfiles[0], k, type(v).__name__), unit="MuonGun")
                    if generator is None:
                        generator = v
                    else:
                        generator += v
        #print generator
        f.close()
        return generator
    generator = harvest_generators(infiles)

    def track_energy(frame):
        energy = []
        energy.append(frame["SplineMPEMuEXDifferential"].energy+frame["MillipedeDepositedEnergy"].value)
        energy.append(frame["SplineMPEMuEXDifferential"].energy+frame["L5MonopodFit4"].energy)
        energy.append(frame["L5MonopodFit4"].energy+frame["MillipedeDepositedEnergy"].value)
        frame["TrackEnergy"]=dataclasses.I3Double(min(energy))
    tray.Add(track_energy)

    def remove_background(frame):
        if frame["IsHese"].value==True:
            return update_mese_selection_precut(frame, True)
        if frame["IsCascade_reco"].value==False and ( (numpy.cos(frame["SplineMPEMuEXDifferential"].dir.zenith)>0.4 and frame["SplineMPEMuEXDifferential"].energy<1e4) or numpy.cos(frame["SplineMPEMuEXDifferential"].dir.zenith)>0.6 and frame["SplineMPEMuEXDifferential"].energy<3e4):
            print("Likely a background event (spline)... removing...",  frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, False)
        if frame["IsCascade"].value==True and frame["TrackFit"].pos.z>370.:
            print("Potentially coincident cascade removed...", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, False)
    tray.Add(remove_background)

    def printy(frame):
        del frame["MillipedeDepositedEnergy"]
        del frame["TrackEnergy"]
        if "MuonLosses" in frame:
            losses = frame["MuonLosses"]
            first_loss_found=False
            deposited_energy = 0
            for i in range(0,len(losses)):
                if losses[i].pos.z<500 and losses[i].pos.z>-500 and math.sqrt(losses[i].pos.x*losses[i].pos.x+losses[i].pos.y*losses[i].pos.y)<550:
                    deposited_energy = deposited_energy + losses[i].energy
        else:
            deposited_energy = 0.
        frame["MillipedeDepositedEnergy"] = dataclasses.I3Double(deposited_energy)
        frame["TrackEnergy"] = dataclasses.I3Double(deposited_energy)
        if frame["IsHese"].value==False and frame["IsCascade_reco"].value==False and deposited_energy==0.:
            return update_mese_selection_precut(frame, False)

        if frame["IsCascade_reco"]==True and frame["L5MonopodFit4"].energy>6e4:
            del frame["IsCascade_reco"]
            del frame["IsTrack_reco"]
            if frame["TrackLength"]>550:
                frame["IsCascade_reco"]=icetray.I3Bool(False)
                frame["IsTrack_reco"]=icetray.I3Bool(True)
            else:
                frame["IsCascade_reco"]=icetray.I3Bool(True)
                frame["IsTrack_reco"]=icetray.I3Bool(False)

        if frame["IsCascade_reco"].value==False and frame["TrackEnergy"].value>6e4:
            del frame["IsCascade_reco"]
            del frame["IsTrack_reco"]
            if frame["TrackLength"]>550:
                frame["IsCascade_reco"]=icetray.I3Bool(False)
                frame["IsTrack_reco"]=icetray.I3Bool(True)
            else:
                frame["IsCascade_reco"]=icetray.I3Bool(True)
                frame["IsTrack_reco"]=icetray.I3Bool(False)

        if frame["IsCascade_reco"].value==True and abs(frame["L5MonopodFit4"].pos.z)>550.:
            return update_mese_selection_precut(frame, False)

        if frame["IsHese"].value==True and frame["IsHESE_ck"].value==False and frame["CascadeFilter"].value==False: #and frame["L4VetoLayer1"].value<10:
            del frame["IsHese"]
            frame["IsHese"]=icetray.I3Bool(False)
        if frame["IsCascade"].value==True and frame["IsCascade_reco"].value==True and (frame["L5MonopodFit4"].pos.z<-500 or frame["L5MonopodFit4"].pos.z>500 or numpy.sqrt(frame["L5MonopodFit4"].pos.x*frame["L5MonopodFit4"].pos.x+frame["L5MonopodFit4"].pos.y*frame["L5MonopodFit4"].pos.y)>550):
            print("Cascade event outside of detector... ",  frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            del frame["IsCascade"]
            frame["IsCascade"]=icetray.I3Bool(False)
        if frame["IsCascade"].value==True and frame["IsCascade_reco"].value==False and (frame["L4VetoTrackMilliOfflineVetoCharge"].value>2 or frame['L4VetoTrackMarginMilliOfflineSide'].value<125):
            del frame["IsCascade"]
            frame["IsCascade"]=icetray.I3Bool(False)
        if frame["IsUpgoingMuon"].value==True:
            #print "UpMu event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        elif frame["IsHese"].value==True:
            print("HESE event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, True)
        elif frame["IsCascade"].value==True and frame["L4VetoTrackOfflineVetoCharge"].value<6 and frame["L4VetoTrackL5OfflineVetoCharge"].value<2:
            #print "Cascade event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        elif frame["IsCascade"].value==True and frame["L4VetoTrackOfflineVetoCharge"].value<2 and frame["L4VetoTrackL5OfflineVetoCharge"].value<3:
            #print "Cascade event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        else:
            if ( ( frame['L4UpgoingTrackOfflineVetoCharge'].value > 10 and frame['L4UpgoingTrackOfflineVetoCharge'].value > \
            frame['L4VetoTrackOfflineVetoCharge'].value and frame['L4UpgoingTrackOfflineVetoChannels'].value > 3 ) or \
            ( frame['L4UpgoingTrackSplitVetoCharge'].value > 10 and frame['L4UpgoingTrackSplitVetoCharge'].value > \
            frame['L4VetoTrackSplitVetoCharge'].value and frame['L4UpgoingTrackSplitVetoChannels'].value > 3 )  or \
            ( frame['L4UpgoingTrackMilliOfflineVetoCharge'].value > 10 and frame['L4UpgoingTrackMilliOfflineVetoCharge'].value > \
            frame['L4VetoTrackMilliOfflineVetoCharge'].value and frame['L4UpgoingTrackMilliOfflineVetoChannels'].value > 3 and \
             frame['L4UpgoingTrackSplitVetoCharge'].value > 6 ) )\
                and frame["TrackFit"].dir.zenith>1.5 and frame["MuonFilter"].value==True:# and frame["OnlineL2Filter"].value==True:
                del frame["IsUpgoingMuon"]
                del frame["IsCascade"]
                frame["IsUpgoingMuon"]=icetray.I3Bool(True)
                frame["IsCascade"]=icetray.I3Bool(False)
                #print "UpMu event!", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
                return update_mese_selection_precut(frame, True)
            else:
                return update_mese_selection_precut(frame, False)

    tray.Add(printy, streams=[icetray.I3Frame.Physics])

    def bug_filter(frame):
        #if frame["I3EventHeader"].run_id==16 and frame["I3EventHeader"].event_id==17967:
        #    return False
        #if frame["I3EventHeader"].run_id==16780 and frame["I3EventHeader"].event_id==34774:
        #    return False
        print(frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
    tray.Add(bug_filter, streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])

    tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeight_Hoerandel5',
        Model=MuonGun.load_model('Hoerandel5_atmod12_SIBYLL'),
        Generator=generator)

    tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeight_GaisserH4a',
        Model=MuonGun.load_model('GaisserH4a_atmod12_SIBYLL'),
        Generator=generator)

    tray.AddModule(passed_final_track_cut, 'passed_final_track_cut')


#---------------------------------------------------------------
# Scripts for data
#---------------------------------------------------------------
@icetray.traysegment
def cascade_l5_cuts_data(tray, name):

    def track_energy(frame):
        energy = []
        energy.append(frame["SplineMPEMuEXDifferential"].energy+frame["MillipedeDepositedEnergy"].value)
        energy.append(frame["SplineMPEMuEXDifferential"].energy+frame["L5MonopodFit4"].energy)
        energy.append(frame["L5MonopodFit4"].energy+frame["MillipedeDepositedEnergy"].value)
        if 'TrackEnergy' in frame:
            frame.Delete('TrackEnergy')
        frame["TrackEnergy"]=dataclasses.I3Double(min(energy))
    tray.Add(track_energy)

    def remove_background(frame):
        if frame["IsHese"].value==True:
            return update_mese_selection_precut(frame, True)
        if frame["IsCascade_reco"].value==False and ( (numpy.cos(frame["SplineMPEMuEXDifferential"].dir.zenith)>0.4 and frame["SplineMPEMuEXDifferential"].energy<1e4) or numpy.cos(frame["SplineMPEMuEXDifferential"].dir.zenith)>0.6 and frame["SplineMPEMuEXDifferential"].energy<3e4):
            print("Likely a background event (spline)... removing...",  frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, False)
        if frame["IsCascade"].value==True and frame["TrackFit"].pos.z>370.:
            print("Potentially coincident cascade removed...", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, False)
    tray.Add(remove_background)

    def printy(frame):
        del frame["MillipedeDepositedEnergy"]
        del frame["TrackEnergy"]
        if "MuonLosses" in frame:
            losses = frame["MuonLosses"]
            first_loss_found=False
            deposited_energy = 0
            for i in range(0,len(losses)):
                if losses[i].pos.z<500 and losses[i].pos.z>-500 and math.sqrt(losses[i].pos.x*losses[i].pos.x+losses[i].pos.y*losses[i].pos.y)<550:
                    deposited_energy = deposited_energy + losses[i].energy
        else:
            deposited_energy = 0.
        frame["MillipedeDepositedEnergy"] = dataclasses.I3Double(deposited_energy)
        frame["TrackEnergy"] = dataclasses.I3Double(deposited_energy)
        if frame["IsHese"].value==False and frame["IsCascade_reco"].value==False and deposited_energy==0.:
            return update_mese_selection_precut(frame, False)

        if frame["IsCascade_reco"]==True and frame["L5MonopodFit4"].energy>6e4:
            del frame["IsCascade_reco"]
            del frame["IsTrack_reco"]
            if frame["TrackLength"]>550:
                frame["IsCascade_reco"]=icetray.I3Bool(False)
                frame["IsTrack_reco"]=icetray.I3Bool(True)
            else:
                frame["IsCascade_reco"]=icetray.I3Bool(True)
                frame["IsTrack_reco"]=icetray.I3Bool(False)

        if frame["IsCascade_reco"].value==False and frame["TrackEnergy"].value>6e4:
            del frame["IsCascade_reco"]
            del frame["IsTrack_reco"]
            if frame["TrackLength"]>550:
                frame["IsCascade_reco"]=icetray.I3Bool(False)
                frame["IsTrack_reco"]=icetray.I3Bool(True)
            else:
                frame["IsCascade_reco"]=icetray.I3Bool(True)
                frame["IsTrack_reco"]=icetray.I3Bool(False)

        if frame["IsCascade_reco"].value==True and abs(frame["L5MonopodFit4"].pos.z)>550.:
            return update_mese_selection_precut(frame, False)

        if frame["IsHese"].value==True and frame["IsHESE_ck"].value==False and frame["CascadeFilter"].value==False: #and frame["L4VetoLayer1"].value<10:
            del frame["IsHese"]
            frame["IsHese"]=icetray.I3Bool(False)
        if frame["IsCascade"].value==True and frame["IsCascade_reco"].value==True and (frame["L5MonopodFit4"].pos.z<-500 or frame["L5MonopodFit4"].pos.z>500 or numpy.sqrt(frame["L5MonopodFit4"].pos.x*frame["L5MonopodFit4"].pos.x+frame["L5MonopodFit4"].pos.y*frame["L5MonopodFit4"].pos.y)>550):
            print("Cascade event outside of detector... ",  frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            del frame["IsCascade"]
            frame["IsCascade"]=icetray.I3Bool(False)
        if frame["IsCascade"].value==True and frame["IsCascade_reco"].value==False and (frame["L4VetoTrackMilliOfflineVetoCharge"].value>2 or frame['L4VetoTrackMarginMilliOfflineSide'].value<125):
            del frame["IsCascade"]
            frame["IsCascade"]=icetray.I3Bool(False)
        if frame["IsUpgoingMuon"].value==True:
            #print "UpMu event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        elif frame["IsHese"].value==True:
            print("HESE event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, True)
        elif frame["IsCascade"].value==True and frame["L4VetoTrackOfflineVetoCharge"].value<6 and frame["L4VetoTrackL5OfflineVetoCharge"].value<2:
            #print "Cascade event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        elif frame["IsCascade"].value==True and frame["L4VetoTrackOfflineVetoCharge"].value<2 and frame["L4VetoTrackL5OfflineVetoCharge"].value<3:
            #print "Cascade event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        else:
            if ( ( frame['L4UpgoingTrackOfflineVetoCharge'].value > 10 and frame['L4UpgoingTrackOfflineVetoCharge'].value > \
            frame['L4VetoTrackOfflineVetoCharge'].value and frame['L4UpgoingTrackOfflineVetoChannels'].value > 3 ) or \
            ( frame['L4UpgoingTrackSplitVetoCharge'].value > 10 and frame['L4UpgoingTrackSplitVetoCharge'].value > \
            frame['L4VetoTrackSplitVetoCharge'].value and frame['L4UpgoingTrackSplitVetoChannels'].value > 3 )  or \
            ( frame['L4UpgoingTrackMilliOfflineVetoCharge'].value > 10 and frame['L4UpgoingTrackMilliOfflineVetoCharge'].value > \
            frame['L4VetoTrackMilliOfflineVetoCharge'].value and frame['L4UpgoingTrackMilliOfflineVetoChannels'].value > 3 and \
             frame['L4UpgoingTrackSplitVetoCharge'].value > 6 ) )\
                and frame["TrackFit"].dir.zenith>1.5 and frame["MuonFilter"].value==True:# and frame["OnlineL2Filter"].value==True:
                del frame["IsUpgoingMuon"]
                del frame["IsCascade"]
                frame["IsUpgoingMuon"]=icetray.I3Bool(True)
                frame["IsCascade"]=icetray.I3Bool(False)
                #print "UpMu event!", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
                return update_mese_selection_precut(frame, True)
            else:
                return update_mese_selection_precut(frame, False)

    tray.Add(printy, streams=[icetray.I3Frame.Physics])
    tray.AddModule(passed_final_track_cut, 'passed_final_track_cut')


#---------------------------------------------------------------
# Scripts for nugen
#---------------------------------------------------------------
@icetray.traysegment
def cascade_l5_cuts_nugen(tray, name):

    # from fix_inice_selection_weight import fix_inice_selection_weight
    # #tray.AddModule(fix_inice_selection_weight, "fix", MCTreeName="I3MCTree")

    def track_energy(frame):
        energy = []
        energy.append(frame["SplineMPEMuEXDifferential"].energy+frame["MillipedeDepositedEnergy"].value)
        energy.append(frame["SplineMPEMuEXDifferential"].energy+frame["L5MonopodFit4"].energy)
        energy.append(frame["L5MonopodFit4"].energy+frame["MillipedeDepositedEnergy"].value)
        if 'TrackEnergy' in frame:
            frame.Delete('TrackEnergy')
        frame["TrackEnergy"]=dataclasses.I3Double(min(energy))
    #tray.Add(track_energy)

    def remove_background(frame):
        if frame["IsHese"].value==True:
            return update_mese_selection_precut(frame, True)
        if frame["IsCascade_reco"].value==False and ( (numpy.cos(frame["SplineMPEMuEXDifferential"].dir.zenith)>0.4 and frame["SplineMPEMuEXDifferential"].energy<1e4) or numpy.cos(frame["SplineMPEMuEXDifferential"].dir.zenith)>0.6 and frame["SplineMPEMuEXDifferential"].energy<3e4):
            print("Likely a background event (spline)... removing...",  frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, False)
        if frame["IsCascade"].value==True and frame["TrackFit"].pos.z>370.:
            print("Potentially coincident cascade removed...", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, False)
    tray.Add(remove_background)

    def printy(frame):
        del frame["MillipedeDepositedEnergy"]
        del frame["TrackEnergy"]
        if "MuonLosses" in frame:
            losses = frame["MuonLosses"]
            first_loss_found=False
            deposited_energy = 0
            for i in range(0,len(losses)):
                if losses[i].pos.z<500 and losses[i].pos.z>-500 and math.sqrt(losses[i].pos.x*losses[i].pos.x+losses[i].pos.y*losses[i].pos.y)<550:
                    deposited_energy = deposited_energy + losses[i].energy
        else:
            deposited_energy = 0.
        frame["MillipedeDepositedEnergy"] = dataclasses.I3Double(deposited_energy)
        frame["TrackEnergy"] = dataclasses.I3Double(deposited_energy)
        if frame["IsHese"].value==False and frame["IsCascade_reco"].value==False and deposited_energy==0.:
            return update_mese_selection_precut(frame, False)

        if frame["IsCascade_reco"]==True and frame["L5MonopodFit4"].energy>6e4:
            del frame["IsCascade_reco"]
            del frame["IsTrack_reco"]
            if "TrackLength" in frame:
                if frame["TrackLength"]>550:
                    frame["IsCascade_reco"]=icetray.I3Bool(False)
                    frame["IsTrack_reco"]=icetray.I3Bool(True)
                else:
                    frame["IsCascade_reco"]=icetray.I3Bool(True)
                    frame["IsTrack_reco"]=icetray.I3Bool(False)
            else:
                frame["IsCascade_reco"]=icetray.I3Bool(True)
                frame["IsTrack_reco"]=icetray.I3Bool(False)

        if frame["IsCascade_reco"].value==False and frame["TrackEnergy"].value>6e4:
            del frame["IsCascade_reco"]
            del frame["IsTrack_reco"]
            if "TrackLength" in frame:
                if frame["TrackLength"]>550:
                    frame["IsCascade_reco"]=icetray.I3Bool(False)
                    frame["IsTrack_reco"]=icetray.I3Bool(True)
                else:
                    frame["IsCascade_reco"]=icetray.I3Bool(True)
                    frame["IsTrack_reco"]=icetray.I3Bool(False)
            else:
                frame["IsCascade_reco"]=icetray.I3Bool(True)
                frame["IsTrack_reco"]=icetray.I3Bool(False)
        if frame["IsCascade_reco"].value==True and abs(frame["L5MonopodFit4"].pos.z)>550.:
            return update_mese_selection_precut(frame, False)

        if frame["IsHese"].value==True and frame["IsHESE_ck"].value==False and frame["CascadeFilter"].value==False: #and frame["L4VetoLayer1"].value<10:
            del frame["IsHese"]
            frame["IsHese"]=icetray.I3Bool(False)

        if frame["IsHese"].value==True and frame["IsHESE_ck"].value==False and frame["CascadeFilter"].value==False: #and frame["L4VetoLayer1"].value<10:
            del frame["IsHese"]
            frame["IsHese"]=icetray.I3Bool(False)

        if frame["IsCascade"].value==True and frame["IsCascade_reco"].value==True and (frame["L5MonopodFit4"].pos.z<-500 or frame["L5MonopodFit4"].pos.z>500 or numpy.sqrt(frame["L5MonopodFit4"].pos.x*frame["L5MonopodFit4"].pos.x+frame["L5MonopodFit4"].pos.y*frame["L5MonopodFit4"].pos.y)>550):
            print("Cascade event outside of detector... ",  frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            del frame["IsCascade"]
            frame["IsCascade"]=icetray.I3Bool(False)

        if frame["IsCascade"].value==True and frame["IsCascade_reco"].value==False and (frame["L4VetoTrackMilliOfflineVetoCharge"].value>2 or frame['L4VetoTrackMarginMilliOfflineSide'].value<125):
            del frame["IsCascade"]
            frame["IsCascade"]=icetray.I3Bool(False)

        if frame["IsUpgoingMuon"].value==True:
            #print "UpMu event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
            return update_mese_selection_precut(frame, True)
        elif frame["IsHese"].value==True:
            print("HESE event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id)
            return update_mese_selection_precut(frame, True)
        elif "L4VetoTrackL5OfflineVetoCharge" in frame:
            if frame["IsCascade"].value==True and frame["L4VetoTrackOfflineVetoCharge"].value<6 and frame["L4VetoTrackL5OfflineVetoCharge"].value<2:
                #print "Cascade event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
                return update_mese_selection_precut(frame, True)
            elif frame["IsCascade"].value==True and frame["L4VetoTrackOfflineVetoCharge"].value<2 and frame["L4VetoTrackL5OfflineVetoCharge"].value<3:
                #print "Cascade event: ", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
                return update_mese_selection_precut(frame, True)
        else:
            if ( ( frame['L4UpgoingTrackOfflineVetoCharge'].value > 10 and frame['L4UpgoingTrackOfflineVetoCharge'].value > \
            frame['L4VetoTrackOfflineVetoCharge'].value and frame['L4UpgoingTrackOfflineVetoChannels'].value > 3 ) or \
            ( frame['L4UpgoingTrackSplitVetoCharge'].value > 10 and frame['L4UpgoingTrackSplitVetoCharge'].value > \
            frame['L4VetoTrackSplitVetoCharge'].value and frame['L4UpgoingTrackSplitVetoChannels'].value > 3 )  or \
            ( frame['L4UpgoingTrackMilliOfflineVetoCharge'].value > 10 and frame['L4UpgoingTrackMilliOfflineVetoCharge'].value > \
            frame['L4VetoTrackMilliOfflineVetoCharge'].value and frame['L4UpgoingTrackMilliOfflineVetoChannels'].value > 3 and \
             frame['L4UpgoingTrackSplitVetoCharge'].value > 6 ) )\
                and frame["TrackFit"].dir.zenith>1.5 and frame["MuonFilter"].value==True:# and frame["OnlineL2Filter"].value==True:
                del frame["IsUpgoingMuon"]
                del frame["IsCascade"]
                frame["IsUpgoingMuon"]=icetray.I3Bool(True)
                frame["IsCascade"]=icetray.I3Bool(False)
                #print "UpMu event!", frame["I3EventHeader"].run_id, frame["I3EventHeader"].event_id
                return update_mese_selection_precut(frame, True)
            else:
                return update_mese_selection_precut(frame, False)

    tray.Add(printy, streams=[icetray.I3Frame.Physics])

    def check_type(frame):
        del frame["IsCascade_true"]
        del frame["IsTrack_true"]
        tree = frame["I3MCTree"]

        is_track = False
        for p in tree:
            if (p.type==-13 or p.type==13) and p.energy>500.:
                is_track=True
                print("Outgoing Muon with at least 100 GeV, classifying as track", p.energy)
                break

        if is_track==True:
            print("Outgoing Muon with at least 100 GeV, classifying as track")
            frame["IsCascade_true"]= icetray.I3Bool(False)
            frame["IsTrack_true"]= icetray.I3Bool(True)
        else:
            frame["IsCascade_true"]= icetray.I3Bool(True)
            frame["IsTrack_true"]= icetray.I3Bool(False)
    tray.Add(check_type)
    tray.AddModule(passed_final_track_cut, 'passed_final_track_cut')

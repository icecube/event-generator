
from math import sqrt
import numpy as np
from scipy.spatial import Delaunay

from icecube import icetray, dataclasses, dataio, simclasses


# Main extraction method.
@icetray.traysegment
def StoppingMuonLabels(tray, name='StoppingMuonLabels'):

    extr_feat = ExtractedFeatures()

    # Calculates labels from data, puts them into extr_feat
    tray.AddModule(GetLabels, name+'Getlabels', Features=extr_feat)

    tray.AddModule(
        AddFeaturesToI3, name+"AddFeatures", Features=extr_feat)


"""
Detector Geometry
-----------------
* ``DET_HULL``: Delaunay triangulation of shape of a slightly (100 m) enlarged detector.

* ``BARE_DET_HULL``: Delaunay triangulation of shape of detector.
"""
DET_HULL = Delaunay(
    np.array([[-694.06, -137.15, 600.0], [680.69, 213.94, 600.0],
              [-300.93, -637.68, 600.0], [439.82, -519.92, 600.0],
              [373.47, 591.8, 600.0], [147.82, 532.61, 600.0],
              [66.19, 625.76, 600.0], [-436.86, 548.81, 600.0],
              [-694.06, -137.15, -600.0], [680.69, 213.94, -600.0],
              [-300.93, -637.68, -600.0], [439.82, -519.92, -600.0],
              [373.47, 591.8, -600.0], [147.82, 532.61, -600.0],
              [66.19, 625.76, -600.0], [-436.86, 548.81, -600.0]]))

BARE_DET_HULL = Delaunay(
    np.array([
        [-256.14, -521.08, -504.40], [361.00, -422.83, -504.71],
        [576.37, 170.92, -510.18], [338.44, 463.72, -498.50],
        [101.04, 412.79, -500.51], [22.11, 509.50, -499.56],
        [-347.88, 451.52, -502.62], [-570.90, -125.14, -504.62],
        [-256.14, -521.08, 496.03], [361.00, -422.83, 499.51],
        [576.37, 170.92, 494.04], [338.44, 463.72, 505.72],
        [101.04, 412.79, 504.42], [22.11, 509.50, 504.66],
        [-347.88, 451.52, 502.01], [-570.90, -125.14, 499.60]
    ]))

"""
Computational Constants
-----------------------
* ``PE_THRESHOLD``: Number of photo electrons to be detected from a muon in order to be considered observed.

* ``N_ITER``: Number of iterations to determine the travel length.
"""
PE_THRESHOLD = 5
N_ITER = 20



# # # # # # # # # #
# General Classes #
# # # # # # # # # #

class Feature:
    """This class contains a single feature using a list.

    Attributes
    ----------
    feature : object
              Object stored in this feature.

    name : str
           Name of the feature.

    index : int
            Index number of the feature.

    role : {"attribute", "label"}
           Role of the feature.

    Methods
    -------
    get()
        Returns the content of the feature.
    """
    def __init__(self, name, index, data, role):
        self.feature = data
        self.name = name
        self.index = index
        self.role = role

    def get(self):
        return self.feature


class ExtractedFeatures:
    """Concatenation of Feature objects.

    Attributes
    ----------
    feature_dict : dict
                   Dictionary of Feature objectds.

    number_features : int
                      Number of features.

    Methods
    -------
    add(key, data, role="attribute")
        Adds a Feature to the feature dictionary.

    print_stats()
        Prints the contents of this object.

    get(feature)
        Gets the contents of a certain feature inside this object.

    get_row()
        Gets the last row.
    """
    def __init__(self):
        self.feature_dict = {}
        self.number_features = 0

    def __str__(self):
        return "ExtractedFeatures Object"

    def __repr__(self):
        return "<ExtractedFeatures Object>"

    # Adds data to a Feature identified by key.
    def add(self, key, data, role="attribute"):
        self.feature_dict[key] = Feature(key, self.number_features, data,
                                          role)

    # Print the lengths of feature lists with their name. For debugging mainly.
    def print_stats(self):
        for feature in self.feature_dict.values():
            print("%s, %i" % (feature.name, feature.length))

    def get(self, feature):
        return self.feature_dict[feature].get()

    def get_row(self):
        row_label = {}
        row_attr = {}
        for key, feature in self.feature_dict.items():
            if feature.role == "label":
                row_label[key] = float(feature.get())
            if feature.role == "attribute":
                row_attr[key] = float(feature.get())
        return (row_label, row_attr)


class Trajectory:
    """Class that contains a parametrization of a trajectory.

    Attributes
    ----------
    v : array, shape = [3,]
        Directional vector of the trajectory.

    w : array, shape = [3,]
        Pivot vector of the trajectory.

    zenith : float
             Zenith angle corresponding to the trajectory.

    Methods
    -------
    eval(t)
        Evaluates the position for a given position t.

    get_distance(p)
        Calculates the distance of a point p to the trajectory.

    closest_approach()
        Calculates the distance of the origin to the trajectory.

    project_onto(p)
        Projects a point p onto the trajectory.

    get_intersection(start, end)
        Performs a bisection algorithm with start and end as initial values.

    travel_length()
        Calculates the travel length of a trajectory.
    """
    def __init__(self, vx, vy, vz, wx, wy, wz, zenith=0.0):
        self.v = np.array([vx, vy, vz])
        self.w = np.array([wx, wy, wz])
        self.zenith = zenith

    def __str__(self):
        return "Trajectory Object"

    def __repr__(self):
        return "<Trajectory Object>"

    def eval(self, t):
        return self.v * t + self.w

    def get_distance(self, p):
        return sqrt(np.sum(np.cross(p - self.w, self.v)**2))

    def closest_approach(self):
        return self.get_distance(np.array([0, 0, 0]))

    def project_onto(self, p):
        return np.sum((p - self.w) * self.v)

    def get_intersection(self, start, end):
        t1 = start
        t2 = end
        t = (t1 + t2) / 2.0
        for k in range(N_ITER):
            if DET_HULL.find_simplex(self.eval(t)) >= 0:
                t2 = t
            else:
                t1 = t
            t = (t1 + t2) / 2.0
        return t

    def travel_length(self):
        start_t = -1000.0
        end_t = 1000.0
        middle_t = 0.0
        while (BARE_DET_HULL.find_simplex(self.eval(middle_t)) >= 0):
            if (self.v[2] > 0.0):
                middle_t -= 10.0
            else:
                middle_t += 10.0
        return (self.get_intersection(start_t, middle_t),
                self.get_intersection(end_t, middle_t))



# # # # # # # # # #
# Topology Method #
# # # # # # # # # #


def make_muon(p, prim, pe_counts):
    """There's no nice way to do this. Either it's handsome, but slow, or fast, but
    ugly. I settled for latter.

    Parameters
    ----------
    p : I3Particle
        I3 Particle Object that corresponds to a muon.

    prim : int
           The running number of the primary.

    Returns
    -------
    muon : array, shape = [23,]
           A numpy array containing everything you need to know about your favorite muon. Entries:

            ========   ======================   =======
            Element    Content                  Type
            ========   ======================   =======
            0          Zenith                   float
            1          Azimuth                  float
            2 to 4     v                        float
            5 to 7     w                        float
            8          length                   float
            9          energy                   float
            10         ptype (pdg_encoding)     int
            11         prim from above          int
            12         stop_r                   float
            13         stop_z                   float
            14         bool(stops or not)       bool
            15         bool(stops in or not)    bool
            16 to 18   entry point              float
            19 to 21   exit point               float
            22         travelled length         float
            23         minor ID                 int
            24         major ID                 int
            25         pe_count                 int
            ========   ======================   =======
    """
    muon = np.zeros(26)
    muon[0] = p.dir.zenith
    muon[1] = p.dir.azimuth
    muon[2:5] = np.array([p.dir.x, p.dir.y, p.dir.z])
    muon[5:8] = np.array([p.pos.x, p.pos.y, p.pos.z])
    muon[8] = p.length
    muon[9] = p.energy
    muon[10] = p.pdg_encoding
    muon[11] = prim
    muon[12:16] = check_stopping(muon[2:5], muon[5:8], muon[8])
    (muon[16:19], muon[19:22], muon[22]) = get_entry_exit(muon[2:5], muon[5:8],
                                                          p.length)
    muon[23] = p.id.minorID
    muon[24] = p.id.majorID
    try:
        muon[25] = pe_counts[(p.id.majorID, p.id.minorID)]
    except:
        muon[25] = 0
    return muon

def check_stopping(v, w, l):
    """Checks whether a muon is stopping strictly inside the detector.

    Parameters
    ----------
    v : array, shape = [3,]
        Direction vector of the track (must be normalized).

    w : array, shape = [3,]
        Pivot vector of the track.

    l : float
        Length of the track.

    Returns
    -------
    stop_r : float
             The r-component of the stopping point.

    stop_z : float
             the z-component of the stopping point.

    stop : bool
           Whether or not the muon stops in the detector volume.

    stop_dc : bool
              Whether or not the muon stops in deep core.
    """
    stop_point = v * l + w
    stop = DET_HULL.find_simplex(stop_point) >= 0
    stop_in = BARE_DET_HULL.find_simplex(stop_point) >= 0
    stop_r = np.linalg.norm(stop_point[:2])
    stop_z = stop_point[2]
    return stop_r, stop_z, stop, stop_in

def get_cut(v, w, start, end):
    """A bisection algorithm that estimates an intersection between a the
    detector
    volume and a track. One day I'm going to make this prettier.

    Parameters
    ----------
    v : array, shape = [3,]
        Direction vector of the track (must be normalized).

    w : array, shape = [3,]
        Pivot vector of the track.

    start : float
            One initial value for bisection.

    end : float
          The other initial value.

    Returns
    -------
    t : float
        t value of intersection point.
    """
    t1 = start
    t2 = end
    t = (t1 + t2) / 2.0
    for k in range(20):
        if DET_HULL.find_simplex(v * t + w) >= 0:
            t2 = t
        else:
            t1 = t
        t = (t1 + t2) / 2.0
    return t

def get_entry_exit(v, w, length):
    """Estimates entry and exit point of a track.

    Parameters
    ----------
    v : array, shape = [3,]
        Direction vector of the track (must be normalized).

    w : array, shape = [3,]
        Pivot vector of the track.

    length : float
             Length of the track.

    Returns
    -------
    entry : array, shape = [3,]
            Entry point.

    exit : array, shape = [3,]
           Exit point.

    travel : float
             Travelled length.
    """
    start_t = 0.0
    end_t = length
    middle_t = 0.0
    while (~(DET_HULL.find_simplex(v * middle_t + w) >= 0)):
        middle_t += 10.0
        if middle_t > end_t:
            entry_t = float("nan")
            exit_t = float("nan")
            travel = 0.0
            return v * entry_t + w, v * exit_t + w, travel
    entry_t = get_cut(v, w, start_t, middle_t)
    exit_t = get_cut(v, w, end_t, middle_t)
    travel = abs(entry_t - exit_t)
    return v * entry_t + w, v * exit_t + w, travel

def get_muon_properties(mc_tree, pe_counts):
    """Builds a table of the properties of all muons in an mc_tree.

    Parameters
    ----------
    mc_tree : I3MCTree
              The MC Tree.

    Returns
    -------
    muons : array, shape = [23, N]
            A table of the properties of all N muons.
    """
    primaries = mc_tree.get_primaries()
    muons = []
    prim_number = 0
    for p in primaries:
        for d in mc_tree.get_daughters(p):
            muons += [make_muon(d, prim_number, pe_counts)]
        prim_number += 1

    return np.array(muons)

def visited_muons(muons):
    """Does the muon actually enter the detector volume at all? This question
    is answered here.

    Parameters
    ----------
    muons : array, shape = [23, N]
            Muon table, see :meth:`make_muon` for more information on that.

    Returns
    -------
    visited : array, shape = [N,]
              Whether or not each of the muons has seen some IceCube madness.
    """
    return (muons[:, 25] >= PE_THRESHOLD)

def get_coincidence(muons):
    """Number of coincident primaries.

    Parameters
    ----------
    muons : array, shape = [23, N]
            Muon table, see :meth:`make_muon` for more information on that.
    Returns
    -------
    coincidence : int
                  The coincidence number.
    """
    return len(list(set(muons[:, 11])))

def decide_label(muons):
    """Decides whether a frame is considered a stopping event or not.

    Parameters
    ----------
    muons : array, shape = [23, N]
            Muon table, see :meth:`make_muon` for more information on that.

    Returns
    -------
    stopping : bool
               Whether it's a stopping event or not.

    stopping_dc : bool
                  Whether it's a deep core stopping event or not.
    """
    n_mu_stop = np.sum(muons[:, 15]) > 0
    all_stop = muons[visited_muons(muons), 14].all() == True
    all_stop_in = muons[visited_muons(muons), 15].all() == True
    return (all_stop & n_mu_stop), (all_stop_in & n_mu_stop)

def pe_count(mcpe_map):
    """Seeks through an I3MCPESeriesMap and counts how many PEs each particle
    produced.

    Parameters
    ----------
    mcpe_map: I3MCPESeriesMap Object
              The Monte Carlo Pulse Series Map.

    Returns
    -------
    pre_counts: dict
                The counts in the scheme (major_id, minor_id): counts.
    """
    counts = {}
    for m in mcpe_map:
        try:
            key = (m[1][0].ID.majorID, m[1][0].ID.minorID)
            try:
                counts[key] += len(m[1])
            except:
                counts[key] = len(m[1])
        except:
            pass
    return counts



# # # # # # # # # #
# Icetray Modules #
# # # # # # # # # #

#==============================================================================
# GetLabels
#==============================================================================
class GetLabels(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "Features",
            "Object from class ExtractedFeatures to contain extracted features.",
            ExtractedFeatures())

    def add(self, key, value, role="label"):
        self.feature.add(key, value, role)

    def Configure(self):
        print("GetLabels Starting...")
        self.feature = self.GetParameter("Features")
        self.frame_index = 0
        self.file_index = 0

    def Physics(self, frame):
        mep_t = dataclasses.get_most_energetic_primary(frame[
            "I3MCTree"])
        traj = Trajectory(
            mep_t.dir.x, mep_t.dir.y,
            mep_t.dir.z, mep_t.pos.x,
            mep_t.pos.y, mep_t.pos.z)
        self.add("zenith_true", mep_t.dir.zenith)
        self.add("azimuth_true", mep_t.dir.azimuth)
        self.add("energy_mep", mep_t.energy)

        try:
            pe_counts = pe_count(frame["I3MCPESeriesMap"])
        except:
            pe_counts = 0
        muon_bunches = get_muon_properties(frame["I3MCTree"], pe_counts)

        self.add("coincidence", get_coincidence(muon_bunches))
        visited = visited_muons(muon_bunches)
        stopping = muon_bunches[:, 15] == True
        energy_total = np.mean(muon_bunches[:, 9])

        if np.sum(visited) > 0:
            stopr = np.mean(muon_bunches[visited, 12])
            stopz = np.mean(muon_bunches[visited, 13])
            nmust = np.sum(muon_bunches[visited, 15])
            energy_stop = np.mean(muon_bunches[stopping, 9])
            stop_det, stop_dc = decide_label(muon_bunches)
        else:
            stopr = np.NaN
            stopz = np.NaN
            nmust = 0
            energy_stop = np.NaN
            stop_det = False
            stop_dc = False
        self.add("true_stop_r", stopr)
        self.add("true_stop_z", stopz)
        self.add("energy_stop", energy_stop)
        self.add("energy_total", energy_total)
        self.add("n_mu", len(muon_bunches))
        self.add("n_mu_stop", nmust)
        self.add("label_det", stop_det)
        self.add("label_in", stop_dc)
        self.add("frame_index", self.frame_index)
        self.frame_index += 1
        self.PushFrame(frame)

    def DAQ(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("Finished GetLabels.")


#==============================================================================
# AddFeaturesToI3
#==============================================================================
class AddFeaturesToI3(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("Features", "Feature Class Object.")

    def Configure(self):
        self.feature = self.GetParameter("Features")

    def Physics(self, frame):
        labels, attributes = self.feature.get_row()
        frame["Hoinka_Labels"] = dataclasses.I3MapStringDouble(labels)
        for key, value in attributes.items():
            prefixed_key = "Hoinka_" + key
            frame[prefixed_key] = dataclasses.I3MapStringDouble({prefixed_key:
                                                                 value})
        self.PushFrame(frame)

    def DAQ(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("AddFeaturesToI3 Finished.")


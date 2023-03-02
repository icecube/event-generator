from __future__ import print_function, division
import numpy as np
from uncertainties import umath, ufloat

from I3Tray import I3Tray
from icecube import icetray, dataclasses


def add_dnn_track_calc(frame,
                       nn_base_name='DeepLearning',
                       direction_key='PrimaryDirection',
                       x_pull_correction=1.38,
                       y_pull_correction=1.38,
                       z_pull_correction=1.37,
                       normalize_z=True,
                       ):

    if nn_base_name[-1] != '_':
        nn_base_name += '_'

    # get DNN_reco prediction
    dir_x_dnn = - frame[nn_base_name + direction_key + 'X'].value
    dir_y_dnn = - frame[nn_base_name + direction_key + 'Y'].value
    dir_z_dnn = - frame[nn_base_name + direction_key + 'Z'].value

    dir_x_unc_dnn = np.abs(
                frame[nn_base_name + direction_key + 'X_uncertainty'].value)
    dir_y_unc_dnn = np.abs(
                frame[nn_base_name + direction_key + 'Y_uncertainty'].value)
    dir_z_unc_dnn = np.abs(
                frame[nn_base_name + direction_key + 'Z_uncertainty'].value)

    # pull correct
    dir_x_unc_dnn *= x_pull_correction
    dir_y_unc_dnn *= y_pull_correction
    dir_z_unc_dnn *= z_pull_correction

    # create ufloat
    u_dir_x = ufloat(dir_x_dnn, dir_x_unc_dnn)
    u_dir_y = ufloat(dir_y_dnn, dir_y_unc_dnn)
    u_dir_z = ufloat(dir_z_dnn, dir_z_unc_dnn)

    # calculate zenith and azimuth
    dir_length = umath.sqrt(u_dir_x**2 + u_dir_y**2 + u_dir_z**2)

    u_dir_x_normed = u_dir_x / dir_length
    u_dir_y_normed = u_dir_y / dir_length
    if normalize_z:
        u_dir_z_normed = u_dir_z / dir_length

    if np.abs(u_dir_z.nominal_value) > 1.:
        u_dir_z /= np.abs(u_dir_z.nominal_value)

    zenith_dnn = umath.acos(u_dir_z)
    azimuth_dnn = (umath.atan2(u_dir_y_normed, u_dir_x_normed)
                   + 2 * np.pi) % (2 * np.pi)

    sigma_dnn = np.sqrt(zenith_dnn.std_dev**2 + azimuth_dnn.std_dev**2 *
                        np.sin(zenith_dnn.nominal_value)**2) / np.sqrt(2)

    azimuth = azimuth_dnn.nominal_value
    zenith = zenith_dnn.nominal_value
    return azimuth, zenith, sigma_dnn


def combine_nn_info(frame,
                    nn_base_name='DeepLearning',
                    direction_key='PrimaryDirection',
                    energy_key='PrimaryEnergy',
                    output_name='DeepLearning',
                    use_calculated_direction=False,
                    base_particle=None,
                    use_energy=True,
                    use_time=True,
                    use_vertex=True,
                    use_direction=True,
                    x_pull_correction=1.38,
                    y_pull_correction=1.38,
                    z_pull_correction=1.37,
                    normalize_z=True,
                    ):

    if nn_base_name[-1] != '_':
        nn_base_name += '_'

    # calculate azimuth, zenith and track uncertainty
    azimuth, zenith, sigma_dnn = add_dnn_track_calc(
                                        frame=frame,
                                        nn_base_name=nn_base_name,
                                        direction_key=direction_key,
                                        x_pull_correction=x_pull_correction,
                                        y_pull_correction=y_pull_correction,
                                        z_pull_correction=z_pull_correction,
                                        normalize_z=normalize_z,
                                        )
    frame[nn_base_name + 'PrimaryAzimuthCalculated'] = dataclasses.I3Double(
                                                                azimuth)
    frame[nn_base_name + 'PrimaryZenithCalculated'] = dataclasses.I3Double(
                                                                zenith)
    frame[nn_base_name + 'TrackUnc'] = dataclasses.I3Double(sigma_dnn)

    # create I3Particle
    if base_particle is None:
        particle = dataclasses.I3Particle()
    else:
        particle = dataclasses.I3Particle(frame[base_particle])

    if nn_base_name + energy_key in frame and use_energy:
        particle.energy = frame[nn_base_name + energy_key].value

    if nn_base_name + 'VertexTime' in frame and use_time:
        particle.time = frame[nn_base_name + 'VertexTime'].value

    if nn_base_name + 'VertexX' in frame and use_vertex:
        particle.pos = dataclasses.I3Position(
            frame[nn_base_name + 'VertexX'].value,
            frame[nn_base_name + 'VertexY'].value,
            frame[nn_base_name + 'VertexZ'].value,
            )

    if nn_base_name + direction_key + 'X' in frame and use_direction:
        if use_calculated_direction:
            particle.dir = dataclasses.I3Direction(zenith, azimuth)
        else:
            particle.dir = dataclasses.I3Direction(
                frame[nn_base_name + direction_key + 'X'].value,
                frame[nn_base_name + direction_key + 'Y'].value,
                frame[nn_base_name + direction_key + 'Z'].value,
                )
    frame[output_name+'_I3Particle'] = particle

    # combine I3Doubles
    len_base = len(nn_base_name)
    nn_vars = {}
    for k in frame.keys():
        if nn_base_name == k[:len_base]:
            if isinstance(frame[k], dataclasses.I3Double):
                nn_vars[k[len_base:]] = frame[k].value
                del frame[k]

    # delete primary direction
    if nn_base_name + direction_key in frame:
        del frame[nn_base_name + direction_key]

    frame[output_name] = dataclasses.I3MapStringDouble(nn_vars)

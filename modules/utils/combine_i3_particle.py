from icecube import dataclasses


def add_combined_i3_particle(tray, cfg, name='add_combined_i3_particle',
                             cfg_key='combine_i3_particle',
                             output_key='combine_i3_particle_output_name',
                             ):
    """Add a new I3Particle to the frame combined from others.

    Config setup:

        combine_i3_particle: {
            pos_x_name: ,
            pos_y_name: ,
            pos_z_name: ,
            dir_name: ,
            time_name: ,
            energy_name: ,
            combine_i3_particle_output_name: 'Combined_I3Particle', [optional]
        }

    Parameters
    ----------
    tray : I3Tray
        The I3Tray to which the modules should be added.
    cfg : dict
        A dictionary with all configuration settings.
    name : str, optional
        Name of the tray segment.
    """
    if cfg_key in cfg:
        if output_key not in cfg:
            cfg[output_key] = 'Combined_I3Particle'

        def combine_i3_particle(frame, output_name,
                                pos_x_name=None,
                                pos_y_name=None,
                                pos_z_name=None,
                                dir_name=None,
                                time_name=None,
                                energy_name=None,
                                shape='Cascade',
                                ):
            particle = dataclasses.I3Particle()
            if pos_x_name is not None:
                particle.pos.x = frame[pos_x_name].pos.x
            if pos_y_name is not None:
                particle.pos.y = frame[pos_y_name].pos.y
            if pos_z_name is not None:
                particle.pos.z = frame[pos_z_name].pos.z
            if dir_name is not None:
                particle.dir = frame[dir_name].dir
            if time_name is not None:
                particle.time = frame[time_name].time
            if energy_name is not None:
                particle.energy = frame[energy_name].energy

            particle.fit_status = dataclasses.I3Particle.FitStatus.OK
            particle.shape = getattr(
                dataclasses.I3Particle.ParticleShape, shape)

            frame[output_name] = particle

        tray.AddModule(combine_i3_particle, name,
                       output_name=cfg[output_key],
                       **cfg[cfg_key])
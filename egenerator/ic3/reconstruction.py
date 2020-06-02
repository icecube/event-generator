import os
from icecube import dataclasses, icetray

from egenerator.ic3.configurator import I3ManagerConfigurator
from egenerator.manager.reconstruction.tray import ReconstructionTray


class EventGeneratorReconstruction(icetray.I3ConditionalModule):

    """Class to apply Event-Generator model.

    """

    def __init__(self, context):
        """Class to apply Event-Generator Model.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox('OutBox')

        # Required settings
        self.AddParameter('seed_keys',
                          'A seed or list of seeds to use.')
        self.AddParameter('model_names',
                          'A model name or list of model names '
                          'defining which model or ensemble of models to '
                          'apply. The full path is given by `model_base_dir` +'
                          ' model_name if `model_base_dir` is provided. '
                          'Otherwise `model_names` must define the full '
                          'model path.')

        # Optional settings
        self.AddParameter('model_base_dir',
                          'The base directory in which the models are located.'
                          ' The full path is given by `model_base_dir` + '
                          ' model_name if `model_base_dir` is provided. '
                          'Otherwise `model_names` must define the full '
                          'model path.',
                          '/data/user/mhuennefeld/exported_models/egenerator')
        self.AddParameter('output_key',
                          'The output base key to which results will be saved',
                          'EventGenerator')
        self.AddParameter('pulse_key',
                          'The pulses to use for the reconstruction.',
                          'InIceDSTPulses')
        self.AddParameter('dom_exclusions_key',
                          'The DOM exclusions to use.',
                          'BadDomsList')
        self.AddParameter('time_exclusions_key',
                          'The time exclusions to use.',
                          None)
        self.AddParameter('add_circular_err',
                          'Add circularized angular uncertainty estimate.',
                          False)
        self.AddParameter('add_covariances',
                          'Add calculation of covariance matrices via Hessian '
                          'matrix evaluated at the best fit point.',
                          False)
        self.AddParameter('label_key',
                          'Only relevant if labels are being loaded. '
                          'The key from which to load labels.',
                          'LabelsDeepLearning')
        self.AddParameter('snowstorm_key',
                          'Only relevant if labels are being loaded. '
                          'The key from which to load snowstorm parameters.',
                          'SnowstormParameters')

        # Reconstruction specific optional settings
        self.AddParameter('minimize_in_trafo_space',
                          'Perform minimization in normalized coordinates if '
                          'True (this is usually desired).',
                          True)
        self.AddParameter('minimize_parameter_dict',
                          'A dictionary with elements fit_parameter: boolean '
                          'that indicates whether the parameter will be fit '
                          '(if set to True) or if it will be held constant '
                          '(if set to False). Values for unspecified '
                          'parameters will default to '
                          '`minimize_parameter_default_value`',
                          {})
        self.AddParameter('minimize_parameter_default_value',
                          'The default value for parameters not defined in '
                          'the `minimize_parameter_dict`: '
                          'a dictionary with elements fit_parameter: boolean '
                          'that indicates whether the parameter will be fit '
                          '(if set to True) or if it will be held constant '
                          '(if set to False). Values for unspecified '
                          'parameters will default to '
                          '`minimize_parameter_default_value`',
                          True)
        self.AddParameter('reco_optimizer_interface',
                          'The reconstruction interface to use. Options are: '
                          '"scipy" or "tfp" (tensorflow_probability).',
                          'scipy')
        self.AddParameter('scipy_optimizer_settings',
                          'Only relevant if `reco_optimizer_interface` is set '
                          ' to "scipy". '
                          'Defines settings for scipy optimizer',
                          {'method': 'BFGS'})
        self.AddParameter('tf_optimizer_settings',
                          'Only relevant if `reco_optimizer_interface` is set '
                          ' to "tfp". '
                          'Defines settings for tensorflow optimizer',
                          {'method': 'bfgs_minimize', 'x_tolerance': 0.001})

    def Configure(self):
        """Configures Module and loads model from file.
        """
        self.seed_keys = self.GetParameter('seed_keys')
        self.model_names = self.GetParameter('model_names')
        self.model_base_dir = self.GetParameter('model_base_dir')
        self.output_key = self.GetParameter('output_key')
        self.pulse_key = self.GetParameter('pulse_key')
        self.dom_exclusions_key = self.GetParameter('dom_exclusions_key')
        self.time_exclusions_key = self.GetParameter('time_exclusions_key')
        self.add_circular_err = self.GetParameter('add_circular_err')
        self.add_covariances = self.GetParameter('add_covariances')
        self.label_key = self.GetParameter('label_key')
        self.snowstorm_key = self.GetParameter('snowstorm_key')

        # Reconstruction specific settings
        self.minimize_in_trafo_space = \
            self.GetParameter('minimize_in_trafo_space')
        self.minimize_parameter_default_value = \
            self.GetParameter('minimize_parameter_default_value')
        self.minimize_parameter_dict = \
            self.GetParameter('minimize_parameter_dict')
        self.reco_optimizer_interface = \
            self.GetParameter('reco_optimizer_interface')
        self.scipy_optimizer_settings = \
            self.GetParameter('scipy_optimizer_settings')
        self.tf_optimizer_settings = self.GetParameter('tf_optimizer_settings')

        if isinstance(self.seed_keys, (list, tuple)):
            raise NotImplementedError('Multiple seeds not yet supported')

        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]

        manager_dirs = []
        for name in self.model_names:
            if self.model_base_dir is not None:
                manager_dirs.append(os.path.join(self.model_base_dir, name))
            else:
                manager_dirs.append(name)

        # Build and configure SourceManager
        self.manager_configurator = I3ManagerConfigurator(
            manager_dirs=manager_dirs,
            reco_config_dir=None,
            load_labels=False,
            misc_setting_updates={
                'seed_names': [self.seed_keys],
            },
            label_setting_updates={
                'label_key': self.label_key,
                'snowstorm_key': self.snowstorm_key,
            },
            data_setting_updates={
                'pulse_key': self.pulse_key,
                'dom_exclusions_key': self.dom_exclusions_key,
                'time_exclusions_key': self.time_exclusions_key,
            }
        )
        self.manager = self.manager_configurator.manager
        self.loss_module = self.manager_configurator.loss_module

        for model in self.manager.models:
            num_vars, num_total_vars = model.num_variables
            msg = '\nNumber of Model Variables:\n'
            msg += '\tFree: {}\n'
            msg += '\tTotal: {}'
            print(msg.format(num_vars, num_total_vars))

        # ------------------------------
        # Gather Reconstruction Settings
        # ------------------------------

        # get a list of parameters to fit
        fit_paramater_list = [self.minimize_parameter_default_value
                              for i in
                              range(self.manager.models[0].num_parameters)]
        for name, value in self.minimize_parameter_dict.items():
            fit_paramater_list[self.models[0].get_index(name)] = value

        # parameter input signature
        parameter_tensor_name = 'x_parameters'

        # -------------------------
        # Build reconstruction tray
        # -------------------------

        # create reconstruction tray
        self.reco_tray = ReconstructionTray(
            manager=self.manager, loss_module=self.loss_module)

        # add reconstruction module
        self.reco_tray.add_module(
            'Reconstruction',
            name='reco',
            fit_paramater_list=fit_paramater_list,
            seed_tensor_name=self.seed_keys,
            minimize_in_trafo_space=self.minimize_in_trafo_space,
            parameter_tensor_name=parameter_tensor_name,
            reco_optimizer_interface=self.reco_optimizer_interface,
            scipy_optimizer_settings=self.scipy_optimizer_settings,
            tf_optimizer_settings=self.tf_optimizer_settings,
        )

        # add covariance module
        if self.add_covariances:
            self.reco_tray.add_module(
                'CovarianceMatrix',
                name='covariance',
                fit_paramater_list=fit_paramater_list,
                seed_tensor_name=self.seed_keys,
                reco_key='reco',
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
            )

        # add circularized angular uncertainty estimation module
        if self.add_circular_err:
            if self.add_covariances:
                covariance_key = 'covariance'
            else:
                covariance_key = None

            self.reco_tray.add_module(
                'CircularizedAngularUncertainty',
                name='CircularizedAngularUncertainty',
                fit_paramater_list=fit_paramater_list,
                seed_tensor_name=self.seed_keys,
                reco_key='reco',
                covariance_key=covariance_key,
                minimize_in_trafo_space=self.minimize_in_trafo_space,
                parameter_tensor_name=parameter_tensor_name,
            )

    def Physics(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current P-Frame.
        """

        # get data batch
        data_batch = self.manager.data_handler.get_tensor_from_frame(frame)

        # reconstruct data
        results = self.reco_tray.execute(data_batch)

        # --------------
        # write to frame
        # --------------

        # write best fit results to frame
        best_fit = results['reco']['result'][0]

        result_dict = dataclasses.I3MapStringDouble()
        for i, name in enumerate(self.manager.models[0].parameter_names):
            result_dict[name] = best_fit[i]

        # add an I3Particle
        particle = dataclasses.I3Particle()
        particle.energy = result_dict['energy']
        particle.time = result_dict['time']
        particle.pos = dataclasses.I3Position(
            result_dict['x'], result_dict['y'], result_dict['z'])
        particle.dir = dataclasses.I3Direction(
            result_dict['zenith'], result_dict['azimuth'])

        # write covariance Matrices to frame
        if self.add_covariances:
            for name, value in results['covariance']:
                if name == 'runtime':
                    result_dict['runtime_covariance'] = value
                else:
                    frame[self.output_key+'_cov_matrix_'+name] = \
                        dataclasses.I3Matrix(value)
        else:
            # write covariance matrix from minimizer to frame
            pass

        if self.add_circular_err:
            result_dict['circular_unc'] = \
                results['CircularizedAngularUncertainty']['circular_unc']
            result_dict['runtime_circular_err'] = \
                results['CircularizedAngularUncertainty']['runtime']

        # save to frame
        frame[self.output_key] = result_dict
        frame[self.output_key+'_I3Particle'] = particle

        # push frame to next modules
        self.PushFrame(frame)

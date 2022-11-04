import os
import tensorflow as tf

from icecube import dataclasses, icetray
from icecube.icetray.i3logging import log_info, log_warn

from egenerator.utils import inspect
from egenerator.utils.configurator import ManagerConfigurator
from egenerator.manager.reconstruction.tray import ReconstructionTray


class EventGeneratorVisualizeBestFit(icetray.I3ConditionalModule):

    """Module to visualize Event-Generator best fit.

    """

    def __init__(self, context):
        """Module to visualize Event-Generator best fit.

        Parameters
        ----------
        context : TYPE
            Description
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddOutBox('OutBox')

        # Required settings
        self.AddParameter(
            'reco_key',
            'The key to which the reconstruction result from a prior '
            'call to `EventGeneratorReconstruction` was written to.')
        self.AddParameter(
            'model_names',
            'A model name or list of model names defining which model or '
            'ensemble of models to apply. The full path is given by '
            '`model_base_dir` + model_name if `model_base_dir` is provided. '
            'Otherwise `model_names` must define the full model path.')
        self.AddParameter(
            'output_dir',
            'The output directoy to which the plots will be written to.')

        # Optional settings
        self.AddParameter(
            'model_base_dir',
            'The base directory in which the models are located.  The full '
            'path is given by `model_base_dir` + model_name if '
            '`model_base_dir` is provided. Otherwise `model_names` must '
            'define the full model path.',
            '/data/user/mhuennefeld/exported_models/egenerator')
        self.AddParameter(
            'pulse_key',
            'The pulses to use for the reconstruction. Note: '
            'pulses in exclusion windows must have already been excluded!',
            'InIceDSTPulses')
        self.AddParameter(
            'dom_exclusions_key',
            'The DOM exclusions to use.',
            'BadDomsList')
        self.AddParameter(
            'time_exclusions_key',
            'The time exclusions to use.',
            None)
        self.AddParameter(
            'num_threads',
            'Number of threads to use for tensorflow. This will '
            'be passed on to tensorflow where it is used to set '
            'the "intra_op_parallelism_threads" and '
            '"inter_op_parallelism_threads" settings. If a '
            'value of zero (default) is provided, the system '
            'uses an appropriate number. Note: when running '
            'this as a job on a cluster you might want to limit '
            '"num_threads" to the amount of allocated CPUs.',
            1)
        self.AddParameter(
            'n_doms_x', 'Number of DOMs to plot along x-axis.', 5)
        self.AddParameter(
            'n_doms_y', 'Number of DOMs to plot along y-axis.', 5)
        self.AddParameter(
            'dom_pdf_kwargs',
            'Additional keyword arguments passed on to `plot_dom_pdf`.', {})
        self.AddParameter(
            'dom_cdf_kwargs',
            'Additional keyword arguments passed on to `plot_dom_cdf`.', {})
        self.AddParameter(
            'pdf_file_template',
            'The file template name to which the PDF will be saved to',
            'dom_pdf_{run_id:06d}_{event_id:06d}.png'
        )
        self.AddParameter(
            'cdf_file_template',
            'The file template name to which the CDF will be saved to',
            'dom_cdf_{run_id:06d}_{event_id:06d}.png'
        )

    def Configure(self):
        """Configures Module and loads model from file.
        """
        self.reco_key = self.GetParameter('reco_key')
        self.model_names = self.GetParameter('model_names')
        self.model_base_dir = self.GetParameter('model_base_dir')
        self.output_dir = self.GetParameter('output_dir')
        self.pulse_key = self.GetParameter('pulse_key')
        self.dom_exclusions_key = self.GetParameter('dom_exclusions_key')
        self.time_exclusions_key = self.GetParameter('time_exclusions_key')
        self.num_threads = self.GetParameter('num_threads')
        self.n_doms_x = self.GetParameter('n_doms_x')
        self.n_doms_y = self.GetParameter('n_doms_y')
        self.pdf_file_template = self.GetParameter('pdf_file_template')
        self.cdf_file_template = self.GetParameter('cdf_file_template')
        self.dom_pdf_kwargs = self.GetParameter('dom_pdf_kwargs')
        self.dom_cdf_kwargs = self.GetParameter('dom_cdf_kwargs')

        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]

        manager_dirs = []
        for name in self.model_names:
            if self.model_base_dir is not None:
                manager_dirs.append(os.path.join(self.model_base_dir, name))
            else:
                manager_dirs.append(name)

        # configure tensorfow settings
        ManagerConfigurator.confifgure_tf(
            num_threads=self.num_threads)

        # extract model parameter names
        parameter_names = inspect.extract_model_parameters(manager_dirs[0])

        # Build and configure SourceManager
        self.manager_configurator = ManagerConfigurator(
            manager_dirs=manager_dirs,
            reco_config_dir=None,
            load_labels=False,
            misc_setting_updates={
                'seed_names': [self.reco_key],
                'seed_parameter_names': parameter_names,
            },
            data_setting_updates={
                'pulse_key': self.pulse_key,
                'dom_exclusions_key': self.dom_exclusions_key,
                'time_exclusions_key': self.time_exclusions_key,
            },
            num_threads=self.num_threads,
        )
        self.manager = self.manager_configurator.manager
        self.loss_module = self.manager_configurator.loss_module

        if 'I3ParticleMapping' in self.manager.configuration.config['config']:
            self.i3_mapping = self.manager.configuration.config['config'][
                'I3ParticleMapping']
        else:
            self.i3_mapping = None

        for model in self.manager.models:
            num_vars, num_total_vars = model.num_variables
            msg = '\nNumber of Model Variables:\n'
            msg += '\tFree: {}\n'
            msg += '\tTotal: {}'
            log_info(msg.format(num_vars, num_total_vars))

        # -------------------------
        # Build reconstruction tray
        # -------------------------

        # create reconstruction tray
        self.reco_tray = ReconstructionTray(
            manager=self.manager, loss_module=self.loss_module)

        self.reco_tray.add_module(
            'VisualizePulseLikelihood',
            name='VisualizePulseLikelihood',
            reco_key=self.reco_key,
            output_dir=self.output_dir,
            n_doms_x=self.n_doms_x,
            n_doms_y=self.n_doms_y,
            pdf_file_template=self.pdf_file_template,
            cdf_file_template=self.cdf_file_template,
            dom_pdf_kwargs=self.dom_pdf_kwargs,
            dom_cdf_kwargs=self.dom_cdf_kwargs,
        )

    def Physics(self, frame):
        """Apply Event-Generator model to physics frames.

        Parameters
        ----------
        frame : I3Frame
            The current P-Frame.
        """

        # get data batch
        n, data_batch = self.manager.data_handler.get_data_from_frame(frame)
        assert n == 1, "Currently only 1-event at a time is supported"

        # collect event meta data
        if 'I3EventHeader' in frame:
            header = frame['I3EventHeader']
            event_header = {
                'run_id': header.run_id,
                'sub_run_id': header.sub_run_id,
                'event_id': header.event_id,
                'sub_event_id': header.sub_event_id,
                'sub_event_stream': header.sub_event_stream,
                'start_time': header.start_time.mod_julian_day_double,
                'end_time': header.end_time.mod_julian_day_double,
                'date_string': str(header.start_time.date_time.date()),
                'time_string': str(header.start_time.date_time.time()),
            }
        else:
            event_header = None

        # run tray and make plots
        results = self.reco_tray.execute(data_batch, event_header=event_header)

        # push frame to next modules
        self.PushFrame(frame)

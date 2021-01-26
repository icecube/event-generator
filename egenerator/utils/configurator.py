import os
import tensorflow as tf

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager
from egenerator.utils.build_components import build_loss_module
from egenerator.data.modules.misc.seed_loader import SeedLoaderMiscModule


class ManagerConfigurator:

    def __init__(self, manager_dirs,
                 reco_config_dir=None,
                 load_labels=False,
                 misc_setting_updates={},
                 label_setting_updates={},
                 data_setting_updates={},
                 additional_loss_modules=None,
                 replace_existing_loss_modules=False,
                 configure_tensorflow=True,
                 num_threads=0,
                 tf_random_seed=1337):
        """Set up and configure the SourceManager object.

        Parameters
        ----------
        manager_dirs : str or list of str
            List of model manager directories.
            If more than one model directory is passed, an ensemble of models
            will be used.
        reco_config_dir : str, optional
            The model manager from which to import the reconstruction settings.
            If None is provided, the first passed element of `manager_dirs`
            will be used.
        load_labels : bool, optional
            If True, labels will be loaded from the frame (these must exist).
            If False, no labels will be loaded.
        misc_setting_updates : dict, optional
            A dictionary with setting values to overwrite.
        label_setting_updates : dict, optional
            A dictionary with setting values to overwrite.
        data_setting_updates : dict, optional
            A dictionary with setting values to overwrite.
        additional_loss_modules : list of dict, optional
            A list of dictionaries which define the loss modules to add.
            If `replace_existing_loss_modules` is True, then the original
            loss modules of the event-generator model are replaced. Otherwise
            the loss modules defined here will be appended.
        replace_existing_loss_modules : bool, optional
            Only relevant if `additional_loss_modules` is provided.
            If True, original event-generator loss modules are discarded.
            If False, the loss modules defined in `additional_loss_modules`
            will be appended to the loss modules of the event-generator model.
        configure_tensorflow : bool, optional
            If True, tensorflow will be configured. Note, this may only be done
            once. Hence, if the tensorflow session is already configured, then
            pass False.
        num_threads : int, optional
            Only relevant if `configure_tensorflow` is True.
            Number of threads to use for tensorflow settings
            `intra_op_parallelism_threads` and `inter_op_parallelism_threads`.
            If zero (default), the system picks an appropriate number.
        tf_random_seed : int, optional
            Only relevant if `configure_tensorflow` is True.
            Random seed for tensorflow.
        """
        if isinstance(manager_dirs, str):
            manager_dirs = [manager_dirs]

        if reco_config_dir is None:
            reco_config_dir = [manager_dirs[0]]
        else:
            reco_config_dir = [reco_config_dir]

        reco_config_file = os.path.join(reco_config_dir[0], 'reco_config.yaml')

        if configure_tensorflow:
            self.confifgure_tf(
                num_threads=num_threads, tf_random_seed=tf_random_seed)

        # read in reconstruction config file
        setup_manager = SetupManager([reco_config_file])
        self.config = setup_manager.get_config()
        data_handler_settings = self.config['data_handler_settings']

        # ------------------
        # Create loss module
        # ------------------
        if additional_loss_modules is not None:

            # make sure this is a list of dictionaries
            if isinstance(additional_loss_modules, dict):
                additional_loss_modules = [additional_loss_modules]

            if replace_existing_loss_modules:
                self.config['loss_module_settings'] = additional_loss_modules
            else:
                if isinstance(self.config['loss_module_settings'], dict):
                    loss_modules = [self.config['loss_module_settings']]
                else:
                    loss_modules = self.config['loss_module_settings']
                self.config['loss_module_settings'] = (
                    loss_modules + additional_loss_modules
                )

        self.loss_module = build_loss_module(
            self.config['loss_module_settings'])

        # ------------------
        # Create misc module
        # ------------------
        reco_config = self.config['reconstruction_settings']
        if ('seed_names' in misc_setting_updates and
                misc_setting_updates['seed_names'] != ['x_parameters']):
            misc_module = SeedLoaderMiscModule()
            misc_settings = dict(
                seed_names=[reco_config['seed']],
                seed_parameter_names=reco_config['seed_parameter_names'],
                float_precision=reco_config['seed_float_precision'],
                missing_value=reco_config['seed_missing_value'],
                missing_value_dict=reco_config['seed_missing_value_dict'],
            )
            misc_settings.update(misc_setting_updates)
            misc_module.configure(config_data=None, **misc_settings)

            # create nested dictionary of modified sub components in order to
            # swap out the loaded misc_module of the data_handler sub component
            modified_sub_components = {'data_handler': {
                'misc_module': misc_module,
            }}
        else:
            # The parameter labels are being used as a seed, so we do not need
            # to create a modified misc module
            modified_sub_components = {}

        if not load_labels or label_setting_updates:
            if not load_labels:
                label_config = {
                    'label_module': 'dummy.DummyLabelModule',
                    'label_settings': {},
                }
            else:
                label_config = dict(data_handler_settings)
                label_config['label_settings'].update(label_setting_updates)

            LabelModuleClass = misc.load_class(
                'egenerator.data.modules.labels.{}'.format(
                            label_config['label_module']))
            label_module = LabelModuleClass()
            label_module.configure(config_data=None,
                                   **label_config['label_settings'])

            if 'data_handler' in modified_sub_components:
                modified_sub_components['data_handler']['label_module'] = \
                    label_module
            else:
                modified_sub_components['data_handler'] = {
                    'label_module': label_module
                }

        if data_setting_updates:
            data_config = dict(data_handler_settings)

            DataModuleClass = misc.load_class(
                'egenerator.data.modules.data.{}'.format(
                            data_config['data_module']))
            data_module = DataModuleClass()
            data_config['data_settings'].update(data_setting_updates)
            data_module.configure(config_data=None,
                                  **data_config['data_settings'])

            if 'data_handler' in modified_sub_components:
                modified_sub_components['data_handler']['data_module'] = \
                    data_module
            else:
                modified_sub_components['data_handler'] = {
                    'data_module': data_module
                }

        # -----------------------------
        # Create and load Model Manager
        # -----------------------------

        # load models from config files
        models = []
        for manager_dir in manager_dirs:

            # adjust manager_dir
            config_i = SetupManager(
                [os.path.join(manager_dir, 'reco_config.yaml')]).get_config()
            config_i['model_manager_settings']['config']['manager_dir'] = \
                manager_dir

            # load manager objects and extract models and a data_handler
            model_manger,  _, data_handler, data_transformer = build_manager(
                config_i,
                restore=True,
                modified_sub_components=modified_sub_components,
                allow_rebuild_base_sources=False,
            )
            models.extend(model_manger.models)

        # build manager object
        manager, models, data_handler, data_transformer = build_manager(
                                self.config,
                                restore=False,
                                models=models,
                                data_handler=data_handler,
                                data_transformer=data_transformer,
                                allow_rebuild_base_sources=False)

        # save manager
        self.manager = manager

    @staticmethod
    def confifgure_tf(num_threads=0, tf_random_seed=1337):
        """Configures tensorflow

        Set memory growth of GPUs to true, sets random seed and specifies
        number of threads for inter and intra op parallelism.

        Parameters
        ----------
        num_threads : int, optional
            Number of threads to use for tensorflow settings
            `intra_op_parallelism_threads` and `inter_op_parallelism_threads`.
            If zero (default), the system picks an appropriate number.
        tf_random_seed : int, optional
            Random seed for tensorflow.
        """
        tf.random.set_seed(tf_random_seed)

        # limit GPU usage
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # limit number of CPU threads
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)

import os
import tensorflow as tf
from copy import deepcopy

from egenerator import misc
from egenerator.settings.setup_manager import SetupManager
from egenerator.utils.build_components import build_manager
from egenerator.utils.build_components import build_loss_module
from egenerator.data.modules.misc.seed_loader import SeedLoaderMiscModule


class I3ManagerConfigurator:

    def __init__(self, manager_dirs,
                 reco_config_dir=None,
                 load_labels=False,
                 misc_setting_updates={},
                 label_setting_updates={},
                 data_setting_updates={}):
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

        Raises
        ------
        ValueError
            Description
        """
        if reco_config_dir is None:
            reco_config_dir = [manager_dirs[0]]
        else:
            reco_config_dir = [reco_config_dir]

        reco_config_file = os.path.join(reco_config_dir[0], 'reco_config.yaml')

        # limit GPU usage
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # read in reconstruction config file
        setup_manager = SetupManager([reco_config_file])
        config = setup_manager.get_config()

        # ------------------
        # Create loss module
        # ------------------
        self.loss_module = build_loss_module(config)

        # ------------------
        # Create misc module
        # ------------------
        reco_config = config['reconstruction_settings']
        if misc_setting_updates['seed_names'] == ['x_parameters']:
            # The parameter labels are being used as a seed, so we do not need
            # to create a modified misc module
            modified_sub_components = {}
        else:
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

        if not load_labels or 'modified_label_module' in reco_config:
            if not load_labels:
                label_config = {
                    'label_module': 'dummy.DummyLabelModule',
                    'label_settings': {},
                }
            else:
                label_config = reco_config['modified_label_module']
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

        if 'modified_data_module' in reco_config:
            data_config = reco_config['modified_data_module']
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
                modified_sub_components=deepcopy(modified_sub_components),
                allow_rebuild_base_sources=False,
            )
            models.extend(model_manger.models)

        # build manager object
        manager, models, data_handler, data_transformer = build_manager(
                                config,
                                restore=False,
                                models=models,
                                data_handler=data_handler,
                                data_transformer=data_transformer,
                                allow_rebuild_base_sources=False)

        # save manager
        self.manager = manager

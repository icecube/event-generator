import os
import xgboost
import yaml
from datetime import datetime

# python 3
try:
    from datetime import timezone
    UTC_TIMEZONE = timezone.utc
# python 2
except ImportError:
    import pytz
    UTC_TIMEZONE = pytz.utc

from . import version_control
from . import misc


class XGBoostModelWrapper:

    """Wrapper class around an XGBoost Model.

    The wrapper class allows for saving additional meta data.

    Attributes
    ----------
    column_description : list of str
        The description of the n values of the data tensor X that is used as
        input to the BDT. This describes the names of frame ojbects and
        column/atrribute names. The structure is as follows:
        [
            frame_keys_1: column_keys_1,
            frame_keys_2: column_keys_2,
            ...
            frame_keys_n: column_keys_m,
        ]
        where `frame_keys_i` and `column_keys_i` are lists of str. These define
        the frame key and column/attribute name. Additional alternative names
        can be provided to account for different naming of keys. The first
        match will be chosen.
    meta_data : dict
        A dictionary containing meta data.
    model : XGBoost model
        An trained XGBoost model instance.
    n_jobs : int, optional
        Number of jobs to run in parralel. This setting is passed on to the
        model class.
    """

    def __init__(
            self,
            xgboost_model=None,
            column_description=None,
            meta_data={},
            n_jobs=1,
            ):
        """Initialize the wrapper instance.

        The wrapper instance can either be instantiated with a model, or it
        can be created without. A second call to `load_model` can then be used
        to load a model from disk.

        Parameters
        ----------
        xgboost_model : None, optional
            The trained xgboost model instance.
        column_description : list of str
            The description of the n values of the data tensor X that is used
            as input to the BDT. This describes the names of frame ojbects and
            column/atrribute names. The structure is as follows:
            [
                frame_keys_1: column_keys_1,
                frame_keys_2: column_keys_2,
                ...
                frame_keys_n: column_keys_m,
            ]
            where `frame_keys_i` and `column_keys_i` are lists of str. These
            define the frame key and column/attribute name. Additional
            alternative names can be provided to account for different naming
            of keys. The first match will be chosen.
        meta_data : dict, optional
            A dictionary containing meta data.
        n_jobs : int, optional
            Number of jobs to run in parralel. This setting is passed on to the
            model class.
        """
        meta_data = dict(meta_data)
        if xgboost_model is None:
            assert column_description is None, 'Expected None!'
        else:
            assert column_description is not None, 'Define columns!'

        assert 'column_description' not in meta_data, meta_data
        assert 'model_class' not in meta_data, meta_data

        self.model = xgboost_model
        self.column_description = column_description
        self.n_jobs = n_jobs
        self.meta_data = meta_data
        self.meta_data['column_description'] = self.column_description

    def save_model(self, model_directory):
        """Save the model to the specified directory.

        Parameters
        ----------
        model_directory : str
            The path to the model directory.
        """
        if self.model is None:
            raise ValueError(
                'Model is not configured yet. Either instantiate '
                'with a model, or load one from disk.'
            )

        # create copy of the meta data dict, so that we can add aditional data
        meta_data = dict(self.meta_data)

        # save model class
        meta_data['model_class'] = misc.get_full_class_string_of_object(
            self.model)

        # get pip environment information
        meta_data['git_short_sha'] = str(version_control.short_sha)
        meta_data['git_sha'] = str(version_control.sha)
        meta_data['git_origin'] = str(version_control.origin)
        meta_data['git_uncommited_changes'] = \
            version_control.uncommitted_changes
        meta_data['pip_installed_packages'] = \
            version_control.installed_packages

        # create time stamp
        now = datetime.now(UTC_TIMEZONE)
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        meta_data['creation_date'] = dt_string

        if not os.path.exists(model_directory):
            print('Creating directory: {}'.format(model_directory))
            os.makedirs(model_directory)

        # save xgboost model
        model_file = os.path.join(model_directory, 'model.json')
        self.model.save_model(model_file)

        # save meta data
        meta_file = os.path.join(model_directory, 'meta.yaml')
        with open(meta_file, 'w') as f:
            yaml.dump(meta_data, f, default_flow_style=False)

    def load_model(self, model_directory):
        """Load the model from the specified directory.

        Parameters
        ----------
        model_directory : str
            The path to the model directory.
        """
        if self.model is not None:
            raise ValueError('Model is already configured!')

        # load meta data
        meta_file = os.path.join(model_directory, 'meta.yaml')
        with open(meta_file, 'r') as stream:
            self.meta_data = dict(yaml.safe_load(stream))
        self.column_description = self.meta_data['column_description']

        # load xgboost model
        model_file = os.path.join(model_directory, 'model.json')
        self.model = misc.load_class(self.meta_data['model_class'])(
            n_jobs=self.n_jobs)
        self.model.load_model(model_file)

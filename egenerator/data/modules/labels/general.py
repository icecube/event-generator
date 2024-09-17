from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor


class GeneralLabelModule(BaseComponent):
    """This is a label module that loads general snowstorm labels."""

    def __init__(self, logger=None):
        """Initialize cascade module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(GeneralLabelModule, self).__init__(logger=logger)

    def _configure(
        self,
        config_data,
        trafo_log,
        float_precision,
        parameter_names,
        label_key="LabelsDeepLearning",
        snowstorm_key="SnowstormParameterDict",
        snowstorm_parameters=[],
    ):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : None, str, or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.
        trafo_log : None or bool or list of bool
            Whether or not to apply logarithm on cascade parameters.
            If a single bool is given, this applies to all labels. Otherwise
            a list of bools corresponds to the labels in the order:
                x, y, z, zenith, azimuth, energy, time
            Snowstorm parameters must not be defined here. No logarithm will be
            applied to the snowstorm parameters.
        float_precision : str
            The float precision as a str.
        parameter_names : list[str]
            A list of labels to load from the label_key.
            These labels are added before the snowstorm keys if provided.
        label_key : str, optional
            The name of the key under which the labels are saved.
        snowstorm_key : str, optional
            The name of the key under which the snowstorm parameters are saved.
            If `snowstorm_key` is None, no snowstorm parameters will be loaded.
            Instead a default value of 1. will be assigned to each of the
            `snowstorm_parameters` defined.
        snowstorm_parameters : list[str], optional
            The names of the snowstorm parameters. These must exist in the
            SnowStormParameter dict specified in `snowstorm_key`.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed directly as parameters into the configure method,
            as these are automatically gathered. Components passed as lists,
            tuples, and dicts are also collected, unless they are nested
            deeper (list of list of components will not be detected).
            The dependent_sub_components may also be left empty for these
            passed and detected sub components.
            Deeply nested sub components or sub components created within
            (and not directly passed as an argument to) this component
            must be added manually.
            Settings that need to be defined are:
                class_string:
                    misc.get_full_class_string_of_object(self)
                settings: dict
                    The settings of the component.
                mutable_settings: dict, default={}
                    The mutable settings of the component.
                check_values: dict, default={}
                    Additional check values.
        dict
            The data of the component. Contains:
                'label_tensors': DataTensorList
                    The tensors of type 'label' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        TypeError
            Description
        ValueError
            Description
        """

        # extend trafo log for snowstorm parameters: fill with False
        if trafo_log is None:
            trafo_log_ext = None
        else:
            if isinstance(trafo_log, bool):
                trafo_log_ext = [trafo_log] * len(parameter_names)
            else:
                trafo_log_ext = list(trafo_log)
            trafo_log_ext.extend([False] * len(snowstorm_parameters))

        parameter_dict = {}
        for i, parameter_name in enumerate(parameter_names):
            parameter_dict[parameter_name] = i

        data = {
            "parameter_names": parameter_names,
            "parameter_dict": parameter_dict,
        }
        data["label_tensors"] = DataTensorList(
            [
                DataTensor(
                    name="x_parameters",
                    shape=[
                        None,
                        len(parameter_names) + len(snowstorm_parameters),
                    ],
                    tensor_type="label",
                    dtype=float_precision,
                    trafo=True,
                    trafo_log=trafo_log_ext,
                )
            ]
        )

        if isinstance(config_data, DataTensorList):
            if config_data != data["label_tensors"]:
                msg = "Tensors are wrong: {!r} != {!r}"
                raise ValueError(
                    msg.format(config_data, data["label_tensors"])
                )
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(
                config_data=config_data,
                trafo_log=trafo_log,
                float_precision=float_precision,
                label_key=label_key,
                snowstorm_key=snowstorm_key,
                snowstorm_parameters=snowstorm_parameters,
            ),
        )
        return configuration, data, {}

    def get_data_from_hdf(self, file, *args, **kwargs):
        """Get label data from hdf file.

        Parameters
        ----------
        file : str
            The path to the hdf file.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no label data is loaded.

        Raises
        ------
        ValueError
            Description
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        # open file
        f = pd.HDFStore(file, "r")

        parameters = []
        try:
            _labels = f[self.configuration.config["label_key"]]
            for label in self.data["parameter_names"]:
                parameters.append(_labels[label])

            snowstorm_key = self.configuration.config["snowstorm_key"]
            snowstorm_params = self.configuration.config[
                "snowstorm_parameters"
            ]
            num_events = len(parameters[0])

            if len(snowstorm_params) > 0:
                if snowstorm_key is not None:
                    _snowstorm_params = f[snowstorm_key]
                    for key in snowstorm_params:
                        parameters.append(_snowstorm_params[key])
                        assert len(_snowstorm_params[key]) == num_events
                else:
                    # No Snowstorm key is provided: add dummy values
                    for key in snowstorm_params:
                        parameters.append(np.ones(num_events))

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning("Skipping file: {}".format(file))
            return None, None
        finally:
            f.close()

        # format cascade parameters
        dtype = getattr(np, self.configuration.config["float_precision"])
        parameters = np.array(parameters, dtype=dtype).T
        num_events = len(parameters)

        return num_events, (parameters,)

    def get_data_from_frame(self, frame, *args, **kwargs):
        """Get label data from frame.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no label data is loaded.
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        parameters = []
        try:
            _labels = frame[self.configuration.config["label_key"]]
            for label in self.data["parameter_names"]:
                parameters.append(np.atleast_1d(_labels[label]))

            snowstorm_key = self.configuration.config["snowstorm_key"]
            snowstorm_params = self.configuration.config[
                "snowstorm_parameters"
            ]
            num_events = len(parameters[0])

            if len(snowstorm_params) > 0:
                if snowstorm_key is not None:
                    _snowstorm_params = frame[snowstorm_key]
                    for key in snowstorm_params:
                        snowstorm_param = np.atleast_1d(_snowstorm_params[key])
                        assert len(snowstorm_param) == num_events
                        parameters.append(snowstorm_param)

                else:
                    # No Snowstorm key is provided: add dummy values
                    for key in snowstorm_params:
                        parameters.append(np.ones(num_events))

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning("Skipping frame: {}".format(frame))
            return None, None

        # format cascade parameters
        dtype = getattr(np, self.configuration.config["float_precision"])
        parameters = np.array(parameters, dtype=dtype).T
        num_events = len(parameters)

        return num_events, (parameters,)

    def create_data_from_frame(self, frame, *args, **kwargs):
        """Create label data from frame.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no label data is created.
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        return self.get_data_from_frame(frame, *args, **kwargs)

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write label data to I3Frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.data['data_tensors']).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        pass

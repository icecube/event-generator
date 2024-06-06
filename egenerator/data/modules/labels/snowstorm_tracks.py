from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor


class SnowstormTrackGeneratorLabelModule(BaseComponent):
    """This is a label module that loads the snowstorm track labels."""

    def __init__(self, logger=None):
        """Initialize track module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(SnowstormTrackGeneratorLabelModule, self).__init__(logger=logger)

    def _configure(
        self,
        config_data,
        trafo_log,
        float_precision,
        num_cascades=5,
        label_key="EventGeneratorMuonTrackLabels",
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
            Whether or not to apply logarithm on parameters.
            If a single bool is given, this applies to all labels. Otherwise
            a list of bools corresponds to the labels in the order:
                zenith, azimuth,
                track_anchor_x, track_anchor_y, track_anchor_z,
                track_anchor_time, track_energy,
                track_distance_start, track_distance_end,
                track_stochasticity,
                cascade_0000_energy,
                cascade_{i:04d}_energy, cascade_{i:04d}_distance,
            Snowstorm parameters must not be defined here. No logarithm will be
            applied to the snowstorm parameters.
        float_precision : str
            The float precision as a str.
        num_cascades : int, optional
            Number of cascades along the track.
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

        # sanity checks:
        if num_cascades < 0:
            raise ValueError(
                "Num cascades {} must be positive!".format(num_cascades)
            )

        # compute number of parameters
        if num_cascades == 0:
            num_params = 10
        elif num_cascades == 1:
            num_params = 11
        else:
            num_params = 11 + (num_cascades - 1) * 2

        # create list of parameter names which is needed for data loading
        parameter_names = [
            "zenith",
            "azimuth",
            "track_anchor_x",
            "track_anchor_y",
            "track_anchor_z",
            "track_anchor_time",
            "track_energy",
            "track_distance_start",
            "track_distance_end",
            "track_stochasticity",
        ]
        if num_cascades >= 1:
            parameter_names.append("cascade_0000_energy")

            if num_cascades > 1:
                for i in range(1, num_cascades):
                    parameter_names.append("cascade_{:04d}_energy".format(i))
                    parameter_names.append("cascade_{:04d}_distance".format(i))

        parameter_dict = {}
        for i, parameter_name in enumerate(parameter_names):
            parameter_dict[parameter_name] = i

        # extend trafo log for snowstorm parameters: fill with False
        if isinstance(trafo_log, bool):
            trafo_log_ext = [trafo_log] * num_params
        else:
            trafo_log_ext = list(trafo_log)
        trafo_log_ext.extend([False] * len(snowstorm_parameters))

        data = {
            "parameter_dict": parameter_dict,
            "parameter_names": parameter_names,
        }
        data["label_tensors"] = DataTensorList(
            [
                DataTensor(
                    name="x_parameters",
                    shape=[None, num_params + len(snowstorm_parameters)],
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
                num_cascades=num_cascades,
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

        track_parameters = []
        try:
            _labels = f[self.configuration.config["label_key"]]
            for label in self.data["parameter_names"]:
                track_parameters.append(_labels[label])

            snowstorm_key = self.configuration.config["snowstorm_key"]
            snowstorm_params = self.configuration.config[
                "snowstorm_parameters"
            ]
            num_events = len(track_parameters[0])

            if len(snowstorm_params) > 0:
                if snowstorm_key is not None:
                    _snowstorm_params = f[snowstorm_key]
                    for key in snowstorm_params:
                        track_parameters.append(_snowstorm_params[key])
                        assert len(_snowstorm_params[key]) == num_events
                else:
                    # No Snowstorm key is provided: add dummy values
                    for key in snowstorm_params:
                        track_parameters.append(np.ones(num_events))

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning("Skipping file: {}".format(file))
            return None, None
        finally:
            f.close()

        # format track parameters
        dtype = getattr(np, self.configuration.config["float_precision"])
        track_parameters = np.array(track_parameters, dtype=dtype).T
        num_events = len(track_parameters)

        return num_events, (track_parameters,)

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

        track_parameters = []
        try:
            _labels = frame[self.configuration.config["label_key"]]
            for label in self.data["parameter_names"]:
                track_parameters.append(np.atleast_1d(_labels[label]))

            snowstorm_key = self.configuration.config["snowstorm_key"]
            snowstorm_params = self.configuration.config[
                "snowstorm_parameters"
            ]
            num_events = len(track_parameters[0])

            if len(snowstorm_params) > 0:
                if snowstorm_key is not None:
                    _snowstorm_params = frame[snowstorm_key]
                    for key in snowstorm_params:
                        snowstorm_param = np.atleast_1d(_snowstorm_params[key])
                        assert len(snowstorm_param) == num_events
                        track_parameters.append(snowstorm_param)
                else:
                    # No Snowstorm key is provided: add dummy values
                    for key in snowstorm_params:
                        track_parameters.append(np.ones(num_events))

        except Exception as e:
            self._logger.warning(e)
            self._logger.warning("Skipping frame: {}".format(frame))
            return None, None

        # format track parameters
        dtype = getattr(np, self.configuration.config["float_precision"])
        track_parameters = np.array(track_parameters, dtype=dtype).T
        num_events = len(track_parameters)

        return num_events, (track_parameters,)

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

        raise not NotImplementedError("This module is read-only!")

    def _get_cascade_extension(self, ref_energy, eps=1e-6):
        """
        PPC does its own cascade extension, leaving the showers at the
        production vertex. Reapply the parametrization to find the
        position of the shower maximum, which is also the best approximate
        position for a point cascade.

        Parameters
        ----------
        ref_energy : array_like
            Energy of cascade in GeV.
        eps : float, optional
            Small constant float.

        Returns
        -------
        array_like
            Distance of shower maximum to cascade vertex in meter.
        """

        # Radiation length in meters, assuming an ice density of 0.9216 g/cm^3
        l_rad = 0.358 / 0.9216  # in meter

        """
        Parameters taken from I3SimConstants (for particle e-):
        https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/
        meta-projects/combo/trunk/sim-services/private/
        sim-services/I3SimConstants.cxx
        """
        a = 2.01849 + 0.63176 * np.log(ref_energy + eps)
        b = l_rad / 0.63207

        # Mode of the gamma distribution gamma_dist(a, b) is: (a-1.)/b
        length_to_maximum = np.clip(((a - 1.0) / b) * l_rad, 0.0, float("inf"))
        return length_to_maximum

from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList, DataTensor

try:
    from icecube import dataclasses
except ImportError:
    logging.getLogger(__name__).warning(
        "Could not import icecube. No IceCube support"
    )


class SeedLoaderMiscModule(BaseComponent):
    """This is a misc module that loads seeds for Source Hypotheses."""

    def __init__(self, logger=None):
        """Initialize seed loader module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(SeedLoaderMiscModule, self).__init__(logger=logger)

    def _configure(
        self,
        config_data,
        seed_names,
        seed_parameter_names,
        float_precision,
        missing_value=None,
        missing_value_dict={},
    ):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : str or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.
        seed_names : list of str
            The list of seed names. These must specify data tables in the hdf
            file or keys in the I3Frame.
        seed_parameter_names : list of str or (str, int)
            The column names to extract from the data table in hdf file
            or from the I3Frame object.
            If the entry in the list is a tuple of (str, int), the string will
            be added the specified number of times.
            The values will be extracted and concatenated in the same order
            as the seed_parameter_names.
        float_precision : str
            The float precision as a str.
        missing_value : None, optional
            Optionally, if a parameter does not exist, this value can be
            assigned to it.
        missing_value_dict : dict, optional
            Optionally, a dictionary with default values to assign to a certain
            parameter if it does not exist. Structure: parameter_name: value

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
                'misc_tensors': DataTensorList
                    The tensors of type 'misc' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        ValueError
            Description
        """

        if not isinstance(config_data, (type(None), str, DataTensorList)):
            raise ValueError("Unknown type: {!r}".format(type(config_data)))

        num_params = 0
        for name in seed_parameter_names:
            if isinstance(name, str):
                num_params += 1
            else:
                num_params += name[1]

        # create data tensor list
        tensor_list = []
        for seed_name in seed_names:
            tensor_list.append(
                DataTensor(
                    name=seed_name,
                    shape=[None, num_params],
                    tensor_type="misc",
                    dtype=float_precision,
                )
            )

        data = {"misc_tensors": DataTensorList(tensor_list)}

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={"config_data": config_data},
            mutable_settings={
                "seed_names": seed_names,
                "seed_parameter_names": seed_parameter_names,
                "float_precision": float_precision,
                "missing_value": missing_value,
                "missing_value_dict": missing_value_dict,
            },
        )
        return configuration, data, {}

    def get_data_from_hdf(self, file):
        """Get misc data from hdf file.

        Parameters
        ----------
        file : str
            The path to the hdf file.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors or None
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
            Returns None if no misc data is loaded.
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        # open file
        f = pd.HDFStore(file, "r")

        seeds = []
        num_events_list = []
        missing_value = self.configuration.config["missing_value"]
        missing_value_dict = self.configuration.config["missing_value_dict"]
        for tensor in self.data["misc_tensors"].list:
            seed_name = tensor.name

            seed_parameters = []
            try:
                _labels = f[seed_name]
                for names in self.configuration.config["seed_parameter_names"]:
                    if isinstance(names, str):
                        names = [names]
                    else:
                        names = [names[0]] * names[1]
                    for name in names:
                        if name in _labels:
                            seed_parameters.append(_labels[name].values)
                        elif name in missing_value_dict:
                            num_events = len(seed_parameters[-1])
                            seed_parameters.append(
                                [missing_value_dict[name]] * num_events
                            )
                        elif missing_value is not None:
                            num_events = len(seed_parameters[-1])
                            seed_parameters.append(
                                [missing_value] * num_events
                            )
                        else:
                            raise KeyError(
                                "Could not find key {!r}".format(name)
                            )

            except Exception as e:
                self._logger.warning(e)
                self._logger.warning("Skipping file: {}".format(file))
                return None, None
            finally:
                f.close()

            # format cascade parameters
            dtype = getattr(np, self.configuration.config["float_precision"])
            seed_parameters = np.array(seed_parameters, dtype=dtype).T

            seeds.append(seed_parameters)
            num_events = len(seed_parameters)
            num_events_list.append(num_events)

        # check if number of events matches across all events
        if not np.all([n == num_events for n in num_events_list]):
            msg = "Not all event numbers match: {!r} for seeds: {!r}."
            raise ValueError(
                msg.format(
                    num_events_list, self.configuration.config["seed_names"]
                )
            )

        return num_events, seeds

    def get_data_from_frame(self, frame, *args, **kwargs):
        """Get misc data from frame.

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
            Returns None if no misc data is loaded.
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        seeds = []
        num_events_list = []
        missing_value = self.configuration.config["missing_value"]
        missing_value_dict = self.configuration.config["missing_value_dict"]
        for tensor in self.data["misc_tensors"].list:
            seed_name = tensor.name

            seed_parameters = []
            try:
                _labels = frame[seed_name]
                for names in self.configuration.config["seed_parameter_names"]:
                    if isinstance(names, str):
                        names = [names]
                    else:
                        names = [names[0]] * names[1]
                    for name in names:
                        if isinstance(_labels, dataclasses.I3Particle):
                            if hasattr(_labels, name):
                                value = getattr(_labels, name)
                            elif name in ["x", "y", "z"]:
                                value = getattr(_labels.pos, name)
                            elif name in ["zenith", "azimuth"]:
                                value = getattr(_labels.dir, name)
                            elif name in missing_value_dict:
                                value = missing_value_dict[name]
                            elif missing_value is not None:
                                value = missing_value
                            else:
                                raise KeyError(
                                    "Could not find key {!r}".format(name)
                                )
                            seed_parameters.append([value])
                        elif name in _labels:
                            seed_parameters.append([_labels[name]])
                        elif name in missing_value_dict:
                            seed_parameters.append([missing_value_dict[name]])
                        elif missing_value is not None:
                            seed_parameters.append([missing_value])
                        else:
                            raise KeyError(
                                "Could not find key {!r}".format(name)
                            )

            except KeyError as e:
                self._logger.warning(e)
                self._logger.warning("Skipping frame: {}".format(frame))
                return None, None

            # format cascade parameters
            dtype = getattr(np, self.configuration.config["float_precision"])
            seed_parameters = np.array(seed_parameters, dtype=dtype).T

            seeds.append(seed_parameters)
            num_events = len(seed_parameters)
            num_events_list.append(num_events)

        # check if number of events matches across all events
        if not np.all([n == num_events for n in num_events_list]):
            msg = "Not all event numbers match: {!r} for seeds: {!r}."
            raise ValueError(
                msg.format(
                    num_events_list, self.configuration.config["seed_names"]
                )
            )

        return num_events, seeds

    def create_data_from_frame(self, frame, *args, **kwargs):
        """Create misc data from frame.

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
            Returns None if no misc data is created.
        """
        if not self.is_configured:
            raise ValueError("Module not configured yet!")

        return self.get_data_from_frame(frame, *args, **kwargs)

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write misc data to I3Frame.

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

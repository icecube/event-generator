from __future__ import division, print_function

import logging
import numpy as np
import pandas as pd

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration


class GeneralFilterModule(BaseComponent):
    """Filter module that enables filtering events based on their labels."""

    def __init__(self, logger=None):
        """Initialize filter module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(GeneralFilterModule, self).__init__(logger=logger)

    def _configure(self, constraints, **kwargs):
        """Configure Component class instance

        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        constraints : list of constraints
            A list of filter constraints where a constraint is a
            tuple (key, column, op, threshold) with:
                key: str
                    Table name in HDF5 file
                column: str
                    Column name in HDF5 table.
                op: str
                    The comparison operation to use. Must be one of
                    '<', '>', '==', '<=', '>='
                threshold: data type of column
                    The constrain threshold value.

            An event passes the filter, if all constraints evaluate to True.
        **kwargs
            Arbitrary keyword arguments.

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
            The data of the component.
            Return None if the component has no data.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        settings = dict(kwargs)
        settings["constraints"] = constraints

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=settings,
        )
        return configuration, {}, {}

    def get_event_filter_mask_from_hdf(
        self, file, tensors, num_events, batch, *args, **kwargs
    ):
        """Calculate event filter mask from hdf file.

        Parameters
        ----------
        file : str
            The path to the input hdf5 file.
        tensors : DataTensorList
            The data tensor list that describes the data input tensors.
        num_events : int
            The number of loaded events.
        batch : tuple of array-like
            The data that needs to be filtered.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        array_like
            An array of bool indicating whether an event passed the filter
            (True) or whether it is filtered out (False).
            Shape: [num_events]
        """
        filter_mask = np.ones(num_events, dtype=bool)

        constraints = self.configuration.config["constraints"]
        for key, column, op, threshold in constraints:

            if isinstance(column, int):
                # Assume this is an I3VectorDouble
                with pd.HDFStore(file, "r") as f:
                    values = f[key]["item"].values
                    indices = f[key]["vector_index"].values
                    values = values[indices == column]
            else:
                # Assume this is an I3MapStringDouble
                with pd.HDFStore(file, "r") as f:
                    values = f[key][column]

            if op == ">":
                constraint_mask = values > threshold
            elif op == ">=":
                constraint_mask = values >= threshold
            elif op == "<=":
                constraint_mask = values <= threshold
            elif op == "<":
                constraint_mask = values < threshold
            elif op == "==":
                constraint_mask = values == threshold
            else:
                raise ValueError("Unknown operation: {}".format(op))

            filter_mask = np.logical_and(filter_mask, constraint_mask)

        return filter_mask

    def get_event_filter_mask_from_frame(
        self, frame, tensors, num_events, batch, *args, **kwargs
    ):
        """Calculate event filter mask from frame.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to compute the filter mask.
        tensors : DataTensorList
            The data tensor list that describes the data input tensors.
        num_events : int
            The number of loaded events.
        batch : tuple of array-like
            The data that needs to be filtered.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        array_like
            An array of bool indicating whether an event passed the filter
            (True) or whether it is filtered out (False).
            Shape: [num_events]
        """
        filter_mask = np.ones(num_events, dtype=bool)

        constraints = self.configuration.config["constraints"]
        for key, column, op, threshold in constraints:
            if isinstance(column, int):
                # Assume `column` is an index in to a vector
                values = np.atleast_1d(frame[key][column])
            else:
                value_table = frame[key]
                if key in value_table:
                    values = np.atleast_1d(value_table[key])
                else:
                    values = np.atleast_1d(getattr(value_table, key))

            if op == ">":
                constraint_mask = values > threshold
            elif op == ">=":
                constraint_mask = values >= threshold
            elif op == "<=":
                constraint_mask = values <= threshold
            elif op == "<":
                constraint_mask = values < threshold
            elif op == "==":
                constraint_mask = values == threshold
            else:
                raise ValueError("Unknown operation: {}".format(op))

            filter_mask = np.logical_and(filter_mask, constraint_mask)

        return filter_mask

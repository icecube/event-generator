from __future__ import division, print_function

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration


class DummyFilterModule(BaseComponent):

    """This is a dummy filter module that does not filter any events.
    """

    def _configure(self, **kwargs):
        """Configure Component class instance

        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
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
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=kwargs)
        return configuration, {}, {}

    def filter_data(self, tensors, num_events, batch):
        """Get data from hdf file.

        Parameters
        ----------
        tensors : DataTensorList
            The data tensor list that describes the data input tensors.
        num_events : int
            The number of loaded events.
        batch : tuple of array-like
            The data that needs to be filtered.

        Returns
        -------
        int
            The number of events left after filtering.
        tuple of array-like
            The filtered data.
        """
        return num_events, batch

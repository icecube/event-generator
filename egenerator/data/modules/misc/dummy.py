from __future__ import division, print_function

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList


class DummyMiscModule(BaseComponent):

    """This is a dummy misc module that does not load any misc data.
    """

    def _configure(self, config_data):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : str or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.

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
        """

        if not isinstance(config_data, (type(None), str, DataTensorList)):
            raise ValueError('Unknown type: {!r}'.format(type(config_data)))

        data = {
            'misc_tensors': DataTensorList([])
        }

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={'config_data': config_data})
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
        return None, (None,)

from __future__ import division, print_function

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration
from egenerator.data.tensor import DataTensorList


class DummyWeightModule(BaseComponent):

    """This is a dummy weight module that does not load any weight data.
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
            as these are automatically gathered.The dependent_sub_components
            may also be left empty. This is later filled by the base class
            from the returned sub components dict.
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
                'weight_tensors': DataTensorList
                    The tensors of type 'weight' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """

        if not isinstance(config_data, (type(None), str, DataTensorList)):
            raise ValueError('Unknown type: {!r}'.format(type(config_data)))

        data = {
            'weight_tensors': DataTensorList([])
        }

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings={'config_data': config_data})
        return configuration, data, {}

    def get_data_from_hdf(self, file):
        """Get weight data from hdf file.

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
            Returns None if no weight data is loaded.
        """
        return None, (None,)

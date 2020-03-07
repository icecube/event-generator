from __future__ import division, print_function

from egenerator.data.modules.base import BaseModule
from egenerator.data.tensor import DataTensorList


class DummyWeightModule(BaseModule):

    """This is a dummy weight module that does not load any weight data.
    """

    def _initialize(self, *args, **kwargs):
        """Initialize Module class.
        This is an abstract method and must be implemented by derived class.

        If there are skip_check_keys, e.g. config keys that do not have to
        match, they must be defined here.
        Any settings used within the module must be saved to 'self.settings'.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        pass

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
        DataTensorList
            The tensors of type 'weight' that will be loaded.
        """

        if not isinstance(config_data, (type(None), str, DataTensorList)):
            raise ValueError('Unknown type: {!r}'.format(type(config_data)))

        self.tensors = DataTensorList([])
        return self.tensors

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

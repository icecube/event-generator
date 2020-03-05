from __future__ import division, print_function

from egenerator.data.modules.base import BaseModule


class DummyFilterModule(BaseModule):

    """This is a dummy filter module that does not filter any events.
    """

    def _initialize(self, *args, **kwargs):
        """Initialize Module class.
        This is an abstract method and must be implemented by derived class.

        If there are skip_check_keys, e.g. config keys that do not have to
        match, they must be defined here.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        pass

    def _configure(self, *args, **kwargs):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        pass

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

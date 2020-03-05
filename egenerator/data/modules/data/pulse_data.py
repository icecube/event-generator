from __future__ import division, print_function

from egenerator.data.modules.base import BaseModule


class PulseDataModule(BaseModule):

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_data_from_hdf(self, file, *args, **kwargs):
        """Get data from hdf file.

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
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).

        """
        raise NotImplementedError()

    def get_data_from_frame(self, frame, *args, **kwargs):
        """Get data from I3Frame.
        This will only return tensors of type 'data'.

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
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        """
        raise NotImplementedError()

    def create_data_from_frame(self, frame, *args, **kwargs):
        """Create data from I3Frame.
        This will only return tensors of type 'data'.

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
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        """
        raise NotImplementedError()

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.
        This will only write tensors of type 'data' to frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        raise NotImplementedError()

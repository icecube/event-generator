from __future__ import division, print_function


class BaseModule(object):

    """Summary

    Attributes
    ----------
    is_configured : bool
        Indicates whether the module is configured.
    settings : dict
        All settings which define the module.
    skip_check_keys : list
        A list of settings that do not need to match when comparing to modules.
    tensors : DataTensorList
        A list of data tensors that belong to this module.
    """

    @property
    def is_configured(self):
        return self._is_configured

    @property
    def tensors(self):
        return self._tensors

    @property
    def settings(self):
        return self._settings

    @property
    def skip_check_keys(self):
        return self._skip_check_keys

    def __init__(self, *args, **kwargs):
        """Initialize Base Module Class

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self._is_configured = False
        self._tensors = None
        self._skip_check_keys = []
        self._settings = {}
        self._initialize(*args, **kwargs)

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

    def configure(self, *args, **kwargs):
        """Configure Base Module Class

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        if self._is_configured:
            raise ValueError('Module is already configured!')
        return_value = self._configure(*args, **kwargs)
        self._is_configured = True
        return return_value

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

    def get_skip_check_keys(self):
        """Get keys which do not have to match.

        Returns
        -------
        list of str
            Config keys which do not have to match
        """
        if not self._is_configured:
            raise ValueError('Module must first be configured!')
        return sorted(self._skip_check_keys)

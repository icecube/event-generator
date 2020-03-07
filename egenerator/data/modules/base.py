from __future__ import division, print_function


class BaseModule(object):
    def __init__(self, *args, **kwargs):
        """Initialize Base Module Class

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.is_configured = False
        self.skip_check_keys = []
        self.settings = {}
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
        if self.is_configured:
            raise ValueError('Module is already configured!')
        return_value = self._configure(*args, **kwargs)
        self.is_configured = True
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
        if not self.is_configured:
            raise ValueError('Module must first be configured!')
        return sorted(self.skip_check_keys)

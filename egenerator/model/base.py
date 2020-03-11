from __future__ import division, print_function
import os
import logging
import tensorlfow as tf

from egenerator.manager.component import BaseComponent, Configuration


class Model(tf.Module, BaseComponent):

    """Summary
    """

    @property
    def checkpoint(self):
        if self.untracked_data is not None and \
                'checkpoint' in self.untracked_data:
            return self.untracked_data['checkpoint']
        else:
            return None

    def __init__(self, name=None, logger=None):
        """Initializes Model object.

        Parameters
        ----------
        name : str, optional
            Name of the model. This gets passed on to tf.Module.__init__
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        tf.Module.__init__(name=name)
        BaseComponent.__init__(logger=self._logger)

    def _configure(self, **kwargs):
        """Summary

        Parameters
        ----------
        **kwargs
            Keyword arguments that are passed on to virtual method
            _configure_derived_class.

        Returns
        -------
        Configuration object
            The configuration object of the newly configured component.
            This does not need to include configurations of sub components
            which are passed as parameters into the configure method,
            as these are automatically gathered. The dependent_sub_components
            may also be left empty for these passed sub components.
            Sum components created within a component must be added.
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
            This must at least contain the tensor list which must be
            stored under the key 'tensors'.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Deleted Parameters
        ------------------
        config : dict
            Configuration of the DataHandler.
        config_data : str or list of str, optional
            File name pattern or list of file patterns which define the paths
            to input data files. The first of the specified files will be
            read in to obtain meta data.
        """
        pass
        raise NotImplementedError()
        return configuration, component_data, sub_components

    def _configure_derived_class(self, config, config_data=None):
        """Setup the data handler with a test input file.
        This method needs to be implemented by derived class.

        Parameters
        ----------
        config : dict
            Configuration of the data handler.
        config_data : str or list of str, optional
            File name pattern or list of file patterns which define the paths
            to input data files. The first of the specified files will be
            read in to obtain meta data.

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
            The data of the component.
            This must at least contain the data tensor list which must be
            stored under the key 'tensors'.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """
        raise NotImplementedError()

    def _save(self):
        raise NotImplementedError()

    def _load(self):
        raise NotImplementedError()

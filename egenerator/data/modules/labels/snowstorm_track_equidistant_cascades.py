import logging

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.data.modules.labels.general import GeneralLabelModule


class SnowstormTrackEquidistantCascadesLabelModule(GeneralLabelModule):
    """Track Equidistant Cascades

    This is a label module that loads the snowstorm track labels
    for a track that is defined by equidistant cascades.
    """

    def __init__(self, logger=None):
        """Initialize track module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(SnowstormTrackEquidistantCascadesLabelModule, self).__init__(
            logger=logger
        )

    def _configure(
        self,
        config_data,
        trafo_log,
        float_precision,
        num_cascades,
        label_key="LabelsMCTrackSphere",
        additional_labels=None,
        snowstorm_key="SnowstormParameterDict",
        snowstorm_parameters=[],
    ):
        """Configure Module Class
        This is an abstract method and must be implemented by derived class.

        Parameters
        ----------
        config_data : None, str, or DataTensorList
            This is either the path to a test file or a data tensor list
            object. The module will be configured with this.
        trafo_log : list of bool
            Whether or not to apply logarithm on the track parameters.
            The logarithm for the energy losses is taken for each cascade.
            a list of bools corresponds to the labels in the order:
                "entry_zenith", "entry_azimuth", "entry_t", "entry_energy",
                "zenith", "azimuth", "energy_loss"
            Snowstorm parameters must not be defined here. No logarithm will be
            applied to the snowstorm parameters.
        float_precision : str
            The float precision as a str.
        num_cascades : int, optional
            Number of cascades along the track.
        label_key : str, optional
            The name of the key under which the labels are saved.
        additional_labels : list, optional
            A list of additional labels to load from the label_key.
            These labels are added before the snowstorm keys if provided.
        snowstorm_key : str, optional
            The name of the key under which the snowstorm parameters are saved.
            If `snowstorm_key` is None, no snowstorm parameters will be loaded.
            Instead a default value of 1. will be assigned to each of the
            `snowstorm_parameters` defined.
        snowstorm_parameters : list[str], optional
            The names of the snowstorm parameters. These must exist in the
            SnowStormParameter dict specified in `snowstorm_key`.

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
                'label_tensors': DataTensorList
                    The tensors of type 'label' that will be loaded.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.

        Raises
        ------
        ValueError
            Description
        """

        # sanity checks:
        if num_cascades < 1:
            raise ValueError(
                "Num cascades {} must be > 0!".format(num_cascades)
            )

        assert len(trafo_log) == 7, trafo_log
        trafo_log = trafo_log[:6] + [trafo_log[6]] * num_cascades

        # create list of parameter names which is needed for data loading
        parameter_names = [
            "entry_zenith",
            "entry_azimuth",
            "entry_t",
            "entry_energy",
            "zenith",
            "azimuth",
        ]
        for i in range(num_cascades):
            parameter_names.append(f"energy_loss_{i:04d}")

        _, data, _ = super(
            SnowstormTrackEquidistantCascadesLabelModule, self
        )._configure(
            config_data=config_data,
            trafo_log=trafo_log,
            float_precision=float_precision,
            parameter_names=parameter_names,
            label_key=label_key,
            additional_labels=additional_labels,
            snowstorm_key=snowstorm_key,
            snowstorm_parameters=snowstorm_parameters,
        )

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(
                config_data=config_data,
                trafo_log=trafo_log,
                float_precision=float_precision,
                num_cascades=num_cascades,
                label_key=label_key,
                additional_labels=additional_labels,
                snowstorm_key=snowstorm_key,
                snowstorm_parameters=snowstorm_parameters,
            ),
        )
        return configuration, data, {}

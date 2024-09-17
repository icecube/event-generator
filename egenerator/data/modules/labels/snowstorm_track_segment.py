import logging

from egenerator import misc
from egenerator.manager.component import Configuration
from egenerator.data.modules.labels.general import GeneralLabelModule


class SnowstormTrackSegmentGeneratorLabelModule(GeneralLabelModule):
    """This is a label module that loads the snowstorm track labels."""

    def __init__(self, logger=None):
        """Initialize track module

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """

        logger = logger or logging.getLogger(__name__)
        super(SnowstormTrackSegmentGeneratorLabelModule, self).__init__(
            logger=logger
        )

    def _configure(
        self,
        config_data,
        trafo_log,
        float_precision,
        label_key="EventGeneratorMuonTrackLabels_c0000",
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
        trafo_log : None or bool or list of bool
            Whether or not to apply logarithm on parameters.
            If a single bool is given, this applies to all labels. Otherwise
            a list of bools corresponds to the labels in the order:
                zenith, azimuth,
                track_anchor_x, track_anchor_y, track_anchor_z,
                track_anchor_time, track_energy,
                track_distance_start, track_distance_end,
                track_stochasticity,
                cascade_0000_energy,
                cascade_{i:04d}_energy, cascade_{i:04d}_distance,
            Snowstorm parameters must not be defined here. No logarithm will be
            applied to the snowstorm parameters.
        float_precision : str
            The float precision as a str.
        label_key : str, optional
            The name of the key under which the labels are saved.
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
        TypeError
            Description
        ValueError
            Description
        """

        parameter_names = [
            "track_anchor_x",
            "track_anchor_y",
            "track_anchor_z",
            "zenith",
            "azimuth",
            "track_energy",
            "track_anchor_time",
            "track_distance_end",
            "track_stochasticity",
        ]

        _, data, _ = super(
            SnowstormTrackSegmentGeneratorLabelModule, self
        )._configure(
            config_data=config_data,
            trafo_log=trafo_log,
            float_precision=float_precision,
            parameter_names=parameter_names,
            label_key=label_key,
            snowstorm_key=snowstorm_key,
            snowstorm_parameters=snowstorm_parameters,
        )

        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(
                config_data=config_data,
                trafo_log=trafo_log,
                float_precision=float_precision,
                label_key=label_key,
                snowstorm_key=snowstorm_key,
                snowstorm_parameters=snowstorm_parameters,
            ),
        )
        return configuration, data, {}

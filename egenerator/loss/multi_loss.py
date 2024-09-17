import logging

from egenerator import misc
from egenerator.manager.component import BaseComponent, Configuration


class MultiLossModule(BaseComponent):
    """Multi loss module that combines the loss of multiple loss modules.

    A loss component that is used to compute the loss. The component
    must provide a
    loss_module.get_loss(data_batch_dict, result_tensors, tensors, **kwargs)
    method.
    """

    def __init__(self, logger=None):
        """Initializes MultiLossModule object.

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(MultiLossModule, self).__init__(logger=self._logger)

    def _configure(self, loss_modules):
        """Configure the MultiLossModule component instance.

        Parameters
        ----------
        loss_modules : list of loss modules
            A list of loss module components.

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
            The data of the component.
        dict
            A dictionary of dependent sub components. This is a dictionary
            of sub components that need to be saved and loaded recursively
            when the component is saved and loaded.
            Return None if no dependent sub components exist.
        """

        dependent_sub_components = {}
        for i, module in enumerate(loss_modules):
            dependent_sub_components["loss_modules_{:04d}".format(i)] = module

        # create configuration object
        configuration = Configuration(
            class_string=misc.get_full_class_string_of_object(self),
            settings=dict(),
        )

        return configuration, {}, dependent_sub_components

    def get_loss(
        self,
        data_batch_dict,
        result_tensors,
        tensors,
        model,
        parameter_tensor_name="x_parameters",
        reduce_to_scalar=True,
        sort_loss_terms=False,
        **kwargs
    ):
        """Get the scalar loss for a given data batch and result tensors.

        Parameters
        ----------
        data_batch_dict : dict of tf.Tensor
            parameters : tf.Tensor
                A tensor which describes the input parameters of the source.
                This fully defines the source hypothesis. The tensor is of
                shape [-1, n_params] and the last dimension must match the
                order of the parameter names (model.parameter_names),
            pulses : tf.Tensor
                The input pulses (charge, time) of all events in a batch.
                Shape: [-1, 2]
            pulses_ids : tf.Tensor
                The pulse indices (batch_index, string, dom) of all pulses in
                the batch of events.
                Shape: [-1, 3]
        result_tensors : dict of tf.Tensor
            A dictionary of output tensors.
            This  dictionary must at least contain:

                'dom_charges': the predicted charge at each DOM
                               Shape: [-1, 86, 60, 1]
                'pulse_pdf': The likelihood evaluated for each pulse
                             Shape: [-1]
        tensors : DataTensorList
            The data tensor list describing the input data
        model : Model
            The model object used to calculate the result tensors.
        parameter_tensor_name : str, optional
            The name of the parameter tensor to use. Default: 'x_parameters'.
        reduce_to_scalar : bool, optional
            If True, the individual terms of the log likelihood loss will be
            reduced (aggregated) to a scalar loss.
            If False, a list of tensors will be returned that contain the terms
            of the log likelihood. Note that each of the returned tensors may
            have a different shape.
        sort_loss_terms : bool, optional
            If true, the loss terms will be sorted and aggregated in three
            types of loss terms (this requires `reduce_to_scalar` == False):
                scalar: shape []
                    scalar loss for the whole batch of events
                event: shape [n_batch]
                    vector loss with one value per event
                dom: shape [n_batch, 86, 60]
                    tensor loss with one value for each DOM and event
        **kwargs
            Arbitrary keyword arguments. These will be passed on to
            the get_loss function of the loss modules.

        Returns
        -------
        tf.Tensor or list of tf.Tensor
            if `reduce_to_scalar` is True:
                Scalar loss
                Shape: []
            else:
                List of tensors defining the terms of the log likelihood
        """
        loss_terms = []
        loss = None
        for key in sorted(self.sub_components.keys()):
            loss_module = self.sub_components[key]
            loss_i = loss_module.get_loss(
                data_batch_dict=data_batch_dict,
                result_tensors=result_tensors,
                tensors=tensors,
                model=model,
                parameter_tensor_name=parameter_tensor_name,
                reduce_to_scalar=reduce_to_scalar,
                sort_loss_terms=sort_loss_terms,
                **kwargs
            )
            if reduce_to_scalar:
                if loss is None:
                    loss = loss_i
                else:
                    loss += loss_i
            else:
                if sort_loss_terms:
                    assert len(loss_i) == 3, (loss_module, loss_i)

                    if loss_terms == []:
                        loss_terms = loss_i
                    else:
                        for term_index in range(3):
                            loss_terms[term_index] += loss_i[term_index]
                else:
                    loss_terms.extend(loss_i)

        if reduce_to_scalar:
            return loss
        else:
            return loss_terms

from __future__ import division, print_function
import numpy as np
from copy import deepcopy


class DataTensor(object):

    def __init__(self, name, shape, tensor_type, dtype, exists=True,
                 vector_info=None, trafo=False, trafo_reduce_axes=(),
                 trafo_log=None, trafo_batch_axis=0, **specs):
        """Class for specifying data input tensors.

        Parameters
        ----------
        name : str
            The name of the data tensor.
        shape : tuple of int or None
            The shape of the tensor. Unkown shapes can be specified with None.
        tensor_type : str
            The type of tensor: 'data', 'label', 'weight', 'misc'
        dtype : str
            The data type of the tensor as str.
            getattr(np, dtype) and getattr(tf, dtype) must exist.
        exists : bool, optional
            Indicates whether this data tensor is being loaded.
        vector_info : dict, optional
            A dictionary containing further information regarding the vector
            type. If 'vector_info' is None, it is assumed, that the tensor is
            not a vector type, e.g. it is in event-structure
            (first axis corresponds to batch id). If the tensor is a vector,
            vector_info must contain the following:
                'type': str
                    The type of the vector: 'index', 'value'
                'reference': str
                    The tensor it references. For an index tensor this is the
                    name of the tensor for which it describes the indices. For
                    the value tensor, this is the name of the tensor that
                    specifies the indices.
        trafo : bool, optional
            If True, a transformation for this tensor will be built by the
            data trafo class.
        trafo_reduce_axes : tuple of int, optional
            This list indicates along which axes the mean and std. deviation
            should be the same. This is useful, for instance, when all DOMs
            should be treated equally, even though the tensor has separate
            entries for each DOM.
        trafo_log : bool, tuple of bool, optional
            Whether or not to perform logarithm on values during
            transformation. if trafo_log is a bool and True, the logarithm will
            be applied to the complete tensor. If it is a list of bool, the
            logarithm will be applied to
                values[..., i] if trafo_log[i] is True
        trafo_batch_axis : int, optional
            The axis which defines the batch dimension. The mean and variance
            will be calculated by reducing over this axis.
        **specs
            Description

        Raises
        ------
        ValueError
            Description
        """
        self.name = name
        self.shape = shape
        self.type = tensor_type
        self.dtype = dtype
        self.exists = exists
        self.vector_info = vector_info
        self.trafo = trafo
        self.trafo_reduce_axes = trafo_reduce_axes
        self.trafo_log = trafo_log
        self.trafo_batch_axis = trafo_batch_axis
        self.specs = specs

        # sanity checks
        if self.type not in ['data', 'label', 'weight', 'misc']:
            raise ValueError('Unknown type: {!r}'.format(self.type))

        if not isinstance(self.dtype, str):
            raise ValueError('{!r} != {!r}'.format(type(self.type), str))

        if not hasattr(np, self.dtype):
            raise ValueError('Invalid dtype str: {!r}'.format(self.type))

        if self.vector_info is not None:
            if self.vector_info['type'] not in ['index', 'value']:
                raise ValueError('Unknown vector type: {!r}'.format(
                    self.vector_info['type']))

    def __eq__(self, other):
        """Check for equality of two data tensors.

        Parameters
        ----------
        other : DataTensor object
            The other data tensor to compare against.

        Returns
        -------
        bool
            True, if both data tensors are equal
        """
        if self.__class__ != other.__class__:
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Check for inequality of two data tensors.
        Overrides the default implementation (unnecessary in Python 3)

        Parameters
        ----------
        other : DataTensor object
            The other data tensor to compare against.

        Returns
        -------
        bool
            True, if data tensors are unequal
        """
        return not self.__eq__(other)


class DataTensorList(object):

    def __init__(self, data_tensors):
        """Create a data tensor list object

        Parameters
        ----------
        data_tensors : list of DataTensor objects
            A list of data tensor objects.
        """
        data_tensors = deepcopy(data_tensors)

        # sort the data tensors according to their name
        names = [tensor.name for tensor in data_tensors]
        sorted_indices = np.argsort(names)
        sorted_data_tensors = data_tensors[sorted_indices]

        self.list = sorted_data_tensors
        self.names = []
        self.shapes = []
        self.types = []
        self.exists = []
        self._name_dict = {}
        self._index_dict = {}
        for i, data_tensor in enumerate(self.list):
            self.names.append(data_tensor.name)
            self.shapes.append(data_tensor.shape)
            self.types.append(data_tensor.type)
            self.exists.append(data_tensor.exists)
            self._name_dict[data_tensor.name] = i
            self._index_dict[i] = data_tensor.name

    def __eq__(self, other):
        """Check for equality of two data tensors.

        Parameters
        ----------
        other : DataTensor object
            The other data tensor to compare against.

        Returns
        -------
        bool
            True, if both data tensors are equal
        """
        if self.__class__ != other.__class__:
            return False

        for t1, t2 in zip(self.list, other.list):
            if not t1 == t2:
                return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Check for inequality of two data tensors.
        Overrides the default implementation (unnecessary in Python 3)

        Parameters
        ----------
        other : DataTensor object
            The other data tensor to compare against.

        Returns
        -------
        bool
            True, if data tensors are not equal
        """
        return not self.__eq__(other)

    def __getitem__(self, name):
        """Get a tensor by its 'name' via the [] operator

        Parameters
        ----------
        name : str
            The name of the tensor which will be returned.

        Returns
        -------
        TensorData object
            The tensor data object
        """
        return self.list[self.get_index(name)]

    def serialize(self):
        raise NotImplementedError()

    def deserialize(self):
        raise NotImplementedError()

    def get_index(self, name):
        """Get the index of the tensor 'name'

        Parameters
        ----------
        name : str
            The name of the tensor for which to return the index.

        Returns
        -------
        int
            The index of the specified tensor.
        """
        return self._name_dict[name]

    def get_name(self, index):
        """Get the name of the tensor number 'index'

        Parameters
        ----------
        index : int
            The index of the specified tensor.

        Returns
        -------
        str
            The name of the specified tensor.

        """
        return self._index_dict[index]

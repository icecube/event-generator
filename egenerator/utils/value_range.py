import tensorflow as tf


class BaseValueRange:

    def __init__(self, scale=1.0, offset=0.0):
        """Initialize the ValueRange object

        This object applies a value range function to the input tensor
        to ensure that the output is within a specified range.

        Parameters
        ----------
        scale : float
            The scale factor to multiply the input by.
        offset : float
            The offset to add to the scaled input.
        """
        self.scale = scale
        self.offset = offset

    def __call__(self, x):
        """Apply value range function to the input x

        Parameters
        ----------
        x : tf.Tensor
            The input tensor to apply the value range function to.

        Returns
        -------
        tf.Tensor
            The output tensor after applying the value range function.
        """
        if self.scale != 1.0:
            x = x * self.scale
        if self.offset != 0.0:
            x = x + self.offset
        return x


class EluValueRange(BaseValueRange):

    def __init__(self, scale=1.0, offset=0.0, min_value=0.0):
        """Initialize the EluValueRange object

        This object applies the Exponential Linear Unit (ELU) function
        to the input tensor and adds an offset to the output to ensure
        that the output is always greater than the specified `min_value`.

        Parameters
        ----------
        scale : float
            The scale factor to multiply the input by.
        offset : float
            The offset to add to the scaled input.
        min_value : float
            The minimum value for the output of the value range function.
        """
        super().__init__(scale=scale, offset=offset)
        self.min_value = min_value + 1.0

    def __call__(self, x):
        """Apply value range function to the input x

        Parameters
        ----------
        x : tf.Tensor
            The input tensor to apply the value range function to.

        Returns
        -------
        tf.Tensor
            The output tensor after applying the value range function.
        """
        x = super().__call__(x)
        return tf.nn.elu(x) + self.min_value

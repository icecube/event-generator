
class CircularizedAngularUncertainty:

    def __init__(self, manager, loss_module, tf_functions, **settings):
        """Initialize module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        tf_functions : dict
            Dictionary of created tf functions. These are saved such that they
            may be reused without having to create new ones.
            Structure is as follows:

            tf_functions = {
                func_name1: [(settings1, func1), (settings2, func2)],
                func_name2: [(settings3, func3), (settings4, func4)],
            }

            where func1 and func2 are based of the same function, but use
            different settings. Sampe applies to func3 and func4.
        **settings
            Description

        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    def execute(self, data_batch, results):
        """Execute module for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of tf.Tensors
            A data batch which consists of a tuple of tf.Tensors.
        results : dict
            A dictrionary with the results of previous modules.

        Returns
        -------
        TYPE
            Description
        """

        return result

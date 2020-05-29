
class MarkovChainMonteCarlo:

    def __init__(self, manager, loss_module, function_cache, **settings):
        """Initialize module and setup tensorflow functions.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        function_cache : FunctionCache object
            A cache to store and share created concrete tensorflow functions.
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

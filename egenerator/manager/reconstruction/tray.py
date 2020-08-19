import tensorflow as tf
import timeit
from egenerator import misc


class ReconstructionTray:

    """ReconstructionTray Class

    A class that combines a set of reconstruction modules which are to be run
    on a batch of data. The ReconstructionTray provides the abilitly to share
    created concrete tensorflow functions between the modules without having
    to instantiate new ones. Results from one module can also be used in the
    other depending on execution order.

    General workflow:

        1.) Set up the reconstruction tray:
            - Instantiate a reconstruction tray with a given ModelManager and
              LossModule object
            - Add desired reconstruction modules
        2.) Execute reconstruction modules by calling execute() with a batch
            of data.

    Attributes
    ----------
    function_cache : FunctionCache object
        A cache to store and share created concrete tensorflow functions.
    loss_module : LossModule object
        The LossModule object to use for the reconstruction steps.
    manager : Manager object
            The SourceManager object.
    module_names : list of str
        A list of the module names.
    modules : list
        A list of reconstruction modules that will be executed.
    """

    def __init__(self, manager, loss_module):
        """Initialize Reconstruction Tray with a Manager and LossModule.

        Parameters
        ----------
        manager : Manager object
            The SourceManager object.
        loss_module : LossModule object
            The LossModule object to use for the reconstruction steps.
        """

        # set up a cache for concrete tensorflow functions
        self.function_cache = FunctionCache()

        # create list to store reconstruction modules
        self.modules = []
        self.module_names = []

        self.manager = manager
        self.loss_module = loss_module

    def add_module(self, module, name=None, **settings):
        """Add a reconstruction module to the reconstruction tray.

        Parameters
        ----------
        module : str
            The name of the module class to add. Will be loaded as:
            egenerator.manager.reconstruction.modules.{module}
        name : str, optional
            The name under which to save the results for this reconstruction
            module. This must be a unique name and cannot match any of the
            other added modules. If none is provided, the module name will be:
                module_{index:04d}
        **settings
            An arbitrary number of keyword arguments that will be passed on
            to the initializer of the reconstruction module.
        """

        # load module class
        ModuleClass = misc.load_class(
            'egenerator.manager.reconstruction.modules.{}'.format(module))

        if name is None:
            name = 'module_{:04d}'.format(len(self.module_names))

        if name in self.module_names:
            raise ValueError('Name {} already exists!'.format(name))

        # set up module
        module = ModuleClass(
            manager=self.manager,
            loss_module=self.loss_module,
            function_cache=self.function_cache,
            **settings
        )

        # save module in internal list (needs to keep order)
        self.modules.append(module)
        self.module_names.append(name)

    def execute(self, data_batch):
        """Execute reconstruction models for a given batch of data.

        Parameters
        ----------
        data_batch : tuple of tf.Tensors
            A data batch which consists of a tuple of tf.Tensors.
        """

        # create a container for the results
        results = {}

        # reconstruct batch: run each module in the order they were created
        for name, module in zip(self.module_names, self.modules):

            # run module
            start_t = timeit.default_timer()
            module_results = module.execute(data_batch, results)
            end_t = timeit.default_timer()

            if 'runtime' in module_results:
                raise ValueError('Module results must not contain "runtime"!')

            module_results['runtime'] = end_t - start_t

            results[name] = module_results

        return results


class FunctionCache:
    """Data Structure to keep a cache of created concrete tensorflow functions.

    Attributes
    ----------
    functions : dict
        Dictionary of created tf functions. These are saved such that they may
        be reused without having to create new ones.
        Structure is as follows:

        tf_functions = {
            function_name1: [(settings1, function1), (settings2, function2)],
            function_name2: [(settings3, function3), (settings4, function4)],
        }

        where function1 and function2 are based of the same function, but use
        different settings. Sampe applies to function3 and function4.

    """

    def __init__(self):
        """Initialize data structure
        """
        self.functions = {}

    def add(self, function, settings):
        """Add a function to the local cache

        Parameters
        ----------
        function : function
            The function to add to the local cache.
        settings : dict
            A dictionary of settings that were used to create the function.

        Raises
        ------
        KeyError
            If function with specified settings already exists in cache.
        """
        if function.__name__ in self.functions:

            # check if the function already exists
            for function_entry in self.functions[function.__name__]:
                if settings == function_entry[0]:
                    msg = 'Function {} with settings {} already exists!'
                    raise KeyError(msg.format(
                        function.__name__, settings))

            # append to list
            self.functions[function.__name__].append((settings, function))
        else:
            # add the new function
            self.functions[function.__name__] = [(settings, function)]

    def get(self, function_name, settings):
        """Retrieve a function with the specified settings from the cache.

        Parameters
        ----------
        function_name : str
            The name of the function.
        settings : dict
            A dictionary of settings that were used to create the function.

        Returns
        -------
        function or None
            If the function with the specified settings exists in the cache,
            it will be returned. Otherwise None will be returned.
        """
        if function_name in self.functions:

            # get the function if it already exists
            for function_entry in self.functions[function_name]:
                if settings == function_entry[0]:
                    return function_entry[1]

        # no suitable function exists in cache
        return None

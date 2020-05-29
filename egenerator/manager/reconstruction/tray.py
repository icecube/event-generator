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
    tf_functions : dict
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
        self.tf_functions = {}

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
            module. This must be a unique name and cannot match any of the other
            added modules. If none is provided, the module name will be:
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
            tf_functions=self.tf_functions,
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
            results[name] = module.execute(data_batch, results)

        return results

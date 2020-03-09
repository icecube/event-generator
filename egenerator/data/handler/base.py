from __future__ import division, print_function
import os
import time
import multiprocessing
import numpy as np
import glob
import resource
import ruamel.yaml as yaml
from copy import deepcopy
import logging

from egenerator.manager.component import BaseComponent
from egenerator.data.tensor import DataTensorList


class BaseDataHandler(BaseComponent):

    """The basic data handler class.
    All data handlers must be derived from this class.

    Attributes
    ----------
    config : config : dict
            Configuration of the DataHandler.
    is_setup : bool
        If True, the data handler is setup and ready to be used.
    logger : logging.logger
        The logger to use for logging.
    skip_check_keys : list, optional
        List of keys in the config that do not need to be checked, e.g.
        that may change.
    tensors : DataTensorList
        A list of DataTensor objects. These are the tensors the data
        handler will create and load. They must always be in the same order
        and have the described settings.
    """

    @property
    def config(self):
        return self._config

    @property
    def skip_check_keys(self):
        return self._skip_check_keys

    @property
    def tensors(self):
        return self._tensors

    def __init__(self, logger=None):
        """Initializes DataHandler object.

        Parameters
        ----------
        logger : logging.logger, optional
            A logging instance.
        """
        self._logger = logger or logging.getLogger(__name__)
        super(BaseDataHandler, self).__init__(logger=self._logger)

        # # create empty member variables
        # self._tensors = None
        # self._config = None
        # self._skip_check_keys = None

        # keep track of multiprocessing processes and managers
        self._untracked_data['mp_processes'] = []
        self._untracked_data['mp_managers'] = []

        # self._is_setup = False

    def _assign_settings(self, data_tensors, config, skip_check_keys,
                         submodules_need_to_be_configured):
        """Assign the configuration settings of the data handler.

        Parameters
        ----------
        data_tensors : DataTensorList
            A list of DataTensor objects. These are the tensors the data
            handler will create and load. They must always be in the same order
            and have the described settings.
        config : dict
            Configuration of the DataHandler.
        skip_check_keys : list
            List of keys in the config that do not need to be checked, e.g.
            that may change.
        submodules_need_to_be_configured : bool
            Indicates whether the sub modules need to be configured. This
            is the case, if the data handler configuration is being loaded
            from file. If instead the data handler was configured via setup,
            the sub modules have already been run via _setup and must not
            be setup again.

        Raises
        ------
        ValueError
            if data handler is already set up or if wrong data type is passed.
        """
        if self.is_configured:
            raise ValueError('The data handler is already set up!')

        # define data tensors
        if not isinstance(data_tensors, DataTensorList):
            raise ValueError(
                'Unsupported type: {!r}'.format(type(data_tensors)))

        self._tensors = data_tensors
        self._config = dict(deepcopy(config))
        self._skip_check_keys = list(deepcopy(skip_check_keys))
        if submodules_need_to_be_configured:
            self._assign_settings_of_derived_class(
                self._tensors, self._config, self._skip_check_keys)
        self._is_setup = True

    def _assign_settings_of_derived_class(self, tensors, config,
                                          skip_check_keys):
        """Perform operations to setup and configure the derived class.
        This is only necessary when the data handler is loaded
        from file. In this case the setup methods '_setup' and 'setup' have
        not run.

        Parameters
        ----------
        tensors : DataTensorList
            A list of DataTensor objects. These are the tensors the data
            handler will create and load. They must always be in the same order
            and have the described settings.
        config : dict
            Configuration of the DataHandler.
        skip_check_keys : list
            List of keys in the config that do not need to be checked, e.g.
            that may change.
        """
        raise NotImplementedError

    def check_if_setup(self, msg='Data handler needs to be set up first!'):
        """Checks if the data handler is setup.

        Raises
        ------
        ValueError
            If the data handler is not set up yet.
        """
        if not self.is_configured:
            raise ValueError(msg)

    def setup(self, config, test_data=None, check_config=True):
        """Setup the datahandler with a test input file.

        Parameters
        ----------
        config : dict
            Configuration of the DataHandler.
        test_data : str or list of str, optional
            File name pattern or list of file patterns which define the paths
            to input data files. The first of the specified files will be
            read in to obtain meta data.
        check_config : bool, optional
            If True, the config passed in to this function and the one obtained
            from the setup data handler will be checked to see if they are
            equal.

        Raises
        ------
        ValueError
            If check_config is True and configs do not match.
        """
        if self.is_configured:
            raise ValueError('The data handler is already set up!')

        if test_data is not None:
            if isinstance(test_data, list):
                test_input_data = []
                for input_pattern in test_data[:3]:
                    test_input_data.extend(glob.glob(input_pattern))
            else:
                test_input_data = glob.glob(test_data)
            test_data = test_input_data

        data_tensors, config_new, skip_check_keys = \
            self._setup(config, test_data)

        # check if settings match
        if check_config and config != config_new:
            raise ValueError('{!r} != {!r}'.format(config, config_new))

        self._assign_settings(data_tensors, config_new, skip_check_keys,
                              submodules_need_to_be_configured=False)

    def _setup(self, config, test_data=None):
        """Setup the datahandler with a test input file.
        This method needs to be implemented by derived class.

        Parameters
        ----------
        config : dict
            Configuration of the DataHandler.
        test_data : list of str, optional
            List of valid file paths to input data files. The first of the
            specified files will be read in to obtain meta data and to setup
            the data handler.

        Returns
        -------
        DataTensorList
            A list of DataTensor objects. These are the tensors the data
            handler will create and load. They must always be in the same order
            and have the described settings.
        dict
            Configuration of the DataHandler.
        list
            List of keys in the config that do not need to be checked, e.g.
            that may change.
        """
        raise NotImplementedError()

    # def load(self, file):
    #     """Load the data handler configuration from file.

    #     Parameters
    #     ----------
    #     file : str
    #         The file path from which the data handler configuration will be
    #         loaded.
    #     """
    #     with open(file, 'r') as stream:
    #         data_dict = yaml.load(stream, Loader=yaml.Loader)

    #     # deserialize data_tensors
    #     data_dict['data_tensors'] = DataTensorList(data_dict['data_tensors'])

    #     self._assign_settings(submodules_need_to_be_configured=True,
    #                           **data_dict)

    # def save(self, output_file, overwrite=False):
    #     """Save the data handler configuration to the specified output_file.
    #     A new data handler can be set up with this exported configuration by
    #     calling load(output_file)

    #     Parameters
    #     ----------
    #     output_file : str
    #         The file path to the output file to which the settings will be
    #         saved.
    #     overwrite : bool, optional
    #         If true, overwrite files if they exist, otherwise raise an error.
    #     """
    #     self.check_if_setup()

    #     output_dir = os.path.dirname(output_file)
    #     if not os.path.isdir(output_dir):
    #         self._logger.info('Creating directory {!r}'.format(output_dir))
    #         os.makedirs(output_dir)

    #     if os.path.exists(output_file):
    #         if overwrite:
    #             self._logger.info('Overwriting file {!r}'.format(output_file))
    #         else:
    #             raise IOError('File {!r} already exists!'.format(output_file))

    #     data_dict = {}
    #     data_dict['skip_check_keys'] = self._skip_check_keys
    #     data_dict['config'] = self._config
    #     data_dict['data_tensors'] = self._tensors.serialize()

    #     with open(output_file, 'w') as yaml_file:
    #         yaml.dump(data_dict, yaml_file)

    def _check_data_structure(self, data, check_vector_tensors=False):
        """Check data structure.

        Note: this only checks if the length of tensors and their shapes match.

        Parameters
        ----------
        data : tuple of array-like
            The data tuple to be checked.
        check_vector_tensors : bool, optional
            If True, the data only consists of tensors of type 'data'. Only
            these will be checked.

        Raises
        ------
        ValueError
            If the shape or length of tensors do not match.
        """

        if check_vector_tensors:
            raise NotImplementedError()

        # check rank
        if len(data) != self._tensors.len:
            raise ValueError('Length {!r} and {!r} do not match!'.format(
                len(data), len(self._tensors.names)))

        # check shape
        for values, tensor in zip(data, self._tensors.list):
            if tensor.exists and tensor.vector_info is None:
                if len(values.shape) != len(tensor.shape):
                    raise ValueError(
                        'Rank {!r} and {!r} do not match for {}'.format(
                                len(values.shape), len(tensor.shape),
                                tensor.name))
                for s1, s2 in zip(values.shape, tensor.shape):
                    if s2 is not None and s1 != s2:
                        raise ValueError(
                            'Shapes {!r} and {!r} do not match for {}!'.format(
                                values.shape, tensor.shape, tensor.name))

            if tensor.vector_info is not None:
                # vector tensors: todo write check for this
                self._logger.warning('Not checking vector tensor {!r}'.format(
                    tensor.name))

    def get_data_from_hdf(self, file, *args, **kwargs):
        """Get data from hdf file.

        Parameters
        ----------
        file : str
            The path to the hdf file.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).

        """
        self.check_if_setup()

        num_events, data = self._get_data_from_hdf(file, *args, **kwargs)
        if num_events is None and data is None:
            return None, None
        self._check_data_structure(data)
        return num_events, data

    def _get_data_from_hdf(self, file, *args, **kwargs):
        """Get data from hdf file. This method needs to be implemented by
        derived class.

        Parameters
        ----------
        file : str
            The path to the hdf file.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        int
            Number of events.
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).

        """
        raise NotImplementedError()

    def get_data_from_frame(self, frame, *args, **kwargs):
        """Get data from I3Frame.
        This will only return tensors of type 'data'.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).
        """
        self.check_if_setup()

        num_events, data = self._get_data_from_frame(frame, *args, **kwargs)
        if num_events is None and data is None:
            return None, None
        self._check_data_structure(data, only_data_tensors=True)
        return num_events, data

    def _get_data_from_frame(self, frame, *args, **kwargs):
        """Get data from I3Frame. This method needs to be implemented by
        derived class.
        This will only return tensors of type 'data'.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).
        """
        raise NotImplementedError()

    def create_data_from_frame(self, frame, *args, **kwargs):
        """Create data from I3Frame.
        This will only return tensors of type 'data'.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).
        """
        self.check_if_setup()

        num_events, data = self._create_data_from_frame(frame, *args, **kwargs)
        if num_events is None and data is None:
            return None, None
        self._check_data_structure(data, only_data_tensors=True)
        return num_events, data

    def _create_data_from_frame(self, frame, *args, **kwargs):
        """Create data from I3Frame. This method needs to be implemented by
        derived class.
        This will only return tensors of type 'data'.

        Parameters
        ----------
        frame : I3Frame
            The I3Frame from which to get the data.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).
        """
        raise NotImplementedError()

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.
        This will only write tensors of type 'data' to frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.check_if_setup()

        self._write_data_to_frame(data, frame, *args, **kwargs)

    def _write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame. This method needs to be implemented by
        derived class.
        This will only write tensors of type 'data' to frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self._tensors).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        raise NotImplementedError()

    def batch_to_event_structure(self, values, indices, num_events):
        """Restructures values which are provided as a list [-1, ...] over
        a complete batch, to shape [n_events, n_per_event]
        (note: n_per_event is not constant!) according to an indices array.

        Parameters
        ----------
        values : array_like
            The array to be restructrured: restructuring is performed along
            axis 0.
        indices : array_like
            Must be the same length as values and the first entry on axis 1
            must specify the batch id. The batch id must range from 0 to
            num_events.
        num_events : int
            The number of events in the batch.

        Returns
        -------
        TYPE
            Description
        """
        values_list = []
        indices_list = []
        for i in range(num_events):
            mask = indices[:, 0] == i
            values_list.append(values[mask])
            indices_list.append(indices[mask])
        return values_list, indices_list

    def get_batch_generator(self, input_data, batch_size,
                            sample_randomly=True,
                            pick_random_files_forever=True,
                            file_capacity=1,
                            batch_capacity=5,
                            num_jobs=1,
                            num_add_files=0,
                            num_repetitions=1,
                            num_splits=None,
                            verbose=False,
                            biased_selection_module=None,
                            *args, **kwargs):
        """Get a python generator object that creates data input batches.

        This is a multiprocessing data iterator.
        There are 3 levels:

            1) A number of 'num_jobs' workers load files from the file list
               into memory and extract the DOM input data, labels, and misc
               data if defined.
               The events (input data, labels, misc data) of the loaded
               file is then queued onto a multiprocessing queue named
               'data_batch_queue'.

            2) Another worker aggregates the events of several files
               (number of files defined by 'num_add_files') together
               by dequeing elements from the 'data_batch_queue'.
               It then creates batches from these events
               (randomly if sample_randomly == True ).
               These batches are then put onto the 'final_batch_queue'.
               Elements in the 'final_batch_queue' now include 'batch_size'
               many events ( tuples of dom_responses, cascade_parameters).

            3) The third level consists of the actual generator object.
               It pops elements off of the 'final_batch_queue' and yields
               these as the desired batches of
               (input data, labels, misc data).

        Parameters
        ----------
        input_data : str or list of str
            File name pattern or list of file patterns which define the paths
            to input data files.
        batch_size : int
            Number of events per batch.
        sample_randomly : bool, optional
            If True, random files and events will be sampled.
            If False, file list and events will not be shuffled.
                Although the order will most likely stay the same, this
                can not be guaranteed, since batches are queued as soon as the
                workers finish loading and processing the files.
        pick_random_files_forever : bool, optional
            If True, random files are sampled from the file list in an infinite
                loop.
            If False, all files will only be loaded once. The 'num_repetitions'
                key defines how many times the events of a file will be used.
        file_capacity : int, optional
            Defines the maximum size of the queue which holds the loaded and
            processed events of a whole file.
        batch_capacity : int, optional
            Defines the maximum size of the batch queue which holds the batches
            of size 'batch_size'. This queue is what is used to obtain the
            final batches, which the generator yields.
        num_jobs : int, optional
            Number of jobs to run in parrallel to load and process input files.
        num_add_files : int, optional
            Defines how many files are additionaly loaded at once.
            Batches will be generated among events of these
            (1 + num_add_files) files
        num_repetitions : int, optional
            Number of times the events in a loaded file are to be used, before
            new files are loaded.
        num_splits : int, optional
            If num_splits is given, the loaded file will be divided into
            num_splits chunks of about equal size. This can be useful when
            the input files contain a lot of events, since the multiprocessing
            queue can not handle elements of arbitrary size.
        verbose : bool, optional
            If True, verbose output with additional information on queues.
        biased_selection_module : object, optional
            A biased selection object that implements a 'select(data)'
            method which applies the biased selection to the data and returns
            the selected (biased) data.
        *args
            Variable length argument list that are passed on to class method
            get_data_from_hdf.
        **kwargs
            Arbitrary keyword arguments that are passed on to class method
            get_data_from_hdf.

        Returns
        -------
        generator
            A generator object which yields batches of input data.
            The returned object is a tuple of array-like tensors with the
            specifications as defined in the DataTensorList (self._tensors).

        Raises
        ------
        ValueError
            Description
        """

        self.check_if_setup()

        if isinstance(input_data, list):
            file_list = []
            for input_pattern in input_data:
                file_list.extend(glob.glob(input_pattern))
        else:
            file_list = glob.glob(input_data)

        # define shared memory variables
        num_files_processed = multiprocessing.Value('i')
        processed_all_files = multiprocessing.Value('b')
        data_left_in_queue = multiprocessing.Value('b')

        # initialize shared variables
        num_files_processed.value = 0
        processed_all_files.value = False
        data_left_in_queue.value = True

        # create Locks
        file_counter_lock = multiprocessing.Lock()

        # create and randomly fill file_list queue
        file_list_queue = multiprocessing.Manager().Queue(maxsize=0)

        number_of_files = 0
        if sample_randomly:
            np.random.shuffle(file_list)

        if not pick_random_files_forever:
            # Only go through given file list once
            for file in file_list:
                number_of_files += 1
                file_list_queue.put(file)

        # create data_batch_queue
        data_batch_queue = multiprocessing.Manager().Queue(
                                                    maxsize=file_capacity)

        # create final_batch_queue
        final_batch_queue = multiprocessing.Manager().Queue(
                                                    maxsize=batch_capacity)

        # keep references to managers alive, such that these do not shut
        # down until the BaseDataHandler object gets garbage collected
        self._untracked_data['mp_managers'].append(file_list_queue)
        self._untracked_data['mp_managers'].append(data_batch_queue)
        self._untracked_data['mp_managers'].append(final_batch_queue)

        def file_loader(seed):
            """Helper Method to load files.

            Loads a file from the file list, processes the data and creates
            the tuple of input tensors.
            It then puts these on the 'data_batch_queue' multiprocessing queue.

            Parameters
            ----------
            seed : int
                The seed of the local random stat.
                Note: the the batch generator is *not* deterministic, since
                all processes use the same queue. Due to differing
                runtimes, the order of the processed files may change.
            """
            local_random_state = np.random.RandomState(seed)

            while not processed_all_files.value:

                # get file
                if pick_random_files_forever:
                    file = local_random_state.choice(file_list)
                else:
                    with file_counter_lock:
                        if not file_list_queue.empty():
                            file = file_list_queue.get()
                        else:
                            file = None

                if file and os.path.exists(file):

                    if verbose:
                        usage = resource.getrusage(resource.RUSAGE_SELF)
                        msg = "{} {:02.1f} GB. file_list_queue:" \
                              " {}. data_batch_queue: {}"
                        self._logger.debug(msg.format(
                            file, usage.ru_maxrss / 1024.0 / 1024.0,
                            file_list_queue.qsize(), data_batch_queue.qsize()))
                    num_events, data = \
                        self.get_data_from_hdf(file, *args, **kwargs)

                    # biased selection
                    if biased_selection_module is not None:
                        data = biased_selection_module.select(data)

                    if data is not None:

                        if num_splits is None:

                            # put batch in queue
                            data_batch_queue.put((num_events, data))
                        else:

                            # split data into several smaller chunks
                            # (Multiprocessing queue can only handle
                            #  a certain size)
                            split_indices_list = np.array_split(
                                    np.arange(num_events),  num_splits)

                            for split_indices in split_indices_list:

                                batch = []
                                for tensor in data:
                                    if tensor is None:
                                        batch.append(None)
                                    else:
                                        batch.append(tensor[split_indices])

                                # put batch in queue
                                data_batch_queue.put((len(split_indices),
                                                      batch))
                else:
                    self._logger.warning(
                        'File {!r} does not exist.'.format(file))

                if not pick_random_files_forever:
                    with file_counter_lock:
                        num_files_processed.value += 1
                        if num_files_processed.value == number_of_files:
                            processed_all_files.value = True

        def fill_event_list(data_batch, event_list, exists, num_events):
            """Fills an event_list with a given data_batch.
            """
            for i, tensor in enumerate(self._tensors.list):

                # check if data exists
                if data_batch[i] is None:
                    exists[i] = False

                if tensor.vector_info is not None:
                    if tensor.vector_info['type'] == 'value':
                        # we are collecting value tensors together with
                        # the indices tensors, so skip for now
                        continue
                    elif tensor.vector_info['type'] == 'index':
                        # get value tensor
                        value_index = self._tensors.get_index(
                            tensor.vector_info['reference'])
                        values = data_batch[value_index]
                        indices = data_batch[i]

                        if values or indices is None:
                            assert values == indices, '{!r} != {!r}'.format(
                                values, indices)
                            event_list[value_index].append(None)
                            event_list[i].append(None)
                        else:
                            # This data input is a vector type and must be
                            # restructured to event shape
                            values, indices = self.batch_to_event_structure(
                                                values, indices, num_events)

                            event_list[value_index].extend(values)
                            event_list[i].extend(indices)
                else:
                    # This data input is already in event structure and
                    # must simply be concatenated along axis 0.
                    event_list[i].append(data_batch[i])

        def data_queue_iterator(sample_randomly):
            """Helper Method to create batches.

            Takes (1 + num_add_files) many elements off of the
            'data_batch_queue' (if available). This corresponds to taking
            events of (1 + num_add_files) many files.
            Batches are then generated from these events.

            Parameters
            ----------
            sample_randomly : bool
                If True, a random event order will be sampled.
                If False, events will not be shuffled.
            """
            if not pick_random_files_forever:
                with file_counter_lock:
                    if processed_all_files.value and data_batch_queue.empty():
                        data_left_in_queue.value = False

            # reset event batch
            size = 0
            batch = [[] for v in self._tensors.names]

            while data_left_in_queue.value:
                # create lists and concatenate at end
                # (faster than concatenating in each step)
                exists = [True for v in self._tensors.names]
                event_list = [[] for v in self._tensors.names]

                # get a new set of events from queue and fill
                num_events, data_batch = data_batch_queue.get()
                current_queue_size = num_events
                fill_event_list(data_batch, event_list, exists, num_events)

                n_files = 0
                while ((n_files < num_add_files or current_queue_size <
                        np.sqrt(max(0, num_repetitions - 1)) * batch_size) and
                       data_left_in_queue.value):

                    # avoid dead lock and delay for a bit
                    time.sleep(0.1)

                    if (data_batch_queue.qsize() > 1 or
                            not data_batch_queue.empty()):
                        num_events, data_batch = data_batch_queue.get()
                        n_files += 1
                        current_queue_size += num_events
                        fill_event_list(data_batch, event_list, exists,
                                        num_events)

                # concatenate into one numpy array:
                for i, tensor in enumerate(self._tensors.list):
                    if exists[i] and tensor.vector_info is None:
                        event_list[i] = np.concatenate(event_list[i], axis=0)

                queue_size = current_queue_size
                if verbose:
                    self._logger.debug('queue_size:', queue_size)

                # num_repetitions:
                #   potentially dangerous for batch_size approx file_size
                for epoch in range(num_repetitions):
                    if not sample_randomly:
                        shuffled_indices = range(queue_size)
                    else:
                        shuffled_indices = np.random.permutation(queue_size)

                    # loop through shuffled events and accumulate them
                    for index in shuffled_indices:

                        # add event to batch
                        for i, tensor in enumerate(self._tensors.list):
                            if exists[i]:
                                if tensor.vector_info is None:
                                    batch[i].append(event_list[i][index])
                                elif tensor.vector_info['type'] != 'index':
                                    batch[i].append(event_list[i][index])
                                else:
                                    # correct batch index for vector indices
                                    indices = event_list[i][index]
                                    indices[:, 0] = size
                                    batch[i].append(indices)
                            else:
                                batch[i].append(None)
                        size += 1

                        # check if we have enough for a complete batch
                        if size == batch_size:

                            # create numpy arrays from input tensors
                            for i, tensor in enumerate(self._tensors.list):
                                if exists[i]:
                                    if tensor.vector_info is None:
                                        batch[i] = np.asarray(batch[i])

                                    else:
                                        batch[i] = np.concatenate(batch[i],
                                                                  axis=0)
                                else:
                                    batch[i] = np.asarray(batch[i])

                            if verbose:
                                usage = resource.getrusage(
                                            resource.RUSAGE_SELF).ru_maxrss
                                msg = "{:02.1f} GB. file_list_queue: {}." \
                                      " data_batch_queue: {}. " \
                                      "final_batch_queue: {}"
                                self._logger.debug(msg.format(
                                    usage / 1024.0 / 1024.0,
                                    file_list_queue.qsize(),
                                    data_batch_queue.qsize(),
                                    final_batch_queue.qsize()))
                            final_batch_queue.put(batch)

                            # reset event batch
                            size = 0
                            batch = [[] for v in self._tensors.names]

                if not pick_random_files_forever:
                    with file_counter_lock:
                        if (processed_all_files.value and
                                data_batch_queue.empty()):
                            data_left_in_queue.value = False

            # collect leftovers and put them in an (incomplete) batch
            if size > 0:
                # create numpy arrays from input tensors
                for i, tensor in enumerate(self._tensors.list):
                    if exists[i]:
                        if tensor.vector_info is None:
                            batch[i] = np.asarray(batch[i])

                        else:
                            batch[i] = np.concatenate(batch[i], axis=0)
                    else:
                        batch[i] = np.asarray(batch[i])
                final_batch_queue.put(batch)

        def batch_iterator():
            """Create batch generator

            Yields
            ------
            np.ndarry, np.ndarray
                Returns a batch of DOM_responses and cascade_parameters.
                dom_responses: [batch_size, x_dim, y_dim, z_dim, num_bins]
                cascade_parameters: [batch_size, num_cascade_parameters]
            """
            while data_left_in_queue.value or not final_batch_queue.empty():
                batch = final_batch_queue.get()
                yield batch

        # create processes
        for i in range(num_jobs):
            process = multiprocessing.Process(target=file_loader, args=(i,))
            process.daemon = True
            process.start()
            self._untracked_data['mp_processes'].append(process)

        process = multiprocessing.Process(target=data_queue_iterator,
                                          args=(sample_randomly,))
        process.daemon = True
        process.start()
        self._untracked_data['mp_processes'].append(process)

        return batch_iterator()

    def kill(self):
        """Kill Multiprocessing queues and workers
        """
        for process in self._untracked_data['mp_processes']:
            process.terminate()

        time.sleep(0.1)
        for process in self._untracked_data['mp_processes']:
            process.join(timeout=1.0)

        self._untracked_data['mp_managers'] = []

    def __del__(self):
        self.kill()

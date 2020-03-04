from __future__ import division, print_function
import os
import multiprocessing
import numpy as np
import glob
import resource
import ruamel.yaml as yaml
from copy import deepcopy
import logging


class DataTensor(object):

    def __init__(self, name, shape, tensor_type, dtype, vector_info=None,
                 trafo=False, trafo_reduce_axes=None, trafo_log=None, **specs):
        """Class for specifying data input tensors.

        Parameters
        ----------
        name : str
            The name of the data tensor.
        shape : tuple of int or None
            The shape of the tensor. Unkown shapes can be specified with None.
        tensor_type : str
            The type of tensor: 'data', 'label', 'weight', 'misc'
        dtype : np.dtype or tf.dtype
            The data type of the tensor.
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
            Description
        trafo_reduce_axes : None, optional
            Description
        trafo_log : None, optional
            Description
        **specs
            Description
        """
        self.name = name
        self.shape = shape
        self.type = tensor_type
        self.dtype = dtype
        self.vector_info = vector_info
        self.trafo = trafo
        self.trafo_reduce_axes = trafo_reduce_axes
        self.trafo_log = trafo_log
        self.specs = specs

        # sanity checks
        if self.vector_info is not None:
            if self.vector_info['type'] not in ['index', 'value']:
                raise ValueError('Unknown vector type: {!r}'.format(
                    self.vector_info['type']))


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
        self._name_dict = {}
        self._index_dict = {}
        for i, data_tensor in enumerate(self.list):
            self.names.append(data_tensor.name)
            self.shapes.append(data_tensor.shape)
            self.types.append(data_tensor.type)
            self._name_dict[data_tensor.name] = i
            self._index_dict[i] = data_tensor.name

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


class BaseDataHandler(object):

    def __init__(self, config, data_tensors, settings={}, skip_check_keys=[],
                 logger=None):
        """Initializes DataHandler object.

        Parameters
        ----------
        config : dict
            Dictionary containing all settings as read in from config file.
            This config must contain all settings needed to configure the
            data handler.
        data_tensors : DataTensorList
            A list of DataTensor objects. These are the tensors the data
            handler will create and load. They must always be in the same order
            and have the described settings.
        settings : dict, optional
            Settins of the DataHandler.
        skip_check_keys : list, optional
            List of keys in the settings that do not need to be checked, e.g.
            that may change.
        logger : None, optional
            Description

        Raises
        ------
        ValueError
            Description
        """
        self.logger = logger or logging.getLogger(__name__)

        # read and save config
        self._config = dict(deepcopy(config))

        # create object settings
        self.settings = settings
        self.skip_check_keys = skip_check_keys

        # define data tensors
        if not isinstance(data_tensors, DataTensorList):
            raise ValueError(
                'Unsupported type: {!r}'.format(type(data_tensors)))
        self.tensors = data_tensors

        # keep track of multiprocessing processes
        self._mp_processes = []

        self.is_setup = False

    def get_data_from_hdf(self, file, *args, **kwargs):
        """Get data from hdf file. This method needs to be implemented by
        inheriting class.

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
            DataTensorList (self.tensors).

        """
        raise NotImplementedError

    def get_data_from_frame(self, frame, *args, **kwargs):
        """Get data from I3Frame. This method needs to be implemented by
        inheriting class

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
            DataTensorList (self.tensors).
        """
        raise NotImplementedError

    def create_data_from_frame(self, frame, *args, **kwargs):
        """Create data from I3Frame. This method needs to be implemented by
        inheriting class

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
            DataTensorList (self.tensors).
        """
        raise NotImplementedError

    def write_data_to_frame(self, data, frame, *args, **kwargs):
        """Write data to I3Frame.

        Parameters
        ----------
        data : tuple of array-like tensors
            The input data (array-like) as specified in the
            DataTensorList (self.tensors).
        frame : I3Frame
            The I3Frame to which the data is to be written to.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        raise NotImplementedError

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
            specifications as defined in the DataTensorList (self.tensors).

        Raises
        ------
        ValueError
            Description
        """

        if not self.is_setup:
            raise ValueError('DataHandler needs to be set up first!')

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
                        self.logger.debug(msg.format(
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
                    self.logger.warning(
                        'File {!r} does not exist.'.format(file))

                if not pick_random_files_forever:
                    with file_counter_lock:
                        num_files_processed.value += 1
                        if num_files_processed.value == number_of_files:
                            processed_all_files.value = True

        def fill_event_list(data_batch, event_list, exists, num_events):
            """Fills an event_list with a given data_batch.
            """
            for i, tensor in enumerate(self.tensors.list):

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
                        value_index = self.tensors.get_index(
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
            batch = [[] for v in self.tensors.names]

            while data_left_in_queue.value:
                # create lists and concatenate at end
                # (faster than concatenating in each step)
                exists = [True for v in self.tensors.names]
                event_list = [[] for v in self.tensors.names]

                # get a new set of events from queue and fill
                num_events, data_batch = data_batch_queue.get()
                current_queue_size = num_events
                fill_event_list(data_batch, event_list, exists, num_events)

                while (current_queue_size <
                       np.sqrt(max(0, num_repetitions - 1)) * batch_size and
                       data_left_in_queue.value):

                    # avoid dead lock and delay for a bit
                    time.sleep(0.1)

                    for i in range(num_add_files):
                        if (data_batch_queue.qsize() > 1 or
                                not data_batch_queue.empty()):
                            num_events, data_batch = data_batch_queue.get()
                            current_queue_size += len(num_events)
                            fill_event_list(data_batch, event_list, exists,
                                            num_events)

                # concatenate into one numpy array:
                for i, tensor in enumerate(self.tensors.list):
                    if exists[i] and tensor.vector_info is None:
                        event_list[i] = np.concatenate(event_list[i], axis=0)

                queue_size = current_queue_size
                if verbose:
                    self.logger.debug('queue_size:', queue_size)

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
                        for i, tensor in enumerate(self.tensors.list):
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
                            for i, tensor in enumerate(self.tensors.list):
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
                                self.logger.debug(msg.format(
                                    usage / 1024.0 / 1024.0,
                                    file_list_queue.qsize(),
                                    data_batch_queue.qsize(),
                                    final_batch_queue.qsize()))
                            final_batch_queue.put(batch)

                            # reset event batch
                            size = 0
                            batch = [[] for v in self.tensors.names]

                if not pick_random_files_forever:
                    with file_counter_lock:
                        if (processed_all_files.value and
                                data_batch_queue.empty()):
                            data_left_in_queue.value = False

            # collect leftovers and put them in an (incomplete) batch
            if size > 0:
                # create numpy arrays from input tensors
                for i, tensor in enumerate(self.tensors.list):
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
            self._mp_processes.append(process)

        process = multiprocessing.Process(target=data_queue_iterator,
                                          args=(sample_randomly,))
        process.daemon = True
        process.start()
        self._mp_processes.append(process)

        return batch_iterator()

    def kill(self):
        """Kill Multiprocessing queues and workers
        """
        for process in self._mp_processes:
            process.terminate()

        time.sleep(0.1)
        for process in self._mp_processes:
            process.join(timeout=1.0)

    def __del__(self):
        self.kill()

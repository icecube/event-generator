import os
import numpy as np
import xgboost
import timeit
from collections import deque

from icecube import icetray, dataclasses

from .bdt_wrapper import XGBoostModelWrapper


class ApplyXGBoostModel(icetray.I3ConditionalModule):

    """Module to apply an XGBoost model.

    Attributes
    ----------
    batch_size : int, optional
        The number of events to accumulate and pass through the BDT in
        parallel. A higher batch size than 1 can usually improve recontruction
        runtime, but will also increase the memory footprint.
    config : dict
        Dictionary with configuration settings
    data_handler : dnn_reco.data_handler.DataHanlder
        A data handler object. Handles nn model input meta data and provides
        tensorflow placeholders.
    data_transformer : dnn_reco.data_trafo.DataTransformer
        The data transformer.
    model : XGBoost or sklearn model
        The BDT model
    """

    def __init__(self, context):
        """Initialize DeepLearningReco Module
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            'model_path',
            'Path to the model. This must be a model that was previously'
            'exported from a XGBoostModelWrapper instance via the `save_model`'
            ' method.'
        )
        self.AddParameter(
            'batch_size',
            'The number of events to accumulate and pass through the BDT in '
            'parallel. A higher batch size than 1 can usually improve '
            'recontruction runtime, but will also increase the memory '
            'footprint.',
            128,
        )
        self.AddParameter(
            'output_key',
            'Frame key to which the result will be written. If none is '
            'provided, the key will be written to the model base name.',
            None,
        )

    def Configure(self):
        """Configure module and load XGBoost model.
        """
        self._model_path = self.GetParameter('model_path')
        self._batch_size = self.GetParameter('batch_size')
        self._output_key = self.GetParameter('output_key')

        if self._output_key is None:
            self._output_key = 'BDT_{}'.format(
                os.path.basename(self._model_path))

        # create and load model
        self._model_wrapper = XGBoostModelWrapper()
        self._model_wrapper.load_model(self._model_path)
        self._model = self._model_wrapper.model
        self._n_features = len(self._model_wrapper.column_description)

        # create variables and frame buffer for batching
        self._frame_buffer = deque()
        self._pframe_counter = 0
        self._batch_event_index = 0
        self._feature_batch = np.empty([self._batch_size, self._n_features])
        self._runtime_batch = np.empty([self._batch_size])
        self._y_pred_batch = None
        self._runtime_prediction = None

    def Process(self):
        """Process incoming frames.

        Pop frames and put them in the frame buffer.
        When a physics frame is popped, accumulate the input data to form
        a batch of events. Once a full batch of physics events is accumulated,
        perform the prediction and push the buffered frames.
        The Physics method can then write the results to the physics frame
        by using the results:
            self._y_pred_batch
            self._runtime_prediction, self._runtime_preprocess_batch
        and the current event index self._batch_event_index
        """
        frame = self.PopFrame()

        # put frame on buffer
        self._frame_buffer.append(frame)

        # check if the current frame is a physics frame
        if frame.Stop == icetray.I3Frame.Physics:

            # add input data for this event
            start_time = timeit.default_timer()
            self._feature_batch[self._pframe_counter] = self._get_data(frame)
            self._runtime_batch[self._pframe_counter] = (
                timeit.default_timer() - start_time)

            self._pframe_counter += 1

            # check if we have a full batch of events
            if self._pframe_counter == self._batch_size:

                # we have now accumulated a full batch of events so
                # that we can perform the prediction
                self._process_frame_buffer()

    def Finish(self):
        """Run prediciton on last incomplete batch of events.

        If there are still frames left in the frame buffer there is an
        incomplete batch of events, that still needs to be passed through.
        This method will run the prediction on the incomplete batch and then
        write the results to the physics frame. All frames in the frame buffer
        will be pushed.
        """
        if self._frame_buffer:

            # there is an incomplete batch of events that we need to complete
            self._process_frame_buffer()

    def _process_frame_buffer(self):
        """Performs prediction for accumulated batch.
        Then writes results to physics frames in frame buffer and eventually
        pushes all of the frames in the order they came in.
        """
        self._perform_prediction(size=self._pframe_counter)

        # reset counters and indices
        self._batch_event_index = 0
        self._pframe_counter = 0

        # push frames
        while self._frame_buffer:
            fr = self._frame_buffer.popleft()

            if fr.Stop == icetray.I3Frame.Physics:

                # write results at current batch index to frame
                self._write_to_frame(fr, self._batch_event_index)

                # increase the batch event index
                self._batch_event_index += 1

            self.PushFrame(fr)

    def _perform_prediction(self, size):
        """Perform the prediction for a batch of events.

        Parameters
        ----------
        size : int
            The size of the current batch.
        """
        if size > 0:
            start_time = timeit.default_timer()

            self._y_pred_batch = self._model.predict_proba(
                self._feature_batch[:size])

            self._runtime_prediction = (
                timeit.default_timer() - start_time) / size
        else:
            self._y_pred_batch = None
            self._runtime_prediction = None

    def _write_to_frame(self, frame, batch_event_index):
        """Writes the prediction results of the given batch event index to
        the frame.

        Parameters
        ----------
        frame : I3Frame
            The physics frame to which the results should be written to.
        batch_event_index : int
            The batch event index. This defines which event in the batch is to
            be written to the frame.
        """

        # Write prediction and uncertainty estimate to frame
        results = {}
        for i, pred in enumerate(self._y_pred_batch[batch_event_index]):

            name = 'pred_{:03d}'.format(i)

            # save prediction
            results[name] = float(pred)

        # write time measurement to frame
        results['runtime_prediction'] = self._runtime_prediction
        results['runtime_preprocess'] = self._runtime_batch[batch_event_index]

        # write to frame
        frame[self._output_key] = dataclasses.I3MapStringDouble(results)

    def _get_data(self, frame):
        """Get input features of BDT from the current frame.

        Parameters
        ----------
        frame : I3Frame
            The current physics frame.

        Returns
        -------
        list
            A list of the collected input features.
        """
        x_data = []
        for keys, cols in self._model_wrapper.column_description:
            value = None
            for key in keys:
                try:
                    obj = frame[key]
                except KeyError:
                    continue

                for col in cols:
                    try:
                        if isinstance(obj, (dataclasses.I3MapStringDouble,
                                            dataclasses.I3MapStringBool,
                                            dataclasses.I3MapStringInt)):
                            value = obj[col]
                        elif not col:
                            value = obj.value
                        else:
                            value = getattr(obj, col)
                    except AttributeError:
                        if isinstance(obj, dataclasses.I3Particle):
                            if col in ['zenith', 'azimuth']:
                                value = getattr(obj.dir, col)
                            elif col in ['x', 'y', 'z']:
                                value = getattr(obj.pos, col)

                    if value is not None:
                        break
                if value is not None:
                        break

            if value is None:
                raise ValueError('Could not find:', keys, cols)

            x_data.append(value)
        return x_data

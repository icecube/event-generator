import numpy as np
from dnn_cascade_selection.utils.model_wrapper import MLModelWrapper


class DummyBiasFunction:

    """Dummy bias function class
    """

    def __init__(self, settings):
        """Summary

        Parameters
        ----------
        settings : dict
            The settings for this class.
        """
        pass

    def __call__(self, bias_data):
        """Apply Bias Function

        Parameters
        ----------
        bias_data : dict
            Dictionary of bias input data.

        Returns
        -------
        float
            Keep probability: probability with which this event should be kept.
        """
        return 1.

    def add_additional_features(self, df):
        """Add additional features to the bias data dictionary.

        Parameters
        ----------
        df : dict
            The bias data dictionary.

        Returns
        -------
        dict
            The updated bias data dictionary
        """
        eps = 1e-8
        energy_key = 'layer_energy_{:02d}'
        charge_key = 'layer_charge_{:02d}'
        for i in range(4):
            df['entry_x_{:02d}'.format(i)] = df['entry_x'][i]
            df['entry_y_{:02d}'.format(i)] = df['entry_y'][i]
            df['entry_z_{:02d}'.format(i)] = df['entry_z'][i]
            df['exit_z_{:02d}'.format(i)] = df['exit_z'][i]
            df['track_length_{:02d}'.format(i)] = df['track_lengths'][i]
            df['layer_energy_{:02d}'.format(i)] = df['layer_energies'][i]
            df['layer_charge_{:02d}'.format(i)] = float(np.sum(
                df['layer_dom_charges'][i]))

            df['charge_energy_ratio_{:02d}'.format(i)] = (
                df[charge_key.format(i)] / (df[energy_key.format(i)] + eps)
            )

        df['inner_outer_energy_ratio'] = (
            (df['layer_energy_00'] + df['layer_energy_01']) /
            (df['layer_energy_02'] + eps)
        )

        df['inner_charge'] = df['layer_charge_00'] + df['layer_charge_01']
        df['outer_charge'] = df['layer_charge_02'] + df['layer_charge_03']
        df['total_charge'] = df['inner_charge'] + df['outer_charge']

        df['inner_outer_charge_ratio'] = (
            df['inner_charge'] / (df['outer_charge'] + eps)
        )

        df['outer_charge_fraction'] = (
            df['outer_charge'] / (df['total_charge'] + eps)
        )

        df['inner_length'] = df['track_length_00'] + df['track_length_01']
        df['total_length'] = df['inner_length'] + df['track_length_02']

        df['inner_outer_length_ratio'] = (
            (df['inner_length']) /
            (df['track_length_02'] + eps)
        )

        df['outer_length_fraction'] = (
            df['track_length_02'] /
            (df['total_length'] + eps)
        )

        return df


class BDTBiasFunction(DummyBiasFunction):

    """BDT bias function class
    """

    def __init__(self, model_path, probability_bound=1e-5, score_index=1):
        """Summary

        Parameters
        ----------
        model_path : str
            The file path to the BDT model.
        probability_bound : float, optional
            The lower bound on the keep probability. The output of the
            BDT will be clipped to a minimal value as specified here.
        score_index : int, optional
            The index of score values to use from model.predict_proba().
        """

        # create and load model
        self._model_path = model_path
        self._score_index = score_index
        self._probability_bound = probability_bound
        self._model_wrapper = MLModelWrapper()
        self._model_wrapper.load_model(self._model_path)
        self.model = self._model_wrapper.model
        self.n_features = len(self._model_wrapper.column_description)
        self.features = []
        for keys, cols in self._model_wrapper.column_description:
            assert len(keys) == 1, ('Expected only one key:', keys)
            assert len(cols) == 1, ('Expected only one column:', cols)
            self.features.append(cols[0])

    def __call__(self, bias_data):
        """Apply Bias Function

        Parameters
        ----------
        bias_data : dict
            Dictionary of bias input data.

        Returns
        -------
        float
            Keep probability: probability with which this event should be kept.
        """

        # add additional features
        bias_data = self.add_additional_features(bias_data)

        # collect features for BDT input data
        input_data = []
        for feature in self.features:
            input_data.append(bias_data[feature])
        input_data = np.expand_dims(np.asarray(input_data), axis=0)

        # apply BDT
        score = self.model.predict_proba(input_data)[:, self._score_index]

        assert len(score) == 1, score
        score = score[0]

        # The score is not the same as a probability!
        # This would only be the case if the training data distribution
        # and distribution of data it is applied on is exactly the same
        # and if there is no over-training and so on..
        # For now, we will ignore thise and use the score directly
        keep_probability = score

        # make sure the probability is greater zero
        keep_probability = max(keep_probability, self._probability_bound)

        return keep_probability

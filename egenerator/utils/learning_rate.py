import tensorflow as tf

from egenerator import misc


class MultiLearningRateScheduler(tf.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that combines multiple schedulers
    """

    def __init__(self, boundaries, scheduler_settings, name=None):
        """MultiLearningRateScheduler

        The function returns a 1-arg callable to compute the multi learning
        rate schedule when passed the current optimizer step.


        Parameters
        ----------
        boundaries
            A list of `Tensor`s or `int`s or `float`s with strictly
            increasing entries, and with all elements having the same type as
            the optimizer step.
        scheduler_settings : list of dict
            A list of scheduler settings that specify the learning rate
            schedules to use for the intervals defined by `boundaries`.
            It should have one more element than `boundaries`, and all
            schedulers should return the same type.
            Each scheduler_setting dict should contain the following:
                'full_class_string': str
                    The full class string of the scheduler class
                'settings': dict
                    A dictionary of arguments that are passed on to
                    the scheduler class
        name
            A string. Optional name of the operation. Defaults to
            'MultiLearningRateScheduler'.
        """

        super(MultiLearningRateScheduler, self).__init__()

        if len(boundaries) != len(scheduler_settings) - 1:
            raise ValueError(
              "The length of boundaries should be 1 less than the length "
              "of scheduler_settings")

        # create schedulers
        schedulers = []
        for settings in scheduler_settings:
            scheduler_class = misc.load_class(settings['full_class_string'])
            scheduler = scheduler_class(**settings['settings'])
            schedulers.append(scheduler)

        if name is None:
            name = 'MultiLearningRateScheduler'

        self.boundaries = boundaries
        self.scheduler_settings = scheduler_settings
        self.schedulers = schedulers
        self.name = name

    def __call__(self, step):
        if step <= self.boundaries[0]:
            return self.schedulers[0](step)
        if step > self.boundaries[-1]:
            return self.schedulers[-1](step)

        for low, high, scheduler in zip(self.boundaries[:-1],
                                        self.boundaries[1:],
                                        self.schedulers[1:-1]):
            if step > low and step <= high:
                return scheduler(step)

    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "scheduler_settings": self.scheduler_settings,
            "name": self.name
        }

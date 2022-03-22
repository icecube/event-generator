
.. _event_hypothesis:

Constructing an event Hypothesis
********************************

The |egenerator| framework allows to train Neural Networks (NNs) to model
the expected light yield for a given event hypothesis.
The event hypothesis can be completely defined by the user.
In its simplest form, this event hypothesis consists of a list of parameters.
During the training process, these parameters are linked to the expected light
yield in the detector with the help of annotated training data that includes
the measured pulses as well as the true parameters for the given event hypothesis.

Within the |egenerator| framework, event hypotheses are set up in a nested structure.
This allows the user to construct arbitrary complex event hypotheses from simple,
elementary building blocks.
These elementary building blocks could be a single energy deposition and a
track segment, for instance.
This setup then only requires training of one NN for each of these elementary
building blocks.
However, there is no necessity for this decomposition. One could also directly
train a model for a more complex event hypothesis.

Internally, event hypotheses are defined via classes that inherit from the
`Source <https://github.com/icecube/event-generator/blob/master/egenerator/model/source/base.py#L30>`_ base class. The ``Source`` class itself is a daughter class of
the more general `Model <https://github.com/icecube/event-generator/blob/master/egenerator/model/base.py#L14>`_ and `BaseComponent <https://github.com/icecube/event-generator/blob/be1d3a403807ddfe845650f5f9abe0280804473a/egenerator/manager/component.py#L380>`_ classes.
These are abstract classes defined in the |egenerator| framework that keep track
of all configured settings.
This enables complete reproducibility and serialization/deserialization of trained models
and ensures their correct usage once trained.
All of this is done in the background, such that the user does not have to keep track of this.


.. _example_cascade:

Example: Training of a Cascade Hypothesis
*****************************************

Here, an example is shown that demonstrates how the |egenerator| framework
can be used to define and train a model for a simple cascade hypothesis
consisting of 7 parameters (x, y, z, zenith, azimuth, energy, time).

The framework is steered by a general yaml-config file.
In this config file, all necessary settings are defined from training data
specifications, to event hypothesis, likelihood function, and neural network
model architecture.
For this example, we will use the config file provided `here <https://github.com/icecube/event-generator/blob/master/configs/cascade_7param_noise_tw_BFRv1Spice321.yaml>`_.


Configuration Settings
----------------------

The configuration file includes many settings. Here, some of the most important
ones are highlighted.

Model Settings
==============

The config file contains a section title "Model Settings" with a dictionary
of the name ``model_settings``. This is arguably the heart of the configuration
file, as it defines the event hypothesis.
This dictionary is defined in a nested way, where each element must at least
define the ``model_class`` and keyword arguments to that python class via the
``config`` parameter. Instances of ``Multi-Source`` classes will also need
to provide the nested models via the key ``multi_source_bases``.


The general structure of the defined event hypothesis in the config file
looks like the following:

.. code-block:: yaml

    # Settings for the neural network model class
    model_settings: {

        # The source class to use
        'model_class': 'egenerator.model.multi_source.independent.IndependentMultiSource',

        # configuration settings for DefaultMultiCascadeModel object
        config: {
            'sources': {
                'noise': 'noise',
                'cascade': 'cascade',
            },
        },

        # Base sources used by the multi-source class
        multi_source_bases: {

            noise: {
                # The noise source model class to use
                'model_class': 'egenerator.model.source.noise.default.DefaultNoiseModel',
                config: {},
            },

            cascade: {
                # The cascade source model class to use
                'model_class': 'egenerator.model.source.cascade.default.DefaultCascadeModel',

                config: {
                    [...]
                },
            },
        },
    }

At the top-most level, this model is defined via the ``IndependentMultiSource`` class,
which is a class that combines multiple sources assuming that they are independent.
In this case, the model is set up with the defined ``sources`` named "noise" and "cascade".
The configuration for these sub-sources are provided in the ``multi_source_bases`` key.
Essentially this means that a 2-component event hypothesis is defined consisting of
a noise model via the class ``DefaultNoiseModel`` and a cascade via the class ``DefaultCascadeModel``.
Training via this configuration file will therefore train these two individual source definitions at the same time.
Note that these are saved and exported in the same nested structure. Therefore, the trained
cascade component can be used individually later on, if desired.

This example demonstrates a simple nested structure of 2 base sources.
In principle, one can add an arbitrary number of layers to this nested structure.
A more complex example that defines a 2-cascade hypothesis with a previously
trained cascade model is given `here <https://github.com/icecube/event-generator/blob/master/configs/starting_multi_cascade_7param_noise_tw_BFRv1Spice321_low_mem_n002.yaml>`_.
By providing the keys ``load_dir`` and ``data_trafo_settings``, a previously trained
model is loaded instead of starting from scratch.

An additional example for a "track" consisting of 10 cascades is provided `here <https://github.com/icecube/event-generator/blob/master/configs/track_equidistant_cascades_n10_w15_7param_noise_tw_BFRv1Spice321_low_mem.yaml>`_.
As of now, the |egenerator| framework does not scale too well to Multi-Sources with many
individual models. A track definition with 100-1000 independent cascade models is currently
not feasible. This might require some optimizations in the software framework.
A workaround may be to directly define more complex sources for cases such as these, instead
of stacking 100 individual models.


Loss module settings
====================

This defines the loss-function (likelihood) to use. For this example config,
the likelihood is defined via two components.
The first component is an unbinned likelihood for the pulse arrival times,
defined in the function ``unbinned_pulse_time_llh`` of the module ``DefaultLossModule``.
The second component is a negative binomial distribution for the total measured charge
via the function ``negative_binomial_charge_pdf``.
The negative binomial is chosen instead of a simple Poisson distribution, such
that over-dispersion from marginalization over systematic parameters can be accounted for.


.. _example_cascade__training_settings:

Training Settings
=================

The key ``training_settings`` defines the training and learning rate decay
strategy.
In this example, the ``ADAM`` optimizer is used with 3 training steps.
The first training step consisting of 10000 optimization steps uses a learning rate
that starts at 0.01 and then decays linearly to 0.001.
Afterwards, 490000 optimization steps are performed with a fixed learning rate
of 0.001.
Finally, 500000 steps are performed with a learning rate that decays from 0.001
down to 0.000001 with a polynomial of second degree.
In total, the model is trained for 1 million optimization steps as defined
in the ``num_training_iterations`` key.
One optimization step is the forward and backward propagation one batch of
training data.

The key ``save_frequency`` defines at which intervals the model is saved to disk.
The |egenerator| framework keeps track of how many training steps are performed
with which configuration file. You can stop and restart the training procedure
at any time. Concurrent training will pick-up from the last saved checkpoint,
unless the setting ``model_manager_settings['restore_model']`` is set to False.
Note however, that the optimizer settings are not saved to disk. Thus, interrupting
and restarting will mean that the learning rate strategy starts from the beginning.
In any case, all configuration and training settings are saved to together with
the model checkpoint, such that it remains reproducible.




Training and Exporting an Model
-------------------------------

Once the configuration file is created, the model can be trained.
This is done in two steps.
First, the data transformation model must be created with the python script `create_trafo_model.py <https://github.com/icecube/event-generator/blob/master/egenerator/create_trafo_model.py>`_ .
This model performs basic transformations to the event hypothesis input parameters
such that these are normalized and easier to use within the network architecture.
Create the transformation model by running the following command:

.. code-block:: bash

    python create_trafo_model.py /PATH/TO/MY/CONFIG/FILE

This step should be fairly quick to run.
A number of batches will be read in from the training data to obtain summary
statistics on the event hypothesis parameters.

Once this steps completes, the training of the neural networks can begin by
executing the following:

.. code-block:: bash

    python train_model.py /PATH/TO/MY/CONFIG/FILE

As noted above in the :ref:`Training Settings<example_cascade__training_settings>`,
this step may be run as many times and with as many different training settings
as desired.


When training is complete, the model can be exported by running the following
command:

.. code-block:: bash

    python export_manager.py /PATH/TO/MY/CONFIG/FILE -o /PATH/TO/OUTPUT/DIR

The exported model can then be distributed and used within the provided
I3TraySegments as described in the section :ref:`Apply Exported Model<apply_model>`.
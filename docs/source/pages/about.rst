.. IceCube Event-Generator Reconstruction

.. _about:

.. note::
   The documentation provided here is currently under construction.
   Please also visit the slack channel #event-generator for further questions.

About the Project
*****************

The |egenerator| project is a software framework that enables the reconstruction of arbitrary IceCube events via a hybrid method composed of Maximum Likelihood Estimation and Deep Learning.
Neural networks (NNs) are used to model the high-dimensional and complex relation between an
event hypothesis and expected light yield in the detector. Once these NNs are trained and
exported, they can be used in a typical maximum likelihood setting via the provided
``I3TraySegments`` (:ref:`Apply Exported Models<apply_model>`).



**Further Material & Publications**

* `ICRC 2021 <https://pos.sissa.it/395/1065>`_

* `Dissertation <http://dx.doi.org/10.17877/DE290R-24043>`_

* `Galactic Plane Science Paper <https://www.science.org/doi/10.1126/science.adc9818>`_

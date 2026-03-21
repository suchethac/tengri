Distributions
=============

Prior distributions for model parameters. Each distribution provides
``sample``, ``log_prob``, and ``transform`` methods compatible with JAX
tracing.

Uniform
-------

.. autoclass:: tengri.Uniform
   :members:
   :show-inheritance:

Gaussian
--------

.. autoclass:: tengri.Gaussian
   :members:
   :show-inheritance:

LogUniform
----------

.. autoclass:: tengri.LogUniform
   :members:
   :show-inheritance:

LogNormal
---------

.. autoclass:: tengri.LogNormal
   :members:
   :show-inheritance:

StudentT
--------

.. autoclass:: tengri.StudentT
   :members:
   :show-inheritance:

Fixed
-----

.. autoclass:: tengri.Fixed
   :members:
   :show-inheritance:

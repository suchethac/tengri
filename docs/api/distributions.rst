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

Log uniform
-----------

.. autoclass:: tengri.LogUniform
   :members:
   :show-inheritance:

Log normal
----------

.. autoclass:: tengri.LogNormal
   :members:
   :show-inheritance:

Student t
---------

.. autoclass:: tengri.StudentT
   :members:
   :show-inheritance:

Fixed
-----

.. autoclass:: tengri.Fixed
   :members:
   :show-inheritance:

Laplace
-------

.. autoclass:: tengri.Laplace
   :members:
   :show-inheritance:

Sentinels
---------

``FREE`` and ``FIXED`` are not distributions. They are singleton markers used
in the nested-dict grammar of :meth:`tengri.SEDModel.build` to say *"take the
registry's default"* — a default prior for ``FREE``, a default value for
``FIXED``.

Note the deliberate spelling difference from :class:`tengri.Fixed` above:
``Fixed(0.05)`` is a distribution that pins a parameter to the value **you**
give it, whereas ``FIXED`` pins it to the value the **registry** already
carries. They are different objects and are not interchangeable.

.. autodata:: tengri.parameters.sentinels.FREE
   :annotation:
   :noindex:

.. autodata:: tengri.parameters.sentinels.FIXED
   :annotation:
   :noindex:

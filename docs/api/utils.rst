Utilities
=========

Grid construction and helper functions used throughout the pipeline.

The Gaussian-process generators (``compute_field_gp``,
``generate_gp_fourier``, ``generate_gp_batch``, ``gp_from_xi``) are
documented in :doc:`models`, alongside the PSD parameterizations they
consume.

make_log_age_grid
-----------------

.. autofunction:: tengri.make_log_age_grid

Noise Utilities
---------------

.. autofunction:: tengri.compute_effective_noise

.. autofunction:: tengri.compute_std_inv

.. autofunction:: tengri.has_noise_model

Parameter Introspection
-----------------------

Read-only walker over every declared free parameter — per-component
``_params.py`` modules plus the shared ``_NON_SFH_PARAMS`` bucket.
See ``docs/adr/0008-parameter-registry-introspection.md`` for the
design rationale.

list_parameters
~~~~~~~~~~~~~~~

.. autofunction:: tengri.list_parameters

describe_parameter
~~~~~~~~~~~~~~~~~~

.. autofunction:: tengri.describe_parameter

recipe_parameters
~~~~~~~~~~~~~~~~~

.. autofunction:: tengri.recipe_parameters

ParameterRecord
~~~~~~~~~~~~~~~

.. autoclass:: tengri.ParameterRecord
   :members:
   :show-inheritance:

Utilities
=========

Grid construction, GP generation, and helper functions used throughout
the pipeline.

make_log_age_grid
-----------------

.. autofunction:: tengri.make_log_age_grid

compute_field_gp
----------------

.. autofunction:: tengri.compute_field_gp

generate_gp_fourier
-------------------

.. autofunction:: tengri.generate_gp_fourier

generate_gp_batch
-----------------

.. autofunction:: tengri.generate_gp_batch

gp_from_xi
-----------

.. autofunction:: tengri.gp_from_xi

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

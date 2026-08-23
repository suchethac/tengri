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

Noise utilities
---------------

.. autofunction:: tengri.compute_effective_noise

.. autofunction:: tengri.compute_std_inv

.. autofunction:: tengri.has_noise_model

Correlated noise
~~~~~~~~~~~~~~~~

Spectroscopic residuals are rarely white. These build a full covariance as
white noise plus a Gaussian-process kernel, for use with a
:class:`~tengri.NoiseModel` that models correlated errors.

.. autofunction:: tengri.gp_noise_covariance

.. autofunction:: tengri.exp_squared_kernel

.. autofunction:: tengri.matern32_kernel

Bundled data and caches
-----------------------

tengri persists compiled JAX kernels and precomputed photometry tables
between sessions, and resolves bundled data files by walking parent
directories for a ``data/`` folder.

.. autofunction:: tengri.data_path

.. autofunction:: tengri.clear_cache

Batching
--------

.. autofunction:: tengri.vmap_chunked

Parameter introspection
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

Physical constants
------------------

Fundamental physical constants in CGS units.

.. autodata:: tengri.units.C_AA
   :no-value:

.. autodata:: tengri.units.LOG10_ZSUN
   :no-value:

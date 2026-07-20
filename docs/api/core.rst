Core API
========

The core classes that form tengri's high-level interface: defining models,
specifying parameters, generating predictions, and creating mock data.

``ForwardModel`` is the outer shell and the surface inference consumes; the
``SEDModel`` you build is what goes inside it. Build the SED chain, wrap it,
fit it.

ForwardModel
------------

.. autoclass:: tengri.ForwardModel
   :members:
   :show-inheritance:
   :exclude-members: predict_rest_sed, predict_magnitudes, predict_derived,
      predict_sfh_quantities, predict_photometry_components,
      predict_spectrum_components

Population
----------

.. autoclass:: tengri.Population
   :members:
   :show-inheritance:

SEDModel
--------

The ``predict_*`` methods that once returned individual observables are
superseded by the single cached :meth:`~tengri.SEDModel.predict` pass — see
:doc:`predicting-properties`. They still work and still warn, but they are
omitted here rather than presented as current API; the migration table in
``docs/dev/api_migration_v0.x.md`` maps each one to its replacement.

.. autoclass:: tengri.SEDModel
   :members:
   :show-inheritance:
   :exclude-members: predict_rest_sed, predict_obs_sed, predict_magnitudes,
      predict_luminosity, predict_hbeta, predict_derived,
      predict_sfh_quantities, predict_sed_quantities,
      predict_sfh_quantities_components, predict_sed_quantities_components,
      predict_radio_quantities, predict_xray_quantities,
      predict_ionizing_quantities, predict_photometry_components,
      predict_spectrum_components, predict_emission_lines, fit

Parameters
----------

.. autoclass:: tengri.Parameters
   :members:
   :show-inheritance:

Prediction
----------

.. autoclass:: tengri.Prediction
   :members:
   :show-inheritance:

SFHQuantities
-------------

.. autoclass:: tengri.SFHQuantities
   :members:
   :show-inheritance:

SEDQuantities
-------------

.. autoclass:: tengri.SEDQuantities
   :members:
   :show-inheritance:

DerivedQuantities
-----------------

.. autoclass:: tengri.DerivedQuantities
   :members:
   :show-inheritance:

EmissionLines
-------------

.. autoclass:: tengri.EmissionLines
   :members:
   :show-inheritance:

MockData
--------

.. autoclass:: tengri.MockData
   :members:
   :show-inheritance:

generate_mock
-------------

.. autofunction:: tengri.generate_mock

Observation
-----------

.. autoclass:: tengri.Observation
   :members:
   :show-inheritance:

Photometry
----------

.. autoclass:: tengri.Photometry
   :members:
   :show-inheritance:

Spectroscopy
-------------

.. autoclass:: tengri.Spectroscopy
   :members:
   :show-inheritance:

NoiseModel
----------

.. autoclass:: tengri.NoiseModel
   :members:
   :show-inheritance:

VIConfig
--------

.. autoclass:: tengri.VIConfig
   :members:
   :show-inheritance:

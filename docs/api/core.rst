Core API
========

The core classes that form tengri's high-level interface: defining models,
specifying parameters, generating predictions, and creating mock data.

``ForwardModel`` is the outer shell and the surface inference consumes; the
``SEDModel`` you build is what goes inside it. Build the SED chain, wrap it,
fit it.

Forward model
-------------

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

SED model
---------

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

Approximation policies
----------------------

The ``approx=`` argument to :meth:`~tengri.SEDModel.build` selects a
build-time lookup table in place of the exact per-evaluation forward. Each
targets a different output channel and they compose: ``WavePrecomp`` serves
photometry, ``SpectrumPrecomp`` the spectrum, ``FeaturePrecomp`` the emission
lines. ``approx=None`` (the default) keeps the exact wave-grid path.

Wave precomp
~~~~~~~~~~~~

.. autoclass:: tengri.WavePrecomp
   :members:
   :show-inheritance:

Spectrum precomp
~~~~~~~~~~~~~~~~

.. autoclass:: tengri.SpectrumPrecomp
   :members:
   :show-inheritance:

Feature precomp
~~~~~~~~~~~~~~~

.. autoclass:: tengri.FeaturePrecomp
   :members:
   :show-inheritance:

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

Prior predictive
----------------

.. autoclass:: tengri.PriorPredictive
   :members:
   :show-inheritance:

SED result
----------

.. autoclass:: tengri.SEDResult
   :members:
   :show-inheritance:

SFH quantities
--------------

.. autoclass:: tengri.SFHQuantities
   :members:
   :show-inheritance:

SED quantities
--------------

.. autoclass:: tengri.SEDQuantities
   :members:
   :show-inheritance:

Derived quantities
------------------

.. autoclass:: tengri.DerivedQuantities
   :members:
   :show-inheritance:

Emission lines
--------------

.. autoclass:: tengri.EmissionLines
   :members:
   :show-inheritance:

Mock data
---------

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

Instrument
----------

A named bundle of (photometry, spectroscopy, noise) defaults, so a
notebook can say ``Instrument.JWST_NIRCam()`` instead of assembling
those pieces by hand. :func:`~tengri.list_instruments` is the menu.

.. autoclass:: tengri.Instrument
   :members:
   :show-inheritance:

Data
----

One galaxy's measurements, validated against the :class:`~tengri.Observation`
that describes them.

.. autoclass:: tengri.Data
   :members:
   :show-inheritance:

Galaxy
------

.. autoclass:: tengri.Galaxy
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

Line list
---------

.. autoclass:: tengri.LineList
   :members:
   :show-inheritance:

Spectral indices
----------------

A :class:`~tengri.SpectralIndexDef` defines one index;
:class:`~tengri.CompositeIndexDef` combines atomic ones; and
:class:`~tengri.SpectralIndexData` carries the observed values to fit against.

.. autoclass:: tengri.SpectralIndexDef
   :members:
   :show-inheritance:

.. autoclass:: tengri.CompositeIndexDef
   :members:
   :show-inheritance:

.. autoclass:: tengri.SpectralIndexData
   :members:
   :show-inheritance:

Two catalogs ship with tengri, keyed by index name, so most work needs no
hand-built definition — look the name up rather than redefining passbands.

.. autodata:: tengri.observation.spectral_indices.STANDARD_INDICES
   :annotation:
   :noindex:

.. autodata:: tengri.observation.spectral_indices.STANDARD_COMPOSITE_INDICES
   :annotation:
   :noindex:

:func:`tengri.measure.spectral_index` is the exploratory surface;
``measure_index_jax`` is the JIT/vmap-safe one used inside inference. Both
take a **rest-frame** wavelength axis.

.. autofunction:: tengri.measure_index_jax

Filter convention
-----------------

.. autoclass:: tengri.FilterConvention
   :members:
   :show-inheritance:

Noise model
-----------

.. autoclass:: tengri.NoiseModel
   :members:
   :show-inheritance:

VI config
---------

.. autoclass:: tengri.VIConfig
   :members:
   :show-inheritance:

Exceptions
----------

Every exception tengri raises derives from :class:`~tengri.TengriError`, so
``except TengriError`` catches all of them. Each also inherits the built-in
that matches its failure mode — :class:`~tengri.ParameterError` is a
``ValueError``, :class:`~tengri.TengriIOError` an ``OSError`` — so existing
``except ValueError`` handlers keep working.

.. autoexception:: tengri.TengriError
   :members:
   :show-inheritance:

.. autoexception:: tengri.ParameterError
   :members:
   :show-inheritance:

.. autoexception:: tengri.ConfigError
   :members:
   :show-inheritance:

.. autoexception:: tengri.BackendError
   :members:
   :show-inheritance:

.. autoexception:: tengri.InferenceError
   :members:
   :show-inheritance:

.. autoexception:: tengri.TengriIOError
   :members:
   :show-inheritance:

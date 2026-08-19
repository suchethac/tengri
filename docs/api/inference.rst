Inference
=========

Fitting models to data, sampling posteriors, and running hierarchical
inference across galaxy populations.

Inference always goes through a :class:`~tengri.ForwardModel`. The entry point
is :meth:`~tengri.ForwardModel.fit`::

    forward = ForwardModel.build(sed=sed, observation=obs)
    result = forward.fit(flux, noise, method="mcmc_nuts")

:class:`~tengri.Fitter` is the engine underneath. ``forward.fit(...)`` is
exactly ``Fitter(forward, flux, noise).run(...)`` and forwards every extra
keyword to :meth:`~tengri.Fitter.run`, so the shortcut is not a reduced
surface. Reach for ``Fitter`` directly only to inspect engine state such as
``compile_signature()``; holding the object is *not* what buys you compile
reuse, because the compilation caches are keyed on that signature and shared
across instances. For many galaxies, use :class:`~tengri.Catalog`.

Pass a :class:`~tengri.ForwardModel`, not a bare :class:`~tengri.SEDModel`.
``Fitter(sed_model, ...)`` still runs but is deprecated; wrap the SED chain
with ``ForwardModel.build(sed=..., observation=...)`` first.

Posterior
---------

.. autoclass:: tengri.Posterior
   :members:
   :show-inheritance:

Fitter
------

.. autoclass:: tengri.Fitter
   :members:
   :show-inheritance:

Catalog
-------

For many galaxies sharing one forward model, :class:`~tengri.Catalog` is the
table-in/table-out surface. It validates columns, units and the redshift
mechanism at construction — before any compile — then exposes ``.fit()`` and
``.predict()``::

    cat = Catalog(forward, table, flux_unit="cgs_fnu", redshift_col="z")
    result = cat.fit(method="mcmc_hmc", key=jax.random.PRNGKey(0),
                     forward_chunk_size=32)

``key`` is required, not optional — every catalog fit is explicitly seeded.

.. autoclass:: tengri.Catalog
   :members:
   :show-inheritance:

``fit_batch`` is the lower-level alternative: it walks the rows one at a time
without :class:`~tengri.Catalog`'s column validation, which is useful when the
input is already in memory as arrays rather than a table.

.. autofunction:: tengri.fit_batch

Bayesian model averaging
------------------------

Combine fits of *different models* to the same data, weighting each by its
marginal likelihood. Every posterior must carry ``Posterior.log_evidence`` —
fit with an evidence-returning method (``"nss"``, ``"laplace"``, or
``"hmc_is"``).

.. autofunction:: tengri.bma_weights

.. autofunction:: tengri.bma_resample

Hierarchical inference
----------------------

For a population sharing hyperparameters, build a
:class:`~tengri.PopulationSEDModel` and pass it to ``ForwardModel.build`` as
the ``population`` slot — the same engine drives it::

    template = SEDModel.build(...)
    pop = PopulationSEDModel(
        sed=template,
        galaxies=galaxies,
        shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
    )
    forward = ForwardModel.build(population=pop, observation=obs)
    result = forward.fit(method="vi")

Constructing :class:`~tengri.PopulationFitter` directly is deprecated and will
be removed in tengri v1.0 (issue #211); the class remains as the routed
implementation behind the pattern above.

PopulationSEDModel
~~~~~~~~~~~~~~~~~~

.. autoclass:: tengri.PopulationSEDModel
   :members:
   :show-inheritance:

PopulationFitter
~~~~~~~~~~~~~~~~

.. autoclass:: tengri.PopulationFitter
   :members:
   :show-inheritance:

PopulationPosterior
~~~~~~~~~~~~~~~~~~~

.. autoclass:: tengri.PopulationPosterior
   :members:
   :show-inheritance:

Parameter information
---------------------

How much of each posterior mode the data determined, as opposed to the
prior — the measured-versus-prior decomposition of a fit.

.. autofunction:: tengri.parameter_information

.. autoclass:: tengri.ParameterInformation
   :members:

sample_raytrace
---------------

.. autofunction:: tengri.sample_raytrace

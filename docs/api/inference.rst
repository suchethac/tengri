Inference
=========

Fitting models to data, sampling posteriors, and running hierarchical
inference across galaxy populations.

Inference always goes through a :class:`~tengri.ForwardModel`. The entry point
is :meth:`~tengri.ForwardModel.fit`::

    forward = ForwardModel.build(sed=sed, observation=obs)
    result = forward.fit(flux, noise, method="mcmc_nuts")

:class:`~tengri.Fitter` is the engine underneath. ``forward.fit(...)`` is
exactly ``Fitter(forward, flux, noise).run(...)``, so reach for ``Fitter``
directly when you want to keep the object around — to reuse its compilation
cache across a catalog, to inspect ``compile_signature()``, or to drive a
backend with options the shortcut does not expose.

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

Hierarchical inference
----------------------

For a population sharing hyperparameters, build a
:class:`~tengri.PopulationSEDModel` and pass it to ``ForwardModel.build`` as
the ``population`` slot — the same ``Fitter`` drives it::

    template = SEDModel.build(...)
    pop = PopulationSEDModel(
        sed=template,
        galaxies=galaxies,
        shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
    )
    forward = ForwardModel.build(population=pop, observation=obs)
    result = Fitter(forward).run("vi")

Constructing :class:`~tengri.PopulationFitter` directly is deprecated and will
be removed in tengri v1.0 (issue #211); the class remains as the routed
implementation behind the pattern above.

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

sample_raytrace
---------------

.. autofunction:: tengri.sample_raytrace

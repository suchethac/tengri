Discovery and citations
=======================

The introspection surface: browse every model, dust law, SFH variant,
nebular backend, AGN block, inference method, filter, and recipe that
tengri ships; look one up by name; search across all of them; and emit
the exact citations a built model uses. All of these are importable
directly from ``tengri`` (e.g. ``tengri.describe("skirtor")``).

A worked, run-it-yourself tour lives in the
:doc:`discovery notebook </spine/03_discovering_the_menu>`; the citation
verbs are covered narratively on the :doc:`Citing tengri </citation>`
page.

Getting oriented
----------------

Start here on a fresh install: a one-screen overview, an interactive
help panel, and an install/environment health check.

.. autofunction:: tengri.summary

.. autofunction:: tengri.help

.. autofunction:: tengri.doctor

.. autofunction:: tengri.explain

.. autofunction:: tengri.tutorial

.. autofunction:: tengri.examples

.. autofunction:: tengri.print_logo

Look up and search
------------------

``describe`` resolves any name across every menu (and discloses when a
name — e.g. ``skirtor``, both a disc and a torus — is registered in more
than one place). ``search`` matches by name, short description, or
citation across all menus, and routes common concept terms (``"star
formation"``, ``"dust emission"``) to the menu that holds those models.

.. autofunction:: tengri.describe

.. autofunction:: tengri.search

.. autofunction:: tengri.suggest_parameters

Menus
-----

Each ``list_*`` returns a table with a ``name`` column (feed it straight
to :meth:`~tengri.SEDModel.build`), a ``status`` column, the citation,
and a one-line description.

.. autofunction:: tengri.list_all

.. autofunction:: tengri.list_sfh_models

.. autofunction:: tengri.list_age_kernels

.. autofunction:: tengri.list_dust_models

.. autofunction:: tengri.list_dust_laws

.. autofunction:: tengri.list_dust_emission_models

.. autofunction:: tengri.list_nebular_backends

.. autofunction:: tengri.list_agn_models

.. autofunction:: tengri.list_agn_blocks

.. autofunction:: tengri.list_xray_models

.. autofunction:: tengri.list_radio_models

.. autofunction:: tengri.list_radio_blocks

.. autofunction:: tengri.list_shock_models

.. autofunction:: tengri.list_metallicity_modes

.. autofunction:: tengri.list_igm_models

.. autofunction:: tengri.list_inference_methods

.. autofunction:: tengri.list_recipes

.. autofunction:: tengri.list_filters

.. autofunction:: tengri.list_registered_filters

.. autofunction:: tengri.list_synthetic_bands

.. autofunction:: tengri.list_instruments

.. autofunction:: tengri.list_components

.. autofunction:: tengri.list_plots

.. autofunction:: tengri.list_properties

.. autofunction:: tengri.list_filter_conventions

The SSP grids are listed by :func:`~tengri.list_known_ssps` and
:func:`~tengri.list_available_ssps`, documented with the rest of the
stellar population synthesis surface on :doc:`models`.

Per-category describe
---------------------

The universal :func:`~tengri.describe` covers most lookups; these return
the richer, category-specific record (and ``describe_agn_block`` takes a
``category`` because AGN block names are not unique across categories).

.. autofunction:: tengri.describe_sfh_model

.. autofunction:: tengri.describe_dust_law

.. autofunction:: tengri.describe_dust_emission_model

.. autofunction:: tengri.describe_nebular_backend

.. autofunction:: tengri.describe_agn_model

.. autofunction:: tengri.describe_agn_block

.. autofunction:: tengri.describe_inference_method

.. autofunction:: tengri.describe_recipe

.. autofunction:: tengri.describe_property

Custom filter loading
---------------------

In-memory registration, file-based directory loading, and DSPS integration
for user-provided transmission curves.

Curve files may be whitespace- or comma-separated, with or without a header
row; ``#`` comments are stripped. Wavelengths are Angstrom unless you say
otherwise with ``wave_unit="nm"`` or ``"um"``.

State the unit whenever the file is not already in Angstrom. A curve given in
nanometers is a valid array of numbers describing the extreme UV, so it cannot
always be detected: registration falls back to a range heuristic that warns
when a curve lies wholly inside the 100-1340 Angstrom gap where the ISM is
opaque and no bandpass exists. That backstop cannot see micron input at all,
and cannot catch a nanometer set running past 1340 nm without also firing on
GALEX FUV.

The ``$TENGRI_FILTER_DIR`` route has nowhere to record a unit, so files placed
there must already be in Angstrom.

Worked end to end, including an ADU-and-zeropoint table taken to a fit, in
``notebooks/custom_filters_7dt.py``.

.. autofunction:: tengri.register_filter

.. autofunction:: tengri.register_filter_from_file

.. autofunction:: tengri.unregister_filter

.. autofunction:: tengri.load_custom_filter

.. autofunction:: tengri.load_tophat_filter

.. autofunction:: tengri.load_filter_from_dsps_transmission_curve

.. autofunction:: tengri.load_filter_from_dsps_file

.. autofunction:: tengri.load_alma_band

Citations
---------

Component-level citation: a built model, ``Parameters`` spec, or
``Posterior`` carries the citation for every SSP grid, physics block,
and inference backend it uses. Composable AGN models fan out into each
of their block slots (disc, torus, NLR, BLR, Fe II, attenuation).

.. autofunction:: tengri.cite

.. autofunction:: tengri.print_citations

.. autofunction:: tengri.cite_components

.. autofunction:: tengri.print_components_bibtex

.. autofunction:: tengri.cite_all

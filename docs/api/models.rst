Physics models
==============

The building blocks of tengri's forward model: star formation history
parameterizations, power spectral density functions, dust attenuation,
stellar population synthesis, and filter handling.

Star formation history
----------------------

Analytic mean-SFH functional forms. Each takes a log-age grid and shape
parameters, returning SFR in solar masses per year.

.. autofunction:: tengri.tsnorm

.. autofunction:: tengri.dpl

.. autofunction:: tengri.double_powerlaw

.. autofunction:: tengri.snorm

.. autofunction:: tengri.norm

.. autofunction:: tengri.lnorm

.. autofunction:: tengri.delayed_tau

.. autofunction:: tengri.constant

.. autofunction:: tengri.exponential

.. autofunction:: tengri.delayed_exponential

.. autofunction:: tengri.triweight_burst

SFH registry
~~~~~~~~~~~~

.. autofunction:: tengri.resolve_sfh

.. Both are re-exported as ``tengri.SFH_REGISTRY`` / ``tengri.FIELD_MODEL_REGISTRY``.
   Document them at the defining module: autodoc resolves a ``#:`` doc-comment
   against the module named in the directive, and ``tengri/__init__.py`` only
   imports them. Pointed at ``tengri`` these rendered Python's builtin
   ``dict.__doc__`` ("dict() -> new empty dictionary ...") instead.

.. autodata:: tengri.components.stellar.sfh.registry.SFH_REGISTRY

.. autodata:: tengri.components.stellar.sfh.registry.FIELD_MODEL_REGISTRY

Power spectral density
----------------------

DRW (damped random walk) PSD models that govern the burstiness of star
formation histories via Gaussian process priors.

.. autofunction:: tengri.psd_drw

.. autofunction:: tengri.drw_acf

.. autofunction:: tengri.drw_variance

.. autofunction:: tengri.compute_sqrt_power_drw

GP generation
--------------

Functions for generating Gaussian process realizations from PSD parameters
and a latent vector.

.. autofunction:: tengri.compute_field_gp

.. autofunction:: tengri.generate_gp_fourier

.. autofunction:: tengri.generate_gp_batch

.. autofunction:: tengri.gp_from_xi

Dust attenuation
-----------------

Two-component dust attenuation (birth cloud + diffuse ISM) following
Charlot & Fall (2000), with pluggable attenuation curves and geometry models.

.. autofunction:: tengri.two_component_dust

Attenuation curves: ``calzetti``, ``cardelli``, ``kriek_conroy``, ``smc``,
``lmc``, ``power_law``, ``salim``, ``li08``.

.. automodule:: tengri.components.dust.attenuation
   :members: calzetti, cardelli, kriek_conroy, smc, lmc, power_law, wg00_shell, wg00_cloudy, wg00_dusty
   :noindex:

Dust emission
~~~~~~~~~~~~~

Energy-balanced IR dust emission models.

.. automodule:: tengri.components.dust.emission
   :members: modified_blackbody, casey2012
   :noindex:

The grain-physics templates behind the library-backed emission models are
loaded from bundled data files.

.. autofunction:: tengri.load_astrodust_hd23

.. autofunction:: tengri.load_pahspec_draine2021

Dust priors
~~~~~~~~~~~

Redshift-dependent attenuation priors from cosmological simulations.

.. automodule:: tengri.components.dust.priors
   :members: narayanan_prior, narayanan_tau_prior
   :noindex:

AGN
---

Accretion disc, torus, BLR, NLR, and unified AGN models.

.. automodule:: tengri.components.agn.disc
   :members: powerlaw_disc, multicolor_disc, kubota_done_disc
   :noindex:

.. automodule:: tengri.components.agn.blr
   :members: blr_emission
   :noindex:

.. automodule:: tengri.components.agn.nlr
   :members: nlr_emission
   :noindex:

.. automodule:: tengri.components.agn.unified
   :members: unified_agn, multicolor_agn, kubota_done_full_agn, unified_nlr_blr
   :noindex:

Nebular emission
----------------

Nebular line and continuum emission from CLOUDY grids or the Cue neural
network emulator, with optional shock and DIG mixing.

.. automodule:: tengri.components.nebular.shock
   :members: shock_line_ratios, shock_emission_sed
   :noindex:

.. automodule:: tengri.components.nebular.dig
   :members: mix_dig_emission
   :noindex:

Observation models
------------------

Photometry, spectroscopy, calibration, and emission line marginalization.

.. automodule:: tengri.observation.calibration
   :members: calibration_polynomial, marginalize_calibration
   :noindex:

Instrumental broadening — the intrinsic stellar velocity dispersion and the
spectrograph's own line-spread function — is applied to a model spectrum
before it is compared with data.

.. autofunction:: tengri.velocity_broaden

.. autofunction:: tengri.apply_lsf

Radio
------

Star-formation synchrotron and AGN jets. Radio models activate via nested
dictionaries separating star-forming and AGN components (e.g.
``SEDModel.build(..., radio={'sf': {'type': 'bell2003'}, 'agn': {'type': 'powerlaw'}})``)
or compact single-model activation (e.g. ``{'type': 'condon92'}`` combining both).
Discover available models and radio-block options via:

.. autofunction:: tengri.list_radio_models
   :noindex:

.. autofunction:: tengri.list_radio_blocks
   :noindex:

Radio SED models are configured via SEDComponent classes with configurable
parameters for synchrotron spectral index, AGN loudness, and star-formation
efficiency.

.. autoclass:: tengri.components.radio.RadioDPL
   :members:
   :show-inheritance:

.. autoclass:: tengri.components.radio.RadioPowerLawSEDComponent
   :members:
   :show-inheritance:

X-ray
------

X-ray emission from accretion-powered AGN coronae and stellar binary systems
(both high-mass and low-mass). X-ray models activate via dictionary
(e.g. ``SEDModel.build(..., xray={'type': 'simple'})`` or
``xray={'type': 'lopez24'}`` for IR-selected AGN).
Discover available models via:

.. autofunction:: tengri.list_xray_models
   :noindex:

X-ray SED models include AGN corona normalizations tied to either UV-optical
disc emission or mid-infrared dust luminosity, and XRB contributions scaled
by galaxy stellar mass.

.. autoclass:: tengri.components.xray.AGNXRayCoronaSEDComponent
   :members:
   :show-inheritance:

.. autoclass:: tengri.components.xray.XRayAirdSEDComponent
   :members:
   :show-inheritance:

Intergalactic medium
--------------------

Mean IGM attenuation blueward of Lyman-alpha.

.. warning::

   Both functions take **observed-frame** wavelengths, not rest-frame ones.
   Passing a rest-frame grid silently returns the wrong transmission rather
   than raising.

.. autofunction:: tengri.igm_transmission

.. autofunction:: tengri.igm_transmission_madau

.. autofunction:: tengri.igm_transmission_meiksin06

.. automodule:: tengri.components.igm
   :members: igm_transmission_asada25
   :noindex:

Stellar population synthesis
----------------------------

DSPS-based stellar population synthesis: loading SSP grids and computing
effective metallicities.

.. autoclass:: tengri.SSPData
   :members:
   :show-inheritance:

.. autofunction:: tengri.load_ssp_data

.. autofunction:: tengri.effective_metallicity

.. autofunction:: tengri.compute_mass_remaining_fraction

Acquiring an SSP grid
~~~~~~~~~~~~~~~~~~~~~

tengri does not bundle SSP grids; they are downloaded on first use. The usual
sequence is to see what exists, fetch one, then load it.

.. autofunction:: tengri.list_known_ssps

.. autofunction:: tengri.list_available_ssps

.. autofunction:: tengri.download_ssp

.. autofunction:: tengri.load_ssp

Filters
-------

Loading and managing photometric filter transmission curves.

.. autofunction:: tengri.load_filter_set

Spatial profiles
----------------

Surface-brightness profiles and the sub-models that compose them. A
:class:`~tengri.SpatialModel` runs a list of profile components;
:class:`~tengri.SpatialSEDModel` joins a SED chain to a spatial chain so both
are fitted together.

.. autoclass:: tengri.SpatialModel
   :members:
   :show-inheritance:

.. autoclass:: tengri.SpatialSEDModel
   :members:
   :show-inheritance:

.. autoclass:: tengri.Sersic
   :members:
   :show-inheritance:

.. autoclass:: tengri.Exponential
   :members:
   :show-inheritance:

.. autoclass:: tengri.FlatSlab
   :members:
   :show-inheritance:

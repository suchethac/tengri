Physics Models
==============

The building blocks of tengri's forward model: star formation history
parameterisations, power spectral density functions, dust attenuation,
stellar population synthesis, and filter handling.

Star Formation History
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

.. autofunction:: tengri.constant_sfh

.. autofunction:: tengri.exponential_sfh

.. autofunction:: tengri.delayed_exponential_sfh

.. autofunction:: tengri.triweight_burst

SFH Registry
~~~~~~~~~~~~~

.. autofunction:: tengri.resolve_sfh

.. autodata:: tengri.SFH_REGISTRY

.. autodata:: tengri.FIELD_MODEL_REGISTRY

Power Spectral Density
----------------------

DRW (damped random walk) PSD models that govern the burstiness of star
formation histories via Gaussian process priors.

.. autofunction:: tengri.psd_drw

.. autofunction:: tengri.drw_acf

.. autofunction:: tengri.drw_variance

.. autofunction:: tengri.compute_sqrt_power_drw

GP Generation
-------------

Functions for generating Gaussian process realisations from PSD parameters
and a latent vector.

.. autofunction:: tengri.compute_field_gp

.. autofunction:: tengri.generate_gp_fourier

.. autofunction:: tengri.generate_gp_batch

.. autofunction:: tengri.gp_from_xi

.. autofunction:: tengri.make_log_age_grid

Dust Attenuation
-----------------

Two-component dust attenuation (birth cloud + diffuse ISM) following
Charlot & Fall (2000), with pluggable attenuation curves and geometry models.

.. autofunction:: tengri.two_component_dust

Attenuation curves: ``calzetti``, ``cardelli``, ``kriek_conroy``, ``smc``,
``lmc``, ``power_law``, ``salim``, ``li08``.

.. automodule:: tengri.models.dust.attenuation
   :members: calzetti, cardelli, kriek_conroy, smc, lmc, power_law, wg00_shell, wg00_cloudy, wg00_dusty
   :noindex:

Dust Emission
~~~~~~~~~~~~~

Energy-balanced IR dust emission models.

.. automodule:: tengri.models.dust.emission
   :members: modified_blackbody, casey2012
   :noindex:

Dust Priors
~~~~~~~~~~~

Redshift-dependent attenuation priors from cosmological simulations.

.. automodule:: tengri.models.dust.priors
   :members: narayanan_prior, narayanan_tau_prior
   :noindex:

AGN
---

Accretion disc, torus, BLR, NLR, and unified AGN models.

.. automodule:: tengri.models.agn.disc
   :members: powerlaw_disc, multicolor_disc, kubota_done_disc
   :noindex:

.. automodule:: tengri.models.agn.blr
   :members: blr_emission
   :noindex:

.. automodule:: tengri.models.agn.nlr
   :members: nlr_emission
   :noindex:

.. automodule:: tengri.models.agn.unified
   :members: unified_agn, multicolor_agn, kubota_done_full_agn, unified_nlr_blr
   :noindex:

Nebular Emission
----------------

Nebular line and continuum emission from CLOUDY grids or the Cue neural
network emulator, with optional shock and DIG mixing.

.. automodule:: tengri.models.nebular.shock
   :members: shock_line_ratios, shock_emission_sed
   :noindex:

.. automodule:: tengri.models.nebular.dig
   :members: mix_dig_emission
   :noindex:

Observation Models
------------------

Photometry, spectroscopy, calibration, and emission line marginalization.

.. automodule:: tengri.models.observation.calibration
   :members: calibration_polynomial, marginalize_calibration
   :noindex:

Stellar Population Synthesis
----------------------------

DSPS-based stellar population synthesis: loading SSP grids and computing
effective metallicities.

.. autoclass:: tengri.SSPData
   :members:
   :show-inheritance:

.. autofunction:: tengri.load_ssp_data

.. autofunction:: tengri.effective_metallicity

Filters
-------

Loading and managing photometric filter transmission curves.

.. autofunction:: tengri.load_filter_set

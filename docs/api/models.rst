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

Dust
----

Two-component dust attenuation (birth cloud + diffuse ISM) following
Charlot & Fall (2000).

.. autofunction:: tengri.two_component_dust

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

"""Dust attenuation and emission models.

Attenuation
-----------
``attenuation.py`` — generalized two-component model with pluggable curves
and f_obscuration.

Emission
--------
``emission.py`` — IR re-emission models (modified blackbody, Casey 2012,
Dale 2014, Draine & Li 2007, Draine & Li 2014 update, Astrodust+PAH,
BOSA, THEMIS) with energy-balance normalization.

Priors
------
``priors.py`` — redshift-dependent dust attenuation priors from
Narayanan+2018 cosmological RT simulations.
"""

from tengri.models.dust.attenuation import (
    DUST_LAWS,
    get_dust_law,
    precompute_dust_age_mask,
    precompute_dust_age_weights,
    register_dust_law,
    resolve_dust_law,
    single_component_dust,
    single_component_dust_fast,
    two_component_dust,
    two_component_dust_fast,
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)
from tengri.models.dust.drude_profiles import (
    N_PAH_FEATURES,
    SMITH2007_PAH_FEATURES,
    decompose_pah,
    drude_profile,
    pah_template,
)
from tengri.models.dust.emission import (
    DUST_EMISSION_MODELS,
    apply_dust_emission,
    astrodust,
    bosa,
    casey2012,
    cmb_contrast_factor,
    cmb_corrected_temperature,
    compute_absorbed_luminosity,
    compute_absorbed_luminosity_from_tau,
    create_astrodust_from_grid,
    create_bosa_from_grid,
    create_dale2014_from_grid,
    create_dl07_from_grid,
    create_dl14_from_grid,
    create_themis_from_grid,
    energy_balance_split,
    get_emission_model,
    load_astrodust_templates,
    load_bosa_templates,
    load_dl14_templates,
    load_draine_li_templates,
    load_themis_templates,
    magphys_dc08,
    planck_bnu,
    register_astrodust_tabulated,
    register_bosa_tabulated,
    register_dale2014_tabulated,
    register_dl07_tabulated,
    register_dl14_tabulated,
    register_emission_model,
    register_themis_tabulated,
    resolve_emission_model,
    themis,
)
from tengri.models.dust.priors import (
    narayanan_prior,
    narayanan_tau_prior,
)

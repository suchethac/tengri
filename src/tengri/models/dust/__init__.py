"""Dust attenuation and emission models.

Attenuation
-----------
``attenuation.py`` — generalized two-component model with pluggable curves
and f_obscuration.

Emission
--------
``emission.py`` — IR re-emission models (modified blackbody, Casey 2012,
Dale 2014, Draine & Li 2007, Draine & Li 2014 update) with energy-balance
normalization.
"""

from tengri.models.dust.attenuation import (
    DUST_LAWS,
    get_dust_law,
    precompute_dust_age_mask,
    precompute_dust_age_weights,
    register_dust_law,
    single_component_dust,
    single_component_dust_fast,
    two_component_dust,
    two_component_dust_fast,
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)
from tengri.models.dust.emission import (
    DUST_EMISSION_MODELS,
    apply_dust_emission,
    calibrate_dl07_pah_fraction,
    casey2012,
    cmb_contrast_factor,
    cmb_corrected_temperature,
    compute_absorbed_luminosity,
    compute_absorbed_luminosity_from_tau,
    create_dale2014_from_grid,
    create_dl07_from_grid,
    create_dl14_from_grid,
    get_emission_model,
    load_dl14_templates,
    load_draine_li_templates,
    planck_bnu,
    register_dale2014_tabulated,
    register_dl07_tabulated,
    register_dl14_tabulated,
    register_emission_model,
)

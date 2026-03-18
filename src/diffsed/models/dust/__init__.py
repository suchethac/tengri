"""Dust attenuation and emission models.

Attenuation
-----------
``attenuation.py`` — generalized two-component model with pluggable curves
and f_obscuration.  ``charlot_fall.py`` provides legacy power-law-only
functions for the fast precomputed path and existing tests.

Emission
--------
``emission.py`` — IR re-emission models (modified blackbody, Dale 2014,
Draine & Li 2007) with energy-balance normalization.
"""

from diffsed.models.dust.attenuation import (
    DUST_LAWS,
    get_dust_law,
    precompute_dust_age_weights,
    register_dust_law,
    two_component_dust,
    two_component_dust_fast,
)
from diffsed.models.dust.emission import (
    DUST_EMISSION_MODELS,
    apply_dust_emission,
    compute_absorbed_luminosity,
    compute_absorbed_luminosity_from_tau,
    get_emission_model,
    planck_bnu,
    register_emission_model,
)

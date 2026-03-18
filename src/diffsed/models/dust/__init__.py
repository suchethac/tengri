"""Dust attenuation models.

Primary module: ``attenuation.py`` — generalized two-component model with
pluggable curves and f_obscuration. ``charlot_fall.py`` provides legacy
power-law-only functions for the fast precomputed path and existing tests.
"""

from diffsed.models.dust.attenuation import (
    DUST_LAWS,
    get_dust_law,
    precompute_dust_age_weights,
    register_dust_law,
    two_component_dust,
    two_component_dust_fast,
)

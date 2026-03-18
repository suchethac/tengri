"""Dust attenuation models.

The primary module is `two_component_dust`, which generalizes the Charlot &
Fall (2000) birth-cloud + diffuse-ISM framework with pluggable attenuation
curves and f_obscuration (Lower 2022).

The original `charlot_fall` module is preserved for backward compatibility.
"""

from diffsed.models.dust.laws import DUST_LAWS, get_dust_law, register_dust_law
from diffsed.models.dust.two_component_dust import (
    precompute_dust_age_weights,
    two_component_dust,
    two_component_dust_fast,
)

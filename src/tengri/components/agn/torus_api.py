"""AGN torus and wind models (simple, two-temperature, clumpy, SKIRTOR, CAT3D, Silva04).

This module is a re-export grouping. The actual implementations live in
various modules under `tengri.components.agn`. Importing from here is
the canonical physics-grouped path:
`from tengri.components.agn.torus_api import ...`.

For backwards compatibility, use `from tengri.components.agn import ...`
to access these symbols.
"""

from tengri.components.agn.cat3d_wind import cat3d_wind_analytic, create_cat3d_wind_from_grid
from tengri.components.agn.silva04 import create_silva04_from_grid, silva04_analytic
from tengri.components.agn.skirtor import create_skirtor_from_grid, skirtor_analytic
from tengri.components.agn.torus import nenkova_torus, simple_torus, two_temperature_torus

__all__ = [
    "cat3d_wind_analytic",
    "create_cat3d_wind_from_grid",
    "create_silva04_from_grid",
    "create_skirtor_from_grid",
    "nenkova_torus",
    "silva04_analytic",
    "simple_torus",
    "skirtor_analytic",
    "two_temperature_torus",
]

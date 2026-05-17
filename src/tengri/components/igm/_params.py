"""Free-parameter declarations owned by the IGM component.

Single source of truth for the small set of IGM CGM damping-wing knobs
declared by :class:`IGMSEDComponent`. Note: these parameters are NOT
present in ``tengri.parameters._param_defs`` — IGM's declared params
have always been owned by the SEDComponent path; the flat
``Parameters(...)`` registry does not expose them. The two conditional
``igm_patchy`` and ``dla`` registry buckets (``igm_x_HI``,
``igm_bubble_mpc``, ``dla_*``) cover different physics and remain in
``_param_defs.py`` until PR4 consolidates them.
"""

from __future__ import annotations

from tengri.core.component import ParamDeclaration
from tengri.parameters.priors import Fixed

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "igm_z_mid",
        Fixed(7.0),
        "CGM damping-wing sigmoid midpoint redshift [dimensionless]",
    ),
    ParamDeclaration(
        "igm_dz",
        Fixed(0.5),
        "CGM damping-wing sigmoid width [dimensionless]",
    ),
    ParamDeclaration(
        "igm_log_nhi",
        Fixed(20.0),
        "CGM plateau log10(N_HI / cm^-2) [dimensionless]",
    ),
)

__all__ = ["PARAMS"]

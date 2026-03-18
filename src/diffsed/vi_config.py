"""Configuration for NIFTy variational inference (geoVI/MGVI/EVI).

Default values follow Philipp Frank's recommendations (2025 discussion).
EVI = Expansion-point Variational Inference: runs MGVI (linear) for
early iterations then switches to geoVI (nonlinear) for accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Philipp Frank's recommended kwargs for jft.optimize_kl
_DEFAULT_DRAW_LINEAR_KWARGS = {
    "cg_name": "SL",
    "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30},
}

_DEFAULT_NONLINEARLY_UPDATE_KWARGS = {
    "minimize_kwargs": {
        "name": "SN",
        "xtol": 1e-3,
        "cg_kwargs": {"name": None},
        "maxiter": 3,
    },
}

_DEFAULT_KL_KWARGS = {
    "minimize_kwargs": {
        "name": "M",
        "absdelta": 1e-3,
        "cg_kwargs": {"name": "MCG"},
        "maxiter": 10,
    },
}


@dataclass(frozen=True)
class VIConfig:
    """Configuration for geoVI/MGVI/EVI optimize_kl calls.

    Parameters
    ----------
    n_samples : int or callable
        Samples per KL iteration. ``mirror_samples=True`` (default in NIFTy)
        doubles this internally, so 3 → 6 effective samples.
    n_iterations : int
        Number of KL minimization iterations.
    use_vmap : bool
        Use ``jax.vmap`` for ``residual_map`` (faster, slightly more memory).
    evi_linear_fraction : float
        Fraction of iterations using ``linear_resample`` before switching to
        ``nonlinear_resample`` in EVI mode.
    draw_linear_kwargs : dict
        Kwargs for the CG solver that generates each sample.
    nonlinearly_update_kwargs : dict
        Kwargs for the Newton-CG that inverts the coordinate transform.
    kl_kwargs : dict
        Kwargs for the outer KL minimization.
    """

    n_samples: int | Callable = 3
    n_iterations: int = 10
    use_vmap: bool = True
    evi_linear_fraction: float = 0.5
    draw_linear_kwargs: dict = field(
        default_factory=lambda: {
            "cg_name": "SL",
            "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30},
        }
    )
    nonlinearly_update_kwargs: dict = field(
        default_factory=lambda: {
            "minimize_kwargs": {
                "name": "SN",
                "xtol": 1e-3,
                "cg_kwargs": {"name": None},
                "maxiter": 3,
            },
        }
    )
    kl_kwargs: dict = field(
        default_factory=lambda: {
            "minimize_kwargs": {
                "name": "M",
                "absdelta": 1e-3,
                "cg_kwargs": {"name": "MCG"},
                "maxiter": 10,
            },
        }
    )


def evi_sample_mode(n_iterations: int, linear_fraction: float = 0.5):
    """Return a callable sample_mode for EVI: MGVI first, then geoVI.

    Early iterations use ``linear_resample`` (cheap, expansion point far from
    optimum). Later iterations use ``nonlinear_resample`` (accurate near the
    converged expansion point).

    Parameters
    ----------
    n_iterations : int
        Total number of KL iterations.
    linear_fraction : float
        Fraction of iterations to run as MGVI (linear_resample).

    Returns
    -------
    callable
        ``sample_mode(i: int) -> str`` for ``jft.optimize_kl``.
    """
    transition = int(n_iterations * linear_fraction)

    def _mode(i: int) -> str:
        return "linear_resample" if i < transition else "nonlinear_resample"

    return _mode

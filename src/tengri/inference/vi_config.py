# SPDX-License-Identifier: BSD-3-Clause
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

    Attributes
    ----------
    n_samples : int or callable
        Samples per KL iteration (doubled by NIFTy's ``mirror_samples``).
    n_iterations : int
        Number of KL minimization iterations.
    use_vmap : bool
        Whether to use ``jax.vmap`` for residual mapping.
    evi_linear_fraction : float
        Fraction of iterations to use linear (MGVI) before nonlinear (geoVI).
    draw_linear_kwargs : dict
        Conjugate gradient solver configuration for sampling.
    nonlinearly_update_kwargs : dict
        Newton-CG configuration for coordinate transform inversion.
    kl_kwargs : dict
        Outer KL minimization configuration.

    Notes
    -----
    Frozen dataclass configuring the variational inference backend. Key fields:
    ``method`` selects the VI algorithm. ``'vi'`` (NIFTy geoVI) and
    ``'vi_linear'`` (NIFTy MGVI) are the working paths.

    The pure-JAX ``native_vi_nonlinear`` / ``native_vi_linear`` backends are
    registered ``tier="broken"`` (#1287), they segfault on DPL/dense_basis
    photometry mocks, and ``'vi_native'`` was never a registered name at all;
    it raises ``KeyError``. The "~19x faster" figure once quoted here was
    withdrawn: it compared MGVI against geoVI, not native against NIFTy.

    Examples
    --------
    >>> from tengri import VIConfig
    >>> cfg = VIConfig(n_samples=4, n_iterations=50)
    >>> cfg.n_samples
    4
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
        Default: 0.5.

    Returns
    -------
    callable
        ``sample_mode(i: int) -> str`` for ``jft.optimize_kl``.
    """
    transition = int(n_iterations * linear_fraction)

    def _mode(i: int) -> str:
        """Return sample mode: linear before transition, nonlinear after."""
        return "linear_resample" if i < transition else "nonlinear_resample"

    return _mode

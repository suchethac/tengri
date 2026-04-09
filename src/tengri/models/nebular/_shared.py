"""Shared utilities for nebular emission backends.

Functions extracted from individual backends to eliminate duplication.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.models.nebular._constants import (
    _C_CGS,
    _H_PLANCK,
    _LOG10_ZSUN,
    _LOG_OH_OFFSET,
    _LSUN_ERG,
    _LYMAN_LIMIT,
)

# ---------------------------------------------------------------------------
# Ionizing photon rate
# ---------------------------------------------------------------------------


@jax.jit
def compute_qh(ssp_wave: jnp.ndarray, ssp_flux: jnp.ndarray) -> float:
    """Compute ionizing photon rate Q_H from a single SSP spectrum.

    Q_H = integral_{0}^{912A} [L_nu / (h * nu)] d_nu

    .. warning::

        Returns ~0 for wNE (with Nebular Emission) SSP spectra because
        ionizing photons are pre-consumed by CLOUDY during SSP generation.
        This is expected — wNE SSPs already include nebular emission.
        Use non-nebular SSP files if you need Q_H for custom nebular models.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid in Angstrom (increasing).
    ssp_flux : array, shape (n_wave,)
        SSP flux in Lsun/Hz/Msun.

    Returns
    -------
    float
        Q_H in photons/s/Msun.
    """
    nu = _C_CGS / (ssp_wave * 1e-8)  # Hz
    l_nu = ssp_flux * _LSUN_ERG  # erg/s/Hz/Msun
    photon_rate = l_nu / (_H_PLANCK * nu)
    mask = ssp_wave < _LYMAN_LIMIT
    integrand = jnp.where(mask, photon_rate, 0.0)
    qh = -jnp.trapezoid(integrand, nu)
    return jnp.maximum(qh, 0.0)


# Vectorized over (metallicity, age) grid dimensions
compute_qh_grid = jax.vmap(
    jax.vmap(compute_qh, in_axes=(None, 0)),
    in_axes=(None, 0),
)


# ---------------------------------------------------------------------------
# Grid interpolation — piecewise-linear
# ---------------------------------------------------------------------------


def _interp_index_weight(
    x: float,
    grid: jnp.ndarray,
) -> tuple[int, float]:
    """Find bracketing index and interpolation weight for 1D grid.

    Returns (i, w) such that value ≈ grid[i]*(1-w) + grid[i+1]*w.
    Clips to grid bounds.
    """
    x_clipped = jnp.clip(x, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, x_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, len(grid) - 2)
    dx = grid[idx + 1] - grid[idx]
    w = jnp.where(dx > 0, (x_clipped - grid[idx]) / dx, 0.0)
    return idx, w


# ---------------------------------------------------------------------------
# Grid interpolation — triweight kernel (smooth, C²)
# Re-exported from utils.interpolation for backward compatibility.
# ---------------------------------------------------------------------------

from tengri.utils.interpolation import (
    compute_grid_weights,  # noqa: F401
    edges_for_grid,  # noqa: F401
    tw_cuml_kern as _tw_cuml_kern,  # noqa: F401
)

# ---------------------------------------------------------------------------
# Metallicity convention converters
# ---------------------------------------------------------------------------


def neb_logzsol_to_log_z_abs(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) -> log10(Z) absolute (DSPS/CloudyGrid convention)."""
    return logzsol + _LOG10_ZSUN


def neb_logzsol_to_cloudy_logoh(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) -> log10(O/H) on CLOUDY c17.01 solar scale (CB19 convention)."""
    return logzsol + _LOG10_ZSUN - _LOG_OH_OFFSET


def neb_logzsol_to_mappings_zeta(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) -> zeta_O solar-relative (MAPPINGS V convention)."""
    return 10.0**logzsol


# ---------------------------------------------------------------------------
# Continuum fallback
# ---------------------------------------------------------------------------


class NebularContinuumFallback:
    """Wrapper that provides continuum for line-only nebular backends.

    When a backend has ``has_continuum = False``, wrap it with this class
    to automatically supply nebular continuum via a secondary backend or
    analytic approximation.

    Parameters
    ----------
    primary : object
        The line-only backend (CB19, MappingsPhotoStellar, MappingsPhotoAGN,
        ShockEmission wrapper, etc.).
    fallback : object or None
        A continuum-capable backend (CueBackend or CloudyGridBackend).
        Takes priority over ``fallback_mode`` if provided.
    fallback_mode : str
        One of ``"error"`` (raise NebularContinuumUnavailableError) or
        ``"warn"`` (warn and return zeros). Default ``"error"``.

    Notes
    -----
    The analytic nebular continuum (free-free + free-bound + two-photon) is
    NOT implemented here yet — that is Phase N-4b. For now, line-only backends
    can pass a secondary Cue/CloudyGrid instance as ``fallback``.
    """

    def __init__(
        self,
        primary,
        fallback=None,
        fallback_mode: str = "error",
    ) -> None:
        if fallback_mode not in ("error", "warn"):
            raise ValueError("fallback_mode must be 'error' or 'warn'")
        self.primary = primary
        self.fallback = fallback
        self.fallback_mode = fallback_mode
        # Delegate attribute access to the primary backend
        self.has_continuum = fallback is not None
        self.has_free_params = getattr(primary, "has_free_params", False)
        self.name = f"fallback({getattr(primary, 'name', type(primary).__name__)})"

    def __getattr__(self, name: str):
        """Delegate all unknown attributes and methods to the primary backend."""
        return getattr(self.primary, name)

    def predict_nebular_sed(self, *args, **kwargs) -> jnp.ndarray:
        """Lines from primary backend + continuum from fallback (if configured).

        The primary backend's ``predict_nebular_sed`` returns line emission only.
        If a fallback is configured, its continuum is added. Otherwise, this
        returns the primary result (lines only, no continuum).

        Returns
        -------
        jnp.ndarray
            Nebular SED on the SSP wavelength grid (erg/s/Hz).
        """
        from tengri.models.nebular._protocol import NebularContinuumUnavailableError

        lines_sed = self.primary.predict_nebular_sed(*args, **kwargs)

        if self.fallback is not None and hasattr(self.fallback, "predict_nebular_sed"):
            cont_sed = self.fallback.predict_nebular_sed(*args, **kwargs)
            return lines_sed + cont_sed

        if self.fallback_mode == "error":
            raise NebularContinuumUnavailableError(
                f"{type(self.primary).__name__} provides no nebular continuum. "
                "Pass fallback=CueBackend(...) or fallback=CloudyGridBackend(...) "
                "to NebularContinuumFallback."
            )
        # fallback_mode == "warn"
        import warnings

        warnings.warn(
            f"{type(self.primary).__name__} has no nebular continuum — returning "
            "lines only. Pass fallback= to NebularContinuumFallback to add continuum.",
            UserWarning,
            stacklevel=2,
        )
        return lines_sed

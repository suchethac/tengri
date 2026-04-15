"""Shared utilities for nebular emission backends.

Functions extracted from individual backends to eliminate duplication.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.components.nebular._constants import (
    _C_CGS,
    _H_PLANCK,
    _LOG10_ZSUN,
    _LOG_OH_OFFSET,
    _LSUN_ERG,
    _LYMAN_LIMIT,
)
from tengri.utils.physics_constants import K_BOLTZ as _K_BOLTZ

# ---------------------------------------------------------------------------
# Line placement
# ---------------------------------------------------------------------------


def place_line_profiles(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    obs_wavelengths: jnp.ndarray,
    line_sigma_aa: float,
) -> jnp.ndarray:
    """Place emission lines onto a wavelength grid as Gaussians or delta functions.

    Parameters
    ----------
    line_wavelengths : array, shape (N_lines,)
        Rest-frame line centres in Å.
    line_luminosities : array, shape (N_lines,)
        Line luminosities in any consistent unit (e.g. Lsun or erg/s).
    obs_wavelengths : array, shape (N_wave,)
        Output wavelength grid in Å (rest-frame, increasing).
    line_sigma_aa : float
        Gaussian line width in Å.  ``0`` or negative → delta function placed
        into the nearest wavelength pixel.

    Returns
    -------
    array, shape (N_wave,)
        SED additive contribution in the same units as ``line_luminosities``
        per Hz (i.e. line_luminosities[j] / sigma_nu for Gaussian,
        line_luminosities[j] / dnu for delta function).
    """
    n_wave = obs_wavelengths.shape[0]

    if line_sigma_aa > 0:
        # Vectorised Gaussian profiles: broadcast (n_wave, 1) × (n_lines,)
        # σ_ν = σ_λ[cm] × c / λ[cm]²
        sigma_nu = line_sigma_aa * 1e-8 * _C_CGS / (line_wavelengths * 1e-8) ** 2  # (n_lines,)
        dwave = obs_wavelengths[:, None] - line_wavelengths[None, :]  # (n_wave, n_lines)
        profiles = jnp.exp(-0.5 * (dwave / line_sigma_aa) ** 2)
        profiles = profiles / (jnp.sqrt(2.0 * jnp.pi) * sigma_nu[None, :])
        sed = jnp.sum(line_luminosities[None, :] * profiles, axis=1)  # (n_wave,)
    else:
        # Vectorised delta functions: nearest-pixel placement via scatter-add.
        # (n_wave, n_lines) distance matrix → argmin per line
        indices = jnp.argmin(
            jnp.abs(obs_wavelengths[:, None] - line_wavelengths[None, :]), axis=0
        )  # (n_lines,)
        indices = jnp.clip(indices, 1, n_wave - 2)
        dwave = jnp.abs(obs_wavelengths[indices + 1] - obs_wavelengths[indices - 1]) / 2.0
        dnu = _C_CGS / (obs_wavelengths[indices] * 1e-8) ** 2 * dwave * 1e-8
        sed = (
            jnp.zeros(n_wave, dtype=obs_wavelengths.dtype).at[indices].add(line_luminosities / dnu)
        )

    return sed


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
    # Clamp per-element to prevent float64 trapezoid accumulation overflow for
    # young pure SSPs.  The cap is loose enough (~1e306) that it never fires for
    # physically realistic rates (~1e31) — it only prevents rate×dnu overflow.
    safe_max = jnp.finfo(jnp.float64).max / ssp_wave.shape[0]
    integrand = jnp.where(mask, jnp.minimum(photon_rate, safe_max), 0.0)
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
# Analytic nebular continuum  (Phase N-4b)
# ---------------------------------------------------------------------------

# Case B recombination coefficient at T=10^4 K [cm³/s]
# Storey & Hummer (1995) via pyNeb: alpha_B(1e4 K, ne=100) = 2.585e-13
# Power-law slope: T^{-0.847} (SH95 fit over 5e3–3e4 K, via pyNeb getTotRecombination)
_ALPHA_B_T4: float = 2.585e-13
_ALPHA_B_SLOPE: float = -0.847

# Lyman-α vacuum wavelength [Å]
_LYA_AA: float = 1216.0

# Free-free coefficient [erg cm³/s/Hz] from Osterbrock & Ferland eq 4.16
# ε_ff(ν) = _FF_COEFF * T^{-1/2} * Z² * n_e * n_i * g_ff * exp(-hν/kT)
_FF_COEFF: float = 6.8e-38

# Two-photon constants (Dopita & Osterbrock / pyNeb Continuum.two_photon):
#   α_eff_2s(T) = _ALPHA_EFF_2S_T4 × (T/1e4)^{-0.728}  [cm³/s]  — effective recombination
#   A_2s = 8.226 s^{-1}  — Einstein A coefficient for 2s→1s two-photon decay
# Reference: pyNeb v1.1.30 (Luridiana et al. 2015), Osterbrock & Ferland (2006) eq 4.29
_ALPHA_EFF_2S_T4: float = 0.838e-13
_ALPHA_EFF_2S_SLOPE: float = -0.728
_A_2S: float = 8.226


def compute_analytic_nebular_continuum(
    wave_aa: jnp.ndarray,
    q_h: float,
    log_z_abs: float,
    temperature: float = 1e4,
) -> jnp.ndarray:
    """Analytic nebular continuum: free-free + two-photon.

    Computes the hydrogen nebular continuum for a case B, fully ionized,
    ionization-bounded HII region, normalized to the ionizing photon rate Q_H.
    Includes thermal bremsstrahlung (free-free) and the two-photon continuum
    from the 2s→1s metastable transition (Nussbaumer & Schmutz 1984).

    Free-bound (recombination) continuum is **not** included here — it requires
    tabulated Milne function data (OF06 Chapter 4) and contributes mainly at
    λ < 3646 Å (Balmer limit).  For a line-only fallback, free-free + two-photon
    capture the dominant continuum shape at optical wavelengths.

    Parameters
    ----------
    wave_aa : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame, increasing).
    q_h : float
        Ionizing photon rate in photons/s.
    log_z_abs : float
        log10(Z/Z_sun) absolute metallicity — currently unused (helium and metals
        contribute ~10% to free-free at solar metallicity; reserved for future
        He-contribution scaling).
    temperature : float
        Electron temperature in K.  Default 10^4 K (typical HII region).

    Returns
    -------
    array, shape (n_wave,)
        Nebular continuum SED in erg/s/Hz.

    Notes
    -----
    Normalization via case B recombination (Osterbrock & Ferland 2006, eq 4.3):
        Q_H = α_B · n_e · n_p · V
        ⟹ n_e · n_p · V = Q_H / α_B(T)

    Free-free (OF06 eq 4.16, hydrogen only):
        L_ff(ν) = [Q_H / α_B(T)] · _FF_COEFF · T^{-1/2} · g_ff(ν,T) · exp(-hν/kT)

    Gaunt factor approximation (Draine 2011, eq 10.9):
        g_ff(u) = max(1.0, sqrt(3)/π · ln(2/u)) where u = hν/kT

    Two-photon (Nussbaumer & Schmutz 1984, Osterbrock & Ferland eq 4.29):
        Spectral shape A₂γ(y) = 202 · y(1-y) · [(1-4y(1-y))^0.8 + 0.88(y(1-y))^1.53]
        where y = ν/ν_Lyα = λ_Lyα/λ; defined for 0 < y < 1 (λ > 1216 Å).
        Peak is near y=0.5, i.e. λ ≈ 2 × λ_Lyα ≈ 2432 Å (optical/NUV).

        Normalization follows Osterbrock & Ferland (2006) §4.5 / pyNeb (Luridiana+2015):
            L_2q(ν) = [Q_H / α_B(T)] × (h × c / λ) × A₂γ(y) / A_2s × α_eff_2s(T)
        where:
            α_eff_2s(T) = 0.838e-13 × (T/10^4)^{-0.728}  [cm³/s]
            A_2s = 8.226 s^{-1}  (Einstein A for 2s→1s)

    References
    ----------
    - Osterbrock & Ferland (2006) — Chapter 4 (free-free, free-bound, two-photon)
    - Nussbaumer & Schmutz (1984) — two-photon spectral distribution
    - Storey & Hummer (1995) — tabulated α_B (SH95); slope T^{-0.847}
    - pyNeb v1.1.30 (Luridiana et al. 2015) — α_B, α_eff_2s fit coefficients
    - Draine (2011), "Physics of the Interstellar and Intergalactic Medium"
    """
    nu = _C_CGS / (wave_aa * 1e-8)  # Hz

    # Case B recombination coefficient: α_B(T) = α_B(1e4) × (T/1e4)^{-0.847}
    # Slope -0.847 from Storey & Hummer (1995) fit via pyNeb over 5e3–3e4 K.
    alpha_b = _ALPHA_B_T4 * (temperature / 1.0e4) ** _ALPHA_B_SLOPE

    # n_e · n_p · V = Q_H / α_B
    q_over_alpha = q_h / jnp.maximum(alpha_b, 1.0e-40)

    # ─── Free-free ───────────────────────────────────────────────────────────
    # Osterbrock & Ferland (2006), eq 4.16
    x = _H_PLANCK * nu / (_K_BOLTZ * temperature)  # dimensionless hν/kT
    # Gaunt factor: Draine (2011) eq 10.9 approximation; clip to ≥ 1
    g_ff = jnp.maximum(1.0, jnp.sqrt(3.0) / jnp.pi * jnp.log(2.0 / jnp.maximum(x, 1e-30)))
    gamma_ff = _FF_COEFF * temperature ** (-0.5) * g_ff * jnp.exp(-x)
    L_ff = q_over_alpha * gamma_ff  # erg/s/Hz

    # ─── Two-photon ──────────────────────────────────────────────────────────
    # N&S 1984 shape: y = ν/ν_Lyα = λ_Lyα/λ, valid for 0 < y < 1 (λ > λ_Lyα).
    # Each 2s photon pair spans λ_Lyα < λ < ∞ (peak near λ ≈ 2 × λ_Lyα).
    # Mask y ≥ 1 (λ ≤ λ_Lyα) to zero — photons cannot exceed Lyα energy.
    # JAX NaN-safe pattern: supply y_safe=0.5 in masked pixels so the shape
    # formula evaluates to a finite value before the jnp.where mask is applied.
    y_raw = _LYA_AA / jnp.maximum(wave_aa, _LYA_AA + 1e-6)  # never > 1 (safe)
    uu = y_raw * (1.0 - y_raw)
    A2q_shape = 202.0 * (uu * (1.0 - 4.0 * uu) ** 0.8 + 0.88 * uu**1.53)
    A2q = jnp.where(wave_aa > _LYA_AA, A2q_shape, 0.0)

    # Normalization (pyNeb / OF06 §4.5):
    # Power emitted per atom per unit-y interval = h × ν_Lyα × A₂γ(y), so:
    #   dL/dν = n_2s V × h × ν_Lyα × A₂γ(y) / ν_Lyα = n_2s V × h × A₂γ(y)... but
    # more cleanly, with y = ν/ν_Lyα:
    #   L_2q(ν) = (Q_H/α_B) × (α_eff_2s/A_2s) × h × y × A₂γ(y)   [erg/s/Hz]
    # where y = _LYA_AA/wave_aa = ν/ν_Lyα (already computed as y_raw).
    alpha_eff_2s = _ALPHA_EFF_2S_T4 * (temperature / 1.0e4) ** _ALPHA_EFF_2S_SLOPE
    L_2q = q_over_alpha * _H_PLANCK * y_raw * A2q / _A_2S * alpha_eff_2s

    return L_ff + L_2q


# ---------------------------------------------------------------------------
# Continuum fallback
# ---------------------------------------------------------------------------


class NebularContinuumFallback:
    """Wrapper that provides continuum for line-only nebular backends.

    When a backend has ``has_continuum = False``, wrap it with this class
    to automatically supply nebular continuum via a secondary backend or
    the built-in analytic approximation.

    Continuum is supplied in priority order at prediction time:

    1. **Secondary backend** (``fallback=CueBackend(...)`` or
       ``fallback=CloudyGridBackend(...)``): full physics, including line
       strengths from the secondary backend's grid.
    2. **Analytic free-free + two-photon continuum** via
       :func:`compute_analytic_nebular_continuum` (Osterbrock & Ferland
       2006 §4.3–4.5): activated automatically when ``ssp_wave`` and
       ``gas_logqion`` are present in the keyword arguments passed to
       :meth:`predict_nebular_sed`.
    3. **Raise** :exc:`~tengri.components.nebular._protocol.NebularContinuumUnavailableError`
       if ``fallback_mode="error"`` and neither Tier 1 nor Tier 2 is
       available.
    4. **Warn and return lines only** if ``fallback_mode="warn"``.

    Parameters
    ----------
    primary : object
        The line-only backend (CB19Backend, MappingsPhotoStellarBackend,
        MappingsPhotoAGNBackend, ShockBackend, etc.).
    fallback : object or None
        A continuum-capable backend (CueBackend or CloudyGridBackend).
        Takes priority over the analytic Tier 2 path when provided.
    fallback_mode : str
        Behaviour when neither a ``fallback`` backend nor the ``ssp_wave``/
        ``gas_logqion`` kwargs are available.  One of ``"error"`` (raise
        NebularContinuumUnavailableError) or ``"warn"`` (emit a warning and
        return lines only).  Default ``"error"``.

    Examples
    --------
    Line-only backend with analytic continuum (Tier 2 via kwargs):

    >>> cb19 = CB19Backend(ssp_data=ssp)
    >>> with_cont = NebularContinuumFallback(cb19, fallback_mode="warn")
    >>> sed = with_cont.predict_nebular_sed(..., ssp_wave=wave, gas_logqion=49.1)

    Line-only backend with full Cue continuum (Tier 1):

    >>> cue = CueBackend(weights_path, ssp_data=ssp)
    >>> with_cont = NebularContinuumFallback(cb19, fallback=cue)
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
        # This wrapper always attempts to provide continuum: either via the
        # fallback backend (Tier 1), analytic approximation (Tier 2), or
        # graceful degradation per fallback_mode (Tier 3/4).
        self.has_continuum = True
        self.has_free_params = getattr(primary, "has_free_params", False)
        self.name = f"fallback({getattr(primary, 'name', type(primary).__name__)})"

    def __getattr__(self, name: str):
        """Delegate all unknown attributes and methods to the primary backend."""
        return getattr(self.primary, name)

    def predict_nebular_sed(self, *args, **kwargs) -> jnp.ndarray:
        """Lines from primary backend + continuum from fallback or analytic.

        Continuum is supplied in priority order:
        1. Secondary backend passed as ``fallback=`` (CueBackend / CloudyGridBackend)
        2. Analytic free-free + two-photon continuum via
           :func:`compute_analytic_nebular_continuum` (if ``ssp_wave`` and
           ``gas_logqion`` are present in ``kwargs``)
        3. Raise :exc:`NebularContinuumUnavailableError` (if ``fallback_mode="error"``)
        4. Warn and return lines only (if ``fallback_mode="warn"``)

        Returns
        -------
        jnp.ndarray
            Nebular SED on the SSP wavelength grid (erg/s/Hz).
        """
        from tengri.components.nebular._protocol import NebularContinuumUnavailableError

        lines_sed = self.primary.predict_nebular_sed(*args, **kwargs)

        # Tier 1: secondary backend
        if self.fallback is not None and hasattr(self.fallback, "predict_nebular_sed"):
            cont_sed = self.fallback.predict_nebular_sed(*args, **kwargs)
            return lines_sed + cont_sed

        # Tier 2: analytic continuum (free-free + two-photon) when possible
        ssp_wave = kwargs.get("ssp_wave")
        gas_logqion = kwargs.get("gas_logqion")
        log_z = kwargs.get("log_z", _LOG10_ZSUN)
        if ssp_wave is not None and gas_logqion is not None:
            q_h = 10.0**gas_logqion
            cont_analytic = compute_analytic_nebular_continuum(ssp_wave, q_h, log_z_abs=log_z)
            return lines_sed + cont_analytic

        if self.fallback_mode == "error":
            raise NebularContinuumUnavailableError(
                f"{type(self.primary).__name__} provides no nebular continuum. "
                "Pass fallback=CueBackend(...) or fallback=CloudyGridBackend(...) "
                "to NebularContinuumFallback, or ensure ssp_wave + gas_logqion are "
                "passed as keyword arguments for analytic continuum."
            )
        # Tier 4: warn and return lines only
        import warnings

        warnings.warn(
            f"{type(self.primary).__name__} has no nebular continuum — returning "
            "lines only. Pass fallback= to NebularContinuumFallback to add continuum.",
            UserWarning,
            stacklevel=2,
        )
        return lines_sed

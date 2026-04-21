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

# ── Line placement ────────────────────────────────────────────────


def place_line_profiles(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    obs_wavelengths: jnp.ndarray,
    line_sigma_aa: float,
) -> jnp.ndarray:
    r"""Place emission lines onto a wavelength grid as Gaussians or delta functions.

    Converts line luminosities (point-like) to spectral luminosity density on a
    wavelength grid by either convolving with a Gaussian profile or placing them
    as delta functions in the nearest pixel. Commonly used to overlay nebular and
    AGN emission lines onto continuum SEDs.

    Parameters
    ----------
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line centres in Å (vacuum wavelength).
    line_luminosities : array, shape (n_lines,)
        Line luminosities in consistent units [erg/s] or [erg/s/Msun].
    obs_wavelengths : array, shape (n_wave,)
        Output wavelength grid in Å (rest-frame, increasing).
    line_sigma_aa : float
        Gaussian line width (FWHM equivalent) in Å. When <= 0, lines are placed as
        delta functions in the nearest pixel. [Å]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density on ``obs_wavelengths``. [erg/s/Hz] or [erg/s/Hz/Msun]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives and are
    vectorised over lines and wavelengths.

    **Gaussian mode** (``line_sigma_aa > 0``):
        Converts wavelength width σ_λ to frequency width via the Jacobian:

        .. math::

            \sigma_\nu = \sigma_\lambda \, \frac{c}{\lambda^2}

        where c is the speed of light and λ is line wavelength. Each line is
        normalized such that the integrated flux equals the input ``line_luminosity``.

    **Delta function mode** (``line_sigma_aa <= 0``):
        Places each line in the nearest wavelength pixel by scatter-add,
        normalised by the local frequency spacing Δν at that pixel. This is fast
        but introduces aliasing artifacts if the wavelength grid is coarse.

    **Upstream**: Implementation adapted from Prospector's line-placement routines
    (Johnson et al. 2021) for JAX differentiability.
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


# ── Ionizing photon rate ──────────────────────────────────────────


@jax.jit
def compute_qh(ssp_wave: jnp.ndarray, ssp_flux: jnp.ndarray) -> float:
    r"""Compute hydrogen-ionizing photon production rate Q_H from an SSP spectrum.

    Integrates the far-UV SSP luminosity density, divided by photon energy, over
    the hydrogen-ionizing frequency range (ν > ν_LL ≈ 13.6 eV, λ < 911.76 Å).
    This rate is essential for all nebular emission models (HII regions, AGN NLR,
    DIG) that depend on ionizing photon supply.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid in Å (rest-frame, increasing).
    ssp_flux : array, shape (n_wave,)
        SSP spectral luminosity density. [erg/s/Hz/Msun]

    Returns
    -------
    float
        Hydrogen-ionizing photon production rate. [photons/s/Msun]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives. Safe inside
    :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

    **Frequency integration**:
        Q_H is computed as:

        .. math::

            Q_H = \int_0^{\nu_{\rm LL}} \frac{L_\nu}{h\nu} \, \mathrm{d}\nu

        where ν_LL = 13.6 eV / h ≈ 3.29 × 10^15 Hz (Lyman limit, λ < 911.76 Å),
        L_ν is the SSP flux [erg/s/Hz/Msun], and h is Planck's constant.

        The integral is computed via trapezoidal quadrature in frequency space
        (not wavelength space) to avoid nonlinear Jacobian effects.

    **Warning — wNE SSPs**:
        Returns ~0 for "with Nebular Emission" (wNE) SSP spectra because CLOUDY
        consumes ionizing photons during SSP generation. If you see Q_H ≈ 0 for
        young SSPs (which should have Q_H > 1e50 photons/s), check that your
        SSP templates are non-nebular variants (BC03, FSPS/Conroy+Gunn models, etc.).

    **Numerical safety**:
        Clamps per-wavelength integrand to prevent float64 overflow during
        trapezoidal accumulation (only relevant for artificially young/pure SSPs
        with Q_H > 1e100). Does not affect physically realistic rates (~1e31).

    References
    ----------
    .. [1] C. Conroy, "Modeling the Panchromatic SED Evolution of Galaxies,"
       ApJ, 647, 201 (2006). arXiv:astro-ph/0604217.
       https://doi.org/10.1086/504612
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


# ── Grid interpolation — piecewise-linear ─────────────────────────


def _interp_index_weight(
    x: float,
    grid: jnp.ndarray,
) -> tuple[int, float]:
    """Find bracketing index and linear interpolation weight for a 1D sorted grid."""
    x_clipped = jnp.clip(x, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, x_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, len(grid) - 2)
    dx = grid[idx + 1] - grid[idx]
    w = jnp.where(dx > 0, (x_clipped - grid[idx]) / dx, 0.0)
    return idx, w


# ── Grid interpolation — triweight kernel (smooth, C²) ────────────
# Re-exported from utils.interpolation for convenience.

from tengri.utils.interpolation import (
    compute_grid_weights,  # noqa: F401
    edges_for_grid,  # noqa: F401
    tw_cuml_kern as _tw_cuml_kern,  # noqa: F401
)

# ── Metallicity convention converters ─────────────────────────────


def neb_logzsol_to_log_z_abs(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) -> log10(Z) absolute (DSPS/CloudyGrid convention)."""
    return logzsol + _LOG10_ZSUN


def neb_logzsol_to_cloudy_logoh(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) -> log10(O/H) on CLOUDY c17.01 solar scale (CB19 convention)."""
    return logzsol + _LOG10_ZSUN - _LOG_OH_OFFSET


def neb_logzsol_to_mappings_zeta(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) -> zeta_O solar-relative (MAPPINGS V convention)."""
    return 10.0**logzsol


# ── Analytic nebular continuum  (Phase N-4b) ──────────────────────

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
    r"""Compute hydrogen nebular continuum: free-free + two-photon emission.

    Predicts the optically-thick, ionization-bounded HII region continuum from
    case B recombination, thermal bremsstrahlung (free-free), and the two-photon
    emission from the 2s metastable transition. This is a Tier 2 analytic
    approximation used as a fallback when full grids (Cue, Cloudy) are unavailable.

    Parameters
    ----------
    wave_aa : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame, increasing).
    q_h : float
        Hydrogen-ionizing photon production rate. [photons/s]
    log_z_abs : float
        Absolute metallicity. [log10(Z/Z_sun)]
        Currently unused; included for forward compatibility (metallicity scaling
        of free-free via He/metal opacity is reserved for a future update).
    temperature : float, optional
        Electron temperature in K. Default: 10^4 K (typical HII region). [K]

    Returns
    -------
    array, shape (n_wave,)
        Nebular continuum spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives with no
    Python-level branching on traced values.

    **Case B recombination normalization** (Osterbrock & Ferland 2006, §4.3):
        The ionization balance is:

        .. math::

            Q_H = \alpha_B(T) \, n_e \, n_p \, V

        where α_B(T) is the case-B recombination coefficient and V is the HII
        region volume. Rearranging:

        .. math::

            n_e \, n_p \, V = \frac{Q_H}{\alpha_B(T)}

        Temperature dependence (Storey & Hummer 1995):

        .. math::

            \alpha_B(T) = \alpha_B(10^4\,\mathrm{K}) \, \left(\frac{T}{10^4}\right)^{-0.847}

        with α_B(10^4 K) ≈ 2.585 × 10^{-13} cm^3/s.

    **Free-free (bremsstrahlung) continuum** (OF06, §4.4, eq. 4.16):
        Thermal radiation from electron–ion collisions:

        .. math::

            L_{\mathrm{ff}}(\nu) = \frac{Q_H}{\alpha_B(T)} \, \epsilon_{\mathrm{ff}}(\nu,T)
                \quad \text{where} \quad
                \epsilon_{\mathrm{ff}} = k_{\mathrm{ff}} \, T^{-1/2} \,
                g_{\mathrm{ff}}(\nu,T) \, e^{-h\nu/(k_B T)}

        Gaunt factor approximation (Draine 2011, §10.4, eq. 10.9):

        .. math::

            g_{\mathrm{ff}}(u) = \max\left(1, \frac{\sqrt{3}}{\pi} \ln\frac{2}{u}\right)
                \quad \text{where} \quad u = \frac{h\nu}{k_B T}

    **Two-photon continuum** (Nussbaumer & Schmutz 1984, OF06 §4.5, eq. 4.29):
        Recombination to the 2s metastable level produces two-photon pairs
        (2s → 1s emission is forbidden as a single photon). Spectral shape:

        .. math::

            A_{2\gamma}(y) = 202 \, y(1-y) \,
                \left[(1-4y(1-y))^{0.8} + 0.88(y(1-y))^{1.53}\right]

        where y = ν/ν_{Ly\alpha} = λ_{Ly\alpha}/λ ∈ (0,1), defined only for
        λ > λ_{Ly\alpha} (≈ 1216 Å). Peak intensity near y ≈ 0.5 (λ ≈ 2432 Å,
        optical/NUV). Normalization:

        .. math::

            L_{2\gamma}(\nu) = \frac{Q_H}{\alpha_B(T)} \, \frac{\alpha_{\mathrm{eff}}^{2s}(T)}{A_{2s}}
                \, h \, y \, A_{2\gamma}(y)

        where α^{eff}_{2s}(T) = 0.838 × 10^{-13} (T/10^4)^{-0.728} cm^3/s is
        the effective recombination coefficient to 2s, and A_{2s} = 8.226 s^{-1}
        is the Einstein A coefficient for the forbidden 2s → 1s transition.

    **Approximation flags**:
        - **Missing continua**: Free-bound (recombination) continuum is omitted
          (contributes λ < 3646 Å, Balmer limit). This is acceptable for optical
          SEDs but underestimates UV flux shortward of the Balmer limit.
        - **Hydrogen only**: Neglects helium and metal free-free opacity (~10% at
          solar metallicity); reserved for a future metallicity-dependent update.
        - **Single temperature**: Assumes uniform electron temperature T=10^4 K.
          Physically, HII region temperature varies with ionization parameter and
          ISM conditions; this approximation is valid for log(U) ∈ [−3, −1].

    **Validity**: Use this function when Cloudy or Cue grids are unavailable.
    For science-grade nebular fitting, prefer CloudyGridBackend or CueBackend.

    References
    ----------
    .. [1] D. E. Osterbrock and G. J. Ferland, "Astrophysics of Gaseous Nebulae
       and Active Galactic Nuclei," 2nd edn. (University Science Books, 2006).
       Chapter 4: Nebular Continuum and Line Emission.
    .. [2] H. Nussbaumer and W. Schmutz, "The two-photon continuum of HeII and
       the He+ f-value problem," A&A, 138, 495 (1984).
    .. [3] P. J. Storey and D. G. Hummer, "Recombination coefficients for H II and
       HeII," MNRAS, 272, 41 (1995). https://doi.org/10.1093/mnras/272.1.41
    .. [4] B. T. Draine, "Physics of the Interstellar and Intergalactic Medium"
       (Princeton University Press, 2011). Section 10.4: Gaunt factors.
    .. [5] V. Luridiana, C. Morisset, and R. A. Shaw, "PyNeb: A Python Package
       for Analysing Emission Lines from Ionised Nebulae," A&A, 573, A42 (2015).
       arXiv:1412.6345. https://doi.org/10.1051/0004-6361/201323152
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


# ── Continuum fallback ────────────────────────────────────────────


class NebularContinuumFallback:
    """Wrapper that adds nebular continuum to line-only nebular backends.

    Many nebular backends (CB19, MAPPINGS, Shock) produce emission lines only,
    without continuum. This wrapper provides missing continuum in a prioritised
    fallback chain: (1) secondary physics backend, (2) analytic free-free +
    two-photon (§4.3–4.5 of Osterbrock & Ferland 2006), (3) error or warning
    graceful degradation.

    Parameters
    ----------
    primary : NebularBackend
        A line-only nebular backend (has_continuum=False). Must implement
        predict_nebular_sed().
    fallback : NebularBackend, optional
        A continuum-capable backend (CueBackend, CloudyGridBackend) to use
        if ``ssp_wave`` and ``gas_logqion`` are unavailable. Default: None.
    fallback_mode : str, optional
        Fallback behaviour if neither backend nor analytical continuum is
        available. One of "error" (raise NebularContinuumUnavailableError)
        or "warn" (emit warning, return lines only). Default: "error".

    Attributes
    ----------
    has_continuum : bool
        Always True; this wrapper guarantees continuum provision (via one of
        three tiers) or graceful failure.
    has_free_params : bool
        Inherited from primary backend.
    name : str
        Identifier string (e.g., "fallback(CB19Backend)").

    Notes
    -----
    **Continuum supply chain** at prediction time:

    1. **Tier 1 (Secondary backend)**: If ``fallback`` is not None and has
       ``predict_nebular_sed``, use it to compute full continuum + lines.
       Integrated with the primary backend's lines.

    2. **Tier 2 (Analytic approximation)**: If ``ssp_wave`` and ``gas_logqion``
       are provided as kwargs to predict_nebular_sed(), compute analytic
       free-free + two-photon via compute_analytic_nebular_continuum().
       This requires ``ssp_wave`` [Å] and ``gas_logqion`` [log10(Q_H)].

    3. **Tier 3/4 (Graceful degradation)**: If neither Tier 1 nor 2 is
       available, either raise NebularContinuumUnavailableError (fallback_mode="error")
       or emit a UserWarning and return lines only (fallback_mode="warn").

    Examples
    --------
    >>> from tengri.components.nebular import CB19Backend, CueBackend, NebularContinuumFallback
    >>> ssp = load_ssp_data(...)
    >>> cb19 = CB19Backend(ssp_data=ssp)
    >>> with_continuum = NebularContinuumFallback(cb19, fallback_mode="warn")
    >>> # Predicts CB19 lines + analytic continuum if ssp_wave provided
    >>> sed = with_continuum.predict_nebular_sed(
    ...     neb_logzsol=-0.5, ..., ssp_wave=wave, gas_logqion=49.2
    ... )
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
        """Predict nebular SED: lines from primary, continuum from fallback chain.

        Retrieves emission lines from the primary backend and adds nebular
        continuum via the priority fallback chain (secondary backend, analytic,
        or graceful degradation).

        Returns
        -------
        jnp.ndarray, shape (n_wave,)
            Nebular spectral luminosity density on SSP wavelength grid.
            [erg/s/Hz]

        Raises
        ------
        NebularContinuumUnavailableError
            If ``fallback_mode="error"`` and neither secondary backend nor
            analytic continuum (via ssp_wave + gas_logqion) is available.

        Warns
        -----
        UserWarning
            If ``fallback_mode="warn"`` and continuum is unavailable. Returns
            lines only in this case.

        Notes
        -----
        **Fallback chain execution**:

        1. Call primary.predict_nebular_sed(*args, **kwargs) to get lines
        2. If fallback backend is available, add its continuum output
        3. Else if ssp_wave and gas_logqion in kwargs, compute analytic continuum
        4. Else apply fallback_mode (error or warn)
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

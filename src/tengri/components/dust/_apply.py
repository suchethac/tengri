# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation application: two-component and single-screen transmission.

Applies the ``DUST_LAWS`` k(lambda) curves to build the Charlot & Fall (2000)
two-component (birth-cloud + diffuse) and single-screen attenuation, the
age-weight precompute, birth/diffuse law-param resolution, and the Lyman-cutoff
mask. Pure JAX; resolves curves by name from :mod:`..laws._registry` at call
time, so it never imports the law functions directly (no import cycle).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp

from tengri.components.dust.laws._registry import (
    resolve_dust_law,
)


def precompute_dust_age_weights(
    age_grid: jnp.ndarray,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
) -> jnp.ndarray:
    r"""Precompute the birth-cloud sigmoid weight.

    Call once at Model init; pass result to ``two_component_dust_fast``.

    Parameters
    ----------
    age_grid : array_like, shape (n_ages,)
        Stellar population ages. [yr]
    t_birth : float
        Birth cloud dispersal age. [yr] Default: 1e7 (10 Myr).
    transition_width : float
        Sigmoid width in dex. [dimensionless] Default: 0.3.

    Returns
    -------
    ndarray, shape (n_ages,)
        Sigmoid weight: 1 for young stars (t < t_birth), 0 for old. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The weight is:

    .. math::

        w(t_{\text{age}}) = \sigma\left(-\frac{\log_{10} t_{\text{age}} - \log_{10} t_{\text{birth}}}{\Delta_{\text{trans}}}\right)

    where :math:`\sigma(x) = 1/(1 + e^{-x})` is the logistic sigmoid.
    """
    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    return jax.nn.sigmoid(-(log_age - log_t_birth) / transition_width)


def precompute_dust_age_mask(
    age_grid: jnp.ndarray,
    t_birth: float = 1e7,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Precompute hard young/old masks for fast two-CSP dust decomposition.

    Uses a hard threshold at ``t_birth`` instead of a smooth sigmoid.
    This is the original Charlot & Fall (2000) formulation and enables
    a fast path where dust is factored out of the age sum entirely.

    Parameters
    ----------
    age_grid : array_like, shape (n_ages,)
        Stellar population ages. [yr]
    t_birth : float
        Birth cloud dispersal age. [yr] Default: 1e7 (10 Myr).

    Returns
    -------
    young_mask : ndarray, shape (n_ages,)
        1.0 for young ages (< t_birth), 0.0 for old. [dimensionless]
    old_mask : ndarray, shape (n_ages,)
        1.0 for old ages (≥ t_birth), 0.0 for young. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The original Charlot & Fall (2000) model uses a hard cutoff instead of a sigmoid.
    This returns complementary masks: young_mask + old_mask = 1 everywhere.
    """
    young = (age_grid < t_birth).astype(age_grid.dtype)  # preserve input precision
    return young, 1.0 - young


#: Two-component attenuation-law parameters that may be set per-component.
#: Maps the law-function keyword to ``(flat_param_name, default)``. The
#: per-component flat names are ``<flat_param_name>_bc`` / ``_diff``.
_TWO_COMPONENT_LAW_PARAMS: tuple[tuple[str, str, float], ...] = (
    ("n_slope", "dust_slope", -0.7),
    ("dust_bump_strength", "dust_bump_strength", 0.0),
    ("dust_delta", "dust_delta", 0.0),
    ("dust_Rv", "dust_Rv", 3.1),
)

#: User-facing per-component short name -> attenuation-law kwarg. Used by the
#: builder grammar to accept ``slope_bc`` / ``slope_diff`` / ``delta_bc`` etc.
#: in the ``dust`` group and route them onto the per-component overrides.
TWO_COMPONENT_OVERRIDE_KEYS: dict[str, str] = {
    "slope": "n_slope",
    "bump_strength": "dust_bump_strength",
    "delta": "dust_delta",
    "Rv": "dust_Rv",
}


def resolve_bc_diff_law_params(
    params: Mapping,
    bc_overrides: Mapping | None = None,
    diff_overrides: Mapping | None = None,
    live_shape_params: frozenset[str] | None = None,
) -> tuple[dict, dict]:
    """Split shared dust law parameters into birth-cloud and diffuse law dicts.

    For each two-component law parameter (slope, bump, delta, Rv), the value is
    the shared ``dust_<x>`` from ``params``, unless a per-component override is
    supplied in ``bc_overrides`` / ``diff_overrides`` (keyed by law-function
    kwarg, e.g. ``n_slope``). The overrides are the static per-component
    settings carried on :class:`DustSEDComponentConfig`. This is the single
    source of truth shared by every stellar two-component attenuation path, so
    they cannot diverge.

    A parameter nobody asked for is **omitted** rather than defaulted, so the
    selected law's own published default stands — see ``live_shape_params``.

    Parameters
    ----------
    params : Mapping
        Flat ``dust_*`` parameter mapping (JAX scalars or floats).
    bc_overrides, diff_overrides : Mapping, optional
        Per-component law-kwarg overrides (e.g. ``{"n_slope": -1.0}`` for the
        FSPS birth-cloud convention). Always honored: an override *is* a
        request, whatever the provenance of the shared parameter.
    live_shape_params : frozenset of str, optional
        Flat names a caller actually asked for, resolved from spec provenance
        by :meth:`SEDModel._requested_law_shape_params` (#1808). Names outside
        this set are left out of the returned dicts. ``None`` keeps the
        historical behavior of passing all four unconditionally, for the
        direct callers that have no spec to ask.

    Returns
    -------
    bc_params, diff_params : dict
        Keyword dicts ready to splat into an attenuation-law function (keys are
        law-function kwargs, e.g. ``n_slope``).

    Notes
    -----
    **JIT-compatible**: yes — only dict construction and ``Mapping.get``; the
    values pass through untouched (traced arrays stay traced).

    Passing all four unconditionally was #1833. The spec declares ONE shared
    ``dust_bump_strength`` / ``dust_delta``, both ``Fixed(0.0)``, while each law
    carries its paper's value in its own signature (``kriek_conroy``
    ``dust_bump_strength=1.0``; ``narayanan_z`` and ``tea``
    ``dust_delta=-0.2``). Injecting the shared zero deleted the 2175 Å Drude
    bump that Kriek & Conroy (2013) Eqn 3 exists to add, so ``two_component``
    silently returned a different law from the one selected — measured at 128%
    on the SED against ``single_component``, which had already been fixed by
    #1808. This is that fix reaching its second caller.
    """
    bc_overrides = bc_overrides or {}
    diff_overrides = diff_overrides or {}
    bc: dict = {}
    diff: dict = {}
    for law_kw, flat_name, default in _TWO_COMPONENT_LAW_PARAMS:
        requested = live_shape_params is None or flat_name in live_shape_params
        shared = params.get(flat_name, default) if requested else None
        for target, overrides in ((bc, bc_overrides), (diff, diff_overrides)):
            if law_kw in overrides:
                target[law_kw] = overrides[law_kw]
            elif requested:
                target[law_kw] = shared
    return bc, diff


def apply_lyman_cutoff(
    k: jnp.ndarray, wavelength: jnp.ndarray, cutoff_aa: float = 0.0
) -> jnp.ndarray:
    r"""Zero an attenuation curve below a wavelength cutoff (Lyman-limit clip).

    Sets :math:`k(\lambda) = 0` for :math:`\lambda < \lambda_{\rm cut}`, leaving
    the curve untouched elsewhere. The standard choice is the hydrogen Lyman
    limit (912 Å): far-UV photons are absorbed by H ionization before reaching
    dust grains, so CIGALE's ``dustatt_modified_starburst`` zeros its curve
    there (``a_vs_ebv`` clips at 91.2 nm). tengri's ``calzetti`` / ``leitherer02``
    polynomials instead *extrapolate* through the FUV by default; this helper
    is the opt-in that reproduces the CIGALE behavior.

    Parameters
    ----------
    k : ndarray, shape (n_wave,)
        Attenuation curve :math:`k(\lambda) = A_\lambda / A_V`. [dimensionless]
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    cutoff_aa : float, optional
        Cutoff wavelength. [Å] Default ``0.0`` -> no-op (``wavelength >= 0`` is
        always true), so passing ``0.0`` disables the clip without a Python
        branch.

    Returns
    -------
    ndarray, shape (n_wave,)
        Curve with values below ``cutoff_aa`` set to zero.

    Notes
    -----
    **JIT-compatible**: yes — a single ``jnp.where``; ``cutoff_aa`` is a static
    Python float (never a traced parameter), so no ``TracerBoolConversionError``
    risk. **Gradient-safe**: yes — ``jnp.where`` on a static mask.
    """
    return jnp.where(wavelength >= cutoff_aa, k, 0.0)


def two_component_dust(
    wavelength: jnp.ndarray,
    age_grid: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = 0.0,
    t_birth: float = 1e7,
    transition_width: float = 0.3,
    bc_params: dict | None = None,
    diff_params: dict | None = None,
    lyman_cutoff_aa: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Two-component dust attenuation following Charlot & Fall (2000) with smooth age transition.

    Separates dust into birth-cloud (young stars) and diffuse ISM (all stars) components
    with independent optical depths and attenuation curves. Transition between components
    uses a smooth sigmoid in log-age, enabling automatic differentiation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    age_grid : array_like, shape (n_ages,)
        Stellar population ages. [yr]
    tau_v1 : float
        Birth-cloud V-band optical depth (at 5500 Å). [dimensionless]
        Note: tengri applies ``tau_bc`` internally but exposes ``tau_v1`` after normalizing
        by attenuation curve slope. See docs/known_bugs.md (CROSSVAL-01) for cross-code comparison.
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc : str, optional
        Attenuation curve name for birth cloud. Default: "power_law". Resolved from ``DUST_LAWS`` registry.
    law_diff : str, optional
        Attenuation curve name for diffuse ISM. Default: "power_law".
    f_obscuration : float, optional
        Fraction of unattenuated sightlines in clumpy geometry (Lower 2022). [dimensionless, in [0, 1]]
        Default: 0.0 (uniform screen).
    t_birth : float, optional
        Birth-cloud dispersal age (sigmoid center). [yr] Default: 1e7 (10 Myr).
    transition_width : float, optional
        Sigmoid transition width in dex. [dimensionless] Default: 0.3 (~5-20 Myr range).
    bc_params : dict, optional
        Per-component overrides for the **birth-cloud** law (e.g.
        ``{"n_slope": -1.0}``). Merged on top of ``**law_params``, so any key
        absent here falls back to the shared value. Enables FSPS-style
        independent indices (birth cloud ``dust1_index`` ≠ diffuse
        ``dust_index``). Default ``None`` → shared parameters.
    diff_params : dict, optional
        Per-component overrides for the **diffuse ISM** law. Same merge
        semantics as ``bc_params``. Default ``None`` → shared parameters.
    lyman_cutoff_aa : float, optional
        Zero both attenuation curves below this wavelength. [Å] Default ``0.0``
        -> disabled (the polynomial extrapolates through the FUV). Set to
        ``912.0`` to match CIGALE's Lyman-limit clip (see
        :func:`apply_lyman_cutoff`).
    **law_params
        Shared keyword arguments passed to both attenuation curve functions
        (e.g., ``n_slope``, ``dust_bump_strength``, ``dust_delta``,
        ``dust_Rv``). ``bc_params`` / ``diff_params`` override these
        per-component.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission factor T(λ, t_age), where T ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives and safe for ``jax.jit``.

    **Gradient-safe**: yes — differentiable everywhere; smooth sigmoid age transition preserves gradients
    through the birth-cloud boundary.

    The total optical depth is:

    .. math::

        \tau(\lambda, t_{\text{age}}) = w(t_{\text{age}}) \cdot \tau_{{\rm V,BC}} \cdot k_{\rm BC}(\lambda)
        + \tau_{{\rm V,ISM}} \cdot k_{\rm ISM}(\lambda)

    where :math:`w(t_{\text{age}})` is the sigmoid weight:

    .. math::

        w(t_{\text{age}}) = \sigma\left(-\frac{\log_{10} t_{\text{age}} - \log_{10} t_{\text{birth}}}{\Delta_{\text{trans}}}\right)

    and :math:`\sigma(x) = 1/(1 + e^{-x})` is the logistic sigmoid. The transmission is then:

    .. math::

        T(\lambda, t_{\text{age}}) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp[-\tau(\lambda, t_{\text{age}})]

    where :math:`f_{\rm obs}` is the unattenuated sightline fraction.

    **Upstream**: Implements the Charlot & Fall (2000) two-component framework [1]_ with sigmoid age transition
    following tengri's differentiable design. Birth-cloud + diffuse ISM separation enables realistic modeling
    of age-dependent dust geometry in galaxies.

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250

    .. [2] S. Lower et al., "How Well Can We Measure Galaxy Dust Attenuation Curves?
       The Impact of the Assumed Star-dust Geometry Model in SED Fitting,"
       ApJ, 931, 14 (2022). arXiv:2203.00074.
       https://doi.org/10.3847/1538-4357/ac6959

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import two_component_dust
    >>> wave = jnp.linspace(1000.0, 30000.0, 300)
    >>> ages = jnp.logspace(6.0, 10.14, 64)
    >>> T = two_component_dust(wave, ages, tau_v1=1.0, tau_v2=0.3)
    >>> T.shape
    (64, 300)
    """
    # Per-component law parameters: shared ``law_params`` with optional
    # ``bc_params`` / ``diff_params`` overlays. Each overlay only replaces the
    # keys it names, so callers can steepen the birth cloud (FSPS
    # ``dust1_index=-1.0``) without touching the diffuse ISM.
    bc_kw = {**law_params, **(bc_params or {})}
    diff_kw = {**law_params, **(diff_params or {})}
    k_bc = resolve_dust_law(law_bc)(wavelength, **bc_kw)
    k_diff = resolve_dust_law(law_diff)(wavelength, **diff_kw)
    # Optional Lyman-limit clip: zero the curve below ``lyman_cutoff_aa`` (CIGALE
    # parity). ``cutoff_aa=0.0`` is a no-op, so the default leaves the FUV
    # extrapolation in place.
    k_bc = apply_lyman_cutoff(k_bc, wavelength, lyman_cutoff_aa)
    k_diff = apply_lyman_cutoff(k_diff, wavelength, lyman_cutoff_aa)

    log_age = jnp.log10(jnp.maximum(age_grid, 1.0))
    log_t_birth = jnp.log10(t_birth)
    weight = jax.nn.sigmoid(-(log_age - log_t_birth) / transition_width)

    tau_lambda = weight[:, None] * tau_v1 * k_bc[None, :] + tau_v2 * k_diff[None, :]

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)


def two_component_dust_separable(
    wavelength: jnp.ndarray,
    dust_age_weights: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc_fn: Callable,
    law_diff_fn: Callable,
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Optimized two-component dust attenuation with factorized age-independent term.

    Exploits the exponential factorization exp(a + b) = exp(a) · exp(b) to separate
    the diffuse ISM component from the age-dependent outer product. The diffuse
    exponentiation operates on (n_wave,) instead of (n_ages, n_wave), saving one full-grid
    exponential. Accepts pre-resolved law functions to avoid dict lookups in hot code.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    dust_age_weights : array_like, shape (n_ages,)
        Pre-computed sigmoid birth-cloud weights from ``precompute_dust_age_weights``.
        Computed once at Model init and cached.
    tau_v1 : float
        Birth-cloud V-band optical depth. [dimensionless]
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc_fn : Callable
        Pre-resolved birth-cloud attenuation function (e.g., ``resolve_dust_law("calzetti")``).
    law_diff_fn : Callable
        Pre-resolved diffuse ISM attenuation function.
    f_obscuration : float, optional
        Unattenuated sightline fraction. [dimensionless, in [0, 1]] Default: 0.0.
    **law_params
        Keyword arguments passed to both law functions.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission T(λ, t_age) ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Performance**: Reduces memory traffic by ~40% on (n_ages, n_wave) grids
    relative to ``two_component_dust`` because the diffuse exponential is computed
    on (n_wave,) and broadcast rather than materialized as (n_ages, n_wave).
    Significant speedup on CPU; moderate benefit on GPU (memory bandwidth more abundant).

    The transmission factorizes as:

    .. math::

        T(\lambda, t_{\text{age}}) = T_{\rm BC}(\lambda, t_{\text{age}}) \cdot T_{\rm ISM}(\lambda)

    where

    .. math::

        T_{\rm BC}(\lambda, t_{\text{age}}) = f_{\rm obs} + (1 - f_{\rm obs}) \, \exp[-w(t_{\text{age}}) \, \tau_{\rm V,BC} \, k_{\rm BC}(\lambda)]

    .. math::

        T_{\rm ISM}(\lambda) = \exp[-\tau_{\rm V,ISM} \, k_{\rm ISM}(\lambda)]

    The ISM component is computed once on (n_wave,) and then broadcast with the age-dependent
    birth-cloud term, avoiding the full (n_ages, n_wave) grid in intermediate storage.

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    k_bc = law_bc_fn(wavelength, **law_params)
    k_diff = law_diff_fn(wavelength, **law_params)

    # Diffuse ISM: age-independent → (n_wave,) exp instead of (n_age, n_wave)
    diffuse_trans = jnp.exp(-tau_v2 * k_diff)  # (n_wave,)

    # Birth cloud: age-dependent outer product → (n_age, n_wave)
    bc_trans = jnp.exp(-dust_age_weights[:, None] * tau_v1 * k_bc[None, :])

    # Combine: broadcast (n_age, n_wave) * (n_wave,) avoids materializing
    # the full (n_age, n_wave) diffuse array
    transmission = bc_trans * diffuse_trans[None, :]

    return f_obscuration + (1.0 - f_obscuration) * transmission


def two_component_dust_fast(
    wavelengths: jnp.ndarray,
    dust_age_weights: jnp.ndarray,
    tau_v1: float,
    tau_v2: float,
    law_bc: str = "power_law",
    law_diff: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Fast dust attenuation using precomputed age weights.

    Avoids recomputing the birth-cloud age sigmoid every call. Used by
    both the fused kernel (at effective wavelengths) and the exact path
    (at the full wavelength grid).

    The output dtype follows the input ``wavelengths`` dtype, so passing float32
    arrays halves memory traffic on the ``(n_ages, n_wave)`` intermediates
    (~1.6x speedup on CPU). That is a property of this function, not of the
    model: nothing hands it float32 wavelengths unless the whole run is in pure
    float32 (``jax.enable_x64(False)``). In particular
    ``forward_dtype="float32"`` does not — it casts nothing (#1433).

    Parameters
    ----------
    wavelengths : array_like, shape (n_wave,)
        Evaluation wavelengths (rest-frame). [Å] Can be the full
        SSP grid or just the filter effective wavelengths.
    dust_age_weights : array_like, shape (n_ages,)
        Pre-computed sigmoid weights from ``precompute_dust_age_weights``.
        Computed once at Model init.
    tau_v1 : float
        Birth-cloud V-band optical depth. [dimensionless]
    tau_v2 : float
        Diffuse ISM V-band optical depth. [dimensionless]
    law_bc : str
        Attenuation curve name for birth cloud. [dimensionless] Default: "power_law".
        Looked up in ``DUST_LAWS`` registry.
    law_diff : str
        Attenuation curve name for diffuse ISM. Default: "power_law".
    f_obscuration : float
        Fraction of unattenuated sightlines. [dimensionless, in [0, 1]] Default: 0.0 (Lower 2022).
    **law_params
        Passed to curve functions: ``n_slope``, ``dust_bump_strength``,
        ``dust_delta``, ``dust_Rv``, etc.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative attenuation factor in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.
    """
    k_bc = resolve_dust_law(law_bc)(wavelengths, **law_params)
    k_diff = resolve_dust_law(law_diff)(wavelengths, **law_params)

    tau_lambda = dust_age_weights[:, None] * tau_v1 * k_bc[None, :] + tau_v2 * k_diff[None, :]

    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_lambda)


# ── Single-component dust model (uniform screen) ──────────────────


def single_component_dust(
    wavelength: jnp.ndarray,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Single-component (uniform foreground screen) dust attenuation.

    Applies a single attenuation curve at uniform optical depth to all stellar ages.
    Age-independent, enabling factorization out of stellar population integration.
    Simpler but less realistic than two-component models; useful for low-precision fits
    or high-redshift galaxies where birth-cloud/ISM distinction is unresolved.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    tau_v : float
        V-band optical depth at 5500 Å. [dimensionless]
    law : str, optional
        Attenuation curve name, resolved from ``DUST_LAWS`` registry. Default: "power_law".
    f_obscuration : float, optional
        Unattenuated sightline fraction in clumpy geometry (Lower 2022). [dimensionless, in [0, 1]]
        Default: 0.0 (uniform foreground screen).
    **law_params
        Keyword arguments passed to the attenuation curve function
        (e.g., ``n_slope``, ``dust_bump_strength``, ``dust_delta``, ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative transmission T(λ) ∈ [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    The transmission is:

    .. math::

        T(\lambda) = f_{\rm obs} + (1 - f_{\rm obs}) \cdot \exp[-\tau_V \, k(\lambda)]

    where :math:`k(\lambda)` is the normalized attenuation curve with :math:`k(5500 \, \text{\AA}) = 1`,
    :math:`\tau_V` is the V-band optical depth, and :math:`f_{\rm obs}` is the fraction of
    unattenuated sightlines (Lower 2022; default 0 = full screen).

    **Age independence**: Unlike two-component models, there is no age-dependence, so this
    transmission can be factored out of the stellar population age integration, enabling
    faster computation.

    **Geometry**: When :math:`f_{\rm obs} = 0`, this recovers the standard Beer-Lambert
    foreground screen. When :math:`f_{\rm obs} > 0`, it models a clumpy geometry where
    a fraction of photons are unattenuated (Lower 2022).

    References
    ----------
    .. [1] S. Lower et al., "How Well Can We Measure Galaxy Dust Attenuation Curves?
       The Impact of the Assumed Star-dust Geometry Model in SED Fitting,"
       ApJ, 931, 14 (2022). arXiv:2203.00074.
       https://doi.org/10.3847/1538-4357/ac6959
    """
    k = resolve_dust_law(law)(wavelength, **law_params)
    return f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_v * k)


def single_component_dust_fast(
    wavelengths: jnp.ndarray,
    n_ages: int,
    tau_v: float,
    law: str = "power_law",
    f_obscuration: float = 0.0,
    **law_params,
) -> jnp.ndarray:
    r"""Single-component dust attenuation broadcast to (n_ages, n_wave).

    Computes ``exp()`` on the 1-D wavelength grid only, then broadcasts
    to ``(n_ages, n_wave)`` via ``jnp.broadcast_to`` (zero-copy in XLA).
    This is the production path used by the SED pipeline.

    Parameters
    ----------
    wavelengths : array_like, shape (n_wave,)
        Evaluation wavelengths (rest-frame). [Å]
    n_ages : int
        Number of SSP age bins (for output shape). [dimensionless]
    tau_v : float
        V-band optical depth. [dimensionless]
    law : str
        Attenuation curve name (from ``DUST_LAWS`` registry). Default: "power_law".
    f_obscuration : float
        Fraction of unattenuated sightlines. [dimensionless, in [0, 1]] Default: 0.0 (Lower 2022).
    **law_params
        Passed to curve function.

    Returns
    -------
    ndarray, shape (n_ages, n_wave)
        Multiplicative transmission factor in [0, 1]. [dimensionless]
        All age rows are identical (age-independent attenuation).

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Memory efficiency**: Using ``jnp.broadcast_to`` avoids materializing
    the full (n_ages, n_wave) grid in memory; the result is a zero-copy view.
    """
    trans_1d = single_component_dust(
        wavelengths, tau_v=tau_v, law=law, f_obscuration=f_obscuration, **law_params
    )
    return jnp.broadcast_to(trans_1d[None, :], (n_ages, wavelengths.shape[0]))


# ── Witt & Gordon (2000) dust geometry transmission functions ─────
#
# These functions compute the wavelength-dependent transmission T(lambda)
# for different star-dust geometries, given a V-band optical depth tau_V
# and an underlying extinction curve k(lambda).
#
# The key insight from Witt & Gordon (2000, ApJ, 528, 799) is that the
# EFFECTIVE attenuation depends strongly on the spatial distribution of
# dust relative to stars.  A uniform foreground screen (SHELL) produces
# the steepest wavelength dependence; a homogeneous mix (CLOUDY) is
# grayer because high-tau sightlines are self-shielded; a clumpy medium
# (DUSTY) is grayest because photons preferentially escape through
# low-tau channels.
#
# All functions are pure JAX and JIT-compatible.

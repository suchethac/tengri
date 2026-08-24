# SPDX-License-Identifier: BSD-3-Clause
"""Shared emission-component helpers for the SED forward model.

Both the non-fused pipeline (``sed_pipeline.py``) and the fused JIT
kernel (``fused_kernels.py``) call these same functions, guaranteeing
identical physics.  All helpers are **pure functions** taking explicit
arguments (no ``model`` object), so they work inside ``@jax.jit``
closures as well as plain Python.

Each helper computes one emission component and returns the SED
(erg/s/Hz).  Orchestration (branching on ``has_nebular``, component
tracking, wavelength interpolation) stays in the caller.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.physics_constants import C_AA

# Module-level aliases (kept for terse local use)
_C_AA: float = C_AA


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dust attenuation of emission components (nebular / shock)
# ═══════════════════════════════════════════════════════════════════════════


def attenuate_emission(
    sed: jnp.ndarray,
    wave: jnp.ndarray,
    mode: str,
    tau_bc: float,
    tau_diff: float,
    law_bc_fn,
    law_diff_fn,
    *,
    neb_bc_fn=None,
    dust_slope: float = -0.7,
    dust_bump_strength: float = 0.0,
) -> jnp.ndarray:
    """Apply dust attenuation to an emission component's SED.

    Parameters
    ----------
    sed: ndarray, shape (n_wave,)
        Input SED [erg/s/Hz] (before dust).
    wave: ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    mode: str
        Attenuation mode: ``"bc"`` (birth-cloud + diffuse), ``"diff"`` (diffuse only),
        ``"neb"`` (separate BC law + diffuse), or ``"none"`` (no attenuation).
    tau_bc: float
        Birth-cloud V-band optical depth [dimensionless].
    tau_diff: float
        Diffuse V-band optical depth [dimensionless].
    law_bc_fn: callable
        Birth-cloud dust law ``(wave, n_slope, dust_bump_strength) -> k(λ)`` [1/mag].
    law_diff_fn: callable
        Diffuse dust law ``(wave, n_slope, dust_bump_strength) -> k(λ)`` [1/mag].
    neb_bc_fn: callable, optional
        Separate BC law for ``mode="neb"``. Falls back to ``law_bc_fn`` if None.
    dust_slope: float, optional
        Dust law slope parameter. Default -0.7.
    dust_bump_strength: float, optional
        Dust law bump strength parameter. Default 0.0.

    Returns
    -------
    sed_attenuated: ndarray, shape (n_wave,)
        Attenuated SED [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The absorbed luminosity is not computed here; the dust energy balance
    is owned by the dust attenuation components via
    :func:`tengri.forward.energy_balance.bolometric_absorbed` (#922).
    """
    if mode == "none":
        return sed

    dust_kw = {"n_slope": dust_slope, "dust_bump_strength": dust_bump_strength}
    sed_out = sed

    # Birth-cloud attenuation (modes "bc" and "neb").
    # Always compute (exp(-0*k)=1 is a no-op); avoids Python boolean
    # conversion on traced tau_bc which fails inside @jax.jit.
    if mode in ("bc", "neb"):
        bc_fn = neb_bc_fn if (mode == "neb" and neb_bc_fn is not None) else law_bc_fn
        if bc_fn is not None:
            k_bc = bc_fn(wave, **dust_kw)
            sed_out = sed_out * jnp.exp(-tau_bc * k_bc)

    # Diffuse ISM attenuation (all modes except "none").
    # Same: always compute, let XLA optimize exp(-0*k)=1.
    if law_diff_fn is not None:
        k_diff = law_diff_fn(wave, **dust_kw)
        sed_out = sed_out * jnp.exp(-tau_diff * k_diff)

    return sed_out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Shock emission
# ═══════════════════════════════════════════════════════════════════════════


def shock_emission(
    wave: jnp.ndarray,
    sed_so_far: jnp.ndarray,
    shock_frac: float,
    shock_velocity: float = 300.0,
    shock_log_density: float = 0.0,
    shock_b_over_sqrt_n: float = 1.0,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
) -> jnp.ndarray:
    """Synthesize shock emission SED (MAPPINGS V).

    Returns raw shock SED before dust attenuation.

    Parameters
    ----------
    wave: ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    sed_so_far: ndarray, shape (n_wave,)
        Current cumulative SED [erg/s/Hz] for bolometric luminosity estimation.
    shock_frac: float
        Fraction of Halpha luminosity channeled into shocks [dimensionless].
    shock_velocity: float, optional
        Shock velocity [km/s]. Default 300.0.
    shock_log_density: float, optional
        Log10 of electron density [cm^-3]. Default 0.0.
    shock_b_over_sqrt_n: float, optional
        Magnetic parameter B/sqrt(n). Default 1.0.
    shock_abundance: str, optional
        Abundance set ("solar", etc.). Default "solar".
    shock_component: str, optional
        Component to return ("combined", etc.). Default "combined".

    Returns
    -------
    ndarray, shape (n_wave,)
        Shock emission SED [erg/s/Hz] before dust attenuation.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    """
    from tengri.components.nebular.shock import compute_shock_sed

    nu = _C_AA / wave
    l_bol = -jnp.trapezoid(sed_so_far, nu)
    # Order-of-magnitude approximation: L(Hα) ~ 1e-3 × L_bol for a star-
    # forming galaxy. Used only to set the *normalization* of the shock
    # template, the resulting shock SED is then scaled by ``shock_frac``
    # at the call site. Magnitude not validity-ranged against a paper;
    # flagged for replacement with the case-B prediction from the SFH.
    l_halpha_approx = jnp.maximum(l_bol * 1e-3, 1e-30)
    l_shock_halpha = shock_frac * l_halpha_approx

    return compute_shock_sed(
        wave,
        shock_velocity,
        l_shock_halpha,
        shock_log_density=shock_log_density,
        shock_b_over_sqrt_n=shock_b_over_sqrt_n,
        shock_abundance=shock_abundance,
        shock_component=shock_component,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Component emission (nebular / AGN / dust-IR / radio / X-ray)
# ═══════════════════════════════════════════════════════════════════════════


# These five helpers -- ``nebular_emission``, ``agn_emission``,
# ``dust_ir_emission``, ``radio_emission`` and ``xray_emission`` -- were removed.
# Each was a second implementation of a component that the ``components/``
# packages already own (``components.nebular``, ``.agn``, ``.dust``, ``.radio``,
# ``.xray``), and nothing had called them since the component refactor. Two of
# them had already drifted from the live physics: the AGN copy still guarded
# polar dust with a branch (``agn_polar_ebv > 0``) that the live SMC formulation
# does not need -- ``exp(-0.921 * ebv * ...)`` is the identity at ``ebv = 0`` and
# stays JIT-friendly -- and the dust-IR copy forwarded ``dust_alpha_dl14`` under
# the old prefixed spelling. A source-text wiring test matched those dead copies
# rather than the live components, so the drift stayed invisible. Import from
# ``components/`` instead; do not reintroduce a wrapper here (same lesson as the
# IGM shim below).

# ═══════════════════════════════════════════════════════════════════════════
# 4. IGM absorption
# ═══════════════════════════════════════════════════════════════════════════


# ``igm_absorption`` now lives solely in ``tengri.components.igm.igm``, the
# single source of truth for the mean-IGM model dispatch (inoue / madau /
# meiksin06 / asada25) plus the patchy and DLA modifiers (#932). Import it from
# there; this module deliberately no longer defines a wrapper copy (an earlier
# shim here had a stale signature and dropped the new use_dla/dla_* kwargs,
# crashing every IGM-enabled predict_obs_sed call).

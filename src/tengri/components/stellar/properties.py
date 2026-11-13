# SPDX-License-Identifier: BSD-3-Clause
"""Derived stellar properties and their registration in the property catalog.

Each property is a pure function of the :class:`ForwardState` produced by
:class:`StellarSEDComponent`: star-formation history and timing, the SED and
dust energy budget, and ionizing photon output. Importing this module registers
every table with :func:`tengri.forward.properties.register_properties`.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.parameters.timing import timing_frac_params


def _time_weighted_sfr(
    sfr_history: jnp.ndarray,
    sfh_lbt_grid: jnp.ndarray,
    window_yr: float,
) -> jnp.ndarray:
    """Time-weighted SFR over the last ``window_yr`` years.

    Thin wrapper around the canonical helper in
    :mod:`tengri.components.stellar.sfh.sfr_window`. Kept for
    StellarSEDComponent's existing call sites; new code should import
    :func:`time_weighted_sfr` from there directly.
    """
    from tengri.components.stellar.sfh.sfr_window import time_weighted_sfr

    return time_weighted_sfr(sfr_history, sfh_lbt_grid, window_yr)


# ─────────────────────────────────────────────────────────────────────
# Property registration (Phase 1A + Phase 1B)
# ─────────────────────────────────────────────────────────────────────
#
# StellarSEDComponent does NOT inherit from SEDModelComponent, so
# __init_subclass__ auto-collection is unavailable. Properties are
# registered manually at module initialization.

_TINY = 1e-30  # Floor for safe division

# ─ Phase 1A: SFH group ─


def _stellar_mass_fn(state, params):
    """Total stellar mass **formed** by the SFH [Msun].

    Notes
    -----
    This is the time-integral of the SFH, ``10**log_mstar_formed``: *not* the
    mass still alive today. Stellar evolution returns a third to a half of it to
    the ISM, so it runs 1.5-1.9x above
    :func:`_stellar_mass_surviving_fn` on ordinary populations.

    The two names look interchangeable and are not; whichever one a paper
    quotes has to be the one it means.
    """
    log_mstar_formed = jnp.asarray(state.derived["log_mstar_formed"])
    return jnp.power(10.0, log_mstar_formed)


def _stellar_mass_surviving_fn(state, params):
    """Total surviving stellar mass [Msun]."""
    log_mstar = jnp.asarray(state.derived["log_mstar_surviving"])
    return jnp.power(10.0, log_mstar)


def _sfr_100myr_fn(state, params):
    """Star formation rate averaged over past 100 Myr [Msun/yr]."""
    return jnp.asarray(state.derived["sfr_100myr"])


def _sfr_10myr_fn(state, params):
    """Star formation rate averaged over past 10 Myr [Msun/yr]."""
    return jnp.asarray(state.derived["sfr_10myr"])


def _ssfr_fn(state, params):
    """Specific star formation rate (SFR / surviving stellar mass) [1/yr].

    Notes
    -----
    Reads ``log_mstar``, **not** ``log_mstar_surviving``; and that asymmetry with
    :func:`_stellar_mass_surviving_fn` is deliberate, not an oversight. When the
    SSP grid carries no mass-remaining table, "how much mass survives" has no
    answer and the property says NaN; but sSFR is still a meaningful number
    against the formed mass, so it falls back rather than going dark. That is
    exactly what the method this replaces did::

        mass_for_ssfr = jnp.where(jnp.isnan(mass_surviving), mass_formed, mass_surviving)

    and the deprecation shim must stay bit-exact with it (#1049, contract §6).
    """
    log_mstar = jnp.asarray(state.derived["log_mstar"])
    stellar_mass_surviving = jnp.power(10.0, log_mstar)
    sfr_100myr = jnp.asarray(state.derived["sfr_100myr"])
    return sfr_100myr / jnp.maximum(stellar_mass_surviving, _TINY)


def _mass_weighted_age_gyr_fn(state, params):
    r"""Mass-weighted mean age of the stellar population [Gyr].

    .. math::

        t_\mathrm{mw} = \frac{\sum_i w_i\, t_i}{\sum_i w_i}

    :math:`w_i` are the CSP mass weights per SSP age bin [Msun] and :math:`t_i`
    the SSP isochrone ages [yr].

    Notes
    -----
    **JIT-compatible**: yes.

    Weighted on the **SSP age grid** (the stars that actually exist in the SED),
    not by integrating the raw SFH grid. The two are not equivalent: an SFH can
    place stellar mass at lookback times beyond the age of the universe at the
    model's redshift (the orchestrator already warns when it does), and the SED
    truncates that mass while a raw-SFH integral would still count it. Weighting
    the population that the SED actually contains keeps this quantity consistent
    with the spectrum it accompanies; the two definitions differed by ~4.6% and
    were both live under this one name until #1131.

    Shares :func:`~tengri.utils.sed_quantities.compute_mass_weighted_age` with
    ``predict_sfh_quantities`` and ``Prediction.sfh``: one implementation, so
    they cannot drift apart again.
    """
    from tengri.utils.sed_quantities import compute_mass_weighted_age

    weights = jnp.asarray(state.derived["age_weights"])
    ssp_ages_yr = jnp.asarray(state.derived["ssp_ages_yr"])
    return compute_mass_weighted_age(weights, ssp_ages_yr)


def _mass_weighted_metallicity_fn(state, params):
    r"""Mass-weighted mean metallicity [dex, log10(Z/Zsun)].

    .. math::

        \langle \log_{10} Z/Z_\odot \rangle_M =
            \frac{\sum_i \log_{10} Z_i \, m_i}{\sum_i m_i} - \log_{10} Z_\odot

    :math:`m_i = \mathrm{SFR}_i \, \Delta t_i` are the per-bin masses [Msun]
    and :math:`\log_{10} Z_i` the SFH's metallicity history, which is stored
    **absolute** because that is the SSP grid's convention.

    Notes
    -----
    **JIT-compatible**: yes.

    The subtraction is the whole point. Without it this returned absolute
    log10(Z) under a docstring, a registry entry and a published table that all
    said log10(Z/Zsun): 1.85 dex, a factor of 70 in Z. The decisive test is
    not the wording but ``met_mode='delta'``: one metallicity, so a
    mass-weighted mean over it must reproduce the input, and it did not.
    """
    from tengri.utils.conversions import log_z_abs_to_logzsol

    sfh_lbt = jnp.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr_history = jnp.asarray(state.derived["sfr_history"])
    log_z_history = jnp.asarray(state.derived["log_metallicity_history"])
    bin_widths = jnp.gradient(sfh_lbt)
    bin_mass = jnp.maximum(sfr_history * bin_widths, 0.0)
    bin_mass_total = jnp.maximum(jnp.sum(bin_mass), _TINY)
    return log_z_abs_to_logzsol(jnp.sum(log_z_history * bin_mass) / bin_mass_total)


# ─ Phase 1B: SED group ─


def _l_bol_fn(state, params):
    """Bolometric luminosity [Lsun]."""
    from tengri.utils.sed_quantities import compute_bolometric_luminosity

    # Delegate to the canonical reduction rather than re-inlining it: that helper
    # folds 1/L_sun into the integral so the ~1e43 erg/s value is never formed
    # (float32-safe, #1206). ``abs`` preserves the original sign convention;
    # wave ascends, so nu descends and the signed area is negative.
    return jnp.abs(compute_bolometric_luminosity(state.sed_intrinsic, state.wave))


def _l_tir_fn(state, params):
    """Infrared luminosity (8–1000 µm) [Lsun]."""
    from tengri.utils.sed_quantities import compute_l_tir

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_l_tir(sed, wave)


def _l_dust_absorbed_fn(state, params):
    """Dust-absorbed luminosity [Lsun].

    Prefers the ``log_L_absorbed`` companion: the linear ``L_absorbed`` is
    ~3.6e43 erg/s and is ``inf`` in float32, while this answer (~9.5e9 Lsun)
    is comfortably representable there (#1837). ``log_L_absorbed``, not
    ``log_L_ir``: the two agree only when ``dust_eta_balance == 1`` and no
    ``dust_log_L_ir`` override is declared (#1837/#2187-series split).
    """
    from tengri.utils.sed_quantities import derived_luminosity_lsun

    return derived_luminosity_lsun(state.derived, "L_absorbed", "log_L_absorbed")


def _irx_fn(state, params):
    r"""Infrared excess against the **monochromatic 1600 A** UV luminosity [dex].

    .. math::

        \mathrm{IRX} = \log_{10}\!\left(
            \frac{L_\mathrm{TIR}}{(\nu L_\nu)_{1600\,\mathrm{A}}}\right)

    :math:`L_\mathrm{TIR}` is the 8-1000 um dust luminosity [erg/s] and
    :math:`(\nu L_\nu)_{1600}` the monochromatic UV luminosity at rest-frame
    1600 A [erg/s]. This is the anchor of the IRX-beta relation (Meurer et al.
    1999 [1]_) and the definition tengri has reported all along.

    See :func:`_irx_fuv_fn` for the band-averaged FUV variant: the two anchors
    differ by ~0.12 dex and are not interchangeable.

    Notes
    -----
    **JIT-compatible**: yes.

    References
    ----------
    .. [1] Meurer, G. R., Heckman, T. M., & Calzetti, D. 1999, "Dust Absorption
       and the Ultraviolet Luminosity Density at z ~ 3 as Calibrated by Local
       Starburst Galaxies", ApJ, 521, 64. doi:10.1086/307523
    """
    from tengri.utils.sed_quantities import (
        compute_irx,
        compute_l_tir,
        compute_log_uv_luminosity_1600,
    )

    sed = state.sed_intrinsic
    wave = state.wave
    l_tir = compute_l_tir(sed, wave)
    # Log-domain UV anchor: nu*L_nu(1600 A) is ~5e42 erg/s and is not
    # float32-representable, so the linear route returned NaN (#1837).
    return compute_irx(l_tir, log_l_uv_erg=compute_log_uv_luminosity_1600(sed, wave))


def _irx_fuv_fn(state, params):
    r"""Infrared excess against the **band-averaged FUV** luminosity [dex].

    .. math::

        \mathrm{IRX_{FUV}} = \log_{10}\!\left(
            \frac{L_\mathrm{TIR}}{\nu_{1500}\,\langle L_\nu\rangle_\mathrm{FUV}}\right)

    :math:`\langle L_\nu \rangle_\mathrm{FUV}` is the mean :math:`L_\nu` over
    1000-1700 A [erg/s/Hz]; a GALEX-FUV-like window rather than a single
    wavelength; and :math:`\nu_{1500} = c / 1500\,\mathrm{A}` the pivot
    frequency [Hz].

    Notes
    -----
    **JIT-compatible**: yes.

    The pivot frequency takes ``C_AA`` from
    :mod:`tengri.utils.physics_constants`. It previously used a hardcoded
    ``2.998e15`` (the speed of light 1000x too small in [A/s]) which inflated
    every reported IRX by exactly :math:`\log_{10}(1000) = 3` dex (#1131).
    """
    import math

    from tengri.utils.physics_constants import C_AA
    from tengri.utils.scale import log10_magnitude
    from tengri.utils.sed_quantities import compute_fuv_flux, compute_irx, compute_l_tir

    sed = state.sed_intrinsic
    wave = state.wave
    l_tir = compute_l_tir(sed, wave)
    fuv = compute_fuv_flux(sed, wave)
    # The pivot frequency stays in the exponent: ``fuv * C_AA / 1500`` is
    # ~4e42 erg/s and overflows float32, though IRX_FUV is ~-0.7 dex (#1837).
    log_l_uv = log10_magnitude(fuv) + math.log10(C_AA / 1500.0)
    return compute_irx(l_tir, log_l_uv_erg=log_l_uv)


def _uv_slope_beta_fn(state, params):
    """UV slope (β in L_ν ∝ ν^β) [dimensionless]."""
    from tengri.utils.sed_quantities import compute_uv_slope_beta

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_uv_slope_beta(sed, wave)


def _dn4000_fn(state, params):
    """D n4000 break diagnostic [dimensionless]."""
    from tengri.utils.sed_quantities import compute_dn4000

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_dn4000(sed, wave)


def _balmer_break_fn(state, params):
    """Balmer break diagnostic [dimensionless]."""
    from tengri.utils.sed_quantities import compute_balmer_break

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_balmer_break(sed, wave)


def _m_uv_fn(state, params):
    """UV absolute magnitude (1600 Å) [AB mag]."""
    from tengri.utils.sed_quantities import compute_m_uv

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_m_uv(sed, wave)


def _fuv_flux_fn(state, params):
    """FUV flux (1000–1700 Å) [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_fuv_flux

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_fuv_flux(sed, wave)


def _nuv_flux_fn(state, params):
    """NUV flux (1700–3000 Å) [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_nuv_flux

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_nuv_flux(sed, wave)


def _fuv_flux_intrinsic_fn(state, params):
    """Intrinsic FUV flux before dust attenuation [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_fuv_flux

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    if "lnu_age" in derived:
        sed_stellar_intrinsic = jnp.sum(jnp.asarray(derived["lnu_age"]), axis=0)
        wave = state.wave
        return compute_fuv_flux(sed_stellar_intrinsic, wave)
    else:
        return nan_scalar


def _nuv_flux_intrinsic_fn(state, params):
    """Intrinsic NUV flux before dust attenuation [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_nuv_flux

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    if "lnu_age" in derived:
        sed_stellar_intrinsic = jnp.sum(jnp.asarray(derived["lnu_age"]), axis=0)
        wave = state.wave
        return compute_nuv_flux(sed_stellar_intrinsic, wave)
    else:
        return nan_scalar


def _rest_uv_color_fn(state, params):
    """Rest-frame U-V color [AB mag]: top-hats 3200–3900 Å minus 5000–5800 Å on L_nu."""
    from tengri.utils.sed_quantities import compute_rest_uv_color

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_rest_uv_color(sed, wave)


# ─ Phase 1B: Luminosity-weighted SFH properties ─


def _luminosity_weighted_age_gyr_fn(state, params):
    """Luminosity-weighted mean age [Gyr]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "L_age" in derived and "ssp_ages_yr" in derived:
        from tengri.utils.sed_quantities import derived_weights_peak_relative

        # Peak-relative weights: the common factor cancels in the mean, and raw
        # ``L_age`` is ~3.3e42 erg/s, 85 of 93 bins ``inf`` in float32 (#1837).
        L_age = derived_weights_peak_relative(derived, "L_age", "log_L_age")
        ssp_ages_yr = jnp.asarray(derived["ssp_ages_yr"])
        L_total = jnp.maximum(jnp.sum(L_age), _TINY)
        lw_age_yr = jnp.sum(ssp_ages_yr * L_age) / L_total
        return lw_age_yr / 1e9
    else:
        return nan_scalar


def _luminosity_weighted_metallicity_fn(state, params):
    """Luminosity-weighted mean metallicity [dex, log10(Z/Zsun)].

    Notes
    -----
    **JIT-compatible**: yes.

    Weighted on the SSP age grid, then converted out of the grid's absolute
    log10(Z): see :func:`_mass_weighted_metallicity_fn` for why the
    subtraction is not cosmetic.
    """
    from tengri.utils.conversions import log_z_abs_to_logzsol

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "L_age" in derived and "ssp_ages_yr" in derived:
        from tengri.utils.sed_quantities import derived_weights_peak_relative

        # See _luminosity_weighted_age_gyr_fn: scale-free weights (#1837).
        L_age = derived_weights_peak_relative(derived, "L_age", "log_L_age")
        ssp_ages_yr = jnp.asarray(derived["ssp_ages_yr"])
        L_total = jnp.maximum(jnp.sum(L_age), _TINY)

        if "log_metallicity_history" in derived and "sfh_grid_lbt_yr" in derived:
            lz_per_ssp = jnp.interp(
                ssp_ages_yr,
                jnp.asarray(derived["sfh_grid_lbt_yr"]),
                jnp.asarray(derived["log_metallicity_history"]),
            )
            return log_z_abs_to_logzsol(jnp.sum(lz_per_ssp * L_age) / L_total)
        else:
            return nan_scalar
    else:
        return nan_scalar


# ─ Phase 1B: Ionizing group ─


def _log_q_h_fn(state, params):
    """log10 ionizing photon production rate [dex re photons/s]: float32-safe."""
    derived = state.derived
    log_nion = derived.get("log_nion")
    return jnp.asarray(log_nion) if log_nion is not None else jnp.asarray(jnp.nan)


def _xi_ion_fn(state, params):
    """Ionizing photon efficiency [Hz/erg]."""
    from tengri.utils.sed_quantities import compute_xi_ion_from_log_qh

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    sed = state.sed_intrinsic
    if sed is None:
        return nan_scalar
    else:
        log_nion = jnp.asarray(derived.get("log_nion", -jnp.inf))
        return compute_xi_ion_from_log_qh(log_nion, sed, state.wave)


# Register properties in the global registry
from tengri.forward.properties import Property, register_properties

_SFH_PROPERTIES = {
    "stellar_mass": Property(
        units="Msun",
        group="sfh",
        doc="Total stellar mass formed by the SFH: its time-integral, "
        "1.5-1.9x above stellar_mass_surviving",
        fn=_stellar_mass_fn,
    ),
    "stellar_mass_surviving": Property(
        units="Msun",
        group="sfh",
        doc="Total surviving stellar mass",
        fn=_stellar_mass_surviving_fn,
    ),
    "sfr_100myr": Property(
        units="Msun/yr",
        group="sfh",
        doc="Star formation rate averaged over past 100 Myr",
        fn=_sfr_100myr_fn,
    ),
    "sfr_10myr": Property(
        units="Msun/yr",
        group="sfh",
        doc="Star formation rate averaged over past 10 Myr",
        fn=_sfr_10myr_fn,
    ),
    "ssfr": Property(
        units="1/yr",
        group="sfh",
        doc="Specific star formation rate (sfr_100myr / stellar_mass_surviving; "
        "falls back to the formed mass when the SSP has no mass-remaining table)",
        fn=_ssfr_fn,
    ),
    "mass_weighted_age_gyr": Property(
        units="Gyr",
        group="sfh",
        doc="Mass-weighted mean age of stellar population",
        fn=_mass_weighted_age_gyr_fn,
    ),
    "mass_weighted_metallicity": Property(
        units="dex",
        group="sfh",
        doc="Mass-weighted mean metallicity (log10 Z/Zsun)",
        fn=_mass_weighted_metallicity_fn,
    ),
    "luminosity_weighted_age_gyr": Property(
        units="Gyr",
        group="sfh",
        doc="Luminosity-weighted mean age of stellar population",
        fn=_luminosity_weighted_age_gyr_fn,
    ),
    "luminosity_weighted_metallicity": Property(
        units="dex",
        group="sfh",
        doc="Luminosity-weighted mean metallicity (log10 Z/Zsun)",
        fn=_luminosity_weighted_metallicity_fn,
    ),
}

_SED_PROPERTIES = {
    "l_bol": Property(
        units="Lsun",
        group="sed",
        doc="Bolometric luminosity",
        fn=_l_bol_fn,
    ),
    "l_tir": Property(
        units="Lsun",
        group="sed",
        doc="Infrared luminosity (8–1000 µm)",
        fn=_l_tir_fn,
    ),
    "l_dust_absorbed": Property(
        units="Lsun",
        group="sed",
        doc="Dust-absorbed luminosity",
        fn=_l_dust_absorbed_fn,
    ),
    "irx": Property(
        units="dex",
        group="sed",
        doc="Infrared excess, log10(L_TIR / nu*L_nu at 1600 A): the Meurer+99 IRX-beta anchor",
        fn=_irx_fn,
    ),
    "irx_fuv": Property(
        units="dex",
        group="sed",
        doc="Infrared excess against the band-averaged FUV (1000-1700 A), pivoted at 1500 A",
        fn=_irx_fuv_fn,
    ),
    "uv_slope_beta": Property(
        units="",
        group="sed",
        doc="UV slope (β in L_ν ∝ ν^β)",
        fn=_uv_slope_beta_fn,
    ),
    "dn4000": Property(
        units="",
        group="sed",
        doc="D n4000 break diagnostic",
        fn=_dn4000_fn,
    ),
    "balmer_break": Property(
        units="",
        group="sed",
        doc="Balmer break diagnostic",
        fn=_balmer_break_fn,
    ),
    "m_uv": Property(
        units="AB mag",
        group="sed",
        doc="UV absolute magnitude (1600 Å)",
        fn=_m_uv_fn,
    ),
    "fuv_flux": Property(
        units="erg/s/Hz",
        group="sed",
        doc="FUV flux (1000–1700 Å)",
        fn=_fuv_flux_fn,
    ),
    "nuv_flux": Property(
        units="erg/s/Hz",
        group="sed",
        doc="NUV flux (1700–3000 Å)",
        fn=_nuv_flux_fn,
    ),
    "fuv_flux_intrinsic": Property(
        units="erg/s/Hz",
        group="sed",
        doc="Intrinsic FUV flux before dust attenuation",
        fn=_fuv_flux_intrinsic_fn,
    ),
    "nuv_flux_intrinsic": Property(
        units="erg/s/Hz",
        group="sed",
        doc="Intrinsic NUV flux before dust attenuation",
        fn=_nuv_flux_intrinsic_fn,
    ),
    "rest_uv_color": Property(
        units="AB mag",
        group="sed",
        doc="Rest-frame U-V color [AB mag]: top-hats 3200–3900 Å minus 5000–5800 Å",
        fn=_rest_uv_color_fn,
    ),
}

_IONIZING_PROPERTIES = {
    # ``q_h`` (linear photons/s, ~1e56) is retired with no alias (#1206 §C): it
    # overflows float32 at every physical ionizing rate, including zero SFR.
    # ``log_q_h`` is the sole surviving form; ``pred.q_h`` and
    # ``predict_properties(..., names=("q_h",))`` raise ``KeyError`` naming it.
    "log_q_h": Property(
        units="dex",
        group="ionizing",
        doc=(
            "log10(ionizing photon production rate / (photons/s)). "
            "`q_h [photons/s] = 10**log_q_h`."
        ),
        fn=_log_q_h_fn,
    ),
    "xi_ion": Property(
        units="Hz/erg",
        group="ionizing",
        doc="Ionizing photon efficiency",
        fn=_xi_ion_fn,
    ),
}


def _timing_gyr_fn(gyr_name):
    """Property function reading one resolved SFH timing in Gyr (#2748).

    NaN when the active SFH has no timing of that name, so one catalog entry
    per ``*_gyr`` name serves every SFH family.
    """

    def fn(state, params):
        timings = state.derived.get("sfh_timing_gyr")
        if timings is None or gyr_name not in timings:
            return jnp.asarray(jnp.nan)
        return jnp.asarray(timings[gyr_name])

    return fn


_SFH_TIMING_PROPERTIES = {
    gyr_name: Property(
        units="Gyr",
        group="sfh",
        doc=f"Resolved {gyr_name} of the active SFH: frac * age(z) when its fraction "
        "is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing",
        fn=_timing_gyr_fn(gyr_name),
    )
    for gyr_name in sorted(set(timing_frac_params().values()))
}

register_properties("stellar", _SFH_PROPERTIES)
register_properties("stellar", _SED_PROPERTIES)
register_properties("stellar", _IONIZING_PROPERTIES)
register_properties("stellar", _SFH_TIMING_PROPERTIES)

del (
    Property,
    register_properties,
    _SFH_PROPERTIES,
    _SFH_TIMING_PROPERTIES,
    _SED_PROPERTIES,
    _IONIZING_PROPERTIES,
)

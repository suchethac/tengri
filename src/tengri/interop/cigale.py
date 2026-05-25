# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE-equivalence helpers (#357).

The CIGALE reproduction notebook (``reproduction/cigale/01_cigale.py``)
runs a wavelength-resolved consistency audit using only the public
tengri API and a matched CIGALE chain. Two convention mismatches show
up systematically:

1. **SFH normalisation**: CIGALE's
   ``sfhdelayed(tau_main=..., age_main=..., normalise=True)`` integrates
   to exactly 1 M_sun formed. tengri's :func:`delayed_exponential`
   (and :func:`declining_exponential`) instead expose ``log_peak_sfr``
   — the *peak* SFR, not the total mass formed. Users hand-tune
   ``log_peak_sfr`` to "roughly match", which is the dominant ~30 %
   normalisation gap in the audit table.

2. **Dust attenuation**: CIGALE's
   ``dustatt_modified_starburst(E_BV_lines=0.3)`` is a single-knob
   modified-Calzetti parameterisation where lines see the full
   ``E(B-V)_lines`` and the stellar continuum sees a fraction of it.
   tengri's :func:`two_component(tau_bc, tau_diff)` is a
   Charlot-Fall pair parameterised in optical-depth-at-V units.
   The mapping is one-to-one but the conventions are different.

Both helpers below let a notebook author pick CIGALE-equivalent
inputs without re-deriving the algebra each time.
"""

from __future__ import annotations

import math

__all__ = [
    "CIGALE_CALZETTI_AV_OVER_EBV",
    "CIGALE_MODIFIED_STARBURST_EBV_RATIO",
    "cigale_ebv_lines_to_tau",
    "log_peak_sfr_for_mass_formed",
]


#: Calzetti+2000 ``R_V = A_V / E(B-V) = 4.05`` for the stellar continuum.
#: Used by ``dustatt_modified_starburst`` in CIGALE.
CIGALE_CALZETTI_AV_OVER_EBV: float = 4.05

#: Default CIGALE ratio ``E(B-V)_continuum / E(B-V)_lines = 0.44`` (Calzetti+2000).
#: ``dustatt_modified_starburst`` divides the line attenuation between BC
#: and diffuse via this same ratio: the continuum sees 0.44 * E(B-V)_lines.
CIGALE_MODIFIED_STARBURST_EBV_RATIO: float = 0.44


# ----------------------------------------------------------------------
# SFH normalisation
# ----------------------------------------------------------------------


def _delayed_exponential_mass(tau_yr: float, age_yr: float) -> float:
    r"""Mass formed by ``delayed_exponential`` per unit ``peak_sfr``.

    .. math::

        \int_0^\mathrm{age}
            (t/\tau) \, e^{-t/\tau + 1} \, dt
        \;=\; e\,\tau\,
              \bigl[1 - e^{-\mathrm{age}/\tau}\,(1 + \mathrm{age}/\tau)\bigr]

    Units: ``[s_per_yr · yr] = yr`` per ``Msun/yr`` peak → Msun.
    """
    x = age_yr / tau_yr
    return math.e * tau_yr * (1.0 - math.exp(-x) * (1.0 + x))


def _declining_exponential_mass(tau_yr: float, age_yr: float) -> float:
    r"""Mass formed by ``declining_exponential`` per unit ``peak_sfr``.

    .. math::

        \int_0^\mathrm{age}
            e^{-(\mathrm{age}-t_{\mathrm{lb}})/\tau} \, dt_{\mathrm{lb}}
        \;=\; \tau \, \bigl[1 - e^{-\mathrm{age}/\tau}\bigr]
    """
    return tau_yr * (1.0 - math.exp(-age_yr / tau_yr))


def _delayed_tau_mass(tau_yr: float, age_yr: float) -> float:
    r"""Mass formed by ``delayed_tau`` per unit ``norm``.

    .. math::

        \int_0^\mathrm{age} t\,e^{-t/\tau}\,dt
        \;=\; \tau^2 \,
              \bigl[1 - e^{-\mathrm{age}/\tau}\,(1 + \mathrm{age}/\tau)\bigr]
    """
    x = age_yr / tau_yr
    return tau_yr * tau_yr * (1.0 - math.exp(-x) * (1.0 + x))


_SFH_MASS_PER_AMPLITUDE: dict[str, callable] = {
    "delayed_exponential": _delayed_exponential_mass,
    "dexp": _delayed_exponential_mass,
    "declining_exponential": _declining_exponential_mass,
    "tau": _declining_exponential_mass,
    "delayed_tau": _delayed_tau_mass,
}


def log_peak_sfr_for_mass_formed(
    model: str,
    mass_formed_msun: float,
    *,
    tau_gyr: float,
    age_gyr: float,
) -> float:
    r"""Return ``log_peak_sfr`` for a target total stellar mass formed.

    Matches CIGALE's ``sfhdelayed(..., normalise=True)`` convention
    (M_formed = 1 M_sun by default). The returned value plugs straight
    into :func:`tengri.delayed_exponential` /
    :func:`tengri.declining_exponential` /
    :func:`tengri.delayed_tau` so the integrated SFH equals
    ``mass_formed_msun``.

    Parameters
    ----------
    model : {'delayed_exponential', 'dexp', 'declining_exponential', 'tau', 'delayed_tau'}
        Tengri SFH variant. ``'dexp'`` and ``'tau'`` are short aliases
        for ``'delayed_exponential'`` and ``'declining_exponential'``
        respectively.
    mass_formed_msun : float
        Target total mass formed over ``[0, age]``. [Msun]
    tau_gyr : float
        SFH e-folding timescale. [Gyr]
    age_gyr : float
        Galaxy age = lookback time of formation. [Gyr]

    Returns
    -------
    float
        ``log10(peak_sfr_msun_per_yr)`` such that the SFH integrates
        to ``mass_formed_msun``. For ``'delayed_tau'`` this is the
        log of the linear ``norm`` parameter (not a peak SFR).

    Raises
    ------
    ValueError
        If ``model`` is unknown, ``tau_gyr <= 0``, ``age_gyr <= 0``,
        or ``mass_formed_msun <= 0``.

    Examples
    --------
    Match CIGALE's ``sfhdelayed(tau_main=1000, age_main=5000, normalise=True)``:

    >>> from tengri.interop.cigale import log_peak_sfr_for_mass_formed
    >>> log_peak_sfr_for_mass_formed(
    ...     "dexp", mass_formed_msun=1.0, tau_gyr=1.0, age_gyr=5.0
    ... )  # doctest: +ELLIPSIS
    -9.4...

    Notes
    -----
    Derivation: each SFH ``SFR(t) = peak_sfr · S(t; tau, age)`` where
    ``S`` is a peak-normalised shape. Then
    ``M_formed = peak_sfr · ∫ S(t) dt``. The integrals are closed-form
    (see ``_*_mass`` helpers in this module).

    The returned ``log_peak_sfr`` is exact up to floating-point round-off
    — no calibration drift, no numerical integration tolerance.
    """
    if mass_formed_msun <= 0:
        raise ValueError(f"mass_formed_msun must be > 0, got {mass_formed_msun}")
    if tau_gyr <= 0:
        raise ValueError(f"tau_gyr must be > 0, got {tau_gyr}")
    if age_gyr <= 0:
        raise ValueError(f"age_gyr must be > 0, got {age_gyr}")
    try:
        mass_per_amp = _SFH_MASS_PER_AMPLITUDE[model]
    except KeyError as exc:
        valid = sorted(set(_SFH_MASS_PER_AMPLITUDE))
        raise ValueError(f"Unknown SFH model {model!r}. Valid: {', '.join(valid)}.") from exc

    integral_yr = mass_per_amp(tau_gyr * 1e9, age_gyr * 1e9)
    if integral_yr <= 0:
        # Defensive: would require age=0 after the input guard, but be
        # explicit since 10^-inf would silently propagate.
        raise ValueError(
            f"SFH integral is non-positive ({integral_yr:.3e}); check tau/age inputs."
        )
    return math.log10(mass_formed_msun / integral_yr)


# ----------------------------------------------------------------------
# Dust attenuation
# ----------------------------------------------------------------------


def cigale_ebv_lines_to_tau(
    ebv_lines: float,
    *,
    av_over_ebv: float = CIGALE_CALZETTI_AV_OVER_EBV,
    ebv_ratio: float = CIGALE_MODIFIED_STARBURST_EBV_RATIO,
) -> dict[str, float]:
    r"""Map ``E(B-V)_lines`` (CIGALE) onto ``(tau_bc, tau_diff)`` (tengri).

    CIGALE's ``dustatt_modified_starburst(E_BV_lines=...)`` is a
    one-knob modified-Calzetti model where:

    - Nebular lines see the full ``E(B-V)_lines``.
    - Stellar continuum sees ``E(B-V)_continuum = ebv_ratio · E(B-V)_lines``
      (Calzetti+2000 default ``ebv_ratio = 0.44``).
    - Both are converted to optical depth at V via
      ``tau_V = A_V / (2.5 / ln 10) = R_V · E(B-V) · ln 10 / 2.5``.

    Tengri's :func:`two_component` uses Charlot & Fall 2000:

    - ``tau_diff`` applies to the entire stellar SED — equivalent to
      the CIGALE continuum attenuation.
    - ``tau_bc`` applies *additionally* to the young (HII-region) light
      that emits the lines — i.e. the *extra* attenuation lines see
      beyond what the continuum sees.

    The conversion is therefore:

    .. math::

        \tau_{\mathrm{diff}}
            \;=\; \frac{\ln 10}{2.5}\, R_V \,
                  (\mathrm{ebv\_ratio}) \, E(B{-}V)_{\mathrm{lines}}

        \tau_{\mathrm{bc}}
            \;=\; \frac{\ln 10}{2.5}\, R_V \,
                  (1 - \mathrm{ebv\_ratio}) \, E(B{-}V)_{\mathrm{lines}}

    so that ``tau_bc + tau_diff = (ln 10 / 2.5) · R_V · E(B-V)_lines``
    — i.e. the lines see the full CIGALE attenuation, the continuum
    sees only the diffuse component.

    Parameters
    ----------
    ebv_lines : float
        ``E(B-V)`` applied to nebular lines in CIGALE.
        [mag] Typical 0.0 – 1.0.
    av_over_ebv : float, optional
        Calzetti+2000 ``R_V = A_V / E(B-V) = 4.05``. Override only if
        you've changed CIGALE's ``dustatt_modified_starburst`` defaults.
    ebv_ratio : float, optional
        ``E(B-V)_continuum / E(B-V)_lines = 0.44`` (Calzetti+2000).
        Override only if you've changed CIGALE's defaults.

    Returns
    -------
    dict
        ``{"tau_bc": float, "tau_diff": float}`` suitable for
        ``dust={'type': 'two_component', 'law_bc': 'calzetti',
                'law_diff': 'calzetti', 'tau_bc': ..., 'tau_diff': ...}``.

    Examples
    --------
    Match CIGALE's ``dustatt_modified_starburst(E_BV_lines=0.3)``:

    >>> from tengri.interop.cigale import cigale_ebv_lines_to_tau
    >>> sorted(cigale_ebv_lines_to_tau(0.3).items())  # doctest: +ELLIPSIS
    [('tau_bc', 0.6...), ('tau_diff', 0.49...)]

    Notes
    -----
    The mapping assumes both legs of the CIGALE model use Calzetti+2000
    (the ``dustatt_modified_starburst`` default). If CIGALE is run with
    a non-Calzetti continuum law, the conversion is still valid in
    optical-depth-at-V but the wavelength dependence will differ from
    tengri's two-Calzetti default; pick matching ``law_bc`` and
    ``law_diff`` accordingly.
    """
    if ebv_lines < 0:
        raise ValueError(f"ebv_lines must be >= 0, got {ebv_lines}")
    if av_over_ebv <= 0:
        raise ValueError(f"av_over_ebv must be > 0, got {av_over_ebv}")
    if not 0.0 <= ebv_ratio <= 1.0:
        raise ValueError(f"ebv_ratio must be in [0, 1], got {ebv_ratio}")

    # tau_V = ln(10) / 2.5 * A_V = ln(10) / 2.5 * R_V * E(B-V)
    tau_v_lines = (math.log(10) / 2.5) * av_over_ebv * ebv_lines
    tau_diff = ebv_ratio * tau_v_lines
    tau_bc = (1.0 - ebv_ratio) * tau_v_lines
    return {"tau_bc": tau_bc, "tau_diff": tau_diff}

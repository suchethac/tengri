# SPDX-License-Identifier: BSD-3-Clause
"""Surviving stellar mass fraction from stellar evolution.

Computes the fraction of formed mass still in living stars + stellar
remnants as a function of age, for a given IMF. This is independent
of the SSP spectral library and can be computed on the fly for any
age grid.

The calculation uses:

- Main-sequence lifetime: simplified Hurley, Pols & Tout (2000) formula
- Initial-final mass relation: Kalirai et al. (2008) for CO white dwarfs
- Remnant masses: NS = 1.4 Msun, BH = 0.4 * m_init (Fryer+2012 approx.)
- IMF integration: numerical trapezoid rule in log-mass space

Supported IMFs: Chabrier (2003), Salpeter (1955), Kroupa (2001).

Known Limitations
-----------------
Compared to FSPS's mass-remaining (computed from full Padova/PARSEC
isochrone tracks), the internal computation differs by 1-5%:

1. **MS lifetime**: We use a simplified power-law fit instead of the
   full metallicity-dependent isochrone tracks (Padova/PARSEC/MIST).
   This overestimates lifetimes at ~3-30 Myr by ~5%.
2. **IFMR**: Kalirai+2008 is a single linear relation; FSPS uses more
   detailed prescriptions that depend on metallicity and isochrone library.
3. **IMF normalization**: Our Chabrier IMF uses the standard lognormal +
   Salpeter formula, but sub-stellar mass integration limits can shift
   the normalization by ~1-2%.
4. **No metallicity dependence**: This module computes mass-remaining at
   solar metallicity only. The actual metallicity dependence is ~3%
   over -2 < log(Z/Zsun) < +0.2 (see FSPS tests).
5. **No post-AGB / HB tracks**: We use a simple MS turnoff criterion;
   FSPS includes detailed post-MS evolution phases.

These differences are well within the systematic uncertainty from
isochrone library choice (~5-10% between Padova, MIST, PARSEC).
For rigorous work, use the pre-stored FSPS mass-remaining table
(``ssp_mass_remaining`` in the SSP HDF5 file) which is computed from
the same isochrone + IMF as the SSP spectra. This module serves as a
fallback when the FSPS table is not available, or when using a
non-FSPS SSP library.

See Also
--------
Behroozi, Wechsler & Conroy (2013, ApJ, 770, 57) use a similar
simplified approach (their Appendix B), fitting the return fraction
as ``f_return(t) = 0.05 * ln(1 + t / 1.4 Myr)`` for a Chabrier IMF.
Our implementation computes the integral numerically for flexibility
across IMFs, but agrees with their formula to ~2% for Chabrier.

References
----------

- Hurley, Pols & Tout 2000, MNRAS, 315, 543 (MS lifetime)
- Kalirai et al. 2008, ApJ, 676, 594 (WD IFMR)
- Fryer et al. 2012, ApJ, 749, 91 (BH remnant masses)
- Chabrier 2003, PASP, 115, 763 (IMF)
- Kroupa 2001, MNRAS, 322, 231 (IMF)
- Behroozi, Wechsler & Conroy 2013, ApJ, 770, 57 (fitting formula)

"""

import jax
import jax.numpy as jnp

# ── IMF definitions: dn/dlog(m) as a function of mass ─────────────


def _chabrier_imf(log_m: jnp.ndarray) -> jnp.ndarray:
    """Chabrier (2003) IMF: dn/dlog(m).

    Lognormal below 1 Msun, Salpeter power-law above.

    Parameters
    ----------
    log_m : array
        log10(m / Msun).

    Returns
    -------
    array
        IMF weight dn/dlog(m) (unnormalized).
    """
    m = 10.0**log_m

    # Lognormal: dn/dlog(m) ~ exp(-(log(m) - log(0.08))^2 / (2 * 0.69^2))
    lognormal = jnp.exp(-((log_m - jnp.log10(0.08)) ** 2) / (2.0 * 0.69**2))

    # Salpeter above 1 Msun: dn/dlog(m) ~ m^(-1.3) (i.e., dn/dm ~ m^(-2.3))
    salpeter = m ** (-1.3) * (1.0**1.3)  # match at m=1

    # Match normalization at m=1
    lognormal_at_1 = jnp.exp(-((0.0 - jnp.log10(0.08)) ** 2) / (2.0 * 0.69**2))
    salpeter_at_1 = 1.0
    scale = lognormal_at_1 / salpeter_at_1

    return jnp.where(m < 1.0, lognormal, salpeter * scale)


def _salpeter_imf(log_m: jnp.ndarray) -> jnp.ndarray:
    """Salpeter (1955) IMF: dn/dlog(m) ~ m^{-1.35}."""
    m = 10.0**log_m
    return m ** (-1.35)


def _kroupa_imf(log_m: jnp.ndarray) -> jnp.ndarray:
    """Kroupa (2001) broken power-law IMF: dn/dlog(m).

    Slopes: alpha = 0.3 (m < 0.08), 1.3 (0.08-0.5), 2.3 (> 0.5).
    Expressed as dn/dlog(m) ~ m^{-(alpha-1)}.
    """
    m = 10.0**log_m

    # Normalization: continuous at breakpoints
    phi_low = m ** (-(0.3 - 1.0))  # m^0.7
    phi_mid = m ** (-(1.3 - 1.0))  # m^-0.3
    phi_high = m ** (-(2.3 - 1.0))  # m^-1.3

    # Match at 0.08 Msun
    c1 = (0.08**0.7) / (0.08 ** (-0.3))
    # Match at 0.5 Msun
    c2 = c1 * (0.5 ** (-0.3)) / (0.5 ** (-1.3))

    return jnp.where(
        m < 0.08,
        phi_low,
        jnp.where(m < 0.5, c1 * phi_mid, c2 * phi_high),
    )


_IMF_REGISTRY = {
    "chabrier": _chabrier_imf,
    "salpeter": _salpeter_imf,
    "kroupa": _kroupa_imf,
}


# ── Main-sequence lifetime ────────────────────────────────────────


def _ms_lifetime_gyr(mass: jnp.ndarray) -> jnp.ndarray:
    """Main-sequence lifetime in Gyr.

    Simplified Hurley et al. (2000) formula for solar metallicity.
    Accurate to ~20% over 0.1-100 Msun.

    Parameters
    ----------
    mass : array
        Stellar mass in Msun.

    Returns
    -------
    array
        Main-sequence lifetime in Gyr.
    """
    # t_MS ~ 10 Gyr * (m/Msun)^{-2.5} for m > ~1 Msun
    # Flattens for low-mass stars (> Hubble time)
    m_safe = jnp.maximum(mass, 0.08)
    return 10.0 * m_safe ** (-2.5) + 0.1 * m_safe ** (-0.75)


# ── Turnoff mass (inverse of MS lifetime) ─────────────────────────


def _turnoff_mass(age_gyr: jnp.ndarray) -> jnp.ndarray:
    """Main-sequence turnoff mass at a given age.

    Inverts the MS lifetime relation numerically via Newton's method
    (JIT-compatible, 10 iterations).

    Parameters
    ----------
    age_gyr : array
        Stellar population age in Gyr.

    Returns
    -------
    array
        Turnoff mass in Msun.
    """
    # Initial guess from the simple power-law approximation
    t_safe = jnp.maximum(age_gyr, 1e-4)
    m_guess = (t_safe / 10.0) ** (-1.0 / 2.5)
    m_guess = jnp.clip(m_guess, 0.1, 150.0)

    # Newton iterations to solve t_MS(m) = age
    # Analytic derivative: d/dm [10*m^{-2.5} + 0.1*m^{-0.75}] = -25*m^{-3.5} - 0.075*m^{-1.75}
    m = m_guess
    for _ in range(10):
        t_ms = _ms_lifetime_gyr(m)
        dt_dm = -25.0 * m ** (-3.5) - 0.075 * m ** (-1.75)
        dt_dm_safe = jnp.where(jnp.abs(dt_dm) > 1e-30, dt_dm, -1e-30)
        m = m - (t_ms - t_safe) / dt_dm_safe
        m = jnp.clip(m, 0.08, 300.0)

    return m


# ── Remnant mass ──────────────────────────────────────────────────


def _remnant_mass(m_init: jnp.ndarray) -> jnp.ndarray:
    """Stellar remnant mass as a function of initial mass.

    - m < 0.5 Msun: He WD, m_rem ~ m_init (doesn't evolve in Hubble time)
    - 0.5 <= m < 8 Msun: CO WD via Kalirai+2008 IFMR: m_WD = 0.394 + 0.109 * m_init
    - 8 <= m < 25 Msun: neutron star, m_rem = 1.4 Msun (boundary: m < 25 strict)
    - m >= 25 Msun: black hole, m_rem = 0.4 * m_init (Fryer+2012 approx.)

    Parameters
    ----------
    m_init : array
        Initial stellar mass in Msun.

    Returns
    -------
    array
        Remnant mass in Msun.
    """
    # WD from Kalirai+2008
    m_wd = 0.394 + 0.109 * m_init

    # NS
    m_ns = 1.4

    # BH (Fryer+2012 rapid model, simplified)
    m_bh = 0.4 * m_init

    return jnp.where(
        m_init < 0.5,
        m_init,  # He WD / doesn't evolve
        jnp.where(
            m_init < 8.0,
            m_wd,
            jnp.where(m_init < 25.0, m_ns, m_bh),
        ),
    )


# ── Public API ────────────────────────────────────────────────────


def compute_mass_remaining_fraction(
    age_gyr: jnp.ndarray,
    imf: str = "chabrier",
    m_low: float = 0.08,
    m_high: float = 120.0,
    n_mass: int = 500,
) -> jnp.ndarray:
    """Compute surviving mass fraction at each age for a given IMF.

    The surviving mass includes living stars (m < m_turnoff) plus
    remnants (WD, NS, BH) from stars that have died (m > m_turnoff).

    Parameters
    ----------
    age_gyr : array, shape (n_age,)
        Population ages [Gyr], sorted ascending.
    imf : str
        IMF name: ``"chabrier"``, ``"salpeter"``, or ``"kroupa"``.
    m_low : float
        Lower mass limit of the IMF [Msun]. Default 0.08.
    m_high : float
        Upper mass limit of the IMF [Msun]. Default 120.
    n_mass : int
        Number of mass grid points for logarithmic integration. Default 500.

    Returns
    -------
    array, shape (n_age,)
        Fraction of formed mass in living stars + remnants (dimensionless,
        in (0, 1]). Value is 1.0 at age=0 (no stars have died).

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.vmap`` for vectorized age integration.
    **Gradient-safe**: yes.

    This function numerically integrates the IMF in logarithmic mass space
    and uses Newton's method to compute the MS turnoff mass at each age.
    See module docstring for limitations vs. FSPS mass-remaining tables.
    """
    if imf not in _IMF_REGISTRY:
        raise ValueError(f"Unknown IMF '{imf}'. Available: {list(_IMF_REGISTRY.keys())}")

    imf_fn = _IMF_REGISTRY[imf]

    # Mass grid in log-space
    log_m_grid = jnp.linspace(jnp.log10(m_low), jnp.log10(m_high), n_mass)
    m_grid = 10.0**log_m_grid
    d_log_m = log_m_grid[1] - log_m_grid[0]

    # IMF weight at each mass: dn/dlog(m) * m (for mass-weighted integral)
    imf_weight = imf_fn(log_m_grid)

    # Total mass formed: integral of m * dn/dlog(m) * dlog(m)
    total_mass = jnp.sum(m_grid * imf_weight * d_log_m)

    # Remnant masses for each star
    remnant_m = _remnant_mass(m_grid)

    def _surviving_at_age(t_gyr):
        """Integrate surviving mass (living stars + remnants) at a given age."""
        m_to = _turnoff_mass(t_gyr)

        # Living stars: m < m_turnoff, contribute their full mass
        living_mass = jnp.sum(jnp.where(m_grid < m_to, m_grid * imf_weight * d_log_m, 0.0))

        # Dead stars: m >= m_turnoff, contribute remnant mass
        dead_remnant_mass = jnp.sum(
            jnp.where(m_grid >= m_to, remnant_m * imf_weight * d_log_m, 0.0)
        )

        # NaN rather than a clamped zero when the IMF integral vanishes (#1404).
        # ``total_mass`` is the IMF mass integral over a fixed, strictly positive
        # log-mass grid, so it cannot be zero for any real IMF. A 1e-30 floor
        # therefore guards a state that is already broken; and guards it the
        # wrong way: with total_mass zero, living_mass and dead_remnant_mass are
        # zero too, so the clamp returns 0/1e-30 = 0, i.e. a surviving fraction
        # of exactly zero. That is a plausible-looking number ("all mass lost")
        # for a broken IMF, and it would propagate into stellar_mass silently.
        # NaN propagates and gets noticed instead. This is the form used by
        # ``utils/sed_quantities.py``.
        return jnp.where(
            total_mass > 1e-20,
            (living_mass + dead_remnant_mass) / jnp.maximum(total_mass, 1e-30),
            jnp.nan,
        )

    return jax.vmap(_surviving_at_age)(age_gyr)

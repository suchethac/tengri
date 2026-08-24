# SPDX-License-Identifier: BSD-3-Clause
"""Gas-regulator chemical evolution model: Z(t) from SFH.

Derives a time-dependent metallicity history Z(t) from the star formation
history using the closed-box / leaky-box analytic framework. This couples
metallicity to the SFH self-consistently, reducing the free parameter count
by eliminating the need for a separate metallicity parameter.

The model follows the gas regulator framework of Bellstedt et al. (2020,
2021; ProSpect) and Leja et al. (2019; Prospector):

    Closed box:  Z(t) = y * ln(1 / f_gas(t))
    Leaky box:   Z(t) = y / (1 + eta) * ln(1 / f_gas(t))

where y is the nucleosynthetic yield, eta is the mass-loading factor
(outflow rate / SFR), and f_gas is the gas fraction.

References
----------

- Bellstedt et al. 2020 (MNRAS 498, 5581): ProSpect chemical evolution
- Bellstedt et al. 2021 (MNRAS 503, 3309): Shark + ProSpect validation
- Leja et al. 2019 (ApJ 876, 3): Prospector continuity SFH
- Tinsley 1980 (Fundamentals of Cosmic Physics 5, 287): closed-box model

"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.physics_constants import (
    LOG10_ZSUN,
    Z_SUN,
)


def closed_box_metallicity(
    age_yr: jnp.ndarray,
    sfr: jnp.ndarray,
    yield_y: float = 0.03,
    eta_outflow: float = 0.0,
    f_gas_init: float = 0.9,
    return_frac: float = 0.4,
) -> jnp.ndarray:
    r"""Compute metallicity history Z(t) from SFH using closed/leaky box model.

    Self-consistently derives gas-phase metallicity at each age from the star
    formation history using analytic chemical evolution. Supports both closed-box
    (no outflows) and leaky-box (with mass-loading outflows) regimes.

    Parameters
    ----------
    age_yr: array_like, shape (n_age,)
        Lookback time grid [yr]. Convention: youngest (smallest lookback time) first,
        oldest last. This matches tengri's internal log-age grid convention.
    sfr: array_like, shape (n_age,)
        Star formation rate [Msun/yr] at each lookback time.
    yield_y: float, optional
        Nucleosynthetic yield (mass of metals produced per unit mass locked
        in long-lived stars). Default: 0.03 (solar neighborhood for Chabrier IMF;
        Vincenzo et al. 2016). [dimensionless]
    eta_outflow: float, optional
        Mass loading factor: Mdot_out / SFR. Default: 0.0 (closed box, no outflows).
        Typical range: 0.5-3 for dwarf galaxies, 0-0.5 for massive galaxies.
        [dimensionless]
    f_gas_init: float, optional
        Initial gas fraction :math:`M_{\rm gas} / (M_{\rm gas} + M_{\star})` at
        earliest cosmic time. Default: 0.9 (galaxy starts gas-dominated).
        [dimensionless, in (0,1)]
    return_frac: float, optional
        Instantaneous recycling fraction: fraction of formed stellar mass
        returned to the ISM via winds/SNe. Default: 0.4 (Chabrier IMF;
        Leitner & Kravtsov 2011). [dimensionless, in [0,1))

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z/Z_sun) at each lookback time. Clipped to [-4, +1].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The closed-box and leaky-box chemical evolution models compute metallicity
    from the integrated SFR history:

    .. math::

        Z(t) = \frac{y_{\rm eff}}{1} \ln\left( \frac{1}{f_{\rm gas}(t)} \right)

    where :math:`y_{\rm eff} = y / (1 + \eta)` is the effective yield and
    :math:`f_{\rm gas}(t)` is the gas fraction at time t.

    Gas mass evolution:

    .. math::

        M_{\rm gas}(t) = M_{\rm gas,0} - (1 - R) M_{\star,{\rm formed}}(t)
        - \eta M_{\star,{\rm formed}}(t)

    where :math:`M_{\star,{\rm formed}}` is the cumulative integral of SFR,
    :math:`R` is the return fraction, and :math:`\eta` is the mass loading factor.

    **Computation order**: The input grid is in lookback time (youngest first).
    Internally, the code reverses to cosmic time order (oldest first) to integrate
    the SFR and track cumulative stellar mass and gas depletion.

    With :math:`\eta = 0` (eta_outflow=0), this reduces to the standard closed-box model.

    References
    ----------
    .. [1] B. Tinsley, "Fundamentals of Cosmic Physics," in Fundamentals of Cosmic Physics,
       5, 287 (1980).
    .. [2] S. Bellstedt et al., "Galaxy And Mass Assembly (GAMA): a forensic SED
       reconstruction of the cosmic star formation history and metallicity evolution
       by galaxy type," MNRAS, 498, 5581 (2020). arXiv:2005.11917.
       https://doi.org/10.1093/mnras/staa2620
    .. [3] J. Leja et al., "How to Measure Galaxy Star Formation Histories.
       II. Nonparametric Models," ApJ, 876, 3 (2019). arXiv:1811.03637.
       https://doi.org/10.3847/1538-4357/ab133c
    """
    # Reverse to cosmic time order (oldest first)
    sfr_cosmic = sfr[::-1]
    age_cosmic = age_yr[::-1]

    # Time step sizes (in years) along cosmic time
    # Use forward differences: dt[i] = age_cosmic[i] - age_cosmic[i+1]
    # (lookback time decreases along cosmic time, so reversed ages decrease)
    # Actually after reversal, age_cosmic goes from large to small (oldest
    # lookback = largest value, first), so dt = |diff|.
    dt = jnp.abs(jnp.diff(age_cosmic))
    # Prepend a zero for the first bin (no mass formed before the first step)
    dt = jnp.concatenate([jnp.zeros(1), dt])

    # Cumulative stellar mass formed in cosmic time order
    mass_formed_cumulative = jnp.cumsum(sfr_cosmic * dt)

    # Total baryonic mass (gas + stars at t=0)
    # At the earliest time, M_star ~ 0 and f_gas = f_gas_init
    # So M_gas_init = f_gas_init * M_total, and the total mass we need
    # is set so the gas fraction works out correctly.
    # Use the final cumulative mass to set the scale.
    total_mass_formed = mass_formed_cumulative[-1]
    # M_total = M_gas_init / f_gas_init
    # At end: M_gas_final = M_gas_init - (1-R+eta)*M_star_formed_total
    # We need M_gas_init large enough that M_gas stays positive.
    # Set M_total from f_gas_init and the expected final mass:
    # M_total = total_mass_formed * (1 - return_frac + eta_outflow) / (1 - f_gas_init)
    # But protect against zero SFR:
    net_lock_frac = 1.0 - return_frac + eta_outflow
    m_total = jnp.where(
        total_mass_formed > 0,
        total_mass_formed * net_lock_frac / jnp.maximum(1.0 - f_gas_init, 0.01),
        1.0,  # arbitrary scale when no stars form
    )
    m_gas_init = f_gas_init * m_total

    # Gas mass at each step:
    # M_gas(t) = M_gas_init - (1 - R + eta) * M_star_formed(t)
    m_gas = m_gas_init - net_lock_frac * mass_formed_cumulative
    m_gas = jnp.maximum(m_gas, 1e-10 * m_total)  # floor to prevent log(0)

    # Gas fraction
    m_star_net = (1.0 - return_frac) * mass_formed_cumulative
    f_gas = m_gas / (m_gas + m_star_net)
    f_gas = jnp.clip(f_gas, 1e-6, 1.0)

    # Metallicity: leaky-box formula
    y_eff = yield_y / (1.0 + eta_outflow)
    z_metal = y_eff * jnp.log(1.0 / f_gas)
    z_metal = jnp.maximum(z_metal, 1e-8)  # floor at ~10^-8

    # Convert to log10(Z/Zsun)
    log_z_solar = jnp.log10(z_metal / Z_SUN)
    log_z_solar = jnp.clip(log_z_solar, -4.0, 1.0)

    # Reverse back to lookback time order (youngest first)
    return log_z_solar[::-1]


def closed_box_metallicity_anchored(
    age_yr: jnp.ndarray,
    sfr: jnp.ndarray,
    met_logzsol_final: float,
    eta_outflow: float = 0.0,
    return_frac: float = 0.4,
) -> jnp.ndarray:
    """Compute Z(t) anchored to a target present-day metallicity.

    Instead of specifying yield_y and f_gas_init independently, this function
    adjusts f_gas_init so that the final (youngest) metallicity matches
    ``met_logzsol_final``. This is useful for inference where the observed
    metallicity constrains the endpoint.

    Parameters
    ----------
    age_yr: array_like, shape (n_age,)
        Lookback time grid [yr] (youngest first).
    sfr: array_like, shape (n_age,)
        Star formation rate [Msun/yr] at each lookback time.
    met_logzsol_final: float
        Target present-day metallicity [dimensionless], log10(Z/Zsun).
    eta_outflow: float, optional
        Mass loading factor [dimensionless]. Default 0.0 (closed box).
    return_frac: float, optional
        Stellar mass return fraction [dimensionless]. Default 0.4.

    Returns
    -------
    ndarray, shape (n_age,)
        log10(Z/Zsun) at each lookback time. The youngest element will
        approximately equal ``met_logzsol_final``.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives throughout.

    The effective yield is computed from the target metallicity and the
    implied gas fraction at the final time step.
    """
    # Target Z at present day
    z_target = Z_SUN * 10.0**met_logzsol_final

    # We need to find yield_y that produces z_target at the endpoint.
    # The shape of Z(t) is set by the SFH; the yield just scales it.
    # Strategy: run with y=1.0, get the unnormalized Z profile, then
    # rescale so the final value matches.

    # Use a small reference yield to get the f_gas(t) profile shape.
    # yield_y=1.0 produces super-solar metallicities for typical SFHs
    # (Z >> Z_sun), hitting the clip boundary in closed_box_metallicity and
    # distorting the profile shape before the scale factor is applied.
    # yield_y=0.01 (near the Pagel 1997 / Pilyugin+2007 empirical range) keeps
    # the reference profile within the physical [-4, 1] dex clip.
    _YIELD_REF = 0.01
    log_z_ref = closed_box_metallicity(
        age_yr,
        sfr,
        yield_y=_YIELD_REF,
        eta_outflow=eta_outflow,
        f_gas_init=0.9,
        return_frac=return_frac,
    )

    # The youngest element (index 0) is the present-day value
    z_ref_final = Z_SUN * 10.0 ** log_z_ref[0]

    # Scale factor: Z ∝ yield_y in the leaky-box, so scaling by z_target/z_ref_final
    # is equivalent to using yield_y = _YIELD_REF * scale directly.
    scale = jnp.where(z_ref_final > 1e-30, z_target / z_ref_final, 1.0)

    # Apply scaling in linear Z space, then convert back
    z_ref_linear = Z_SUN * 10.0**log_z_ref
    z_scaled = z_ref_linear * scale
    z_scaled = jnp.maximum(z_scaled, 1e-8)

    log_z_solar = jnp.log10(z_scaled / Z_SUN)
    return jnp.clip(log_z_solar, -4.0, 1.0)


def chem_evol_metallicity_on_ssp_grid(
    ssp_log_ages_yr: jnp.ndarray,
    log_age_grid: jnp.ndarray,
    sfr: jnp.ndarray,
    yield_y: float = 0.03,
    eta_outflow: float = 0.0,
    f_gas_init: float = 0.9,
    return_frac: float = 0.4,
) -> jnp.ndarray:
    """Compute Z(t) from SFH and interpolate onto the SSP age grid.

    This is the main entry point for the SED pipeline integration.
    It computes chemical evolution on the GP/SFH grid, then interpolates
    the resulting log(Z/Zsun) onto the SSP log-age grid.

    Parameters
    ----------
    ssp_log_ages_yr: array_like, shape (n_ssp_age,)
        Log10(age/yr) of the SSP templates [dimensionless].
    log_age_grid: array_like, shape (n_grid,)
        Log10(lookback time/yr) grid on which the SFR is defined [dimensionless].
    sfr: array_like, shape (n_grid,)
        Star formation rate [Msun/yr] at each grid point.
    yield_y: float, optional
        Nucleosynthetic yield [dimensionless]. Default 0.03.
    eta_outflow: float, optional
        Mass loading factor [dimensionless]. Default 0.0.
    f_gas_init: float, optional
        Initial gas fraction [dimensionless]. Default 0.9.
    return_frac: float, optional
        Stellar return fraction [dimensionless]. Default 0.4.

    Returns
    -------
    ndarray, shape (n_ssp_age,)
        log10(Z) absolute metallicity on the SSP age grid [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    """
    # Convert log-age grid to linear years
    age_yr = 10.0**log_age_grid

    # Compute Z(t) on the SFH grid
    log_z_solar = closed_box_metallicity(
        age_yr,
        sfr,
        yield_y=yield_y,
        eta_outflow=eta_outflow,
        f_gas_init=f_gas_init,
        return_frac=return_frac,
    )

    # Interpolate onto SSP age grid (both in log-age space)
    log_z_on_ssp = jnp.interp(
        ssp_log_ages_yr,
        log_age_grid,
        log_z_solar,
    )

    # Convert from log10(Z/Zsun) to log10(Z) absolute
    log_z_abs = log_z_on_ssp + LOG10_ZSUN
    return log_z_abs

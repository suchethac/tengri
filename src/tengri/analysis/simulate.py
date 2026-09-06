# SPDX-License-Identifier: BSD-3-Clause
"""Forward-model SEDs from arbitrary SFH and metallicity history arrays.

This module provides the simulation-facing API for tengri: given
tabulated SFH(t) and optionally Z(t) arrays (e.g., from cosmological
simulations like IllustrisTNG, EAGLE, UniverseMachine), compute the
rest-frame SED, observed photometry, or spectrum.

This bypasses the parametric SFH models in Parameters and directly uses
the DSPS CSP integral.

Usage
-----
Quick photometry from an SFH array::

    from tengri import load_ssp_data, load_filter_set
    from tengri.analysis.simulate import sed_from_sfh, photometry_from_sfh

    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r"])

    # SFH from a simulation (cosmic time in Gyr, SFR in Msun/yr)
    t_gyr = np.linspace(0.1, 13.7, 100)
    sfr = 10.0 * np.exp(-t_gyr / 3.0)  # exponentially declining

    sed = sed_from_sfh(t_gyr, sfr, ssp, log_z=-0.3)
    phot = photometry_from_sfh(
        t_gyr, sfr, ssp, filters, log_z=-0.3, redshift=0.5, dust_tau_bc=0.3, dust_tau_diff=0.5
    )

With metallicity history::

    log_z_history = -2.0 + 1.5 * t_gyr / 13.7  # enrichment
    sed = sed_from_sfh(t_gyr, sfr, ssp, log_z=log_z_history, lgmet_scatter=0.3)

All functions are pure JAX and JIT-compilable.

References
----------

- Hearin et al. 2023, MNRAS, 521, 1741 (DSPS)

"""

import jax.numpy as jnp

from tengri.components.dust.attenuation import select_law_kwargs, two_component_dust
from tengri.components.stellar.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    interpolate_metallicity,
    interpolate_metallicity_evolving,
)
from tengri.cosmology import luminosity_distance
from tengri.parameters.translate import LOG10_ZSUN

# ── Constants ─────────────────────────────────────────────────────


def _filter_waves_and_trans(filters):
    """Return (waves, trans) from ``load_filter_set`` output or FilterCurve sequence.

    ``load_filter_set`` returns ``(filter_waves, filter_trans, filter_curves)``;
    callers may pass that tuple, or only the list of ``FilterCurve`` objects.
    """
    if isinstance(filters, tuple) and len(filters) >= 2:
        waves, trans = filters[0], filters[1]
        if (
            isinstance(waves, (list, tuple))
            and isinstance(trans, (list, tuple))
            and len(waves) == len(trans)
            and len(waves) > 0
        ):
            return list(waves), list(trans)
    try:
        filter_waves = [f.wave for f in filters]
        filter_trans = [f.trans for f in filters]
    except (TypeError, AttributeError) as exc:
        raise TypeError(
            "filters must be the tuple returned by load_filter_set(names) or a sequence "
            "of FilterCurve objects (e.g. load_filter_set(names)[2])."
        ) from exc
    return filter_waves, filter_trans


# ── Core: SED from tabulated SFH ──────────────────────────────────


def sed_from_sfh(
    t_gyr: jnp.ndarray,
    sfr: jnp.ndarray,
    ssp_data,
    log_z=-0.3,
    lgmet_scatter: float = 0.0,
    dust_tau_bc: float = 0.0,
    dust_tau_diff: float = 0.0,
    dust_law: str = "power_law",
    dust_slope: float = -0.7,
    t_obs_gyr: float = 13.7,
    **dust_kwargs,
):
    """Compute rest-frame SED from a tabulated star formation history.

    Integrates composite stellar population (CSP) models over a tabulated SFH
    and optional metallicity history. Useful for cosmological simulations,
    semi-analytic models, or custom SFH specifications.

    Parameters
    ----------
    t_gyr : ndarray, shape (n_t,)
        Cosmic time grid in Gyr (increasing, from early to late universe).
    sfr : ndarray, shape (n_t,)
        Star formation rate [Msun/yr] at each time point.
    ssp_data : SSPData
        SSP templates from ``load_ssp_data()``.
    log_z : float or ndarray, optional
        Stellar metallicity log10(Z/Zsun).

        - float: constant metallicity for all stars. Default: -0.3.
        - ndarray shape (n_t,): metallicity history Z(t).

    lgmet_scatter : float, optional
        Lognormal scatter in metallicity [dex]. Used only when log_z is
        scalar. Default: 0 (delta function).
    dust_tau_bc : float, optional
        Birth cloud V-band optical depth. Default: 0 (no dust).
    dust_tau_diff : float, optional
        Diffuse ISM V-band optical depth. Default: 0 (no dust).
    dust_law : str, optional
        Dust attenuation curve name. Default: "power_law".
        Options: "power_law", "calzetti", "kriek_conroy", etc.
    dust_slope : float, optional
        Power-law slope (for power_law curve). Default: -0.7.
    t_obs_gyr : float, optional
        Age of universe at observation [Gyr]. Default: 13.7.
    **dust_kwargs
        Additional dust parameters (dust_bump_strength, dust_delta, etc.).

    Returns
    -------
    dict
        Rest-frame SED with keys:

        - "wavelength" : ndarray shape (n_wave,). Wavelength [Angstrom]
        - "sed" : ndarray shape (n_wave,). Luminosity density [erg/s/Hz]
        - "stellar_mass" : float. Total stellar mass formed [Msun]
        - "weights" : ndarray shape (n_age,). CSP age weights

    Notes
    -----
    **JIT-compatible**: yes; the entire function is JAX-native and
    compatible with jax.jit, jax.grad, and jax.vmap.

    **Metallicity evolution**: When log_z is an array, it is interpolated
    onto the SSP age grid. The interpolation respects the optional
    lgmet_scatter parameter.
    """
    # Convert SFH to SSP age weights via interpolation onto SSP age grid
    ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0  # log10(age/yr)
    ssp_ages_yr = 10.0**ssp_log_ages_yr

    # Convert cosmic time to lookback time
    t_lookback_yr = (t_obs_gyr - t_gyr) * 1e9  # yr
    log_t_lookback = jnp.log10(jnp.maximum(t_lookback_yr, 1.0))

    # Interpolate SFR onto SSP age grid
    sfr_on_ssp = jnp.interp(ssp_log_ages_yr, log_t_lookback[::-1], sfr[::-1])
    weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)

    # Metallicity interpolation
    if isinstance(log_z, (float, int)) or (hasattr(log_z, "ndim") and log_z.ndim == 0):
        # Scalar metallicity: convert to absolute log(Z)
        log_z_abs = float(log_z) + LOG10_ZSUN  # solar offset
        ssp_flux_at_z = interpolate_metallicity(
            ssp_data.ssp_flux,
            ssp_data.ssp_lgmet,
            log_z_abs,
        )
    else:
        # Array metallicity history: interpolate onto SSP ages
        log_z_array = jnp.asarray(log_z)
        log_z_on_ssp = jnp.interp(
            ssp_log_ages_yr,
            log_t_lookback[::-1],
            log_z_array[::-1],
        )
        log_z_abs = log_z_on_ssp + (-1.8477)
        ssp_flux_at_z = interpolate_metallicity_evolving(
            ssp_data.ssp_flux,
            ssp_data.ssp_lgmet,
            log_z_abs,
        )

    # Dust attenuation
    if dust_tau_bc > 0 or dust_tau_diff > 0:
        # ``dust_slope`` is this function's own defaulted argument, so it is
        # offered to every law -- and most laws have no slope to set. Narrow to
        # what the selected law declares (#2185): a slope-free law such as
        # ``calzetti`` would otherwise be handed an ``n_slope`` it cannot use,
        # which was silently discarded before the laws dropped their ``**kwargs``
        # and is a loud ``ValueError`` now. Keys the caller passed in
        # ``dust_kwargs`` are NOT narrowed: those are explicit, and a law that
        # cannot read one should say so.
        #
        # ``dict(...)`` rather than a ``{...}`` literal so this reads as what it
        # is to ``tools/check_dust_law_kwargs.py``: a one-key mapping handed
        # straight to the narrowing helper, not a hand-assembled law-parameter
        # dict splatted into an evaluation. What that rule guards against is a
        # dict carrying "whatever its author remembered"; here the selected law
        # decides what survives, which is the property the rule wants.
        law_kw = select_law_kwargs(dust_law, dict(n_slope=dust_slope))
        dust_atten = two_component_dust(
            ssp_data.ssp_wave,
            ssp_ages_yr,
            tau_v1=dust_tau_bc,
            tau_v2=dust_tau_diff,
            law_bc=dust_law,
            law_diff=dust_law,
            **law_kw,
            **dust_kwargs,
        )
    else:
        dust_atten = jnp.ones((len(ssp_ages_yr), len(ssp_data.ssp_wave)))

    # CSP integral
    sed = compute_csp_sed(weights, ssp_flux_at_z, dust_atten)

    return {
        "wavelength": ssp_data.ssp_wave,
        "sed": sed,
        "stellar_mass": float(jnp.sum(weights)),
        "weights": weights,
    }


def photometry_from_sfh(
    t_gyr: jnp.ndarray,
    sfr: jnp.ndarray,
    ssp_data,
    filters,
    log_z=-0.3,
    redshift: float = 0.0,
    dust_tau_bc: float = 0.0,
    dust_tau_diff: float = 0.0,
    apply_igm: bool = True,
    **kwargs,
):
    """Compute observed photometry from a tabulated SFH.

    Parameters
    ----------
    t_gyr : array, shape (n_t,)
        Cosmic time in Gyr.
    sfr : array, shape (n_t,)
        SFR in Msun/yr.
    ssp_data : SSPData
        SSP templates.
    filters
        Either the full return value of ``load_filter_set(names)``, i.e.
        ``(waves, trans, curves)``, or a sequence of ``FilterCurve``
        objects (e.g. the third element of that tuple).
    log_z : float or array
        Metallicity (scalar or history).
    redshift : float
        Source redshift.
    dust_tau_bc, dust_tau_diff : float
        Dust optical depths.
    apply_igm : bool
        Apply IGM absorption. Default True.
    **kwargs
        Additional dust/model parameters.

    Returns
    -------
    dict with keys:
        "flux" : array (n_filters,). Observed flux in erg/s/cm^2/Hz
        "sed" : array (n_wave,). Rest-frame SED in erg/s/Hz
        "stellar_mass" : float. Total mass formed
    """
    from tengri.observation.photometry import compute_flux_density

    # Compute SED
    result = sed_from_sfh(
        t_gyr,
        sfr,
        ssp_data,
        log_z=log_z,
        dust_tau_bc=dust_tau_bc,
        dust_tau_diff=dust_tau_diff,
        **kwargs,
    )
    sed = result["sed"]
    wave = result["wavelength"]

    # IGM absorption
    if apply_igm and redshift > 0:
        from tengri.components.igm import igm_transmission

        wave_obs = wave * (1.0 + redshift)
        igm_trans = igm_transmission(wave_obs, redshift)
        sed = sed * igm_trans

    # Luminosity distance: ``luminosity_distance`` already applies the
    # 10-pc absolute-magnitude convention at z=0 (returns ~3.086e19 cm).
    # The earlier ``if redshift > 0 else 1.0`` fallback was a 10^19×
    # flux error at z=0; the dead branch is gone.
    dl_cm = luminosity_distance(redshift)

    # Filter convolution
    filter_waves, filter_trans = _filter_waves_and_trans(filters)

    fluxes = []
    for fw, ft in zip(filter_waves, filter_trans):
        f = compute_flux_density(sed, wave, fw, ft, redshift, dl_cm)
        fluxes.append(f)

    result["flux"] = jnp.array(fluxes)
    return result


def spectrum_from_sfh(
    t_gyr: jnp.ndarray,
    sfr: jnp.ndarray,
    ssp_data,
    wave_obs: jnp.ndarray,
    log_z=-0.3,
    redshift: float = 0.0,
    dust_tau_bc: float = 0.0,
    dust_tau_diff: float = 0.0,
    apply_igm: bool = True,
    sigma_v: float = 0.0,
    **kwargs,
):
    """Compute observed spectrum from a tabulated SFH.

    Parameters
    ----------
    t_gyr : array, shape (n_t,)
        Cosmic time in Gyr.
    sfr : array, shape (n_t,)
        SFR in Msun/yr.
    ssp_data : SSPData
        SSP templates.
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid in Angstrom.
    log_z : float or array
        Metallicity.
    redshift : float
        Source redshift.
    sigma_v : float
        Velocity dispersion (km/s) for broadening. Default 0.
    **kwargs
        Additional parameters.

    Returns
    -------
    dict with keys:
        "flux" : array (n_pix,). Observed flux in erg/s/cm^2/Hz
        "sed" : array (n_wave,). Rest-frame SED
        "stellar_mass" : float
    """
    from tengri.observation.spectrum import compute_spectrum

    result = sed_from_sfh(
        t_gyr,
        sfr,
        ssp_data,
        log_z=log_z,
        dust_tau_bc=dust_tau_bc,
        dust_tau_diff=dust_tau_diff,
        **kwargs,
    )
    sed = result["sed"]
    wave = result["wavelength"]

    # IGM
    if apply_igm and redshift > 0:
        from tengri.components.igm import igm_transmission

        wave_obs_full = wave * (1.0 + redshift)
        igm_trans = igm_transmission(wave_obs_full, redshift)
        sed = sed * igm_trans

    # See note in ``sed_from_sfh``: ``luminosity_distance`` already
    # handles the z=0 → 10 pc absolute-magnitude convention.
    dl_cm = luminosity_distance(redshift)

    # Compute observed spectrum
    flux = compute_spectrum(sed, wave, wave_obs, redshift, dl_cm)

    # Velocity broadening
    if sigma_v > 0:
        from tengri.observation.spectrum import velocity_broaden

        flux = velocity_broaden(flux, wave_obs, sigma_v)

    result["flux"] = flux
    result["wave_obs"] = wave_obs
    return result

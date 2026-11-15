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

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from tengri.components.dust.attenuation import select_law_kwargs, two_component_dust
from tengri.components.stellar.component import _mass_scale_lnu, tabulated_history_weights
from tengri.cosmology import age_at_z, luminosity_distance
from tengri.inference.history_ingest import _check_ssp_grid, _mass_weights
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
    t_obs_gyr: float | None = None,
    redshift: float | None = None,
    on_out_of_grid: str = "raise",
    **dust_kwargs,
):
    """Compute rest-frame SED from a tabulated star formation history.

    Builds the same SSP weights as :meth:`~tengri.Catalog.from_histories`
    (:func:`~tengri.components.stellar.component.tabulated_history_weights`), so
    the formed mass and SED agree with the catalog path for the same history.

    Parameters
    ----------
    t_gyr : ndarray, shape (n_t,)
        Cosmic time grid in Gyr (increasing, from early to late universe).
        The SFR is zero outside this grid, and the grid is not clamped to its
        edge values.
    sfr : ndarray, shape (n_t,)
        Star formation rate [Msun/yr] at each time point.
    ssp_data : SSPData
        SSP templates from ``load_ssp_data()``.
    log_z : float or ndarray, optional
        Stellar metallicity log10(Z/Zsun).

        - float: constant metallicity for all stars. Default: -0.3.
        - ndarray shape (n_t,): metallicity history Z(t), on the same nodes as
          ``t_gyr``.

    lgmet_scatter : float, optional
        Lognormal metallicity-distribution width [dex], applied with the same
        kernel as the forward model. Default: 0.
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
        Age of universe at observation [Gyr], the lookback origin. Default:
        ``age_at_z(redshift)``, or ``age_at_z(0)`` when no redshift is given.
    redshift : float, optional
        Source redshift. Sets the default ``t_obs_gyr``.
    on_out_of_grid : {"raise", "warn", "ignore"}, optional
        What to do when a metallicity is outside the SSP grid, where the lookup
        would otherwise clip to the edge. Default: "raise", as in
        :meth:`~tengri.Catalog.from_histories`. The check runs on concrete
        inputs only; under ``jax.jit`` the lookup clips silently.
    **dust_kwargs
        Additional dust parameters (dust_bump_strength, dust_delta, etc.).

    Returns
    -------
    dict
        Rest-frame SED with keys:

        - "wavelength" : ndarray shape (n_wave,). Wavelength [Angstrom]
        - "sed" : ndarray shape (n_wave,). Luminosity density [erg/s/Hz]
        - "stellar_mass" : float. Total stellar mass formed [Msun]
        - "weights" : ndarray shape (n_age,). CSP age weights [Msun]

    Raises
    ------
    ValueError
        If a metallicity is outside the SSP grid under ``on_out_of_grid="raise"``.

    Notes
    -----
    **JIT-compatible**: yes for the weights and SED. The metallicity grid check
    is eager and is skipped for traced inputs.

    Mass older than the oldest SSP template is kept and assigned to that
    template, as in the forward model (#1522, #2714).
    """
    t_gyr = jnp.asarray(t_gyr)
    sfr = jnp.asarray(sfr)
    t_obs = age_at_z(0.0 if redshift is None else redshift) if t_obs_gyr is None else t_obs_gyr

    _check_metallicity_on_grid(t_gyr, sfr, log_z, ssp_data.ssp_lgmet, on_out_of_grid)
    log_z_abs = jnp.asarray(log_z) + LOG10_ZSUN  # solar offset, traced-safe
    joint_weights, total_mass = tabulated_history_weights(
        t_gyr,
        sfr,
        ssp_data.ssp_lg_age_gyr,
        ssp_data.ssp_lgmet,
        t_obs,
        log_z_abs,
        lgmet_scatter,
    )
    ssp_ages_yr = 10.0 ** (ssp_data.ssp_lg_age_gyr + 9.0)

    # Dust attenuation
    def _two_component() -> jnp.ndarray:
        # ``dust_slope`` is this function's own defaulted argument, so it is
        # offered to every law -- and most laws have no slope to set. Narrow to
        # what the selected law declares (#2185): a slope-free law such as
        # ``calzetti`` would otherwise be handed an ``dust_slope`` it cannot use,
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
        law_kw = select_law_kwargs(dust_law, dict(dust_slope=dust_slope))
        return two_component_dust(
            ssp_data.ssp_wave,
            ssp_ages_yr,
            tau_v1=dust_tau_bc,
            tau_v2=dust_tau_diff,
            law_bc=dust_law,
            law_diff=dust_law,
            **law_kw,
            **dust_kwargs,
        )

    ones = jnp.ones((len(ssp_ages_yr), len(ssp_data.ssp_wave)))
    dust_atten = _select_dust(
        jnp.logical_or(dust_tau_bc > 0, dust_tau_diff > 0), _two_component, ones
    )

    # CSP integral: the joint weights sum to one, so the mass enters once here.
    # The factor L_sun is applied last, as in the forward model (#1206).
    ssp_flux_at_age = jnp.einsum("ma,maw->aw", joint_weights, ssp_data.ssp_flux)
    # Grouped through _mass_scale_lnu: a standalone total_mass * L_sun reaches
    # 1.2e46, which is inf in float32.
    sed = _mass_scale_lnu(jnp.einsum("aw,aw->w", ssp_flux_at_age, dust_atten), total_mass)
    weights = joint_weights.sum(axis=0) * total_mass

    return {
        "wavelength": ssp_data.ssp_wave,
        "sed": sed,
        "stellar_mass": _host_scalar(total_mass),
        "weights": weights,
    }


def _check_metallicity_on_grid(t_gyr, sfr, log_z, ssp_lgmet, policy) -> None:
    """Refuse (or warn about) a metallicity outside the SSP grid, as ``from_histories`` does.

    A scalar is one node carrying all the mass; a history is weighted by its
    per-node stellar mass. Skipped for traced inputs, which have no values to check.
    """
    if any(isinstance(x, Tracer) for x in (t_gyr, sfr, log_z)):
        return
    log_z_np = np.asarray(log_z, dtype=float)
    if log_z_np.ndim == 0:
        logzsol = log_z_np.reshape(1, 1)
        mass_w = np.ones((1, 1))
    else:
        logzsol = log_z_np.reshape(1, -1)
        t_np = np.asarray(t_gyr, dtype=float).reshape(1, -1)
        mass_w = _mass_weights(t_np, np.asarray(sfr, dtype=float).reshape(1, -1))
    _check_ssp_grid(logzsol, mass_w, np.asarray(ssp_lgmet), policy)


def _select_dust(gate, attenuate, ones: jnp.ndarray) -> jnp.ndarray:
    """Return ``attenuate()`` where ``gate`` holds, else ``ones``.

    A concrete gate keeps the eager Python branch, so the dust kernel is not built
    for an unattenuated call. A traced gate (a fitted optical depth) goes through
    ``lax.cond``; the attenuated branch is cast to ``ones``' dtype so both branches
    agree.
    """
    if isinstance(gate, Tracer):
        return jax.lax.cond(gate, lambda: attenuate().astype(ones.dtype), lambda: ones)
    return attenuate() if bool(gate) else ones


def _host_scalar(value):
    """Return a concrete scalar as a Python float; keep a traced one as it is."""
    if isinstance(value, Tracer):
        return value
    return float(value)


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
        redshift=redshift,
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
    resample: str = "auto",
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
        Velocity dispersion (km/s) for broadening. Default 0. Applied to
        the observed-frame, pre-IGM flux -- stellar kinematics are
        intrinsic to the source and act before the line-of-sight IGM
        transmission, which can only broaden the instrument's response to
        an edge, never the edge itself (#2589).
    resample : {"auto", "point", "conserving"}
        How the rest-frame SED is turned into pixels, with the meaning and the
        default of :attr:`~tengri.Spectroscopy.resample` (#2530): ``"auto"``
        integrates over each pixel where the pixels are wider than the model
        grid and point-samples otherwise.
    **kwargs
        Additional parameters.

    Returns
    -------
    dict with keys:
        "flux" : array (n_pix,). Observed flux in erg/s/cm^2/Hz
        "sed" : array (n_wave,). Rest-frame SED
        "stellar_mass" : float
    """
    from tengri.observation.spectroscopy import resolve_resample_mode, static_redshift
    from tengri.observation.spectrum import compute_spectrum, compute_spectrum_conserving

    result = sed_from_sfh(
        t_gyr,
        sfr,
        ssp_data,
        log_z=log_z,
        dust_tau_bc=dust_tau_bc,
        dust_tau_diff=dust_tau_diff,
        redshift=redshift,
        **kwargs,
    )
    sed = result["sed"]
    wave = result["wavelength"]

    # See note in ``sed_from_sfh``: ``luminosity_distance`` already
    # handles the z=0 → 10 pc absolute-magnitude convention.
    dl_cm = luminosity_distance(redshift)

    # Compute observed spectrum (pre-IGM: stellar kinematics act on the
    # galaxy's own light, which the IGM has not yet absorbed).
    z_static = static_redshift(redshift)
    conserving = resolve_resample_mode(resample, wave_obs, wave, z_static)
    resampler = compute_spectrum_conserving if conserving else compute_spectrum
    if conserving and sigma_v > 0:
        # The pixel integral is taken of the kinematically broadened model (#2530).
        from tengri.observation.spectrum import broaden_velocity_only

        sed = broaden_velocity_only(sed, wave, sigma_v)
    flux = resampler(sed, wave, wave_obs, redshift, dl_cm)

    # Velocity broadening -- intrinsic to the source, applied BEFORE the
    # line-of-sight IGM transmission (#2589): a galaxy's own kinematics
    # cannot physically broaden an absorption feature imprinted after the
    # light has left the galaxy. Broadening is a constant-width Gaussian in
    # ln(lambda), and redshifting only shifts ln(lambda) by the constant
    # ln(1+z), so broadening this observed-frame, pre-IGM ``flux`` is
    # equivalent to broadening the rest-frame SED before redshifting.
    if sigma_v > 0 and not conserving:
        from tengri.observation.spectrum import velocity_broaden

        flux = velocity_broaden(flux, wave_obs, sigma_v)

    # IGM: multiplies the already-kinematically-broadened observed flux.
    if apply_igm and redshift > 0:
        from tengri.components.igm import igm_transmission

        igm_trans = igm_transmission(wave_obs, redshift)
        flux = flux * igm_trans

    result["flux"] = flux
    result["wave_obs"] = wave_obs
    return result

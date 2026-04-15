"""Diffuse ionized gas (DIG) nebular emission mixing.

DIG is low-density ionized gas between HII regions, characterised by a lower
ionization parameter (logU ~ -4) compared to HII regions (logU ~ -2.5 to -3).
This produces enhanced low-ionization line ratios ([NII]/Ha, [SII]/Ha, [OI]/Ha).

Typical DIG fractions are 30-60% of total Ha in local galaxies
(Reynolds 1984; Haffner et al. 2009; Tacchella et al. 2022).

The mixing model evaluates any nebular backend at two ionization parameters:

    L_total = (1 - f_DIG) * L(logU_HII) + f_DIG * L(logU_DIG)

where logU_DIG = logU_HII + delta_logU (delta_logU is negative, default -1 dex).

When ``neb_dig_frac=0`` (default), this reduces to pure HII emission with
zero overhead when ``neb_dig_frac`` is a Python float 0.0; under JIT with
a traced value, both forward passes execute.

References
----------
- Reynolds 1984, ApJ, 282, 191
- Haffner et al. 2009, RvMP, 81, 969
- Tacchella et al. 2022, ApJ, 926, 134
"""

import jax.numpy as jnp


def mix_dig_emission(
    nebular_backend,
    ssp_wave: jnp.ndarray,
    ssp_weights: jnp.ndarray,
    ssp_log_ages_yr: jnp.ndarray,
    log_z: float,
    neb_logU: float = -3.0,
    neb_logZ_gas: float | None = None,
    neb_fesc: float = 0.0,
    neb_fesc_lya: float = 0.0,
    neb_dig_frac: float = 0.0,
    neb_dig_delta_logU: float = -1.0,
    line_sigma_aa: float = 0.0,
    **kwargs,
) -> jnp.ndarray:
    """Compute nebular SED with DIG mixing.

    Evaluates the nebular backend at two ionization parameters and mixes:

        L_total = (1 - f_DIG) * L(logU_HII) + f_DIG * L(logU_DIG)

    where ``logU_DIG = logU_HII + delta_logU`` (delta_logU is negative).

    When ``neb_dig_frac=0`` (default), this returns pure HII emission.

    Parameters
    ----------
    nebular_backend : CloudyGridBackend or CueBackend
        Any backend with a ``predict_nebular_sed`` method.
    ssp_wave : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    ssp_weights : array, shape (n_age,)
        CSP mass weights per age bin.
    ssp_log_ages_yr : array, shape (n_age,)
        log10(age/yr) of SSP bins.
    log_z : float
        Stellar metallicity log10(Z) (absolute).
    neb_logU : float
        HII region ionization parameter log10(U). Default -3.0.
    neb_logZ_gas : float or None
        Gas-phase metallicity log10(Z_gas/Zsun). If None, the backend
        defaults to matching stellar metallicity.
    neb_fesc : float
        Ionizing photon escape fraction [0, 1]. Default 0.0.
    neb_fesc_lya : float
        Lyman-alpha escape fraction [0, 1]. Default 0.0.
    neb_dig_frac : float
        Fraction of nebular emission from DIG [0, 1]. Default 0.0.
    neb_dig_delta_logU : float
        Offset in log10(U) for DIG relative to HII (dex, negative).
        Default -1.0.
    line_sigma_aa : float
        Gaussian line width in Angstrom. Default 0.0.
    **kwargs
        Additional keyword arguments forwarded to the backend
        (e.g., Cue-specific ionizing spectrum parameters).

    Returns
    -------
    array, shape (n_wave,)
        Total nebular SED (HII + DIG weighted mixture) in erg/s/Hz.
    """
    common_kw = dict(
        ssp_wave=ssp_wave,
        ssp_weights=ssp_weights,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z,
        neb_logZ_gas=neb_logZ_gas,
        neb_fesc=neb_fesc,
        neb_fesc_lya=neb_fesc_lya,
        line_sigma_aa=line_sigma_aa,
        **kwargs,
    )

    # HII component (standard ionization parameter)
    neb_hii = nebular_backend.predict_nebular_sed(neb_logU=neb_logU, **common_kw)

    # Short-circuit: when neb_dig_frac is a Python literal 0.0, skip the extra
    # forward pass entirely.  Under JIT with a traced value both passes execute.
    if isinstance(neb_dig_frac, (int, float)) and neb_dig_frac == 0.0:
        return neb_hii

    # DIG component (lower ionization parameter)
    logU_dig = neb_logU + neb_dig_delta_logU
    neb_dig = nebular_backend.predict_nebular_sed(neb_logU=logU_dig, **common_kw)

    # Linear mix
    return (1.0 - neb_dig_frac) * neb_hii + neb_dig_frac * neb_dig

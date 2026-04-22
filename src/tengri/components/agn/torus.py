"""Toy torus models for AGN infrared emission.

.. warning::
    These are **toy models** using 1-2 temperature modified blackbodies.
    They are NOT radiative transfer results and should NOT be used for
    science.  For production work, use the SKIRTOR templates in
    ``tengri.components.agn.skirtor`` (tabulated from 3D Monte Carlo RT).

Two toy models are provided for testing and fast prototyping:

1. **simple_torus** — single-temperature modified blackbody
   with silicate opacity. 2 free parameters.
2. **two_temperature_torus** — hot + warm dust components. 4 free params.

Both return specific luminosity L_nu in erg/s/Hz. All functions are pure
JAX and JIT-compilable.

References
----------
- Nenkova et al. 2008, ApJ, 685, 147 (CLUMPY torus)
- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR)
- Draine 2003, ARA&A, 41, 241 (silicate opacity)
"""

import warnings

import jax.numpy as jnp

from tengri.components.agn._phys import (
    LSUN_ERG as _LSUN_ERG,
    planck_lnu as _planck_lnu,
    wavelength_to_nu as _wavelength_to_nu,
)

# ── Physical constants (CGS) ──────────────────────────────────────

_MICRON_ANGSTROM = 1e4  # Micron -> Angstrom

# Silicate feature wavelength
_LAMBDA_SI = 9.7 * _MICRON_ANGSTROM  # 9.7 um in Angstrom


# ── Model 1: Simple hot blackbody torus ───────────────────────────


def simple_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float = 0.5,
    agn_T_torus: float = 1000.0,
    agn_tau_torus: float = 5.0,
    agn_tau_beta: float = 1.5,
    **_kwargs,
) -> jnp.ndarray:
    """Simple single-temperature dust torus with silicate opacity.

    L_nu = L_bol * f_torus * B_nu(T_torus) / B_int * (1 - exp(-tau * (9.7um/lam)^beta))

    where B_int normalizes the modified blackbody to integrate to
    L_bol * f_torus.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_torus_frac : float
        Fraction of L_bol re-emitted by torus (covering factor).
        Typical range: 0.1 to 0.9. Default 0.5.
    agn_T_torus : float
        Torus dust temperature [K].
        Typical range: 500 to 1500. Default 1000.
    agn_tau_torus : float
        Optical depth at 9.7 um silicate feature.
        Typical range: 1 to 10. Default 5.
    agn_tau_beta : float
        Power-law index for opacity wavelength dependence.
        Typical range: 1.0 to 2.0. Default 1.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].
    """
    warnings.warn(
        "simple_torus is a toy model (single-temperature MBB, not radiative transfer) "
        "and should NOT be used for science. Use skirtor_analytic from "
        "tengri.components.agn.skirtor for production work.",
        DeprecationWarning,
        stacklevel=2,
    )
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    nu = _wavelength_to_nu(wavelength)

    # Blackbody emission
    b_nu = _planck_lnu(nu, agn_T_torus)

    # Silicate opacity: tau(lambda) = tau_torus * (9.7um / lambda)^beta
    opacity = 1.0 - jnp.exp(
        -agn_tau_torus * (_LAMBDA_SI / jnp.maximum(wavelength, 1.0)) ** agn_tau_beta
    )

    # Modified blackbody shape
    shape = b_nu * opacity

    # Normalize to L_bol * f_torus
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg


# ── Model 2: Two-temperature torus (SKIRTOR-inspired) ─────────────


def two_temperature_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_torus_frac: float = 0.5,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_tau_torus: float = 5.0,
    agn_tau_beta: float = 1.5,
    **_kwargs,
) -> jnp.ndarray:
    """Two-temperature dust torus (hot sublimation + warm outer torus).

    Inspired by SKIRTOR clumpy torus models. The emission is a mixture
    of two modified blackbodies:

        L_nu = f_hot * BB(T_hot) + (1 - f_hot) * BB(T_warm)

    both modified by the same silicate opacity profile.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_torus_frac : float
        Fraction of L_bol re-emitted by torus. Default 0.5.
    agn_T_hot : float
        Hot dust temperature [K], near sublimation.
        Typical range: 1000 to 1500. Default 1200.
    agn_T_warm : float
        Warm dust temperature [K], outer torus.
        Typical range: 200 to 800. Default 300.
    agn_frac_hot : float
        Luminosity fraction in hot component (0 to 1). Default 0.3.
    agn_tau_torus : float
        Optical depth at 9.7 um. Default 5.
    agn_tau_beta : float
        Opacity power-law index. Default 1.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].
    """
    warnings.warn(
        "two_temperature_torus is a toy model (two-temperature MBB, not radiative transfer) "
        "and should NOT be used for science. Use skirtor_analytic from "
        "tengri.components.agn.skirtor for production work.",
        DeprecationWarning,
        stacklevel=2,
    )
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    nu = _wavelength_to_nu(wavelength)

    # Two blackbody components
    b_hot = _planck_lnu(nu, agn_T_hot)
    b_warm = _planck_lnu(nu, agn_T_warm)

    # Silicate opacity
    opacity = 1.0 - jnp.exp(
        -agn_tau_torus * (_LAMBDA_SI / jnp.maximum(wavelength, 1.0)) ** agn_tau_beta
    )

    # Weighted mixture with opacity
    shape = (agn_frac_hot * b_hot + (1.0 - agn_frac_hot) * b_warm) * opacity

    # Normalize
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg


# ── Nenkova+2008 CLUMPY torus ─────────────────────────────────────


def _load_nenkova_data():
    """Load Nenkova+2008 CLUMPY torus templates from FSPS data files.

    Returns ``(wave_aa, fnu_grid, tau_vals)`` where ``fnu_grid`` has shape
    ``(n_wave, n_tau)`` and ``tau_vals`` has shape ``(n_tau,)``.

    Raises
    ------
    FileNotFoundError
        If the data file is not found via SPS_HOME or the default fsps path.
    """
    import os

    import numpy as np

    sps_home = os.environ.get("SPS_HOME", "")
    if not sps_home:
        sps_home = os.path.expanduser("~/Projects/fsps")
    dat_path = os.path.join(sps_home, "dust", "Nenkova08_y010_torusg_n10_q2.0.dat")
    if not os.path.isfile(dat_path):
        raise FileNotFoundError(
            f"Nenkova+2008 data not found at {dat_path}. "
            "Set $SPS_HOME to your FSPS installation directory."
        )

    tau_vals = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0])
    data = np.genfromtxt(dat_path, skip_header=4)
    return data[:, 0], data[:, 1:], tau_vals


def nenkova_torus(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_tau: float = 30.0,
    agn_torus_frac: float = 0.5,
) -> jnp.ndarray:
    """AGN torus emission from Nenkova et al. (2008) CLUMPY templates.

    Interpolates the CLUMPY radiative-transfer torus library in optical depth
    ``agn_tau``, then normalizes to ``agn_torus_frac * L_bol``. This is the
    production-quality alternative to the deprecated toy torus models in this
    module; for science use prefer the SKIRTOR templates in
    ``tengri.components.agn.skirtor``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). AGN bolometric luminosity.
    agn_tau : float, optional
        Equatorial optical depth of the clumpy torus [dimensionless].
        Valid range: 5 to 150. Default 30.
    agn_torus_frac : float, optional
        Fraction of L_bol re-emitted by the torus (covering factor).
        Default 0.5.

    Returns
    -------
    L_nu : jnp.ndarray, shape (n_wave,)
        Specific luminosity [erg s^-1 Hz^-1].

    Notes
    -----
    **Not JIT-compatible** — loads data from disk. For JIT-compatible
    inference, precompute the template on a fixed wavelength grid and
    pass as a static array.

    Data source: ``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat``,
    the same file used by FSPS (Conroy & Gunn 2010) and Prospector
    (Johnson et al. 2021 [2]_).

    References
    ----------
    .. [1] M. Nenkova et al., "AGN Dusty Tori. I. Handling of Clumpy Media,"
       ApJ, 685, 147 (2008). https://doi.org/10.1086/590482
    .. [2] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4295/abef67

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> wave = jnp.logspace(3, 5, 100)
    >>> sed = nenkova_torus(wave, agn_log_lbol=12.0, agn_tau=30)
    >>> sed.shape
    (100,)
    """
    import numpy as np
    from scipy.interpolate import interp1d

    wave_aa, fnu_grid, tau_vals = _load_nenkova_data()

    wave_np = np.asarray(wavelength)
    n_wave = wave_np.shape[0]
    n_tau = len(tau_vals)

    log_wave_data = np.log10(np.maximum(wave_aa, 1e-30))
    log_wave_model = np.log10(np.maximum(wave_np, 1e-30))
    i1 = int(np.argmin(np.abs(wave_np - wave_aa[0])))
    i2 = int(np.argmin(np.abs(wave_np - wave_aa[-1])))

    fnu_interp = np.zeros((n_wave, n_tau))
    for k in range(n_tau):
        log_fnu = np.log10(np.maximum(fnu_grid[:, k], 1e-70))
        log_fnu_model = np.interp(log_wave_model[i1:i2 + 1], log_wave_data, log_fnu)
        fnu_interp[i1:i2 + 1, k] = 10.0 ** log_fnu_model

    tau_interp = interp1d(
        tau_vals, fnu_interp, axis=1,
        bounds_error=False, fill_value=(fnu_interp[:, 0], fnu_interp[:, -1]),
    )
    fnu_at_tau = tau_interp(float(agn_tau))

    fnu_jax = jnp.array(fnu_at_tau)
    nu = _wavelength_to_nu(wavelength)
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(fnu_jax[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_bol_erg = 10.0 ** agn_log_lbol * _LSUN_ERG
    return l_bol_erg * agn_torus_frac * fnu_jax / integral_safe

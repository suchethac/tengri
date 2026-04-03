"""MAPPINGS V shock emission line model.

Interpolates Allen+2008 (ApJS, 178, 20) Table 5 shock model tabulations
(solar abundance, pre-shock density n=1 cm^-3) to compute emission line
luminosities as a function of shock velocity. Lines are placed on an
arbitrary wavelength grid as Gaussians or delta functions.

The key diagnostic signatures of shock-heated gas vs HII regions:
- Enhanced [NII], [SII], [OI] at low/moderate velocities
- [OIII] peaks at intermediate velocities (~300-400 km/s)
- Halpha/Hbeta slightly above Case B due to collisional excitation

References
----------
- Allen et al. 2008, ApJS, 178, 20 (MAPPINGS V shock models)
- Rich et al. 2011, ApJ, 734, 87 (shock diagnostics in galaxies)
"""

import jax.numpy as jnp

# Physical constants
_C_CGS = 2.99792458e10  # cm/s
_LSUN_ERG = 3.828e33  # erg/s

# -----------------------------------------------------------------------
# Allen+2008 Table 5 shock model grid (solar abundance, n=1 cm^-3)
# -----------------------------------------------------------------------

# Shock velocities [km/s] for interpolation grid
_SHOCK_V = jnp.array([100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1000.0])

# Line ratios relative to Hbeta (Allen+2008 Table 5, solar, n=1)
# [OII] 3727 / Hbeta
_R_OII = jnp.array([3.5, 5.2, 4.8, 3.1, 2.4, 1.9, 1.2, 0.8])
# [OIII] 5007 / Hbeta
_R_OIII = jnp.array([0.3, 1.5, 4.2, 5.8, 6.1, 5.5, 3.8, 2.5])
# [OI] 6300 / Hbeta
_R_OI = jnp.array([0.8, 1.2, 0.9, 0.5, 0.3, 0.2, 0.15, 0.1])
# [NII] 6583 / Hbeta
_R_NII = jnp.array([2.5, 3.8, 3.2, 2.1, 1.6, 1.3, 0.9, 0.6])
# [SII] 6716+6731 / Hbeta
_R_SII = jnp.array([2.8, 4.5, 3.5, 2.0, 1.4, 1.0, 0.6, 0.4])
# Halpha / Hbeta (Case B + collisional enhancement)
_R_HA = jnp.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7])

# Rest-frame line wavelengths [Angstrom]
_SHOCK_LINE_WAVELENGTHS = {
    "OII_3727": 3727.0,
    "Hbeta": 4861.0,
    "OIII_4959": 4959.0,  # = OIII_5007 / 2.98
    "OIII_5007": 5007.0,
    "OI_6300": 6300.0,
    "NII_6548": 6548.0,  # = NII_6583 / 2.94
    "Halpha": 6563.0,
    "NII_6583": 6583.0,
    "SII_6716": 6716.0,
    "SII_6731": 6731.0,
}

# Doublet splitting ratios (atomic physics)
_OIII_DOUBLET_RATIO = 2.98  # 5007/4959
_NII_DOUBLET_RATIO = 2.94  # 6583/6548
_SII_DOUBLET_RATIO = 1.0  # 6716/6731 ~ 1 at low density

# Pre-sorted arrays for JIT-compatible line placement
_LINE_NAMES = list(_SHOCK_LINE_WAVELENGTHS.keys())
_LINE_WAVES = jnp.array([_SHOCK_LINE_WAVELENGTHS[n] for n in _LINE_NAMES])
_N_LINES = len(_LINE_NAMES)


def shock_line_ratios(
    shock_velocity: float,
    shock_log_density: float = 1.0,
) -> dict[str, float]:
    """Return line luminosity ratios relative to Hbeta.

    Interpolates Allen+2008 MAPPINGS V shock model tabulations
    (solar abundance, pre-shock density n=1 cm^-3). The density
    parameter is reserved for future grid extension.

    Parameters
    ----------
    shock_velocity : float
        Shock velocity in km/s (clipped to [100, 1000]).
    shock_log_density : float
        Log10 pre-shock density in cm^-3. Currently unused (grid is
        for n=1 only). Reserved for future multi-density grids.

    Returns
    -------
    dict
        Line name -> luminosity ratio relative to Hbeta.
    """
    v_clip = jnp.clip(shock_velocity, 100.0, 1000.0)

    r_oiii_5007 = jnp.interp(v_clip, _SHOCK_V, _R_OIII)
    r_nii_6583 = jnp.interp(v_clip, _SHOCK_V, _R_NII)
    r_sii_total = jnp.interp(v_clip, _SHOCK_V, _R_SII)

    return {
        "OII_3727": jnp.interp(v_clip, _SHOCK_V, _R_OII),
        "Hbeta": jnp.array(1.0),
        "OIII_4959": r_oiii_5007 / _OIII_DOUBLET_RATIO,
        "OIII_5007": r_oiii_5007,
        "OI_6300": jnp.interp(v_clip, _SHOCK_V, _R_OI),
        "NII_6548": r_nii_6583 / _NII_DOUBLET_RATIO,
        "Halpha": jnp.interp(v_clip, _SHOCK_V, _R_HA),
        "NII_6583": r_nii_6583,
        "SII_6716": r_sii_total / (1.0 + 1.0 / _SII_DOUBLET_RATIO),
        "SII_6731": r_sii_total / (1.0 + _SII_DOUBLET_RATIO),
    }


def _shock_line_luminosities_array(
    shock_velocity: float,
    l_shock_halpha: float,
    shock_log_density: float = 1.0,
) -> jnp.ndarray:
    """Compute absolute line luminosities as a flat array.

    Returns luminosities in Lsun, ordered by ``_LINE_NAMES``.
    This is the JIT-friendly version used internally by
    ``shock_emission_sed``.

    Parameters
    ----------
    shock_velocity : float
        Shock velocity in km/s.
    l_shock_halpha : float
        Total shock Halpha luminosity in Lsun.
    shock_log_density : float
        Log10 pre-shock density (reserved).

    Returns
    -------
    array, shape (n_lines,)
        Line luminosities in Lsun.
    """
    ratios = shock_line_ratios(shock_velocity, shock_log_density)

    # Convert from Hbeta-relative to Halpha-relative, then scale
    r_ha = ratios["Halpha"]
    # L_line = (ratio / r_ha) * L_halpha
    luminosities = jnp.array([ratios[name] / r_ha * l_shock_halpha for name in _LINE_NAMES])
    return luminosities


def shock_emission_sed(
    wavelength: jnp.ndarray,
    shock_velocity: float,
    l_shock_halpha: float,
    shock_log_density: float = 1.0,
    line_sigma_aa: float = 0.0,
) -> jnp.ndarray:
    """Compute shock emission line SED.

    Places MAPPINGS V shock emission lines on an arbitrary wavelength
    grid. Lines are represented as Gaussians (if ``line_sigma_aa > 0``)
    or delta functions added to the nearest pixel.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom (rest-frame, increasing).
    shock_velocity : float
        Shock velocity in km/s (100-1000).
    l_shock_halpha : float
        Total shock Halpha luminosity in Lsun.
    shock_log_density : float
        Log10 pre-shock density in cm^-3 (reserved, default 1.0).
    line_sigma_aa : float
        Gaussian line width in Angstrom. 0 = delta function
        (added to nearest wavelength pixel).

    Returns
    -------
    array, shape (n_wave,)
        Shock emission SED in Lsun/Hz.
    """
    line_lum = _shock_line_luminosities_array(shock_velocity, l_shock_halpha, shock_log_density)

    sed = jnp.zeros_like(wavelength)
    n_wave = wavelength.shape[0]

    if line_sigma_aa > 0:
        # Gaussian profiles
        for j in range(_N_LINES):
            lw = _LINE_WAVES[j]
            ll = line_lum[j]
            # sigma_nu = sigma_lambda[cm] * c[cm/s] / lambda[cm]^2
            # line_sigma_aa is in Å; convert to cm with 1e-8 before using in CGS formula.
            sigma_nu = line_sigma_aa * 1e-8 * _C_CGS / (lw * 1e-8) ** 2
            profile = jnp.exp(-0.5 * ((wavelength - lw) / line_sigma_aa) ** 2)
            profile = profile / (jnp.sqrt(2.0 * jnp.pi) * sigma_nu)
            # profile [Hz^{-1}], ll [Lsun] -> contribution [Lsun/Hz] (no unit conversion needed)
            sed = sed + ll * profile
    else:
        # Delta functions: add to nearest pixel
        for j in range(_N_LINES):
            lw = _LINE_WAVES[j]
            ll = line_lum[j]
            idx = jnp.argmin(jnp.abs(wavelength - lw))
            idx = jnp.clip(idx, 1, n_wave - 2)
            # Approximate delta_nu from pixel width
            dwave = jnp.abs(wavelength[idx + 1] - wavelength[idx - 1]) / 2.0
            dnu = _C_CGS / (wavelength[idx] * 1e-8) ** 2 * dwave * 1e-8
            line_flux_density = ll / dnu  # Lsun/Hz
            sed = sed.at[idx].add(line_flux_density)

    return sed

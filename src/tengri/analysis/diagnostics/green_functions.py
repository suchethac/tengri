# SPDX-License-Identifier: BSD-3-Clause
"""Green's functions and window functions for SFH sensitivity.

The Green's function G_lambda(t_age) tells you how much a stellar
population of age t_age contributes to the flux at wavelength lambda
(or through filter lambda). This is the SSP mass-to-light ratio.

The window function W_lambda(t_age) = G_lambda(t_age) * <SFR(t_age)>
weights the Green's function by the mean SFH, telling you which
lookback times actually contribute to the observed flux.

Together these answer:

- "Which ages does H-alpha probe?" (young, ~few Myr)
- "Which ages does the Balmer break probe?" (intermediate, ~100 Myr-1 Gyr)
- "Which ages does the K-band probe?" (old, ~Gyr)
- "At what timescales can I constrain PSD power?"

This is the time-domain complement to the gradient SEDs in saliency.py.
Connects to Munoz+2026 Eq. 11 and Iyer+2024 window function formalism.

Key insight: the Fourier transform of W gives the frequency-domain
sensitivity, directly linking observable bands to PSD timescales.
"""

import jax
import jax.numpy as jnp

from tengri.utils.filter_convention import FilterConvention, filter_weight as _filter_weight
from tengri.utils.scale import representable_denominator


def compute_green_function(
    ssp_flux_at_z,
    ssp_wave,
    filter_wave=None,
    filter_trans=None,
    wave_target=None,
    convention=FilterConvention.BESSELL,
):
    """Compute Green's function G(t_age) for a filter or wavelength.

    G(t_age) = flux contribution per unit stellar mass at age t_age.

    For photometry::

        G = int L_SSP(lambda|t_age) * T(lambda) * w(lambda) dlambda
            / int T(lambda) * w(lambda) dlambda

    with ``w = 1/lambda`` (``BESSELL`` default, matches DSPS/FSPS) or
    ``w = 1/lambda**2`` (``ENERGY``, CIGALE).

    For a single wavelength, ``G = L_SSP(wave_target | t_age)``.

    Parameters
    ----------
    ssp_flux_at_z: array, shape (n_age, n_wave)
        SSP spectra at fixed metallicity.
    ssp_wave: array, shape (n_wave,)
        Wavelength grid (Angstrom).
    filter_wave: array, optional
        Filter wavelength grid.
    filter_trans: array, optional
        Filter transmission.
    wave_target: float, optional
        Single wavelength (Angstrom). Used if no filter provided.

    Returns
    -------
    array, shape (n_age,)
        Green's function G(t_age).
    """
    n_age = ssp_flux_at_z.shape[0]

    if filter_wave is not None and filter_trans is not None:
        # Photometric Green's function
        weight = filter_trans * _filter_weight(filter_wave, convention)
        denom = jnp.trapezoid(weight, filter_wave)
        greens = jnp.zeros(n_age)

        for i in range(n_age):
            ssp_on_filt = jnp.interp(filter_wave, ssp_wave, ssp_flux_at_z[i], left=0.0, right=0.0)
            num = jnp.trapezoid(ssp_on_filt * weight, filter_wave)
            # Derivative-sized floor: this is a denominator, so its VJP needs
            # 1/floor**2 representable, not just floor (#1860).
            greens = greens.at[i].set(num / jnp.maximum(denom, representable_denominator(1e-30)))
        return greens

    elif wave_target is not None:
        # Monochromatic Green's function
        return jax.vmap(lambda ssp: jnp.interp(wave_target, ssp_wave, ssp))(ssp_flux_at_z)

    else:
        raise ValueError("Provide either (filter_wave, filter_trans) or wave_target")


def compute_window_function(green_fn, mean_sfr_on_ages):
    """Compute window function W(t_age) = G(t_age) * <SFR(t_age)>.

    The window function tells you which lookback times actually
    contribute to the observed flux, given the galaxy's mean SFH.

    Parameters
    ----------
    green_fn: array, shape (n_age,)
        Green's function G(t_age).
    mean_sfr_on_ages: array, shape (n_age,)
        Mean SFR evaluated at the SSP age grid (Msun/yr).

    Returns
    -------
    array, shape (n_age,)
        Window function W(t_age) (unnormalized).
    """
    return green_fn * mean_sfr_on_ages


def compute_window_function_fourier(window_fn, ssp_ages_yr):
    """Compute Fourier transform of window function.

    ``|W_tilde(omega)|^2`` tells you the PSD sensitivity: how much power
    at frequency omega contributes to the variance of this observable.

    From Munoz+2026 Eq. 11::

        sigma^2_L = int |W_tilde(omega)|^2 P(omega) d_omega / (2*pi)

    Parameters
    ----------
    window_fn: array, shape (n_age,)
        Window function W(t_age).
    ssp_ages_yr: array, shape (n_age,)
        SSP ages in years (for frequency axis).

    Returns
    -------
    power_transfer: array, shape (n_freq,)
        ``|W_tilde(omega)|^2``: the PSD-to-observable transfer function.
    omega: array, shape (n_freq,)
        Angular frequencies (rad/yr).
    """
    n = len(window_fn)

    # FFT of window function (on the SSP age grid)
    # Use log-spaced ages, so compute dt for proper normalization
    dt = jnp.diff(ssp_ages_yr, prepend=0.0)
    w_tilde = jnp.fft.rfft(window_fn * dt)
    power_transfer = jnp.abs(w_tilde) ** 2

    # Frequency grid
    d_age = jnp.mean(jnp.diff(ssp_ages_yr))
    freqs = jnp.fft.rfftfreq(n, d=d_age)
    omega = 2.0 * jnp.pi * freqs

    return power_transfer, omega


def compute_time_sensitivity_matrix(ssp_flux_at_z, ssp_wave, wavelengths_target):
    """Compute sensitivity of multiple wavelengths to different ages.

    Returns a matrix G(wavelength, age) showing which wavelengths
    are sensitive to which stellar population ages.

    Parameters
    ----------
    ssp_flux_at_z: array, shape (n_age, n_wave)
        SSP spectra at fixed metallicity.
    ssp_wave: array, shape (n_wave,)
        Wavelength grid (Angstrom).
    wavelengths_target: array, shape (n_target,)
        Target wavelengths to evaluate (Angstrom).
        E.g., [1500, 2500, 4000, 5500, 6563, 8000, 16000] for
        FUV, NUV, Balmer break, V-band, H-alpha, I-band, H-band.

    Returns
    -------
    sensitivity: array, shape (n_target, n_age)
        G(wavelength, age) matrix. Each row is the Green's function
        at that wavelength.
    """
    n_target = len(wavelengths_target)
    n_age = ssp_flux_at_z.shape[0]

    sensitivity = jnp.zeros((n_target, n_age))
    for w_idx, wave in enumerate(wavelengths_target):
        g = compute_green_function(ssp_flux_at_z, ssp_wave, wave_target=wave)
        sensitivity = sensitivity.at[w_idx].set(g)

    return sensitivity

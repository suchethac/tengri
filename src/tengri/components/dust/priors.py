# SPDX-License-Identifier: BSD-3-Clause
"""Redshift-dependent dust attenuation priors from Narayanan+2018.

Based on cosmological radiative transfer simulations (SIMBA/Narayanan et al.
2018, ApJ, 869, 70), which show systematic trends in attenuation curve shape
and optical depth with redshift and stellar mass:

- Higher-z galaxies have steeper curves (more negative delta)
- UV bump strength decreases with redshift
- Optical depth scales with stellar mass and redshift

These functions return dicts of Gaussian distributions suitable for direct
use in Parameters kwargs.

References
----------
Narayanan, D., et al. (2018). ApJ, 869, 70.
    "A Theory for the Variation of Dust Attenuation Laws in Galaxies"
"""

from __future__ import annotations


def narayanan_prior(z: float) -> dict:
    """Recommended dust attenuation priors based on Narayanan+2018 z-dependent trends.

    Returns dict of Gaussian distributions for dust_delta and dust_bump_strength,
    suitable for direct use in Parameters.

    Based on Narayanan et al. (2018, ApJ, 869, 70) cosmological RT simulations:

    - Higher-z galaxies have steeper curves (more negative dust_delta)
    - UV bump strength decreases with redshift

    Parameters
    ----------
    z : float
        Source redshift [dimensionless].

    Returns
    -------
    dict
        Keys ``"dust_delta"`` and ``"dust_bump_strength"``, each mapping to a
        ``Gaussian`` distribution [dimensionless]. Suitable for direct use in
        ``Parameters(..., **narayanan_prior(z))``.

    Examples
    --------
    >>> from tengri import Parameters
    >>> from tengri.components.dust.priors import narayanan_prior
    >>> spec = Parameters(..., **narayanan_prior(z=2.0))

    Notes
    -----
    **JIT-compatible**: no, prior specification is a factory-time operation.

    **Gradient-safe**: no, returns static prior distributions.
    """
    from tengri.parameters.priors import Gaussian

    # Narayanan+2018 trends (median from SIMBA simulations):
    delta_mean = -0.2 - 0.1 * z  # steeper at high z
    delta_sigma = 0.15
    bump_mean = max(0.0, 1.0 - 0.15 * z)  # weaker at high z
    bump_sigma = 0.3

    return {
        "dust_delta": Gaussian(delta_mean, delta_sigma),
        "dust_bump_strength": Gaussian(bump_mean, bump_sigma),
    }


def narayanan_tau_prior(z: float, log_mstar: float = 10.0) -> dict:
    """Recommended dust optical depth prior based on Narayanan+2018 scaling.

    The diffuse-ISM optical depth tau_diff scales with stellar mass and
    redshift, reflecting the higher dust content in massive, high-z galaxies.

    Parameters
    ----------
    z : float
        Source redshift [dimensionless].
    log_mstar : float, optional
        log₁₀(M_star / M_sun) [dimensionless]. Default: 10.0.

    Returns
    -------
    dict
        Key ``"dust_tau_diff"`` mapping to a ``Gaussian`` distribution
        [dimensionless]. Suitable for direct use in
        ``Parameters(..., **narayanan_tau_prior(z, log_mstar))``.

    Examples
    --------
    >>> from tengri import Parameters
    >>> from tengri.components.dust.priors import narayanan_tau_prior
    >>> spec = Parameters(..., **narayanan_tau_prior(z=1.5, log_mstar=10.5))

    Notes
    -----
    **JIT-compatible**: no, prior specification is a factory-time operation.

    **Gradient-safe**: no, returns static prior distributions.
    """
    from tengri.parameters.priors import Gaussian

    tau_mean = 0.5 * (10 ** (log_mstar - 10)) ** 0.5 * (1 + z) ** 0.5
    tau_sigma = 0.3 * tau_mean + 0.1

    return {"dust_tau_diff": Gaussian(tau_mean, tau_sigma)}

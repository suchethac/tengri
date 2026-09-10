# SPDX-License-Identifier: BSD-3-Clause
"""Redshift-dependent dust attenuation priors from Narayanan+2018.

Narayanan, Conroy, Davé, Johnson & Popping (2018) post-process the 25 Mpc
MUFASA cosmological hydrodynamic simulation with dust radiative transfer and
publish a median attenuation curve at each integer redshift 0 to 6. Their
Section 5.1 reports two trends across that range:

- the median curves become **grayer** with redshift (the slope grows less
  negative);
- the 2175 Å bump weakens with redshift.

:func:`narayanan_prior` centers Gaussians on those published medians, so a fit
that frees :func:`~tengri.components.dust.attenuation.kriek_conroy`'s slope and
bump starts where the simulation says a galaxy at that redshift sits. The
centers come from the same fitted table
:func:`~tengri.components.dust.attenuation.narayanan_z` interpolates, produced
by ``scripts/fit_narayanan2018_medians.py``; there is one source, not two.

These functions return dicts of Gaussian distributions suitable for direct
use in Parameters kwargs.

References
----------
D. Narayanan, C. Conroy, R. Davé, B. D. Johnson and G. Popping (2018).
    "A Theory for the Variation of Dust Attenuation Laws in Galaxies",
    ApJ, 869, 70. arXiv:1805.06905.
    https://doi.org/10.3847/1538-4357/aaed25
    Median curves: https://bitbucket.org/desika/narayanan_attenuation_laws/
"""

from __future__ import annotations


def narayanan_prior(z: float) -> dict:
    """Kriek & Conroy slope and bump priors centered on the MUFASA median at z.

    Both keys are :func:`~tengri.components.dust.attenuation.kriek_conroy`
    parameters: ``dust_delta`` is its power-law slope and
    ``dust_bump_strength`` the multiplier on its Drude amplitude
    :math:`E_b = 0.85 - 1.9\\,\\delta`. The Gaussian centers are the published
    Narayanan et al. (2018) medians at ``z``, linearly interpolated between the
    integer nodes and held at the end node outside 0 to 6.

    Parameters
    ----------
    z : float
        Source redshift [dimensionless]. Values outside 0 to 6 take the end
        node; the paper tabulates no median beyond that range.

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

    The centers move with redshift the way the paper's Section 5.1 says: the
    slope grows *less* negative (the curve gets grayer), from -0.556 at z = 0
    to +0.080 at z = 6, and the implied bump amplitude weakens, from
    :math:`E_b` = 6.36 to 1.96. Before #2199 this function centered ``dust_delta``
    on ``-0.2 - 0.1 z``, which steepened with redshift, and ``dust_bump_strength``
    on ``max(0, 1 - 0.15 z)``; neither closed form appears in the paper and the
    slope one had the reported trend backwards.

    Use these with ``law='kriek_conroy'``. Pairing them with
    ``law='narayanan_z'`` is not possible and not needed: that law reads no
    slope or bump at all, because it *is* the median curve at z.

    The widths (0.15 in :math:`\\delta`, 0.3 in the bump multiplier) are
    unchanged from before #2199 and are not from the paper, which publishes a
    median curve rather than a dispersion in these two parameters. They are
    deliberately narrower than the curve-to-curve scatter the paper describes;
    widen them for a fit meant to explore it.

    References
    ----------
    .. [1] D. Narayanan, C. Conroy, R. Davé, B. D. Johnson and G. Popping,
       "A Theory for the Variation of Dust Attenuation Laws in Galaxies,"
       ApJ, 869, 70 (2018). arXiv:1805.06905.
       https://doi.org/10.3847/1538-4357/aaed25
       Median curves: https://bitbucket.org/desika/narayanan_attenuation_laws/
    """
    import jax.numpy as jnp

    from tengri.components.dust.attenuation import (
        _NARAYANAN_BUMP_STRENGTH,
        _NARAYANAN_DELTA,
        _NARAYANAN_Z_NODES,
    )
    from tengri.parameters.priors import Gaussian

    # ONE source for the medians: the table narayanan_z interpolates, fitted by
    # scripts/fit_narayanan2018_medians.py to the paper's published curves.
    # Recomputing them here would be a second copy free to drift from the first.
    delta_mean = float(jnp.interp(jnp.asarray(z), _NARAYANAN_Z_NODES, _NARAYANAN_DELTA))
    bump_mean = float(jnp.interp(jnp.asarray(z), _NARAYANAN_Z_NODES, _NARAYANAN_BUMP_STRENGTH))
    delta_sigma = 0.15
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

    **Unverified scaling.** Unlike :func:`narayanan_prior`, whose centers are
    the paper's own published median curves, the
    :math:`\\tau \\propto M_*^{1/2}(1+z)^{1/2}` form here matches no equation in
    Narayanan et al. (2018), which publishes attenuation curve *shapes* and not
    an optical-depth scaling. Treat it as a rough starting point, not a result;
    it is left as it was because #2199 fixed the curve shape and found no
    published number to replace this with.
    """
    from tengri.parameters.priors import Gaussian

    tau_mean = 0.5 * (10 ** (log_mstar - 10)) ** 0.5 * (1 + z) ** 0.5
    tau_sigma = 0.3 * tau_mean + 0.1

    return {"dust_tau_diff": Gaussian(tau_mean, tau_sigma)}

# SPDX-License-Identifier: BSD-3-Clause
"""Reference stellar masses from a model built at one fixed redshift.

A per-galaxy-redshift catalog fit evaluates every galaxy through one program whose
redshift is a runtime input. The independent reference for galaxy ``i`` is the plain
model built at that galaxy's redshift, fitted on its own; these helpers return its
numbers.
"""

import jax.numpy as jnp


def reinserted_mass_mean(model, flux, noise, samples, mass_name, key):
    """Mean log10 mass the fixed-z model's own fitter reinserts into ``samples``.

    Parameters
    ----------
    model : SEDModel
        Built at the galaxy's redshift (a ``Fixed`` value).
    flux, noise : array_like
        The galaxy's photometry and its uncertainty.
    samples : dict
        Posterior draws of every free parameter except the mass.
    mass_name : str
        The profiled mass parameter.
    key : jax.Array
        PRNG key for the conditional mass draw.

    Returns
    -------
    float
        Mean of the reinserted log10 mass draws.
    """
    from tengri.inference.fitter import Fitter
    from tengri.inference.mass_profile import reinsert_profiled_mass
    from tengri.inference.posterior import Posterior

    no_mass = {k: v for k, v in samples.items() if k != mass_name}
    post = Posterior(
        samples=no_mass,
        params={k: jnp.mean(v, axis=0) for k, v in no_mass.items()},
        method="reference",
        wall_time_s=0.0,
        diagnostics={},
        _model=model,
    )
    reinsert_profiled_mass(
        Fitter(model, flux, noise),
        post,
        data=flux,
        noise=noise,
        presence=None,
        line_obs=None,
        line_err=None,
        key=key,
    )
    return float(jnp.mean(post.samples[mass_name]))

"""Nested Sampling algorithms for tengri.

Faithful port of the handley-lab/blackjax nested_sampling branch.
Provides Nested Slice Sampling (NSS) using Hit-and-Run Slice Sampling as
the inner kernel with adaptive covariance tuning.

Available modules
-----------------
- ``base``: Core NS types and generic kernel.
- ``adaptive``: Adaptive NS with inner kernel parameter tuning.
- ``nss``: Nested Slice Sampling top-level API.
- ``integrator``: NSIntegrator for evidence tracking.
- ``slice_sampling``: Hit-and-Run Slice Sampling (from blackjax.mcmc.ss).
- ``utils``: Post-processing (finalise, sample, ess, log_weights).
- ``from_mcmc``: Build NS from MCMC kernels.
- ``particles``: Particle statistics (covariance, etc.).

Usage
-----
>>> from tengri.inference.backends.nested.nss import as_top_level_api
>>> from tengri.inference.backends.nested.utils import finalise, sample, ess
>>> algo = as_top_level_api(logprior_fn, loglik_fn, num_inner_steps=D)
>>> state = algo.init(particles)
>>> state, info = algo.step(key, state)
"""

from tengri.inference.backends.nested import (
    adaptive,
    base,
    from_mcmc,
    integrator,
    nss,
    particles,
    slice_sampling,
    utils,
)

__all__ = [
    "adaptive",
    "base",
    "from_mcmc",
    "integrator",
    "nss",
    "particles",
    "slice_sampling",
    "utils",
]

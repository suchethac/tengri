# SPDX-License-Identifier: BSD-3-Clause
"""Window-adapted GHMC hands the kernel the momentum inverse scale, not the inverse mass (#3318).

BlackJAX's GHMC squares a one-dimensional ``momentum_inverse_scale`` into an inverse mass,
so the window's adapted variance must enter as its square root. Passing the variance made
the hierarchical flat seam sample with an inverse mass equal to the variance squared. The
draws change at a fixed key, which is the named physics change this test pins.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np
import pytest

from tests.contract.test_chain_program import (
    _FLAT_SEAM_N,
    flat_seam_population,  # noqa: F401
)

pytestmark = pytest.mark.regression_bug

#: md5 of the flat seam's float64 ``psd_sigma`` draws before the fix, measured on the
#: pre-fix hand-off at this fixture and key. These draws must not recur.
PRE_FIX_PSD_SIGMA_MD5 = "7740370b713c10a184c9ca1e1ea63555"


def test_flat_seam_ghmc_draws_differ_from_pre_fix(flat_seam_population):  # noqa: F811
    """The two-galaxy seam fixture, GHMC, key ``PRNGKey(0)``: the draws are not the pre-fix draws.

    The chain must stay finite and move, and its step size must be finite.
    """
    from tengri.inference._hierarchical_flat import run_flat_sampler
    from tengri.inference.hierarchical import PopulationFitter

    factory, galaxies = flat_seam_population
    n_warmup, n_burnin, n_samples = _FLAT_SEAM_N
    fitter = PopulationFitter(factory, galaxies, _via_routing=True)
    seam = run_flat_sampler(
        fitter,
        "mcmc_ghmc",
        key=jax.random.PRNGKey(0),
        n_warmup=n_warmup,
        n_burnin=n_burnin,
        n_samples=n_samples,
        allow_unvalidated=True,
        verbose=False,
    )
    sigma = np.asarray(seam.shared_samples["psd_sigma"])
    digest = hashlib.md5(sigma.tobytes()).hexdigest()
    assert not digest.startswith(PRE_FIX_PSD_SIGMA_MD5), digest

    assert np.all(np.isfinite(sigma))
    assert sigma.std() > 1e-6, "the chain did not move"
    assert np.isfinite(seam.diagnostics["step_size"])
    assert np.mean(np.diff(sigma) != 0) > 0.5, "most GHMC steps must move the chain"

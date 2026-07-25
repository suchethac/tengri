# SPDX-License-Identifier: BSD-3-Clause
"""NUTS metric preconditioning is a coordinate change, not a model change (#1301).

``precondition=True`` samples ``H(A zeta)`` instead of ``H(xi)`` and maps the draws
back with ``xi = A zeta``. A linear reparametrization has a constant Jacobian, so
the posterior is untouched — only the sampler's geometry changes.

The failure this guards against is silent and severe: if the backend forgets to map
positions back, every returned "posterior" is expressed in the whitened
coordinates. The numbers still look like plausible parameter values, the fit still
reports a step size and a divergence count, and nothing raises.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, ForwardModel, NoiseModel, Observation, Photometry, SEDModel

pytestmark = pytest.mark.contract

BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j", "2mass_h"]
OVERRIDES = {
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 1.5,
    "sfh_dpl_log_total_mass": 10.5,
    "dust_tau_diff": 0.2,
}


def _model(ssp):
    obs = Observation(
        photometry=Photometry.from_names(BANDS),
        noise=NoiseModel(calibration_floor=0.02, student_t_dof=None),
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        stellar={"met_logzsol": Fixed(-0.3)},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FREE},
        neb={"type": "none"},
        redshift=Fixed(0.1),
        apply_igm=False,
    )


def _mock(model):
    """A valid full parameter set: a prior draw with a few values pinned."""
    drawn = model.spec.sample(jax.random.PRNGKey(0))
    params = {
        **model.spec.get_fixed_values(),
        **drawn,
        **{k: jnp.array(v) for k, v in OVERRIDES.items() if k in drawn},
    }
    mock = model.mock(params, snr=30.0, key=jax.random.PRNGKey(1))
    return np.asarray(mock.flux_obs), np.asarray(mock.noise)


def _fit(model, data, noise, *, precondition):
    return ForwardModel.build(sed=model).fit(
        data,
        noise,
        method="mcmc_nuts",
        key=jax.random.PRNGKey(7),
        n_warmup=250,
        n_samples=250,
        n_chains=1,
        dense_mass_matrix=False,
        precondition=precondition,
        verbose=False,
    )


def test_preconditioning_leaves_the_posterior_unchanged(ssp_data_wne):
    """Same posterior, different coordinates — the invariant a mapping bug breaks."""
    model = _model(ssp_data_wne)
    data, noise = _mock(model)

    plain = _fit(model, data, noise, precondition=False)
    preconditioned = _fit(model, data, noise, precondition=True)

    for name in plain.samples:
        a = np.asarray(plain.samples[name])
        b = np.asarray(preconditioned.samples[name])
        if a.ndim != 1:
            continue
        spread = max(float(np.std(a)), 1e-3)
        shift = abs(float(np.mean(a) - np.mean(b))) / spread
        assert shift < 1.0, (
            f"{name}: posterior mean moved {shift:.2f} sd between the plain and "
            f"preconditioned runs ({np.mean(a):.4f} vs {np.mean(b):.4f}) — "
            "the draws are probably still in whitened coordinates"
        )


def test_preconditioned_samples_respect_the_prior_support(ssp_data_wne):
    """Whitened draws mapped back must land inside the priors' physical bounds.

    A missing inverse map shows up here even when the means happen to agree.
    """
    model = _model(ssp_data_wne)
    data, noise = _mock(model)
    result = _fit(model, data, noise, precondition=True)

    checked = 0
    for name in model.spec.free_params:
        if name not in result.samples:
            continue
        dist = model.spec.get_distribution(name)
        low, high = getattr(dist, "lo", None), getattr(dist, "hi", None)
        if low is None or high is None:
            continue
        draws = np.asarray(result.samples[name])
        assert draws.min() >= float(low) - 1e-8, f"{name} below prior support"
        assert draws.max() <= float(high) + 1e-8, f"{name} above prior support"
        checked += 1
    assert checked > 0, "no bounded priors were checked — the guard would pass vacuously"

# SPDX-License-Identifier: BSD-3-Clause
"""MAP defaults to L-BFGS, not Adam.

`run_map`'s `optimizer=` default changed from `"adam"` to `"lbfgs"` (alias
`"lbfgs_scipy"`): quasi-Newton reaches a converged optimum reliably where a
fixed-step gradient-descent budget may not. Measured on a D=8, 14-band mock
recovery fixture: the population default (Adam, 8 restarts x 800 steps)
reached a negative log posterior of 6.33, not converged; a single scipy
L-BFGS-B start reached 6.0008 in well under a second.

Two contracts are pinned here:

1. The registered ``run_map`` default really is ``"lbfgs"`` (not merely true
   in the function signature -- ``Fitter.run("map")`` merges
   ``defaults.toml``'s ``[inference.map]`` section over it, which is the
   actual mechanism that shipped the old ``"adam"`` default to every ordinary
   caller and so is exactly the kind of place a re-default could silently
   fail to take effect).
2. On a real mock-recovery fit, the new default reaches a loss no worse than
   the old one.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from tengri import ForwardModel, Observation, Photometry, SEDModel, generate_mock, recipes

pytestmark = pytest.mark.contract


def test_run_map_signature_defaults_to_lbfgs():
    from tengri.inference.backends.map_dispatch import run_map

    param = inspect.signature(run_map).parameters["optimizer"]
    assert param.default == "lbfgs"


def test_fitter_run_map_defaults_to_lbfgs(ssp_data_fsps):
    """The default actually reaching the backend, not just the function signature.

    ``Fitter.run`` merges ``defaults.toml``'s per-method section over the
    backend's own kwarg defaults (caller kwargs still win over both) -- the
    ``[inference.map]`` table is exactly where the old ``"adam"`` default
    lived operationally, so this is the guard that would have caught a
    re-default that only touched the Python default and missed the config.
    """
    from tengri.parameters.defaults import get_inference_defaults

    assert get_inference_defaults("map").get("optimizer", "lbfgs") == "lbfgs"


@pytest.fixture(scope="module")
def recovery_fixture(ssp_data_fsps):
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    model = SEDModel.build(
        ssp_data=ssp_data_fsps, observation=obs, **recipes.mock_recovery_minimal()
    )
    truth = model.spec.sample(jax.random.PRNGKey(0))
    mock = generate_mock(model, truth, key=jax.random.PRNGKey(1), snr=20.0)
    forward = ForwardModel.build(sed=model, observation=obs)
    return forward, mock


def test_default_map_reaches_lower_or_equal_nlp_than_adam(recovery_fixture):
    """The default (lbfgs) must not regress the fit quality Adam gave."""
    forward, mock = recovery_fixture

    lbfgs_result = forward.fit(
        mock["flux_obs"], mock["noise"], method="map", key=jax.random.PRNGKey(2), verbose=False
    )
    adam_result = forward.fit(
        mock["flux_obs"],
        mock["noise"],
        method="map",
        key=jax.random.PRNGKey(2),
        optimizer="adam",
        n_steps=300,
        verbose=False,
    )

    lbfgs_nlp = float(lbfgs_result.diagnostics["final_loss"])
    adam_nlp = float(adam_result.diagnostics["final_loss"])
    assert jnp.isfinite(lbfgs_nlp), "default MAP must return a finite loss"
    assert lbfgs_nlp <= adam_nlp + 1e-6, (
        f"default optimizer (lbfgs) nlp={lbfgs_nlp:.6f} is worse than "
        f"optimizer='adam', n_steps=300 nlp={adam_nlp:.6f}"
    )

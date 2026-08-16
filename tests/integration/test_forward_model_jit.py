# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end JIT through ForwardModel (forward-model architecture §9.1).

The new shell must be transparent under jax.jit. A loss function that
wraps forward.predict_observables + a likelihood computation must JIT, run, and
take gradients without error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tests._grad_parity import assert_grad_matches_fd


@pytest.fixture
def sed_model_minimal(synthetic_ssp, simple_observation):
    """Minimal SED model for JIT testing."""
    from tengri import FIXED, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )


@pytest.mark.integration
def test_forward_model_end_to_end_jit(sed_model_minimal, simple_observation) -> None:
    """ForwardModel.predict must be transparent under jax.jit.

    A loss function that wraps ForwardModel.predict and performs a
    likelihood-like computation must JIT-compile, run, and produce
    finite gradients.
    """
    from tengri.forward.forward_model import ForwardModel

    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)

    @jax.jit
    def loss(params):
        pred = forward.predict_observables(params)
        # Expect {"phot_fnu": array(...)}
        if "phot_fnu" in pred:
            return jnp.sum(pred["phot_fnu"] ** 2)
        raise AssertionError(f"Prediction dict missing 'phot_fnu' key; got: {list(pred.keys())}")

    params = {name: jnp.float64(0.5) for name in sed_model_minimal.spec.free_params}
    value = loss(params)
    assert jnp.isfinite(value)

    grads = assert_grad_matches_fd(loss, params)
    assert all(jnp.isfinite(g).all() for g in grads.values())


@pytest.mark.integration
def test_forward_model_predict_output_structure(sed_model_minimal, simple_observation) -> None:
    """ForwardModel.predict returns the correct output structure."""
    from tengri.forward.forward_model import ForwardModel

    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    params = {name: jnp.float64(0.5) for name in sed_model_minimal.spec.free_params}

    pred = forward.predict_observables(params)

    assert isinstance(pred, dict), f"Expected dict, got {type(pred)}"
    assert "phot_fnu" in pred, f"Missing 'phot_fnu'; got keys: {list(pred.keys())}"
    assert jnp.isfinite(pred["phot_fnu"]).all()


@pytest.mark.integration
def test_forward_model_predict_deterministic(sed_model_minimal, simple_observation) -> None:
    """ForwardModel.predict is deterministic (same params → same result)."""
    from tengri.forward.forward_model import ForwardModel

    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    params = {name: jnp.float64(0.5) for name in sed_model_minimal.spec.free_params}

    pred1 = forward.predict_observables(params)
    pred2 = forward.predict_observables(params)

    assert jnp.allclose(pred1["phot_fnu"], pred2["phot_fnu"])


@pytest.mark.integration
def test_forward_model_closure_audit() -> None:
    """ForwardModel.predict must not capture data-file globals (§9.2).

    Architecture spec ``docs/dev/archive/forward-model-architecture.md`` §9.2:
    data files (SSP grids, filter matrices, templates) MUST flow through
    component-owned frozen state, never through free-variable closure
    capture in a ``@jit``-able function. The closure count is a useful
    proxy for "how much hidden state is this function carrying?".
    """
    from tengri.forward.forward_model import ForwardModel

    closure = ForwardModel.predict.__closure__ or ()
    assert len(closure) == 0, (
        f"ForwardModel.predict closes over {len(closure)} free variables; "
        "data files must flow through component state, not closures "
        "(architecture spec §9.2)."
    )

# SPDX-License-Identifier: BSD-3-Clause
"""Probe tests for the standard Fitter path on hierarchical fits.

Bypasses the legacy ``_maybe_population_delegate`` to exercise the
standard NUTS/MAP/VI machinery on a PopulationSEDModel. These tests
discover what breaks in the standard path so the deep removal can
proceed deliberately.

Part of PR #239's plan; deleted before the final removal PR merges.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel

pytestmark = pytest.mark.contract


def _template(synthetic_ssp, simple_observation):
    from tengri import FIXED, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED, "log_peak_sfr": Uniform(-1.0, 3.0)},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=0.05,
    )


def test_fitter_constructs_via_standard_path_for_population(
    synthetic_ssp, simple_observation
) -> None:
    """Fitter(forward, _skip_population_delegate=True) constructs on a hierarchical fit.

    Auto-extracts (N, n_filters) data from pop.batched_data(); spec is
    the PopulationSpecView from #241. The construction completes
    without the legacy PopulationFitter delegation.
    """
    from tengri.inference.fitter import Fitter

    template = _template(synthetic_ssp, simple_observation)
    N = 3
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[
            {"flux_obs": jnp.ones(5) * 1e-18, "noise": jnp.ones(5) * 1e-19} for _ in range(N)
        ],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    fitter = Fitter(forward, _skip_population_delegate=True)
    assert fitter.model is forward
    assert fitter.data.shape == (N, 5)
    assert fitter.noise.shape == (N, 5)
    # Spec is the wrapped PopulationSpecView
    from tengri.parameters._population_view import PopulationSpecView

    assert isinstance(fitter.spec, PopulationSpecView)
    # No delegate
    assert fitter._population_delegate is None


@pytest.mark.xfail(
    reason=(
        "Fitter._initialize_unbounded produces scalar xi per free name. "
        "Hierarchical needs (N,)-shape for per-galaxy params. "
        "Next incremental PR."
    )
)
def test_fitter_loss_fn_evaluates_on_hierarchical(synthetic_ssp, simple_observation) -> None:
    """neg_log_posterior_fn evaluates to a scalar on a hierarchical fit.

    The chi^2 broadcasts over the galaxy axis naturally; the prior
    term sums xi^T xi where xi has (N,) leading shape for per-galaxy
    free params. End-to-end: forward.predict -> chi^2 -> scalar.

    Couplings fixed so far (incremental):
    - PopulationSEDModel.__hash__ → id(self) (compile-cache hashability)
    - PopulationSpecView.resolve_mirrors → template delegation
    - ForwardModel.predict_photometry → routes through self.predict
      so batched output flows instead of falling back to scalar SED

    Remaining: Fitter._initialize_unbounded must produce (N,)-shape
    xi for per-galaxy free params; currently scalar everywhere.
    """
    from tengri.inference.fitter import Fitter

    template = _template(synthetic_ssp, simple_observation)
    N = 3
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[
            {"flux_obs": jnp.ones(5) * 1e-18, "noise": jnp.ones(5) * 1e-19} for _ in range(N)
        ],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    fitter = Fitter(forward, _skip_population_delegate=True)

    # Draw a sample from the spec (gives batched-shape params)
    import jax

    key = jax.random.PRNGKey(0)
    xi_init = fitter._initialize_unbounded(key)

    # Evaluate the loss function on this xi
    loss_fn = fitter._get_or_build_loss_fn()
    loss_value = loss_fn(xi_init, fitter._data_args)
    # Just verify we got a finite scalar
    assert loss_value.shape == ()
    assert jnp.isfinite(loss_value), (
        f"loss is non-finite: {float(loss_value)}; "
        f"likely a shape-mismatch in the chi^2 broadcast or prior penalty"
    )

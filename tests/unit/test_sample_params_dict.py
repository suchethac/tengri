# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :func:`tengri.forward.orchestrator.sample_params_dict`.

The helper closes the user-facing loop: components in → params dict
out, ready for :func:`run_components`. These tests verify:

- Every declared parameter is sampled with the right shape.
- ``overrides`` pin specific keys without re-sampling.
- ``redshift`` (bare-name allowlist) can be supplied via overrides
  even when no component declares it.
- Two calls with the same key produce identical draws.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.forward.orchestrator import (
    merge_declared_parameters,
    run_components,
    sample_params_dict,
    slice_params_for_component,
)


@pytest.fixture
def chain():
    return [RadioSEDComponent(), DustAttenuationSEDComponent(), IGMSEDComponent()]


@pytest.mark.unit
def test_sample_returns_one_value_per_declared_parameter(chain):
    """Every merged parameter has a sampled value in the output."""
    merged = merge_declared_parameters(chain)
    out = sample_params_dict(chain, jax.random.PRNGKey(0))
    for name in merged:
        assert name in out
        assert jnp.ndim(out[name]) == 0  # scalar prior draws


@pytest.mark.unit
def test_overrides_pin_values_without_resampling(chain):
    """Override keys appear verbatim; non-override keys are sampled."""
    out = sample_params_dict(
        chain,
        jax.random.PRNGKey(1),
        overrides={"radio_q_ir": 2.64, "dust_tau_v": 0.3},
    )
    assert float(out["radio_q_ir"]) == pytest.approx(2.64)
    assert float(out["dust_tau_v"]) == pytest.approx(0.3)
    # Untouched key still gets a draw.
    assert "igm_log_nhi" in out


@pytest.mark.unit
def test_overrides_can_supply_bare_redshift(chain):
    """`redshift` is in BARE_NAME_ALLOWLIST and no component declares it.

    The override path must still inject it into the output so the
    caller can drive the pipeline with a fixed redshift.
    """
    out = sample_params_dict(chain, jax.random.PRNGKey(2), overrides={"redshift": 0.7})
    assert float(out["redshift"]) == pytest.approx(0.7)


@pytest.mark.unit
def test_same_key_produces_identical_draws(chain):
    """Determinism: same PRNG key + same components → same params dict."""
    a = sample_params_dict(chain, jax.random.PRNGKey(42))
    b = sample_params_dict(chain, jax.random.PRNGKey(42))
    assert a.keys() == b.keys()
    for k in a:
        assert float(a[k]) == pytest.approx(float(b[k]))


@pytest.mark.unit
def test_end_to_end_drives_run_components(chain):
    """sample_params_dict output feeds run_components without massaging."""
    from tengri.protocols import PipelineState

    params = sample_params_dict(
        chain,
        jax.random.PRNGKey(7),
        overrides={"redshift": 0.5},
    )
    wave = jnp.logspace(2, 8, 256)
    state = PipelineState(
        wave=wave,
        sed_intrinsic=jnp.ones_like(wave) * 1e30,
        sed_observed=jnp.ones_like(wave),
    )
    final = run_components(chain, state, params)
    # All three adapters' contributions should be visible.
    assert final.sed_attenuated is not None
    assert "L_ir" in final.derived
    # IGM modified the observed-frame SED.
    assert final.sed_observed is not None


@pytest.mark.unit
def test_overrides_for_undeclared_non_bare_key_are_silently_dropped(chain):
    """An override for a key no component owns and not in the allowlist
    must NOT leak into the output — that would be a silent contract
    violation. The component-prefix slicer would never see it anyway,
    but the helper itself shouldn't surface it.
    """
    out = sample_params_dict(chain, jax.random.PRNGKey(3), overrides={"unknown_key": 1.0})
    assert "unknown_key" not in out


@pytest.mark.unit
def test_slicing_round_trips(chain):
    """Sampled params dict slices cleanly per component."""
    params = sample_params_dict(chain, jax.random.PRNGKey(11), overrides={"redshift": 0.3})
    for comp in chain:
        sliced = slice_params_for_component(comp, params)
        assert "redshift" in sliced  # bare-allowlist threading
        prefixes = (
            (comp.parameter_prefix,)
            if isinstance(comp.parameter_prefix, str)
            else tuple(comp.parameter_prefix)
        )
        for k in sliced:
            assert k == "redshift" or any(k.startswith(p) for p in prefixes)

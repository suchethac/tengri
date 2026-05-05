# SPDX-License-Identifier: BSD-3-Clause
"""Phase II-2 stellar skeleton tests.

The full physics migration is deferred. These tests validate only the
**design-decision surface**:

- Tuple ``parameter_prefix`` is accepted by the orchestrator.
- :class:`StellarSEDComponent` declares parameters across the three
  prefixes (``sfh_``, ``met_``, ``chem_``).
- :meth:`StellarSEDComponent.apply` raises ``NotImplementedError`` (the
  skeleton is intentionally non-functional).
- :func:`merge_declared_parameters` accepts a mixed-prefix component
  alongside single-prefix components without false-positive collision.
"""

from __future__ import annotations

import pytest

from tengri.components.stellar import (
    StellarSEDComponent,
    StellarSEDComponentConfig,
)
from tengri.core import BARE_NAME_ALLOWLIST, PipelineState, SEDComponent
from tengri.forward.orchestrator import (
    merge_declared_parameters,
    slice_params_for_component,
)


@pytest.mark.unit
def test_skeleton_satisfies_protocol():
    """Skeleton duck-types as :class:`SEDComponent`."""
    assert isinstance(StellarSEDComponent(), SEDComponent)


@pytest.mark.unit
def test_tuple_prefix_is_accepted_by_slicer():
    """slice_params_for_component picks up all three prefixes."""
    comp = StellarSEDComponent()
    params = {
        "redshift": 0.5,
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "met_logzsol": -0.2,
        "chem_yield": 0.02,
        "dust_tau_v": 0.3,  # must NOT leak in
    }
    sliced = slice_params_for_component(comp, params)
    assert "sfh_tsnorm_log_peak_sfr" in sliced
    assert "met_logzsol" in sliced
    assert "chem_yield" in sliced
    assert "redshift" in sliced  # bare-name allowlist
    assert "dust_tau_v" not in sliced


@pytest.mark.unit
def test_declared_parameters_obey_tuple_prefix_rule():
    """Every declared name starts with one of the three prefixes."""
    comp = StellarSEDComponent()
    decls = comp.declared_parameters()
    assert len(decls) > 0
    allowed_prefixes = comp.parameter_prefix
    for decl in decls:
        assert any(decl.name.startswith(p) for p in allowed_prefixes) or (
            decl.name in BARE_NAME_ALLOWLIST
        ), f"{decl.name} violates the prefix rule"


@pytest.mark.unit
def test_merge_with_mixed_prefix_component():
    """merge_declared_parameters accepts a tuple-prefix component."""
    from tengri.components.igm.component import IGMSEDComponent

    merged = merge_declared_parameters([StellarSEDComponent(), IGMSEDComponent()])
    # Stellar contributes both sfh_ and met_ keys.
    assert any(k.startswith("sfh_") for k in merged)
    assert any(k.startswith("met_") for k in merged)
    assert any(k.startswith("igm_") for k in merged)


@pytest.mark.unit
def test_apply_without_ssp_data_raises():
    """Component constructed without ssp_data refuses to run apply()."""
    import jax.numpy as jnp

    comp = StellarSEDComponent()  # ssp_data defaults to None
    state = PipelineState(wave=jnp.linspace(1e3, 1e8, 64))
    with pytest.raises(ValueError, match="requires ssp_data"):
        comp.apply(state, {})


@pytest.mark.unit
def test_apply_unsupported_sfh_model_raises():
    """Phase II-2.5 supports tsnorm/dpl/continuity/dirichlet/dense_basis;
    other registered modes (lnorm, snorm, ...) raise NotImplementedError
    until their orchestrator-vs-legacy equivalence is pinned."""
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    fake_ssp = SSPData(
        ssp_wave=jnp.array([1.0, 2.0, 3.0]),
        ssp_flux=jnp.zeros((1, 1, 3)),
        ssp_lg_age_gyr=jnp.array([0.0]),
        ssp_lgmet=jnp.array([-2.0]),
    )
    # 'lnorm' (lognormal) is registered in SFH_REGISTRY but its parity
    # against legacy has not been pinned by an equivalence test — the
    # component refuses to dispatch.
    comp = StellarSEDComponent(
        config=StellarSEDComponentConfig(sfh_model="lnorm"),
        ssp_data=fake_ssp,
    )
    state = PipelineState(wave=fake_ssp.ssp_wave)
    with pytest.raises(NotImplementedError, match="not yet pinned"):
        comp.apply(state, {})


@pytest.mark.unit
def test_config_validation_unknown_sfh_model():
    """Invalid sfh_model is caught at declared_parameters() time."""
    bad = StellarSEDComponent(config=StellarSEDComponentConfig(sfh_model="this_does_not_exist"))
    with pytest.raises(ValueError, match="not in SFH_REGISTRY"):
        bad.declared_parameters()

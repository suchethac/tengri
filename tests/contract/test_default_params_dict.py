# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :func:`tengri.forward.orchestrator.default_params_dict`.

The deterministic sibling of :func:`sample_params_dict`: components in →
params dict out, every value read off the declared prior's ``default``
instead of drawn from it. It exists so that a caller who wants "this
pipeline at its declared defaults" can ask for it, rather than typing a
literal that copies the declaration and then falls behind it (#1832).

Why a helper the caller invokes, and not an auto-fill inside
:func:`run_components`: filling silently is what made #1706 invisible for
as long as it was. A parameter nobody supplies must still raise. Asking
for the declared defaults is an explicit request; being handed them
behind your back is not.

These tests verify:

- Every declared parameter gets a value, for every component in the registry.
- The value *is* the declared default, not a placeholder.
- ``overrides`` pin keys, including bare-allowlist ``redshift``.
- A declaration carrying no default raises and names itself, rather than
  being dropped — a dropped key is precisely the #1832 failure.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.forward.orchestrator import (
    default_params_dict,
    merge_declared_parameters,
    run_components,
)
from tengri.parameters.priors import Uniform
from tengri.protocols.component import ParamDeclaration


@pytest.fixture
def chain():
    return [RadioSEDComponent(), DustAttenuationSEDComponent(), IGMSEDComponent()]


@pytest.mark.unit
def test_returns_one_value_per_declared_parameter(chain):
    """Every merged parameter has a value in the output."""
    merged = merge_declared_parameters(chain)
    out = default_params_dict(chain)
    for name in merged:
        assert name in out
        assert jnp.ndim(out[name]) == 0  # scalar


@pytest.mark.unit
def test_values_are_the_declared_defaults(chain):
    """The value is read off the declaration, not invented.

    A helper that returned zeros would satisfy "the key is present" and
    silently change the physics of every caller.
    """
    merged = merge_declared_parameters(chain)
    out = default_params_dict(chain)
    for name, prior in merged.items():
        assert float(out[name]) == pytest.approx(float(prior.default)), name


@pytest.mark.unit
def test_free_by_default_parameters_are_supplied_too(chain):
    """The #1832 case: a ``Uniform`` declaration is free, and still has a default.

    ``spec.get_fixed_values()`` supplies only the ``Fixed`` ones, which is
    why the X-ray offsets went missing. This helper reads the declaration,
    so free and fixed are treated alike.
    """
    out = default_params_dict([XRaySEDComponent()])
    assert float(out["xray_det_hmxb"]) == pytest.approx(0.0)
    assert float(out["xray_det_lmxb"]) == pytest.approx(0.0)


@pytest.mark.unit
def test_overrides_pin_values(chain):
    """Override keys appear verbatim; the rest keep their declared defaults."""
    out = default_params_dict(chain, overrides={"radio_q_ir": 2.11, "dust_tau_v": 0.3})
    assert float(out["radio_q_ir"]) == pytest.approx(2.11)
    assert float(out["dust_tau_v"]) == pytest.approx(0.3)
    assert "igm_log_nhi" in out


@pytest.mark.unit
def test_overrides_can_supply_bare_redshift(chain):
    """``redshift`` is in BARE_NAME_ALLOWLIST and no component declares it."""
    out = default_params_dict(chain, overrides={"redshift": 0.7})
    assert float(out["redshift"]) == pytest.approx(0.7)


@pytest.mark.unit
def test_overrides_for_undeclared_non_bare_key_are_dropped(chain):
    """Matches ``sample_params_dict``: an unowned key must not leak through."""
    out = default_params_dict(chain, overrides={"unknown_key": 1.0})
    assert "unknown_key" not in out


@pytest.mark.unit
def test_declaration_without_a_default_raises_and_names_itself():
    """A default-less declaration must fail loudly, not vanish from the dict.

    Dropping it would hand back a dict missing a key the component indexes
    — #1832 exactly, one layer down and harder to see.
    """

    class _StubComponent:
        name = "stub"
        parameter_prefix = "stub_"

        def declared_parameters(self):
            return [ParamDeclaration("stub_knob", Uniform(1.0, 2.0), "no default declared")]

    with pytest.raises(ValueError, match=r"stub_knob"):
        default_params_dict([_StubComponent()])


@pytest.mark.unit
def test_end_to_end_drives_run_components(chain):
    """Output feeds ``run_components`` without massaging, as the sibling does."""
    from tengri.protocols import ForwardState

    params = default_params_dict(chain, overrides={"redshift": 0.5})
    wave = jnp.logspace(2, 8, 256)
    state = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.ones_like(wave) * 1e30,
        sed_observed=jnp.ones_like(wave),
    )
    final = run_components(chain, state, params)
    assert final.sed_attenuated is not None
    assert "L_ir" in final.derived


@pytest.mark.unit
def test_every_registry_component_can_supply_its_own_declarations():
    """Registry-wide: no component may declare a parameter it cannot default.

    Measured at the time of writing: 34 components, 214 declarations, zero
    without a default. This pins that, so a new declaration that forgets
    ``default=`` fails here rather than as a ``KeyError`` from inside
    whichever component happened to index it first.
    """
    from tengri.forward.component_factory import _REGISTRY

    offenders = []
    checked = 0
    for key, cls in sorted(_REGISTRY.items(), key=lambda kv: str(kv[0])):
        try:
            component = cls()
            declared = merge_declared_parameters([component])
        except Exception:  # needs constructor args or data — not this test's subject
            continue
        if not declared:
            continue
        checked += 1
        try:
            supplied = default_params_dict([component])
        except ValueError as exc:
            offenders.append(f"{key}: {exc}")
            continue
        missing = sorted(set(declared) - set(supplied))
        if missing:
            offenders.append(f"{key}: {missing}")

    # Without this the sweep passes vacuously wherever the constructors fail
    # (no data files, say) — a green test that checked nothing. Measured at 28
    # parameter-declaring components that need no constructor args; the floor is
    # deliberately well below that, so it reports emptiness rather than tracking
    # the registry's normal growth and churn.
    assert checked >= 10, f"only {checked} components were actually checked — sweep is empty"
    assert not offenders, "components whose declarations cannot be defaulted:\n" + "\n".join(
        offenders
    )

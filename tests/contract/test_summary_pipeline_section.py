# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Pipeline chain section in ``config.display.summary``.

These tests cover the two new module-level helpers
(``_safe_call``, ``_component_config_summary``) and the defensive
behavior of the Pipeline section when the component chain is
unavailable. They use ``unittest.mock`` so they run with no SSP data
and no JAX compilation cost.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tengri.config.display import (
    _component_config_summary,
    _safe_call,
    summary,
)

pytestmark = pytest.mark.contract

# ── _safe_call ───────────────────────────────────────────────────────


def test_safe_call_returns_tuple_when_method_exists():
    component = SimpleNamespace(publishes=lambda: [SimpleNamespace(name="L_ir")])
    out = _safe_call(component, "publishes")
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert out[0].name == "L_ir"


def test_safe_call_returns_empty_tuple_when_method_missing():
    component = SimpleNamespace()
    assert _safe_call(component, "publishes") == ()


def test_safe_call_swallows_exceptions():
    def boom():
        raise RuntimeError("nope")

    component = SimpleNamespace(publishes=boom)
    assert _safe_call(component, "publishes") == ()


# ── _component_config_summary ────────────────────────────────────────


def test_config_summary_renders_known_keys():
    config = SimpleNamespace(
        sfh_model="dpl",
        metallicity_model="delta",
        n_grid=64,
        # Unknown key — should be ignored:
        not_a_known_key="ignored",
    )
    component = SimpleNamespace(config=config)
    out = _component_config_summary(component)
    assert out.startswith("[")
    assert out.endswith("]")
    assert "sfh_model=dpl" in out
    assert "metallicity_model=delta" in out
    assert "not_a_known_key" not in out


def test_config_summary_caps_at_three_entries():
    config = SimpleNamespace(
        sfh_model="dpl",
        metallicity_model="delta",
        n_grid=64,
        backend="cue",  # 4th key — should be dropped to keep summary readable
    )
    out = _component_config_summary(SimpleNamespace(config=config))
    assert out.count("=") == 3


def test_config_summary_empty_when_no_known_keys():
    config = SimpleNamespace(only_unknown="x")
    assert _component_config_summary(SimpleNamespace(config=config)) == ""


def test_config_summary_empty_when_no_config():
    assert _component_config_summary(SimpleNamespace()) == ""


# ── summary() defensive behavior ────────────────────────────────────


def _minimal_model(*, chain_raises: bool = False, chain=None):
    """Build a MagicMock with the minimum surface ``summary()`` reads."""
    model = MagicMock()
    model.spec.mean_sfh_type = ["dpl"]
    model.spec.nebular_mode = "off"
    model.spec.n_free = 8
    model.ssp_data.ssp_flux.shape = (10, 20, 30)
    model.ssp_data.ssp_wave = [1000.0, 30000.0]
    model.filter_waves = None
    model.z_fixed = 0.05
    model._forward_dtype = "float64"
    model.precomputed.photometry = None
    model.precomputed.spectroscopy = None
    model.precomputed.photometry_ztable = None
    model.hybrid.photometry = None
    model._compositional.photometry = None
    model._dust_emission_model = None
    model._agn_model = None
    model._uses_igm = False
    model._uses_radio = False
    model._uses_xray = False
    model._uses_shock = False
    model.uses_stochastic_sfh = False
    model.n_grid = 0

    if chain_raises:
        model._build_component_chain.side_effect = RuntimeError("broken model")
    else:
        model._build_component_chain.return_value = chain or []
    return model


def test_summary_includes_pipeline_section_with_chain():
    stellar = SimpleNamespace(
        name="stellar",
        config=SimpleNamespace(sfh_model="dpl", n_grid=64),
        publishes=lambda: [SimpleNamespace(name="lnu_age"), SimpleNamespace(name="sfr")],
        requires=lambda: [],
        requires_optional=lambda: [],
    )
    dust = SimpleNamespace(
        name="dust",
        config=SimpleNamespace(law_bc="calzetti", law_diff="calzetti"),
        publishes=lambda: [SimpleNamespace(name="L_ir")],
        requires=lambda: [SimpleNamespace(name="lnu_age")],
        requires_optional=lambda: [],
    )
    model = _minimal_model(chain=[stellar, dust])

    out = summary(model)

    # Section header + per-component rendering
    assert "Pipeline (2 components):" in out
    assert "1. stellar" in out
    assert "2. dust" in out
    # Bracketed config snippet. Agreeing screens read back as the single `law`
    # the grammar accepts, so the summary must not name a key the user did not
    # write -- and must not spend two of its three displayed slots saying
    # calzetti twice.
    assert "sfh_model=dpl" in out
    assert "law=calzetti" in out
    assert "law_bc=" not in out
    assert "law_diff=" not in out
    # Publishes / requires lines
    assert "publishes: lnu_age, sfr" in out
    assert "publishes: L_ir" in out
    assert "reads:     lnu_age" in out


def test_summary_keeps_both_screens_when_the_laws_differ():
    """Differing screens must still print both -- the collapse is not a rename.

    This is the case that makes collapsing the agreeing pair safe. A summary
    that printed only one law here would hide a two-component model whose
    birth-cloud and diffuse screens use different attenuation curves, which is
    a physically distinct model and not a display detail.
    """
    dust = SimpleNamespace(
        name="dust",
        config=SimpleNamespace(law_bc="calzetti", law_diff="cardelli89"),
        publishes=lambda: [SimpleNamespace(name="L_ir")],
        requires=lambda: [],
        requires_optional=lambda: [],
    )

    out = summary(_minimal_model(chain=[dust]))

    assert "law_bc=calzetti" in out
    assert "law_diff=cardelli89" in out
    # Never the collapsed spelling: there is no single law to name.
    assert "law=calzetti" not in out


def test_summary_pipeline_section_defensive_when_chain_raises():
    model = _minimal_model(chain_raises=True)
    out = summary(model)
    assert "unable to build chain: RuntimeError" in out
    # The rest of the summary should still render
    assert "SSP grid" in out
    assert "Parameters:" in out


def test_summary_skips_pipeline_section_when_chain_empty():
    model = _minimal_model(chain=[])
    out = summary(model)
    assert "Pipeline (" not in out
    # Other sections still present
    assert "Parameters:  8 free" in out


#: The line ``summary()`` renders for each optional Protocol method.
_PROTOCOL_LINE = {
    "requires": "reads:",
    "requires_optional": "reads*:",
    "publishes": "publishes:",
}


def test_summary_handles_a_component_implementing_none_of_the_protocol():
    """A component with no Protocol methods at all shouldn't crash."""
    component = SimpleNamespace(name="partial", config=None)
    out = summary(_minimal_model(chain=[component]))
    assert "1. partial" in out
    assert not [lab for lab in _PROTOCOL_LINE.values() if lab in out]


@pytest.mark.parametrize("missing", sorted(_PROTOCOL_LINE))
def test_summary_handles_a_component_missing_one_protocol_method(missing):
    """Exactly the missing method's line disappears; the others still render.

    This was previously parametrized over the same three names while building a
    component with *none* of them, so ``missing`` was never read — three
    identical runs reported as three passes, and the "missing exactly one"
    cases the parametrization named were not covered at all. The all-missing
    case it actually tested now has its own test above.
    """
    methods = {
        "requires": lambda: [SimpleNamespace(name="L_bol")],
        "requires_optional": lambda: [SimpleNamespace(name="L_agn")],
        "publishes": lambda: [SimpleNamespace(name="L_ir")],
    }
    del methods[missing]
    component = SimpleNamespace(name="partial", config=None, **methods)

    out = summary(_minimal_model(chain=[component]))

    assert "1. partial" in out
    rendered = {lab for lab in _PROTOCOL_LINE.values() if lab in out}
    expected = {lab for key, lab in _PROTOCOL_LINE.items() if key != missing}
    assert rendered == expected, (
        f"with {missing}() absent, expected lines {sorted(expected)} but "
        f"summary rendered {sorted(rendered)}"
    )

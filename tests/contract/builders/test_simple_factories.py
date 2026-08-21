# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.builders.{igm,radio,xray} — simple single-axis components.

These three modules share the design — variant-string selectors with a
component-wide param set — so the tests are parametrized across them.

What we pin down here:

1. Every variant in the parser's ``_VALID_*_TYPES`` enum surfaces as a
   factory.
2. Factory output is shape-correct: ``{'type': <variant>, '*': FIXED}``
   on a default call, with per-param overrides honored.
3. Per-param :class:`Distribution` overrides round-trip through the
   parser — the factory dict is byte-equivalent to a hand-written one.
4. Validation rejects typos with a helpful list of valid names.
5. The IGM-specific boolean flags (``patchy`` / ``dla``) and the
   auto-enable behavior when a flag-conditional param is supplied.

Wildcard (``*=FREE``) behavior is intentionally **not** tested for
round-trip activation: the parser does not flip these components'
params via the wildcard. That's a parser-side limitation; the factory
faithfully emits the wildcard, and a hand-written dict behaves
identically.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract

from tengri import FIXED, Uniform, builders, parse_groups


@pytest.mark.parametrize(
    "module_name,representative_variant,sample_short_param",
    [
        ("radio", "condon92", "q_ir"),
        ("xray", "simple", "delta_alpha_ox"),
        ("igm", "inoue14", None),  # IGM tested separately for flags
    ],
)
def test_module_surface_and_default_dict(
    module_name: str,
    representative_variant: str,
    sample_short_param: str | None,
) -> None:
    mod = getattr(builders, module_name)
    assert "none" in mod.available()
    assert representative_variant in mod.available()
    out = getattr(mod, representative_variant)()
    if module_name == "radio":
        # #1980: radio's {'type': name} spelling is retired — the legacy-name
        # factory emits the composable sf/agn resolution instead.
        assert "type" not in out
        assert out["sf"]["type"] and out["agn"]["type"]
    else:
        assert out["type"] == representative_variant
    assert out["all_params"] is FIXED
    if sample_short_param is not None:
        sig = inspect.signature(getattr(mod, representative_variant))
        assert sample_short_param in sig.parameters, (
            f"{module_name}.{representative_variant} signature is missing "
            f"the documented short-form param {sample_short_param!r}"
        )


# ── Per-param override round-trips ────────────────────────────────


def test_radio_q_ir_override_round_trips() -> None:
    """An explicit Distribution on a radio param must surface as FREE."""
    spec = parse_groups(
        sfh={"type": "dpl"},
        radio=builders.radio.condon92(q_ir=Uniform(2.0, 3.0)),
    )
    assert "radio_q_ir" in spec.free_params


def test_xray_delta_alpha_ox_override_round_trips() -> None:
    spec = parse_groups(
        sfh={"type": "dpl"},
        xray=builders.xray.simple(delta_alpha_ox=Uniform(-0.3, 0.3)),
    )
    assert "xray_delta_alpha_ox" in spec.free_params


# ── Validation: typos and bad wildcard ────────────────────────────


def test_unknown_kwarg_raises_with_valid_list() -> None:
    with pytest.raises(TypeError, match="condon92") as exc:
        builders.radio.condon92(q_irrrr=Uniform(2.0, 3.0))
    msg = str(exc.value)
    assert "q_irrrr" in msg
    assert "q_ir" in msg


def test_invalid_wildcard_raises() -> None:
    with pytest.raises(ValueError, match="FREE or FIXED"):
        builders.radio.condon92(_="free")


# ── IGM-specific: boolean flags + auto-enable ─────────────────────


def test_igm_patchy_flag_surfaces_in_output() -> None:
    out = builders.igm.inoue14(patchy=True)
    assert out["patchy"] is True
    assert "dla" not in out  # not set


def test_igm_dla_flag_surfaces_in_output() -> None:
    out = builders.igm.inoue14(dla=True)
    assert out["dla"] is True


def test_igm_dla_param_auto_enables_dla_flag() -> None:
    """Providing a DLA-conditional param must imply ``dla=True``.

    Otherwise the parser would silently ignore the override because the
    DLA block isn't active. The auto-enable saves users from a
    surprising silent-no-op.
    """
    out = builders.igm.inoue14(log_n_hi=Uniform(20.0, 22.0))
    assert out["dla"] is True
    assert out["log_n_hi"] == Uniform(20.0, 22.0)


def test_igm_dla_param_round_trips_to_free() -> None:
    """A DLA param override must produce a free ``dla_*`` parameter."""
    spec = parse_groups(
        sfh={"type": "dpl"},
        igm=builders.igm.inoue14(log_n_hi=Uniform(20.0, 22.0)),
    )
    assert "dla_log_n_hi" in spec.free_params


def test_igm_none_has_no_flags_in_signature() -> None:
    """``igm.none`` is a no-op; no flags or params to expose."""
    sig = inspect.signature(builders.igm.none)
    assert list(sig.parameters) == ["all_params"]


# ── Module-level wiring ───────────────────────────────────────────


@pytest.mark.parametrize("mod_name", ["igm", "radio", "xray"])
def test_module_in_builders_namespace(mod_name: str) -> None:
    assert hasattr(builders, mod_name), f"builders.{mod_name} missing"
    mod = getattr(builders, mod_name)
    assert callable(mod.available)
    assert isinstance(mod.available(), list)
    assert mod.available() == sorted(mod.available())


def test_all_three_factories_emit_type_key() -> None:
    """Sanity check across every variant in all three modules.

    #1980: radio's {'type': name} spelling is retired, so its legacy-name
    factories emit the composable sf/agn axes; igm and xray keep the type key.
    """
    for mod_name in ("igm", "xray"):
        mod = getattr(builders, mod_name)
        for variant in mod.available():
            out = getattr(mod, variant)()
            assert out["type"] == variant
            assert out["all_params"] is FIXED
    for variant in builders.radio.available():
        out = getattr(builders.radio, variant)()
        assert "type" not in out, f"radio.{variant} still emits the retired type key"
        assert out["sf"]["type"] and out["agn"]["type"]
        assert out["all_params"] is FIXED

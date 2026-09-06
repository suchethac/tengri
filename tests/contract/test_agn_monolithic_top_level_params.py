# SPDX-License-Identifier: BSD-3-Clause
"""A monolithic AGN model's parameters ARE the ``agn`` top level (R27).

The nesting guard that routes ``agn_tau_skirtor`` into ``agn={'torus': {...}}``
is a statement about the **composable** model, whose six sub-blocks own the
parameters between them. A non-composable model has no sub-block to nest under
-- and ``agn={'type': 'kd18_agnfitter', 'disc': {...}}`` is refused by the
monolithic/sub-block check -- so applying the guard there leaves a build with no
working spelling at all: two hops, the second a refusal.

Measured before the fix (all 13 non-composable names, one of their own declared
parameters at the top level): 13 of 13 raised
``'agn_log_mbh' is a 'agn.disc' parameter, not a 'agn' one ... Nest it``.

The registry these tests enumerate is
:func:`tengri.components.agn.unified.monolithic_agn_model_names` -- the two
halves of ``agn={'type': ...}``'s non-composable menu: the deprecated preset
names in ``_AGN_PRESETS``, which route through the composable runner with fixed
block selectors, and the self-contained names in ``_SELF_CONTAINED_AGN_MODELS``,
which resolve to their own forward function. Deliberately enumerated, never
hand-listed: a per-type allowlist would pass while a newly registered model
stayed broken.
"""

import warnings

import pytest

pytestmark = pytest.mark.contract

from tengri.components.agn._params import PARAMS
from tengri.components.agn.blocks._consumes import monolithic_agn_declared_params
from tengri.components.agn.unified import monolithic_agn_model_names
from tengri.parameters import DEFAULT, Fixed, parse_groups
from tengri.protocols.component import declared_default

_MONOLITHIC = sorted(monolithic_agn_model_names())


def _parse(agn: dict):
    """Parse a monolithic AGN group, muting deprecation/recipe advisories."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_groups(
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.5),
        )


def test_the_monolithic_registry_is_not_empty():
    """Guard the guard: an empty registry would make every case below vacuous."""
    assert len(_MONOLITHIC) >= 13, _MONOLITHIC
    assert "composable" not in _MONOLITHIC


@pytest.mark.parametrize("model", _MONOLITHIC)
def test_monolithic_model_declares_parameters(model):
    """Every non-composable model declares at least its own normalization knobs."""
    declared = monolithic_agn_declared_params(model)
    assert "agn_log_lbol" in declared, sorted(declared)


@pytest.mark.parametrize("model", _MONOLITHIC)
def test_every_declared_parameter_is_accepted_at_the_agn_top_level(model):
    """Each declared parameter, written flat with its own declared default,
    builds and lands in ``spec.get_fixed_values()`` at that value.

    One build per parameter rather than one build carrying all of them, so a
    single rejected name names itself instead of failing the whole model.
    """
    declared = sorted(monolithic_agn_declared_params(model))
    assert declared, model
    for name in declared:
        value = declared_default(PARAMS, name)
        spec = _parse(
            {"type": model, name: Fixed(value), "all_params": Fixed(DEFAULT)},
        )
        fixed = spec.get_fixed_values()
        assert name in fixed, f"{model}: {name} accepted but absent from get_fixed_values()"
        assert fixed[name] == pytest.approx(value, abs=0.0, rel=1e-12), (
            f"{model}: {name} landed as {fixed[name]!r}, not the {value!r} written"
        )


@pytest.mark.parametrize("model", _MONOLITHIC)
def test_unknown_names_still_raise_for_monolithic_models(model):
    """Widening the accepted set is not the same as accepting anything."""
    with pytest.raises(ValueError, match="agn_not_a_real_parameter"):
        _parse({"type": model, "agn_not_a_real_parameter": Fixed(1.0)})


def test_a_name_the_model_does_not_declare_gets_monolithic_advice():
    """The refusal must not send a monolithic build to a sub-block.

    ``skirtor_stalevski`` swallows ``agn_torus_frac`` through ``**_kwargs`` and
    pins ``frac_agn=1.0`` internally, so the parameter is exactly inert there:
    measured 0.0 relative SED change from 0.05 to 0.95, against 19.7 for
    ``agn_oa_skirtor`` on the same call. Refusing it is right; sending the
    reader to ``agn={'torus': {...}}`` is not, because the monolithic/sub-block
    check refuses that on the next hop.
    """
    assert "agn_torus_frac" not in monolithic_agn_declared_params("skirtor_stalevski")
    with pytest.raises(ValueError) as exc_info:
        _parse({"type": "skirtor_stalevski", "agn_torus_frac": Fixed(1.0)})
    message = str(exc_info.value)
    assert "skirtor_stalevski" in message
    assert "Nest it" not in message, message
    assert "composable" in message, message


def test_composable_builds_keep_the_nesting_guard():
    """R23 is untouched: the sub-block owner still claims its parameters."""
    with pytest.raises(ValueError, match="Nest it"):
        _parse(
            {
                "type": "composable",
                "norm": "independent",
                "disc": {"type": "multicolor"},
                "torus": {"type": "skirtor"},
                "agn_tau_skirtor": Fixed(7.0),
            }
        )


def test_public_menu_lists_every_monolithic_model():
    """``list_agn_models()`` is the advertised menu, so it must cover the
    registry ``agn={'type': ...}`` actually accepts.

    It listed only ``['composable']`` -- every deprecated monolithic name was
    buildable and undiscoverable at the same time.
    """
    import tengri

    listed = {row["name"] for row in tengri.list_agn_models()}
    missing = sorted(set(_MONOLITHIC) - listed)
    assert not missing, f"buildable but absent from the public menu: {missing}"
    assert "composable" in listed


@pytest.mark.parametrize("model", _MONOLITHIC)
def test_describe_agn_model_resolves_every_listed_name(model):
    """The menu and its per-row describe() must agree."""
    import tengri

    row = tengri.describe_agn_model(model)
    assert str(row["citation"]).strip(), model
    assert str(row["short_doc"]).strip(), model


# ──────────────────────────────────────────────────────────────────────────
# R37: an AGN `type` that is not an AGN model is refused at build time.
# ──────────────────────────────────────────────────────────────────────────


def _validated_agn_types() -> set[str]:
    """Every string ``agn={'type': ...}`` accepts, from the registries."""
    from tengri.components.agn.unified import AGN_MODELS

    return set(monolithic_agn_model_names()) | set(AGN_MODELS) | {"none"}


def test_the_r27_contract_covers_exactly_the_validated_type_set():
    """The enumeration above and the set the grammar validates are one set.

    If the validator accepted a name this file does not build, that name would
    have no coverage; if it refused one this file builds, the file would be
    testing a spelling users cannot write.
    """
    assert _validated_agn_types() == set(_MONOLITHIC) | {"composable", "none"}


def test_none_is_the_off_switch_not_a_composable_block_name():
    """``agn={'type': 'none'}`` disables the group; it is not a block name.

    ``'none'`` is registered as a type in all six composable categories, so
    R37's block-name branch claimed the one spelling that means "no AGN at
    all" and answered it with ``agn={'type': 'composable', 'atten': {'type':
    'none', ...}}`` -- a build with an AGN. Every other group in the grammar
    takes the same off switch (``neb``, ``shock``, ``radio``, ``xray``,
    ``igm``, ``dust_attenuation``, ``dust_emission``), and ``_translate_dust``
    names agn among them. Measured at a293dec69: it parsed, and the resulting
    spec carried ``agn_model == 'none'``.
    """
    spec = _parse({"type": "none"})
    assert spec.agn_model == "none"


@pytest.mark.parametrize("block_type", ["fritz", "kd18_agnfitter", "nenkova_agnfitter"])
def test_a_composable_block_name_as_the_model_type_names_the_composable_form(block_type):
    """A block name written as the model type is refused, naming the nesting.

    These build silently and die at the first `predict_photometry` with
    `ValueError: Unknown AGN model` -- `_translate_agn` forwarded any string to
    `agn_model` and deferred validation to predict time. Worse, a parameter
    written beside such a type got the sub-block nesting advice, which is a
    dead end twice over: the nesting is refused for a monolithic build, and the
    type is not a model in the first place. The error now says which it is.
    """
    with pytest.raises(ValueError) as exc_info:
        _parse({"type": block_type, "all_params": Fixed(DEFAULT)})
    message = str(exc_info.value)
    assert block_type in message
    assert "composable" in message
    assert "Nest it" not in message, message


def test_an_unknown_agn_type_is_refused_with_the_model_list():
    """Anything that is neither a model nor a block name gets the menu."""
    with pytest.raises(ValueError) as exc_info:
        _parse({"type": "totally_bogus_xyz", "all_params": Fixed(DEFAULT)})
    message = str(exc_info.value)
    assert "totally_bogus_xyz" in message
    assert "composable" in message


def test_a_near_miss_model_name_gets_a_close_match():
    """A typo on a real model name suggests the model, not the block menu."""
    with pytest.raises(ValueError) as exc_info:
        _parse({"type": "skirtor_stalevsky", "all_params": Fixed(DEFAULT)})
    assert "skirtor_stalevski" in str(exc_info.value)


@pytest.mark.parametrize("model", [*_MONOLITHIC, "composable"])
def test_every_validated_type_still_builds(model):
    """The guard refuses only what it should: every validated name builds."""
    _parse({"type": model, "all_params": Fixed(DEFAULT)})

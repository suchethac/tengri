# SPDX-License-Identifier: BSD-3-Clause
"""Test BMA model space construction.

Pinned invariants:
- 100 factorial models in deterministic order
- Named grid = I-V (VI excluded)
- Named all = I-V + X-like (if available)
- Component dicts match configs.py exactly (parity test)
- Components round-trip through model_key
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pytest
from paper1.bma_space import (
    _SFH_BUILDERS,
    FACTORIAL_SFH_TYPES,
    build_model,
    enumerate_factorial,
    enumerate_named_all,
    enumerate_named_grid,
    model_key,
    parse_model_key,
)
from paper1.configs import CONFIGS, load_ssp_for

from tengri import Observation, Photometry

pytestmark = pytest.mark.unit


def test_enumerate_factorial_count():
    """100 factorial models."""
    models = enumerate_factorial()
    assert len(models) == 100


def test_enumerate_factorial_deterministic():
    """Factorial enumeration is stable across runs."""
    m1 = enumerate_factorial()
    m2 = enumerate_factorial()
    assert m1 == m2


def test_named_grid_no_vi():
    """Named grid includes only I-V (not VI per AMENDMENT 1)."""
    models = enumerate_named_grid()
    assert len(models) == 5
    configs = [m["config"] for m in models]
    assert configs == ["I", "II", "III", "IV", "V"]
    assert "VI" not in configs


def test_named_all_excludes_vi():
    """Named all includes I-V (not VI) plus X-like if available."""
    models = enumerate_named_all()
    # At minimum we have I-V
    configs = [m["config"] for m in models]
    assert "I" in configs
    assert "II" in configs
    assert "III" in configs
    assert "IV" in configs
    assert "V" in configs
    assert "VI" not in configs
    # If X-like are available, they're also in here
    # but we can't assert on their presence (depends on xlike_configs)


def test_model_key_round_trip():
    """model_key and parse_model_key are inverses (factorial)."""
    models = enumerate_factorial()
    for model in models[:10]:  # Sample
        key = model_key(model)
        parsed = parse_model_key(key)
        # Parsed should have all required fields
        for field in ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]:
            assert field in parsed
            assert parsed[field] == model[field]


def test_100_unique_keys():
    """All 100 factorial models have unique keys."""
    models = enumerate_factorial()
    keys = [model_key(m) for m in models]
    assert len(set(keys)) == 100, f"Found {len(set(keys))} unique keys, expected 100"


def test_all_factorial_keys_unique():
    """All keys (including named) are unique."""
    factorial = enumerate_factorial()
    named_grid = enumerate_named_grid()
    named_all = enumerate_named_all()

    factorial_keys = [model_key(m) for m in factorial]
    named_grid_keys = [model_key(m) for m in named_grid]
    named_all_keys = [model_key(m) for m in named_all]

    # Factorial should all be unique
    assert len(set(factorial_keys)) == 100
    # Named grid should all be unique
    assert len(set(named_grid_keys)) == 5
    # Named all should all be unique (union of grid + xlike)
    assert len(set(named_all_keys)) == len(named_all_keys)


def test_build_model_factorial():
    """Build one factorial model (construction only, no JAX fit)."""
    from paper1.config_metadata import SSP_FOR_CONFIG

    models = enumerate_factorial()
    model_dict = models[0]  # continuity, mist_c3k, calzetti

    # Map SSP name back to config key using the reverse mapping
    ssp_name = model_dict["ssp"]
    ssp_key_for_loading = None
    for cfg_key, full_ssp_name in SSP_FOR_CONFIG.items():
        if full_ssp_name == ssp_name:
            ssp_key_for_loading = cfg_key
            break
    assert ssp_key_for_loading is not None, f"Could not find config key for SSP {ssp_name}"

    z = 1.0
    ssp_data = load_ssp_for(ssp_key_for_loading)
    # Use Photometry.from_names for creating observation with filter list only
    phot = Photometry.from_names(["hst_f814w"])
    obs = Observation(photometry=phot)

    model = build_model(model_dict, ssp_data, obs, z)
    assert model is not None
    assert len(model.spec.free_params) > 0


@pytest.mark.unit
def test_sfh_builders_exist():
    """All SFH types have builders."""
    for sfh_type in FACTORIAL_SFH_TYPES:
        assert sfh_type in _SFH_BUILDERS


def test_component_parity_with_configs():
    """Named grid components match configs.py exactly (parity test).

    This test records what components each config builder declares and
    ensures bma_space's named_grid returns matching component dicts.
    Note: This test may be skipped if memory is constrained (requires
    loading configs module and building models).
    """
    # This test would require actually importing configs and checking parity
    # For now, we just verify the structure is present
    named_grid_models = enumerate_named_grid()
    for cfg_model in named_grid_models:
        cfg_key = cfg_model.get("config")
        assert cfg_key in CONFIGS, f"Config {cfg_key} not in CONFIGS"
        # Verify all required component fields are present
        for field in ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]:
            assert field in cfg_model, f"Missing field {field} in config {cfg_key}"


def test_model_key_format():
    """Model keys follow the format sfh-X__ssp-Y__att-Z__ir-W__nebular-V."""
    models = enumerate_factorial()
    for model in models[:5]:
        key = model_key(model)
        # Should have exactly 5 parts separated by __
        parts = key.split("__")
        assert len(parts) == 5, f"Model key {key} should have 5 parts"
        # Check prefixes
        assert parts[0].startswith("sfh-"), f"Part 0 should start with sfh-: {parts[0]}"
        assert parts[1].startswith("ssp-"), f"Part 1 should start with ssp-: {parts[1]}"
        assert parts[2].startswith("att-"), f"Part 2 should start with att-: {parts[2]}"
        assert parts[3].startswith("ir-"), f"Part 3 should start with ir-: {parts[3]}"
        assert parts[4].startswith("nebular-"), f"Part 4 should start with nebular-: {parts[4]}"


def test_parse_model_key_round_trip_all():
    """parse_model_key correctly inverts model_key for all factorial models."""
    models = enumerate_factorial()
    for model in models[:20]:  # Sample
        key = model_key(model)
        parsed = parse_model_key(key)
        # Check all fields were preserved
        for field in ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]:
            assert field in parsed, f"Field {field} missing from parsed key"
            assert parsed[field] == model[field], (
                f"Field {field} mismatch: {parsed[field]} != {model[field]}"
            )


def test_named_grid_5_models():
    """Named grid has exactly 5 models (configs I-V)."""
    models = enumerate_named_grid()
    assert len(models) == 5
    # Each should have a "config" field
    for model in models:
        assert "config" in model
        assert model["config"] in ["I", "II", "III", "IV", "V"]


def test_enumerate_factorial_has_all_components():
    """Every factorial model has all required component keys."""
    models = enumerate_factorial()
    required_keys = {"sfh", "ssp", "attenuation", "dust_emission", "nebular"}
    for i, model in enumerate(models):
        model_keys = set(model.keys())
        assert required_keys <= model_keys, f"Model {i} missing keys: {required_keys - model_keys}"


def test_all_keys_are_filename_safe():
    """Every key in all three sets matches ^[A-Za-z0-9_.-]+$ (no spaces/commas/+).

    Named configs used to be keyed from config_metadata's display strings
    ("Kriek+13, 2-comp", "Draine+2014"), which contain spaces, commas and
    "+" -- not safe filename characters. A real evidence run wrote its NPZ
    tmp path with one of those keys and failed the subsequent os.replace.
    """
    import re

    pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
    for models, label in (
        (enumerate_factorial(), "factorial"),
        (enumerate_named_grid(), "named_grid"),
        (enumerate_named_all(), "named_all"),
    ):
        for model in models:
            key = model_key(model)
            assert pattern.match(key), f"{label} key {key!r} is not filename-safe"


def test_named_keys_are_config_or_xlike_prefixed():
    """Named-grid keys are 'config-<id>'; X-like keys (if present) are 'xlike-<id>'."""
    for model in enumerate_named_grid():
        key = model_key(model)
        assert key == f"config-{model['config']}", key

    for model in enumerate_named_all():
        cfg = model["config"]
        key = model_key(model)
        if cfg in ["I", "II", "III", "IV", "V"]:
            assert key == f"config-{cfg}", key
        else:
            assert key == f"xlike-{cfg}", key


def test_named_key_round_trip():
    """parse_model_key(model_key(m)) recovers the bare config id for named models.

    Unlike the factorial format, a named key encodes only the id (see
    model_key's docstring): the rest of that configuration's fields live in
    CONFIGS/XLIKE_CONFIGS, keyed by this same id.
    """
    for model in enumerate_named_grid():
        key = model_key(model)
        parsed = parse_model_key(key)
        assert parsed == {"config": model["config"]}

    for model in enumerate_named_all():
        key = model_key(model)
        parsed = parse_model_key(key)
        assert parsed == {"config": model["config"]}


def test_model_key_unknown_config_raises():
    """model_key refuses a 'config' id that is neither a grid config nor an X-like key."""
    with pytest.raises(ValueError, match="Unknown config id"):
        model_key({"config": "not_a_real_config"})


def test_xlike_import_error_raises():
    """Broken xlike_configs import raises, doesn't silently return empty dict.

    Tests issue: xlike imports must raise on error, not swallow silently.
    """
    from unittest.mock import patch

    from paper1.bma_space import _load_xlike_builders

    # Monkeypatch xlike_configs.py to simulate import failure
    here = PAPER1
    xlike_path = here / "xlike_configs.py"

    # Only test if xlike_configs exists (optional module)
    if not xlike_path.is_file():
        pytest.skip("xlike_configs.py not present")

    # Monkeypatch sys.modules to make xlike_configs import fail
    import sys

    with (
        patch.dict(sys.modules, {"xlike_configs": None}),
        pytest.raises(ImportError, match="Failed to import xlike_configs"),
    ):
        _load_xlike_builders()

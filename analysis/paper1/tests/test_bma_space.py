# SPDX-License-Identifier: BSD-3-Clause
"""Test BMA model space construction.

Pinned invariants:
- 100 factorial models in deterministic order
- Named grid = I-V (VI excluded)
- Named all = I-V + X-like (if available)
- Component dicts match configs.py exactly
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pytest  # noqa: E402

from paper1.bma_space import (  # noqa: E402
    enumerate_factorial,
    enumerate_named_grid,
    model_key,
    parse_model_key,
    build_model,
    FACTORIAL_SFH_TYPES,
    _SFH_BUILDERS,
)
from paper1.configs import load_ssp_for  # noqa: E402
from tengri import Observation, Photometry  # noqa: E402

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
    """Named grid includes only I-V (not VI)."""
    models = enumerate_named_grid()
    assert len(models) == 5
    configs = [m["config"] for m in models]
    assert configs == ["I", "II", "III", "IV", "V"]
    assert "VI" not in configs


def test_model_key_round_trip():
    """model_key and parse_model_key are inverses."""
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


def test_build_model_factorial(tmp_path):
    """Build one factorial model (no JAX fit, just construction)."""
    models = enumerate_factorial()
    model_dict = models[0]  # continuity, mist_c3k, calzetti

    z = 1.0
    ssp_data = load_ssp_for(model_dict["ssp"][-1])
    phot = Photometry(
        names=["hst_f814w"],
        fnu=[1e-27],
        fnu_err=[0.1e-27],
    )
    obs = Observation(photometry=phot)

    model = build_model(model_dict, ssp_data, obs, z)
    assert model is not None
    assert len(model.spec.free_params) > 0


@pytest.mark.unit
def test_sfh_builders_exist():
    """All SFH types have builders."""
    for sfh_type in FACTORIAL_SFH_TYPES:
        assert sfh_type in _SFH_BUILDERS

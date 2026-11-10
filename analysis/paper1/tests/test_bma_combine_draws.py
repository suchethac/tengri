# SPDX-License-Identifier: BSD-3-Clause
"""Draw handling in the BMA combiner: SFR floor and non-finite draw removal."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.bma_combine import (
    DRAW_QUANTITIES,
    LOG_SFR_FLOOR,
    _clean_draws,
    _extract_percentiles,
    _resample_mixture,
)

pytestmark = pytest.mark.unit


def _cell(n: int = 50) -> dict[str, np.ndarray]:
    x = np.linspace(9.0, 10.0, n)
    return {
        "log_stellar_mass_formed": x,
        "log_stellar_mass_survived": x - 0.2,
        "log_sfr_100myr": x - 9.0,
        "log_sfr_10myr": x - 9.5,
    }


def test_floor_constant_matches_fig05():
    assert LOG_SFR_FLOOR == -10.0


def test_neg_inf_sfr_draw_is_floored_and_percentiles_finite():
    data = _cell()
    data["log_sfr_100myr"][:20] = -np.inf
    draws, n_dropped = _clean_draws(data)
    assert n_dropped == 0
    assert draws["log_sfr_100myr"].shape == (50,)
    assert np.all(draws["log_sfr_100myr"][:20] == LOG_SFR_FLOOR)
    perc = _extract_percentiles(data, "log_sfr_100myr")
    assert perc is not None and np.all(np.isfinite(perc))
    assert perc[0] == LOG_SFR_FLOOR


def test_nan_draws_dropped_jointly_and_counted():
    data = _cell()
    data["log_stellar_mass_formed"][3] = np.nan
    data["log_sfr_10myr"][7] = np.nan
    draws, n_dropped = _clean_draws(data)
    assert n_dropped == 2
    assert {len(v) for v in draws.values()} == {48}
    for q in DRAW_QUANTITIES:
        perc = _extract_percentiles(data, q)
        assert perc is not None and np.all(np.isfinite(perc))


def test_mixture_and_per_model_agree_on_cleaned_draws():
    data = _cell()
    data["log_sfr_100myr"][:5] = -np.inf
    data["log_stellar_mass_formed"][10:13] = np.nan
    post = [{"model_key": "m", "_npz": data}]
    mix = _resample_mixture(post, [0], {"m": 1.0}, 20000, 0)
    for q in DRAW_QUANTITIES:
        per_model = _extract_percentiles(data, q)
        assert mix[q] == pytest.approx(per_model, abs=0.05)
        assert np.all(np.isfinite(mix[q]))
    assert min(mix["log_sfr_100myr"]) >= LOG_SFR_FLOOR


def test_model_with_no_finite_draws_contributes_nothing():
    bad = {q: np.full(5, np.nan) for q in DRAW_QUANTITIES}
    posts = [{"model_key": "a", "_npz": _cell()}, {"model_key": "b", "_npz": bad}]
    mix = _resample_mixture(posts, [0, 1], {"a": 0.5, "b": 0.5}, 500, 1)
    assert np.all(np.isfinite(mix["log_stellar_mass_formed"]))
    assert _extract_percentiles(bad, "log_stellar_mass_formed") is None


def test_non_keyerror_failure_raises():
    class Broken:
        def __getitem__(self, key):
            raise OSError("corrupt archive member")

    with pytest.raises(OSError):
        _extract_percentiles(Broken(), "log_stellar_mass_formed")


def test_missing_quantity_is_none():
    assert _extract_percentiles({"log_sfr_10myr": np.zeros(3)}, "log_sfr_100myr") is None
    assert _extract_percentiles(None, "log_sfr_100myr") is None

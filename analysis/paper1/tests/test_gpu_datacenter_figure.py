# SPDX-License-Identifier: BSD-3-Clause
"""The datacenter GPU figure must not draw a number it cannot measure.

``sherlock_h100_batch.json`` ships an ``aa_control`` block: the spread of
repeated identical measurements. It exists because the headline comparison on
that card is small -- float64 against float32 differs by 2.4% -- and a ratio
inside the repeat noise is not a small effect, it is no measured effect at all.
A figure that annotated one would print a number a rerun could invert the sign
of.

The other tests here pin the file contract. The previous GPU figure kept its
numbers as literals transcribed from a bench report, and the caption then named
hardware the literals were not from; a partial file that renders anyway is the
same failure arriving by a different route.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fig04_gpu_datacenter as figure

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def payload():
    return figure.load(figure.DATA)


def test_the_committed_data_builds(payload):
    """Non-vacuity: the guards must not refuse the real measurements."""
    fig, stats = figure.build(copy.deepcopy(payload))
    assert stats["H100_resolvable"] and stats["Xeon_resolvable"]
    figure.plt.close(fig)


def test_the_crossovers_are_read_from_the_data(payload):
    fig, stats = figure.build(copy.deepcopy(payload))
    figure.plt.close(fig)
    assert stats["crossover_f64"] == (128.0, 512.0)
    assert stats["crossover_f32"] == (512.0, 2048.0)


def test_float64_crosses_earlier_than_float32(payload):
    """The counterintuitive result, pinned so a data swap cannot lose it.

    float64 costs the Xeon roughly twice as much and costs the H100 nothing,
    so the crossover moves the way a reader does not expect.
    """
    fig, stats = figure.build(copy.deepcopy(payload))
    figure.plt.close(fig)
    assert stats["crossover_f64"][0] < stats["crossover_f32"][0]


def test_a_ratio_inside_the_repeat_noise_is_refused(payload):
    """The guard the aa_control block exists for."""
    noisy = copy.deepcopy(payload)
    noisy["aa_control"]["gpu_f64"] = [1.30] * len(noisy["aa_control"]["gpu_f64"])
    with pytest.raises(ValueError) as excinfo:
        figure.build(noisy)
    assert "noise floor" in str(excinfo.value)


@pytest.mark.parametrize(
    "block", ["forward", "gradient", "forward_ms_per_call", "aa_control", "caveats"]
)
def test_a_partial_file_is_refused(payload, tmp_path, block):
    """Rendering from a file missing a block would silently drop a panel."""
    trimmed = copy.deepcopy(payload)
    trimmed.pop(block)
    path = tmp_path / "trimmed.json"
    path.write_text(json.dumps(trimmed))
    with pytest.raises(KeyError):
        figure.load(path)


def test_the_caveats_travel_with_the_data(payload):
    """The three things that must not be claimed are carried, not remembered."""
    joined = " ".join(payload["caveats"]).lower()
    assert "launch" in joined
    assert "asymptotic" in joined
    assert "latency result" in joined

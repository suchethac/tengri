# SPDX-License-Identifier: BSD-3-Clause
"""The star formation history panel must draw the recovery, not only the truth.

``fig01`` is the figure the section introduces as "the mock, the joint
observation drawn from it, and the recovery". Its SFH panel drew one curve,
labeled Truth, because ``plot_sfh`` was only ever handed the truth parameters
-- it could not have drawn a posterior. Seven of the thirty-six free parameters
live in that history (a total mass and six continuity ratios), so the most
structured part of the model went unillustrated, and a reader met a curve in a
recovery figure with nothing to compare it against.

Every mechanical check passed while this was true: the script exited 0, the
convergence gate passed, the published filename was earned and the PDF was
valid and non-empty. Only looking at the figure showed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

pytestmark = pytest.mark.unit

GRID_YR = np.geomspace(1e6, 1.2e10, 40)
FREE = ["sfh_cont_log_total_mass"]


class _State:
    def __init__(self, derived):
        self.derived = derived


class _Model:
    """Returns an SFH whose amplitude is set by the one parameter it reads."""

    def __init__(self):
        self.calls = 0

    def predict_state(self, params):
        self.calls += 1
        amplitude = 10.0 ** (params.get("sfh_cont_log_total_mass", 10.0) - 10.0)
        return _State(
            {"sfh_grid_lbt_yr": GRID_YR, "sfr_history": amplitude * np.ones_like(GRID_YR)}
        )


def _labels(ax):
    return ax.get_legend_handles_labels()[1]


def _posterior(n=200):
    return {"sfh_cont_log_total_mass": np.linspace(9.5, 10.5, n)}


def test_without_a_posterior_only_the_truth_is_drawn():
    """The truth-only render is a legitimate state and must stay available."""
    from paper1.fig01_mock_joint_infer import plot_sfh

    fig, ax = plt.subplots()
    plot_sfh(ax, _Model(), {"sfh_cont_log_total_mass": 10.0})
    assert _labels(ax) == ["Truth"]
    plt.close(fig)


def test_with_a_posterior_the_recovery_is_drawn_beside_the_truth():
    """The defect. A recovery figure has to show the recovery."""
    from paper1.fig01_mock_joint_infer import plot_sfh

    fig, ax = plt.subplots()
    plot_sfh(ax, _Model(), {"sfh_cont_log_total_mass": 10.0}, _posterior(), FREE, n_draws=8)
    labels = _labels(ax)

    assert "Truth" in labels
    assert any("Posterior" in label for label in labels), (
        f"the SFH panel drew no posterior: {labels}"
    )
    assert ax.collections, "no shaded interval was drawn"
    plt.close(fig)


def test_the_band_is_built_from_the_requested_number_of_draws():
    """One predict_state per draw, plus one for the truth."""
    from paper1.fig01_mock_joint_infer import plot_sfh

    model = _Model()
    fig, ax = plt.subplots()
    plot_sfh(ax, model, {"sfh_cont_log_total_mass": 10.0}, _posterior(), FREE, n_draws=12)
    assert model.calls == 13, f"expected 12 draws + 1 truth, got {model.calls}"
    plt.close(fig)


def test_the_band_spans_the_whole_chain_rather_than_its_first_draws():
    """Taking the leading draws would show where the sampler started.

    The stub's amplitude rises monotonically across the chain, so a band built
    from the first few draws sits entirely below one built from all of them.
    """
    from paper1.fig01_mock_joint_infer import plot_sfh

    fig, ax = plt.subplots()
    plot_sfh(ax, _Model(), {"sfh_cont_log_total_mass": 10.0}, _posterior(), FREE, n_draws=10)
    path = ax.collections[0].get_paths()[0].vertices
    top = float(np.max(path[:, 1]))

    # The last draw is 10.5 -> amplitude 10**0.5 = 3.16. A band from the
    # leading draws alone could not reach near it.
    assert top > 2.0, f"the shaded interval tops out at {top:.3f}; it is not spanning the chain"
    plt.close(fig)


def test_an_empty_free_name_list_falls_back_to_truth_only():
    """Absence of names is not a reason to invent a band."""
    from paper1.fig01_mock_joint_infer import plot_sfh

    fig, ax = plt.subplots()
    plot_sfh(ax, _Model(), {"sfh_cont_log_total_mass": 10.0}, _posterior(), [], n_draws=8)
    assert _labels(ax) == ["Truth"]
    plt.close(fig)


def test_the_total_mass_is_not_labeled_as_the_stellar_mass():
    """``sfh_cont_log_total_mass`` is the SFH's time-integral, i.e. mass formed.

    Labeled $\\log M_\\star$ it reads as the surviving mass, which differs by
    0.1948 dex on this mock's truth -- larger than the offsets the panel plots,
    and not the quantity Section 7's published codes report.
    """
    source = (PAPER1 / "fig01_mock_joint_infer.py").read_text()
    index = source.index('"sfh_cont_log_total_mass"')
    label_line = source[index : index + 200]
    assert "M_{\\rm formed}" in label_line, (
        f"the total-mass entry is not labeled as formed mass: {label_line.splitlines()[0]}"
    )


def test_the_figure_actually_hands_the_panel_its_posterior():
    """The defect was the call site, not the helper.

    ``plot_sfh`` was always *able* to draw a band; ``main`` never gave it one.
    Every test above calls the helper directly, so all of them stay green when
    the call site is reverted -- which is exactly how the original bug
    survived. A check that observes the helper cannot see a wiring fault.

    Asserted against the source because reaching the real call site means
    building the kitchen-sink model, a minute of work for a one-line fact.
    """
    source = (PAPER1 / "fig01_mock_joint_infer.py").read_text()
    call = [line for line in source.splitlines() if "plot_sfh(ax_sfh" in line]
    assert len(call) == 1, f"expected one call site, found {len(call)}: {call}"
    assert "posterior" in call[0], (
        f"main() draws the SFH panel without handing it the posterior, so the "
        f"panel can only show the truth: {call[0].strip()}"
    )

# SPDX-License-Identifier: BSD-3-Clause
"""Why the kitchen-sink mock may use the default IGM fold.

``WavePrecomp`` folds IGM transmission into the sub-band quadrature two ways.
The default ``"node"`` evaluates transmission at the quadrature nodes and
multiplies, forming <S><T>; ``"exact"`` carries it inside the bandpass
integral, <S*T>. They differ only where transmission varies within a band,
which at :math:`z=1` means the band containing the Lyman limit at 1824 A
observed.

The writing handoff recorded that this mock needs the exact fold. Measured
against the mock as it is actually built, it does not: the fold moves exactly
one band by 0.227%, which is 0.017 of that band's own 1-sigma depth, and
leaves the other fifteen bit-identical. A shift that small cannot move the
posterior, so the mock is not wrong to run on the default.

That conclusion is a property of this band set and this depth, not of the
code, and it stops holding the moment either changes -- adding GALEX FUV,
which straddles the break far more severely at this redshift, or deepening the
noise until 0.2% is resolvable. These tests exist to fail then, so the choice
is made again on evidence rather than inherited as a habit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR.parents[1]))

# `contract`, matching this directory's other files. Not `slow`: the suite's
# conftest deselects that by default, and a test nobody runs is a test that
# cannot tell anyone the mock's band set has moved. It builds two models, so
# it costs about a minute.
pytestmark = pytest.mark.contract

#: The fold may not move any band by more than this fraction of its own noise.
#: Measured worst case is 0.0168; the bar is a factor of three above it, close
#: enough that a real change in the mock trips it and far enough that it does
#: not chase the last digit of a quadrature.
MAX_SHIFT_IN_SIGMA = 0.05

#: The only band whose transmission is structured at this redshift. Stated so
#: the test can assert the *others* are untouched: a fold that perturbs a flat
#: band is doing something other than what it claims.
STRUCTURED_BAND = "galex_nuv"


@pytest.fixture(scope="module")
def folds():
    """Photometry at the truth parameters under each fold, and the noise."""
    import tengri
    from analysis.paper1 import verify_mock_listing as vml
    from analysis.paper1.fig_mock_joint_infer import TRUTH_NPZ
    from tengri import WavePrecomp

    if not TRUTH_NPZ.is_file():
        pytest.skip(f"mock truth not generated: {TRUTH_NPZ}")
    try:
        ssp = tengri.load_ssp(vml.SSP_NAME)
    except FileNotFoundError:
        pytest.skip(f"SSP library {vml.SSP_NAME!r} is not present on this machine")

    truth = np.load(TRUTH_NPZ, allow_pickle=True)
    names = [str(n) for n in truth["free_params"]]
    params = dict(zip(names, np.asarray(truth["truth_values"], dtype=np.float64)))

    # The band names and the per-band noise come from the truth file while the
    # photometry comes from the live builder, so the two must be the same set
    # in the same order. They silently were not when MOCK_FILTERS was edited
    # without regenerating the mock: the test then compared one band's flux
    # under another band's name and noise, and reported everything fine.
    recorded = [str(b) for b in truth["filters"]]
    assert recorded == list(vml.MOCK_FILTERS), (
        "the mock's band set has moved away from the truth file "
        f"({recorded} vs {list(vml.MOCK_FILTERS)}); regenerate the mock, "
        "because every tolerance below is indexed by this order"
    )

    observation = vml.build_joint_observation()
    real = vml.WavePrecomp

    def photometry(fold):
        vml.WavePrecomp = (
            (lambda: WavePrecomp()) if fold == "node" else (lambda: WavePrecomp(igm_fold="exact"))
        )
        try:
            model = vml.build_mock_model(ssp, observation)
        finally:
            vml.WavePrecomp = real
        # If the truth file and the model disagree about the free set, every
        # number below is computed at the wrong point and the comparison is
        # meaningless. Fail rather than predict on a partial dict.
        assert set(model.spec.free_params) == set(names), (
            "the mock's free parameters have moved away from the truth file; "
            "regenerate the mock before trusting this tolerance"
        )
        return np.asarray(model.predict_photometry(params), dtype=np.float64)

    return {
        "bands": [str(b) for b in truth["filters"]],
        "sigma": np.asarray(truth["phot_sig"], dtype=np.float64),
        "node": photometry("node"),
        "exact": photometry("exact"),
    }


def test_the_probe_separates_the_two_folds(folds):
    """Guards the other two: a fold that changed nothing would pass them both."""
    node, exact = folds["node"], folds["exact"]
    assert not np.array_equal(node, exact), (
        "the two folds gave identical photometry in every band, so this mock "
        "does not exercise the fold at all and the tolerance below is vacuous"
    )


def test_only_the_structured_band_moves(folds):
    """A fold must not perturb bands that carry no IGM structure."""
    moved = [band for band, a, b in zip(folds["bands"], folds["node"], folds["exact"]) if a != b]
    assert moved == [STRUCTURED_BAND], (
        f"the fold moved {moved}, expected only [{STRUCTURED_BAND!r}]. A band "
        "with flat transmission changing means the fold is reaching something "
        "other than the IGM."
    )


def test_the_fold_is_below_the_noise_everywhere(folds):
    """The claim that licenses running this mock on the default fold."""
    shift = np.abs(folds["exact"] - folds["node"]) / folds["sigma"]
    worst = int(np.argmax(shift))
    assert shift[worst] < MAX_SHIFT_IN_SIGMA, (
        f"{folds['bands'][worst]} moves {shift[worst]:.4f} sigma between the "
        f"node and exact IGM folds, at or above the {MAX_SHIFT_IN_SIGMA} bar. "
        "The mock can no longer be run on the default fold without the choice "
        "showing up in the posterior: rebuild it with "
        "WavePrecomp(igm_fold='exact')."
    )

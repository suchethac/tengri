# SPDX-License-Identifier: BSD-3-Clause
"""Regression: Inoue+2014 Lyman-series DLA coefficients match the paper.

The damped-Lyman-alpha (DLA) Lyman-series optical depth of Inoue et al.
(2014, MNRAS 442, 1805) uses two per-line coefficients — ``A_j^DLA1`` (for
the ``lambda_obs < 3 lambda_j`` regime, power 2) and ``A_j^DLA2`` (for
``lambda_obs >= 3 lambda_j``, power 3), tabulated in the paper's Table 2.

A transcription error left ``_A_DLA[:, 1]`` (the second regime) ~2.87-3.65x
too high and ``_A_DLA[:, 0]`` drifting up to 1.27x too high in the line tail,
so the Lyman-continuum transmission at z >= 2 was systematically too low (the
DLA opacity dominates the 800-912 A rest window). The existing crossval test
missed it: it only asserted agreement *above* Lya (rest > 1216 A), excluding
the very region the DLA term governs, and crossval is not run in CI.

Reference values below are the published Inoue+2014 Table 2 columns
``A_DLA_J_1`` / ``A_DLA_J_2`` (the same values eazy-py, BAGPIPES and
Synthesizer ship). This test is self-contained — it pins the coefficients to
the paper, independent of any other code.
"""

import chex
import jax
import numpy as np
import pytest

from tengri.components.igm import igm as igm_mod, igm_transmission

pytestmark = pytest.mark.regression_paper

# Inoue+2014 (MNRAS 442, 1805) Table 2, first 10 Lyman lines (j = 2..11):
# columns lambda_j [A], A_DLA_J_1, A_DLA_J_2.
_INOUE14_TABLE2_DLA = np.array(
    [
        # lambda_j,    A_DLA_J_1,   A_DLA_J_2
        [1215.67, 1.617e-04, 5.390e-05],  # Ly-alpha
        [1025.72, 1.545e-04, 5.151e-05],  # Ly-beta
        [972.537, 1.498e-04, 4.992e-05],
        [949.743, 1.460e-04, 4.868e-05],
        [937.803, 1.429e-04, 4.763e-05],
        [930.748, 1.402e-04, 4.672e-05],
        [926.226, 1.377e-04, 4.590e-05],
        [923.150, 1.355e-04, 4.516e-05],
        [920.963, 1.335e-04, 4.448e-05],
        [919.352, 1.316e-04, 4.385e-05],
    ]
)


def test_a_dla_shape():
    """39 Lyman transitions, two DLA regime coefficients each."""
    chex.assert_shape(igm_mod._A_DLA, (39, 2))


def test_a_dla_matches_inoue14_table2():
    """Both DLA columns match the published Inoue+2014 Table 2 values.

    Regression for the coefficient transcription bug: ``A_j^DLA2`` was stored
    ~2.87x too high (a near-copy of ``A_j^DLA1`` rather than the true ~1/3
    value). Anchored to the paper, not to another SED code.
    """
    a_dla = np.asarray(igm_mod._A_DLA)
    lam = np.asarray(igm_mod._LAMBDA_LYMAN)
    n_check = _INOUE14_TABLE2_DLA.shape[0]

    np.testing.assert_allclose(
        lam[:n_check],
        _INOUE14_TABLE2_DLA[:, 0],
        rtol=1e-4,
        err_msg="Lyman line rest wavelengths disagree with Inoue+2014 Table 2",
    )
    np.testing.assert_allclose(
        a_dla[:n_check, 0],
        _INOUE14_TABLE2_DLA[:, 1],
        rtol=2e-3,
        err_msg="A_j^DLA1 (regime 1, power 2) disagrees with Inoue+2014 Table 2",
    )
    np.testing.assert_allclose(
        a_dla[:n_check, 1],
        _INOUE14_TABLE2_DLA[:, 2],
        rtol=2e-3,
        err_msg="A_j^DLA2 (regime 2, power 3) disagrees with Inoue+2014 Table 2",
    )


def test_a_dla2_is_about_one_third_of_a_dla1():
    """Physical sanity: the paper's A_j^DLA2 is ~1/3 of A_j^DLA1, not ~0.95x.

    Guards against the specific failure mode (col 2 being a scaled copy of
    col 1). Across all 39 lines the ratio sits near 1/3.
    """
    a_dla = np.asarray(igm_mod._A_DLA)
    ratio = a_dla[:, 1] / a_dla[:, 0]
    assert np.all(ratio < 0.5), f"A_DLA2/A_DLA1 should be ~1/3, got max {ratio.max():.3f}"
    assert np.all(ratio > 0.25), f"A_DLA2/A_DLA1 should be ~1/3, got min {ratio.min():.3f}"


def test_lyman_continuum_transmission_z4_not_over_absorbed():
    """End-to-end: the Lyman-continuum transmission at z=4 recovers the paper.

    At z=4, rest 900 A (observed 4500 A) the correct Inoue+2014 transmission
    is ~0.26; the buggy DLA coefficients pushed it down to ~0.18. Assert we are
    back near the paper value (the value BAGPIPES/Synthesizer produce).
    """
    z = 4.0
    rest = np.array([850.0, 900.0])
    wave_obs = rest * (1.0 + z)
    T = np.asarray(igm_transmission(jax.numpy.asarray(wave_obs), z))
    # Paper/BAGPIPES: T(850)~0.094, T(900)~0.265. Buggy: ~0.068, ~0.180.
    assert T[1] > 0.23, f"z=4 rest-900A transmission {T[1]:.3f} too low (DLA over-absorbing)"
    assert T[0] > 0.08, f"z=4 rest-850A transmission {T[0]:.3f} too low (DLA over-absorbing)"

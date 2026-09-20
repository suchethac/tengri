#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The appendix photometry-error table must converge, and reproduce.

Appendix `tab:photometry_error` names measure_photometry_error.py in its
caption, but neither that script nor its JSON had ever been committed, so the
table's numbers could not be checked. Regenerating them reproduced the printed
values; these tests keep that true.

The convergence test is here because the first implementation failed it. Its
sub-bands were carved out of a shared dense grid by masking, which discards
the mass between a sub-band edge and the nearest node. That loss grows with K,
so the measured error ROSE with K -- 0.016, 0.019, 0.050 for SDSS g -- while
still looking like a plausible table of small percentages. A quadrature that
diverges as it is refined is not a quadrature, and nothing else would have
caught it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_photometry_error import SUBBAND_COUNTS, measure_band

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def sdss_g():
    tengri = pytest.importorskip("tengri")
    import numpy as np

    phot = tengri.Photometry.from_names(["sdss_g"])
    return (
        np.asarray(phot.filter_waves[0], dtype=float),
        np.asarray(phot.filter_trans[0], dtype=float),
        str(phot.convention),
    )


def test_subband_error_falls_monotonically_with_k(sdss_g):
    """Refining the quadrature must not make it worse."""
    row = measure_band(*sdss_g)
    errors = [row[f"K_{k}"] for k in SUBBAND_COUNTS]
    for k_lo, k_hi, e_lo, e_hi in zip(
        SUBBAND_COUNTS, SUBBAND_COUNTS[1:], errors, errors[1:], strict=False
    ):
        assert e_hi <= e_lo, (
            f"error rose from {e_lo:.4f}% at K={k_lo} to {e_hi:.4f}% at K={k_hi}: "
            "a quadrature that diverges as it is refined is integrating the "
            "wrong thing, most likely losing mass at the sub-band edges"
        )


def test_the_scheme_ordering_is_the_one_the_appendix_claims(sdss_g):
    """One effective wavelength is worst, Taylor better, sub-bands best."""
    row = measure_band(*sdss_g)
    assert row["A.Phi"] > row["Taylor"] > row["K_3"] > row["K_8"]


def test_reproduces_the_printed_sdss_g_row(sdss_g):
    """Within the tolerance a filter-curve revision can move it.

    Printed: 1.401, 0.269, 0.262, 0.035, 0.014, 0.006. Loose on purpose --
    this pins that the table is reproducible, not that a curve file is frozen.
    """
    row = measure_band(*sdss_g)
    printed = {"A.Phi": 1.401, "Taylor": 0.269, "K_1": 0.262, "K_3": 0.035}
    for key, want in printed.items():
        assert row[key] == pytest.approx(want, rel=0.10), (
            f"{key}: regenerated {row[key]:.4f}% against a printed {want}%"
        )

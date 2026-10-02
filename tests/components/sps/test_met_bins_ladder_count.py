# SPDX-License-Identifier: BSD-3-Clause
"""Metallicity ladders use their own bin count (#2600).

A ``bins`` / ``bins_continuity`` ladder with a bin count other than the
registry default must give every age bin its own metallicity.  The unfixed
code read a fixed count of six, so a ``bins_continuity`` ladder with fewer
bins summed every declared step into every age.

Target metallicities sit ON the ``synthetic_ssp_wide`` lgmet nodes so the
tests pin the ladder arithmetic, not the metallicity kernel.
"""

from __future__ import annotations

from itertools import pairwise

import chex
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.parameters.translate import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

_LOG_AGE_LO, _LOG_AGE_HI = 6.0, 10.14


def _edges(n_bins):
    return np.linspace(_LOG_AGE_LO, _LOG_AGE_HI, n_bins + 1)


def _targets(lgz_nodes, n_bins):
    """Per-bin absolute log Z, youngest bin first, cycling over the nodes."""
    return np.array([lgz_nodes[i % len(lgz_nodes)] for i in range(n_bins)])


def _met_dict(mode, edges, targets):
    """Met dict realizing ``targets`` (youngest first) in the given mode."""
    met = {
        "type": mode,
        "met_bin_edges_log_yr": [float(e) for e in edges],
        "all_params": Fixed(DEFAULT),
    }
    if mode == "bins":
        for i, z in enumerate(targets):
            met[f"bin_{i}"] = Fixed(float(z - LOG10_ZSUN))
    else:
        # base = oldest bin; d_log_z_<i> steps old -> young.
        met["logzsol_base"] = Fixed(float(targets[-1] - LOG10_ZSUN))
        for i, step in enumerate(np.diff(targets[::-1])):
            met[f"d_log_z_{i}"] = Fixed(float(step))
    return met


def _build(ssp, met):
    return SEDModel.build(
        ssp_data=ssp,
        met=met,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(3.0),
            "age_gyr": Fixed(12.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )


def _joint_weights(ssp, met):
    st = _build(ssp, met).predict_state({})
    return (
        np.asarray(st.derived["joint_weights"]),
        np.asarray(st.derived["ssp_ages_yr"]),
    )


def _bin_means(ssp, met, edges):
    """Mean absolute log Z at the SSP age nearest each bin's center."""
    jw, ages = _joint_weights(ssp, met)
    lgz = np.asarray(ssp.ssp_lgmet)
    log_age = np.log10(ages)
    means = []
    for lo, hi in pairwise(edges):
        j = int(np.argmin(np.abs(log_age - 0.5 * (lo + hi))))
        assert lo < log_age[j] < hi, "SSP age grid too coarse for this bin"
        w = jw[:, j] / jw[:, j].sum()
        means.append(float(w @ lgz))
    return np.array(means)


@pytest.mark.parametrize("mode", ["bins", "bins_continuity"])
@pytest.mark.parametrize("n_bins", [2, 3, 6])
def test_each_bin_carries_its_own_metallicity(synthetic_ssp_wide, mode, n_bins):
    ssp = synthetic_ssp_wide
    edges = _edges(n_bins)
    targets = _targets(np.asarray(ssp.ssp_lgmet), n_bins)
    means = _bin_means(ssp, _met_dict(mode, edges, targets), edges)
    np.testing.assert_allclose(means, targets, atol=1e-6)


def test_bins_continuity_three_bin_ladder_gives_three_metallicities(synthetic_ssp_wide):
    ssp = synthetic_ssp_wide
    edges = _edges(3)
    targets = _targets(np.asarray(ssp.ssp_lgmet), 3)
    assert len(set(np.round(targets, 6))) == 3
    means = _bin_means(ssp, _met_dict("bins_continuity", edges, targets), edges)
    assert len(set(np.round(means, 6))) == 3, f"bins collapsed to one Z: {means}"
    np.testing.assert_allclose(means, targets, atol=1e-6)


@pytest.mark.parametrize("n_bins", [2, 3, 6])
def test_bins_and_bins_continuity_agree_on_the_same_ladder(synthetic_ssp_wide, n_bins):
    ssp = synthetic_ssp_wide
    edges = _edges(n_bins)
    targets = _targets(np.asarray(ssp.ssp_lgmet), n_bins)
    jw_bins, _ = _joint_weights(ssp, _met_dict("bins", edges, targets))
    jw_cont, _ = _joint_weights(ssp, _met_dict("bins_continuity", edges, targets))
    chex.assert_trees_all_close(jw_bins, jw_cont, atol=1e-10)


@pytest.mark.parametrize("mode", ["bins", "bins_continuity"])
def test_seven_bin_ladder_raises(synthetic_ssp_wide, mode):
    edges = _edges(7)  # 8 edges -> 7 bins, one more than the declared six
    targets = _targets(np.asarray(synthetic_ssp_wide.ssp_lgmet), 6)
    met = _met_dict(mode, edges, targets)
    with pytest.raises(ValueError, match="6"):
        _build(synthetic_ssp_wide, met)


@pytest.mark.parametrize(("mode", "key"), [("bins", "bin_3"), ("bins_continuity", "d_log_z_3")])
def test_out_of_ladder_bin_key_raises(synthetic_ssp_wide, mode, key):
    edges = _edges(3)  # 3 bins: valid bin_0..2 / d_log_z_0..1
    targets = _targets(np.asarray(synthetic_ssp_wide.ssp_lgmet), 3)
    met = _met_dict(mode, edges, targets)
    met[key] = Fixed(0.1)
    with pytest.raises(ValueError, match=rf"{key}.*3-bin ladder"):
        _build(synthetic_ssp_wide, met)

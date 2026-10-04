# SPDX-License-Identifier: BSD-3-Clause
"""Stars formed before the oldest SSP template are the oldest template (#2714).

On the PARSEC grid (oldest node 12.589 Gyr) at ``z = 0`` (age 13.787 Gyr) both age
kernels dropped every parcel older than the last node and the formed-mass
normalization rescaled each remaining node up: for a constant history over
[0.001, 13.5] Gyr the oldest node held 5.5 % instead of 11.9 % of the mass and the
nodes younger than 1 Gyr 7.50 % instead of 6.99 %. A parcel older than the oldest
template now clamps onto it on both kernels, so the shares equal the analytic
clamp-to-oldest reference.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri.components.stellar.component as C
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri._data_setup import package_or_env_data_path
from tengri.components.stellar.component import SFHBeyondOldestTemplateWarning
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_r"]
KERNELS = ("cic", "dsps")
#: Share tolerance per (kernel, window). cic carries the second-order error of its
#: dense integrand at a step (3.8e-6 and 2.9e-5 measured, 16x smaller at 4x the
#: resolution); the histogram kernel assigns each parcel to one node (zeroth order,
#: #2683), 5.3e-4 here. The "inside_grid" window puts a step beside the 12.589 Gyr
#: node, where that assignment moves 1 % of the mass between two nodes: a property
#: of the kernel that #2714 does not touch, so only cic is held to the reference.
SHARE_TOL = {
    ("cic", "past_oldest"): 1e-5,
    ("cic", "inside_grid"): 1e-4,
    ("dsps", "past_oldest"): 1e-3,
}
#: constant-SFR windows (lookback Gyr): one ending past the oldest node, one inside.
WINDOWS = {"past_oldest": (13.5, 0.001), "inside_grid": (12.0, 0.5)}
#: Photometry of the PARSEC z = 0 constant history after / before the fix, same
#: units (measured with main's source and data): FUV x0.9325, r x0.9499.
PHOT_RATIO = {"galex_fuv": 0.9325, "sdss_r": 0.9499}


@pytest.fixture(scope="module")
def ssp_data_mist():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    path = package_or_env_data_path("fsps_mist_miles_chabrier.h5")
    if not path.is_file():
        pytest.skip(f"SSP data not found: {path}")
    return load_ssp_data(str(path))


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(BANDS))


def _const_model(ssp, obs, kernel, window, z=0.0):
    start, end = window
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(z),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": start,
            "end_gyr": end,
            "age_kernel": kernel,
        },
    )


def _age_marginal(model):
    jw = np.asarray(model.predict_state({}).derived["joint_weights"]).sum(0)
    return jw / jw.sum()


def _clamp_reference(lg_nodes_gyr, window):
    """Node shares of a constant SFR on ``window`` with parcels clamped onto the last node.

    Exact up to Gauss-Legendre round-off: every parcel at lookback t splits
    between its bracketing nodes linear in log10(t); a parcel older than the
    last node goes wholly onto it.
    """
    start, end = window
    nodes = 10.0**lg_nodes_gyr
    xg, wg = np.polynomial.legendre.leggauss(400)
    w = np.zeros(len(nodes))
    # segments: [0, node_0] -> node 0; [node_j, node_j+1] split; [node_last, inf) -> last
    bounds = [(0.0, nodes[0], -1)] + [(nodes[j], nodes[j + 1], j) for j in range(len(nodes) - 1)]
    bounds.append((nodes[-1], np.inf, len(nodes) - 1))
    for lo, hi, j in bounds:
        a, b = max(lo, end), min(hi, start)
        if b <= a:
            continue
        if j == -1:
            w[0] += b - a
        elif not np.isfinite(hi):
            w[-1] += b - a
        else:
            t = 0.5 * (b - a) * xg + 0.5 * (b + a)
            mass = 0.5 * (b - a) * wg
            f = (np.log10(t) - lg_nodes_gyr[j]) / (lg_nodes_gyr[j + 1] - lg_nodes_gyr[j])
            w[j] += np.sum(mass * (1.0 - f))
            w[j + 1] += np.sum(mass * f)
    return w / w.sum()


@pytest.mark.parametrize(("kernel", "window_name"), sorted(SHARE_TOL))
def test_shares_equal_the_clamp_reference(ssp_data_fsps, obs, kernel, window_name):
    lg = np.asarray(ssp_data_fsps.ssp_lg_age_gyr)
    ref = _clamp_reference(lg, WINDOWS[window_name])
    got = _age_marginal(_const_model(ssp_data_fsps, obs, kernel, WINDOWS[window_name]))
    young = 10.0**lg < 1.0
    tol = SHARE_TOL[(kernel, window_name)]
    assert abs(got[-1] - ref[-1]) <= tol, f"{kernel} oldest node {got[-1]:.6f} vs {ref[-1]:.6f}"
    assert abs(got[young].sum() - ref[young].sum()) <= tol, (
        f"{kernel} nodes < 1 Gyr {got[young].sum():.6f} vs {ref[young].sum():.6f}"
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_no_uniform_rescale_of_the_younger_nodes(ssp_data_fsps, obs, kernel):
    """The 6.99 % of the mass younger than 1 Gyr is no longer inflated to 7.50 %."""
    lg = np.asarray(ssp_data_fsps.ssp_lg_age_gyr)
    got = _age_marginal(_const_model(ssp_data_fsps, obs, kernel, WINDOWS["past_oldest"]))
    assert got[10.0**lg < 1.0].sum() == pytest.approx(0.0699, abs=2e-4)
    assert got[-1] == pytest.approx(0.1192, abs=1e-3)


def test_photometry_of_the_constant_history_changes_by_the_stated_amount(ssp_data_fsps, obs):
    """FUV x0.9325 and r x0.9499 against the dropped-and-rescaled weights.

    The dropped 6.75 % of the mass was re-added as young and intermediate nodes
    (bluer than a 12.6 Gyr population); it now sits on the oldest node.
    """
    cic = np.asarray(
        _const_model(ssp_data_fsps, obs, "cic", WINDOWS["past_oldest"]).predict_photometry({})
    )
    # main's values for the same model, computed with the unfixed source.
    before = np.array([8.256804844271512e-13, 3.0655354736032988e-12])
    for band, got, old in zip(BANDS, cic, before, strict=True):
        assert got / old == pytest.approx(PHOT_RATIO[band], abs=5e-4), band


@pytest.mark.parametrize("kernel", KERNELS)
def test_grid_without_a_gap_is_unchanged_to_1e_12(ssp_data_mist, obs, kernel, monkeypatch):
    """MIST (oldest node 14.1 Gyr) at z = 0: the extension adds zero-width parcels only."""
    m = _const_model(ssp_data_mist, obs, kernel, WINDOWS["past_oldest"])
    with_ext = np.asarray(m.predict_state({}).derived["joint_weights"])
    phot_ext = np.asarray(m.predict_photometry({}))
    monkeypatch.setattr(
        C, "_extend_integrand_to_history", lambda fine, tab, ssp, factor=16: (fine, ssp[-1])
    )
    monkeypatch.setattr(
        C,
        "_refined_dsps_sfr_with_fold",
        lambda lb, fn, kw, extent: (fn(lb, **kw), 0.0),
    )
    jax.clear_caches()
    m2 = _const_model(ssp_data_mist, obs, kernel, WINDOWS["past_oldest"])
    # Zero-width parcels and one longer sfh_fn call change the reduction order only.
    np.testing.assert_allclose(
        with_ext, np.asarray(m2.predict_state({}).derived["joint_weights"]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(phot_ext, np.asarray(m2.predict_photometry({})), rtol=1e-12)
    jax.clear_caches()


def _warn_classes(rec):
    return [r for r in rec if issubclass(r.category, SFHBeyondOldestTemplateWarning)]


def test_warning_fires_once_per_build_on_the_gap(ssp_data_fsps, obs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = _const_model(ssp_data_fsps, obs, "cic", WINDOWS["past_oldest"])
    hits = _warn_classes(rec)
    assert len(hits) == 1
    text = str(hits[0].message)
    assert "12.589" in text and f"{float(age_at_z(0.0)):.3f}" in text
    with warnings.catch_warnings(record=True) as rec:  # calls never warn again
        warnings.simplefilter("always")
        m.predict_photometry({})
        jax.jit(lambda: m.predict_photometry({}))()
    assert _warn_classes(rec) == []


@pytest.mark.parametrize(
    ("grid", "z", "fires"),
    [("fsps", 0.0, True), ("fsps", 0.2, False), ("mist", 0.0, False)],
)
def test_warning_is_silent_without_a_gap(ssp_data_fsps, ssp_data_mist, obs, grid, z, fires):
    ssp = ssp_data_fsps if grid == "fsps" else ssp_data_mist
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _const_model(ssp, obs, "cic", WINDOWS["inside_grid"], z=z)
    assert bool(_warn_classes(rec)) is fires


def test_warning_reads_the_lowest_redshift_of_a_free_range(ssp_data_fsps, obs):
    from tengri import Uniform

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            neb={"type": "none"},
            redshift=Uniform(0.0, 0.5),
            sfh={"type": "const", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
        )
    assert len(_warn_classes(rec)) == 1


ALPHA_NODES = jnp.array([-0.2, 0.0, 0.2, 0.4])


@pytest.mark.parametrize("kernel", KERNELS)
def test_alpha_enhanced_grid_has_the_same_shares(ssp_data_fsps, obs, kernel):
    """The alpha axis is collapsed before the kernels, so the shares are the 3-D ones."""
    flux4 = ssp_data_fsps.ssp_flux[:, None, :, :] * (1.0 + 0.4 * ALPHA_NODES)[None, :, None, None]
    ssp4 = ssp_data_fsps._replace(ssp_flux=flux4, ssp_alpha_fe=ALPHA_NODES)
    ref = _clamp_reference(np.asarray(ssp_data_fsps.ssp_lg_age_gyr), WINDOWS["past_oldest"])
    m = SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 13.5,
            "end_gyr": 0.001,
            "age_kernel": kernel,
        },
        met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
    )
    got = _age_marginal(m)
    assert abs(got[-1] - ref[-1]) <= SHARE_TOL[(kernel, "past_oldest")]
    assert abs(got[:-1].sum() - ref[:-1].sum()) <= SHARE_TOL[(kernel, "past_oldest")]


def test_fold_mass_does_not_depend_on_the_last_float_of_age_z():
    """A history reaching age(z) puts the same mass on the oldest row at age(z)(1 +- 1e-12).

    The last tail point is exactly the extent, so no comparison against a
    recomputed 10**log10(t) decides whether it counts.
    """
    ages = jnp.asarray(10.0 ** np.linspace(6.0, np.log10(12.589e9), 40))
    lookback = C._refined_dsps_lookbacks(ages)
    t_obs_yr = float(age_at_z(0.0)) * 1e9
    masses = [
        float(
            C._refined_dsps_sfr_with_fold(
                lookback, lambda t, **_: jnp.ones_like(t), {}, t_obs_yr * (1.0 + d)
            )[1]
        )
        for d in (-1e-12, 0.0, 1e-12)
    ]
    assert masses[0] == pytest.approx(masses[2], rel=1e-10)
    assert masses[1] == pytest.approx(t_obs_yr - float(lookback[-1]), rel=1e-10)


def _table_state(ssp, obs, kernel):
    """A constant tabulated history, 1 Msun/yr over cosmic time 0.287-13.787 Gyr."""
    t_gyr = np.linspace(0.287, float(age_at_z(0.0)), 400)
    m = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={"type": "table", "age_kernel": kernel},
    )
    state = m.predict_state({"sfh_t_gyr": jnp.asarray(t_gyr), "sfh_sfr": jnp.ones_like(t_gyr)})
    jw = np.asarray(state.derived["joint_weights"]).sum(0)
    return jw / jw.sum(), float(state.derived["log_mstar_formed"]), t_gyr


@pytest.mark.parametrize("kernel", KERNELS)
def test_tabulated_history_folds_onto_the_oldest_node(ssp_data_fsps, obs, kernel):
    """The histogram kernel dropped a table's mass beyond the last node (0.03 dex low)."""
    got, log_formed, t_gyr = _table_state(ssp_data_fsps, obs, kernel)
    lg = np.asarray(ssp_data_fsps.ssp_lg_age_gyr)
    lookback_max = float(age_at_z(0.0)) - t_gyr[0]
    ref = _clamp_reference(lg, (lookback_max, 0.0))
    assert abs(got[-1] - ref[-1]) <= 1e-3, f"{kernel} oldest node {got[-1]:.6f} vs {ref[-1]:.6f}"
    young = 10.0**lg < 1.0
    assert abs(got[young].sum() - ref[young].sum()) <= 1e-3
    table_integral = np.log10(1.0 * (t_gyr[-1] - t_gyr[0]) * 1e9)
    assert log_formed == pytest.approx(table_integral, abs=1e-3)

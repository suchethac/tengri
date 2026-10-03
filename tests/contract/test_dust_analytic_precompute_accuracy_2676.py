# SPDX-License-Identifier: BSD-3-Clause
"""Accuracy tests for the analytic dust-emission precompute adapter.

Tests off-node accuracy at default grids, Wien-tail accuracy, band-coverage refusal,
and redshift consistency against exact closures integrated on a wide fine grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def tophat(lo_um, hi_um):
    """Construct a top-hat filter."""
    inner = np.geomspace(lo_um * 1e4, hi_um * 1e4, 400)
    return (
        np.concatenate([[inner[0] * (1 - 1e-6)], inner, [inner[-1] * (1 + 1e-6)]]),
        np.concatenate([[0.0], np.ones_like(inner), [0.0]]),
    )


def exact(model, kw, fw, ft):
    """Exact closure: integrate model on ultra-fine grid."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    WIDE = np.geomspace(1e2, 1e8, 40000)
    s = np.asarray(M[model](jnp.asarray(WIDE), 1.0, **kw), dtype=float)
    return np.trapezoid(np.interp(fw, WIDE, s) * ft / fw, fw) / np.trapezoid(ft / fw, fw)


def lookup(model, kw, fw, ft, **grids):
    """Lookup value from precomputed grid."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    res = adapter.precompute([fw], [ft], 0.0, None, model=model, **grids)
    q = [kw[n] for n in adapter.AXIS_PARAMS[model]]
    return float(np.asarray(adapter.build_lookup(res, model=model)(1.0, *q)).ravel()[0])


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody"])
def test_off_node_accuracy_default_grids(model):
    """Off-node accuracy at default grids: issue query and 12 random points."""
    rng = np.random.RandomState(42)

    # Axis parameter bounds (from prior registry)
    if model == "modified_blackbody":
        axis_bounds = {"dust_T": (10.0, 100.0), "dust_beta_ir": (1.0, 2.5)}
        kw_issue = {"dust_T": 47.3, "dust_beta_ir": 1.65}
    elif model == "casey2012":
        axis_bounds = {
            "dust_T": (10.0, 100.0),
            "dust_beta_ir": (1.0, 2.5),
            "dust_alpha_mir": (0.5, 3.5),
            "dust_lambda_0_um": (50.0, 300.0),
        }
        kw_issue = {
            "dust_T": 47.3,
            "dust_beta_ir": 1.65,
            "dust_alpha_mir": 1.8,
            "dust_lambda_0_um": 130.0,
        }
    elif model == "graybody":
        axis_bounds = {
            "dust_T": (10.0, 100.0),
            "dust_beta_ir": (1.0, 2.5),
            "dust_lambda_0_um": (50.0, 300.0),
        }
        kw_issue = {"dust_T": 47.3, "dust_beta_ir": 1.65, "dust_lambda_0_um": 130.0}

    bands = ((60, 90), (250, 500), (750, 950))

    # Test issue query first
    for b in bands:
        fw, ft = tophat(*b)
        lkp = lookup(model, kw_issue, fw, ft)
        ex = exact(model, kw_issue, fw, ft)
        ratio = lkp / ex
        assert abs(ratio - 1.0) < 1e-3, (
            f"{model} issue query band {b}: lookup/exact = {ratio:.6f}, "
            f"error {abs(ratio - 1.0):.6f}"
        )

    # Test 12 random points within axis bounds
    for _ in range(12):
        kw = {k: rng.uniform(*v) for k, v in axis_bounds.items()}
        for b in bands:
            fw, ft = tophat(*b)
            lkp = lookup(model, kw, fw, ft)
            ex = exact(model, kw, fw, ft)
            ratio = lkp / ex
            assert abs(ratio - 1.0) < 1e-3, (
                f"{model} random point {kw}, band {b}: "
                f"lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
            )


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody"])
def test_on_node_exactness(model):
    """On-node query agrees to 1e-3."""
    if model == "modified_blackbody":
        kw = {"dust_T": 30.0, "dust_beta_ir": 1.5}
        T_grid = np.array([20.0, 30.0, 60.0])
        beta_grid = np.array([1.5, 1.8, 2.0])
        grids = {"T_grid": T_grid, "beta_grid": beta_grid}
    elif model == "casey2012":
        kw = {"dust_T": 25.0, "dust_beta_ir": 1.8, "dust_alpha_mir": 2.0, "dust_lambda_0_um": 150.0}
        T_grid = np.array([25.0, 35.0, 60.0])
        beta_grid = np.array([1.5, 1.8, 2.0])
        alpha_mir_grid = np.array([1.5, 2.0, 2.5])
        lambda_0_um_grid = np.array([100.0, 150.0, 200.0])
        grids = {
            "T_grid": T_grid,
            "beta_grid": beta_grid,
            "alpha_mir_grid": alpha_mir_grid,
            "lambda_0_um_grid": lambda_0_um_grid,
        }
    elif model == "graybody":
        kw = {"dust_T": 20.0, "dust_beta_ir": 1.8, "dust_lambda_0_um": 100.0}
        T_grid = np.array([20.0, 40.0, 60.0])
        beta_grid = np.array([1.5, 1.8, 2.0])
        lambda_0_um_grid = np.array([100.0, 150.0, 200.0])
        grids = {
            "T_grid": T_grid,
            "beta_grid": beta_grid,
            "lambda_0_um_grid": lambda_0_um_grid,
        }

    fw, ft = tophat(250, 500)
    lkp = lookup(model, kw, fw, ft, **grids)
    ex = exact(model, kw, fw, ft)
    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"{model} on-node: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


@pytest.mark.parametrize("model", ["modified_blackbody", "graybody"])
def test_wien_tail_accuracy(model):
    """Wien tail (15 K, 8-24 um) converged to 1e-3."""
    if model == "modified_blackbody":
        kw = {"dust_T": 15.0, "dust_beta_ir": 1.5}
        grids = {
            "T_grid": 15.0 * np.array([0.998, 1.0, 1.002]),
            "beta_grid": 1.5 * np.array([0.998, 1.0, 1.002]),
        }
    elif model == "graybody":
        kw = {"dust_T": 15.0, "dust_beta_ir": 1.5, "dust_lambda_0_um": 200.0}
        grids = {
            "T_grid": 15.0 * np.array([0.998, 1.0, 1.002]),
            "beta_grid": 1.5 * np.array([0.998, 1.0, 1.002]),
            "lambda_0_um_grid": 200.0 * np.array([0.998, 1.0, 1.002]),
        }

    fw, ft = tophat(8, 24)
    lkp = lookup(model, kw, fw, ft, **grids)
    ex = exact(model, kw, fw, ft)
    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"{model} Wien tail: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


def test_band_coverage_refusal():
    """Bands outside rest grid are refused with a ValueError naming band and range."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    # 12-20 mm is outside the default continuum grid (100 Å - 10 mm)
    kw = {"dust_T": 20.0, "dust_beta_ir": 1.5}
    grids = {
        "T_grid": 20.0 * np.array([0.998, 1.0, 1.002]),
        "beta_grid": 1.5 * np.array([0.998, 1.0, 1.002]),
    }

    fw, ft = tophat(120000, 200000)  # 12-20 mm (um in Angstrom)
    with pytest.raises(ValueError) as excinfo:
        lookup("modified_blackbody", kw, fw, ft, **grids)
    err_msg = str(excinfo.value)
    # Should name the band's rest-frame wavelength range
    assert "12" in err_msg or "20" in err_msg or "Filter" in err_msg or "range" in err_msg, (
        f"ValueError should name the band or its range; got: {err_msg}"
    )

    # pah_drude 40-70 um test removed: now covered by test_pah_drude_coverage


def test_pah_drude_coverage():
    """pah_drude in 40-70 um agrees with exact closure to 1e-3."""
    from tengri.components.dust import dust_analytic_precompute as adapter
    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    # The PAH grid must be widened to include 40-70 um; once it is, test it.
    fw, ft = tophat(40, 70)
    res = adapter.precompute([fw], [ft], 0.0, None, model="pah_drude")

    WIDE = np.geomspace(1e2, 1e8, 40000)
    wide_pah = np.asarray(M["pah_drude"](jnp.asarray(WIDE), 1.0), dtype=float)
    ex = np.trapezoid(np.interp(fw, WIDE, wide_pah) * ft / fw, fw) / np.trapezoid(
        ft / fw, fw
    )

    lkp = float(np.asarray(adapter.build_lookup(res, model="pah_drude")(1.0)).ravel()[0])
    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"pah_drude 40-70 um: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


def test_redshift_off_node():
    """z=2 off-node query (issue's first case at redshift)."""
    from tengri.components.dust import dust_analytic_precompute as adapter
    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    z = 2.0
    kw = {"dust_T": 47.3, "dust_beta_ir": 1.65}
    fw, ft = tophat(60, 90)  # observed frame

    # Exact: template evaluated on ultra-fine grid, then interpolated
    WIDE = np.geomspace(1e2, 1e8, 40000)
    fw_rest = fw / (1 + z)  # convert to rest frame
    s = np.asarray(M["modified_blackbody"](jnp.asarray(WIDE), 1.0, **kw), dtype=float)
    ex = np.trapezoid(np.interp(fw_rest, WIDE, s) * ft / fw_rest, fw_rest) / np.trapezoid(
        ft / fw_rest, fw_rest
    )

    # Lookup at redshift z
    res = adapter.precompute([fw], [ft], z, None, model="modified_blackbody")
    lkp = float(
        np.asarray(adapter.build_lookup(res, model="modified_blackbody")(1.0)).ravel()[0]
    )

    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"modified_blackbody z=2: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )

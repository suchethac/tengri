# SPDX-License-Identifier: BSD-3-Clause
"""Folded integrand (#2769) equals the materialized interpolate-then-weight path.

The fold never builds the (..., m) interpolated integrand; it reads the SSP row
through node weights. These tests pin it against the materialized path for
positive SSP-like templates on a union grid that has out-of-bounds nodes, with
and without the Lyman-edge augmentation.
"""

import numpy as np
import pytest

from tengri.utils.filter_convention import FilterConvention
from tengri.utils.grid_interp import (
    _np_trapezoid,
    _vectorized_interp,
    edge_split,
    fold_integrand,
    fold_integrand_T,
    lyc_augment_fold,
    lyc_augment_fold_T,
    lyc_augment_grid_for_step,
    preintegrate_grid,
    preintegrate_transposed,
    subband_quadrature,
    transpose_templates,
)

RTOL = 1e-13
N_SUB = 8
EFF_WAVE = 4000.0


def _scale_atol(expected):
    return RTOL * max(float(np.max(np.abs(expected))), 1e-300)


def _assert_match(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=_scale_atol(expected))


@pytest.fixture(scope="module")
def ssp_like():
    rng = np.random.default_rng(2769)
    wave_src = np.geomspace(91.0, 1.0e6, 240)
    # Positive, smoothly varying "spectra" with a steep blue end, like an SSP.
    templates = (
        np.exp(rng.normal(0.0, 1.0, (3, 4, wave_src.size)))
        * (wave_src[None, None, :] / 5000.0) ** -1.5
    )
    # Union grid: source nodes plus filter nodes, extending past both source ends
    # so some points are out of bounds (zeroed by the materialized path).
    filter_wave = np.linspace(700.0, 2500.0, 57)
    grid = np.sort(
        np.concatenate(
            [
                np.exp(rng.uniform(np.log(30.0), np.log(1.2e6), 300)),
                wave_src[::3],
                filter_wave,
            ]
        )
    )
    tw = rng.uniform(0.05, 1.0, grid.size)
    filter_trans = rng.uniform(0.1, 1.0, filter_wave.size)
    return {
        "wave_src": wave_src,
        "templates": templates,
        "grid": grid,
        "tw": tw,
        "filter_wave": filter_wave,
        "filter_trans": filter_trans,
    }


def _materialized(grid, tw, wave_src, templates):
    return _vectorized_interp(grid, wave_src, templates) * tw


def test_out_of_bounds_nodes_exist(ssp_like):
    grid, wave = ssp_like["grid"], ssp_like["wave_src"]
    assert np.any(grid < wave[0]) and np.any(grid > wave[-1])


def test_fold_trapezoid_matches_materialized(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    fold = fold_integrand(grid, tw, wave, tmpl)
    mat = _materialized(grid, tw, wave, tmpl)
    _assert_match(fold.trapezoid(), _np_trapezoid(mat, grid, axis=-1))


def test_fold_moment_matches_materialized(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    lam_minus_eff = grid / 1.7 - 1234.5
    fold = fold_integrand(grid, tw, wave, tmpl)
    mat = _materialized(grid, tw, wave, tmpl)
    _assert_match(
        fold.trapezoid(lam_minus_eff),
        _np_trapezoid(mat * lam_minus_eff, grid, axis=-1),
    )


def test_fold_dot_matches_materialized_for_random_weights(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    rng = np.random.default_rng(7)
    weights = rng.uniform(-1.0, 1.0, (grid.size, 5))
    fold = fold_integrand(grid, tw, wave, tmpl)
    mat = _materialized(grid, tw, wave, tmpl)
    _assert_match(fold.dot(weights), mat @ weights)


@pytest.mark.parametrize("edge_frac", [0.05, 0.3, 0.8])
def test_fold_edge_split_matches_materialized(ssp_like, edge_frac):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    edge = float(grid[0] + edge_frac * (grid[-1] - grid[0]))
    fold = fold_integrand(grid, tw, wave, tmpl)
    mat = _materialized(grid, tw, wave, tmpl)
    below_f, above_f = edge_split(fold, grid, edge)
    below_m, above_m = edge_split(mat, grid, edge)
    _assert_match(below_f, below_m)
    _assert_match(above_f, above_m)


@pytest.mark.parametrize("lyc_edge", [None, 911.76 * 1.8])
def test_fold_subband_matches_materialized(ssp_like, lyc_edge):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    fold = fold_integrand(grid, tw, wave, tmpl)
    mat = _materialized(grid, tw, wave, tmpl)
    denom = _np_trapezoid(tw, grid)
    phi_f, nodes_f = subband_quadrature(grid, tw, fold, denom, N_SUB, EFF_WAVE, lyc_edge)
    phi_m, nodes_m = subband_quadrature(grid, tw, mat, denom, N_SUB, EFF_WAVE, lyc_edge)
    _assert_match(phi_f, phi_m)
    _assert_match(nodes_f, nodes_m)


def test_lyc_augmented_fold_matches_dense_augmentation(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    fw, ft = ssp_like["filter_wave"], ssp_like["filter_trans"]
    edge = float(wave[60] * 1.0000007 + 0.25)  # inside a source bracket cell
    conv = FilterConvention.BESSELL
    grid_d, integ_d, tw_d = lyc_augment_grid_for_step(grid, tw, wave, tmpl, fw, ft, edge, conv)
    grid_f, fold_f, tw_f = lyc_augment_fold(grid, tw, wave, tmpl, fw, ft, edge, conv)
    np.testing.assert_array_equal(grid_f, grid_d)
    np.testing.assert_array_equal(tw_f, tw_d)
    assert grid_f.size == grid.size + 2
    _assert_match(fold_f.trapezoid(), _np_trapezoid(integ_d, grid_d, axis=-1))
    below_f, above_f = edge_split(fold_f, grid_f, edge)
    below_d, above_d = edge_split(integ_d, grid_d, edge)
    _assert_match(below_f, below_d)
    _assert_match(above_f, above_d)
    denom = _np_trapezoid(tw_d, grid_d)
    phi_f, nodes_f = subband_quadrature(grid_f, tw_f, fold_f, denom, N_SUB, EFF_WAVE, edge)
    phi_d, nodes_d = subband_quadrature(grid_d, tw_d, integ_d, denom, N_SUB, EFF_WAVE, edge)
    _assert_match(phi_f, phi_d)
    _assert_match(nodes_f, nodes_d)


def test_lyc_augmented_fold_unbracketed_edge_falls_back(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    fw, ft = ssp_like["filter_wave"], ssp_like["filter_trans"]
    edge = float(wave[-1] * 2.0)  # beyond the source: no bracket cell
    grid_f, fold_f, _ = lyc_augment_fold(grid, tw, wave, tmpl, fw, ft, edge)
    np.testing.assert_array_equal(grid_f, grid)
    _assert_match(fold_f.trapezoid(), _np_trapezoid(_materialized(grid, tw, wave, tmpl), grid, -1))


def test_transposed_layout_is_contiguous_and_row_major_in_templates(ssp_like):
    tmpl = ssp_like["templates"]
    src_T = transpose_templates(tmpl)
    assert src_T.flags.c_contiguous
    assert src_T.shape == (tmpl.shape[-1], int(np.prod(tmpl.shape[:-1])))
    np.testing.assert_array_equal(src_T, tmpl.reshape(-1, tmpl.shape[-1]).T)


def test_fold_T_matches_materialized_for_trapezoid_and_dot(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    src_T = transpose_templates(tmpl)
    fold = fold_integrand_T(grid, tw, wave, src_T, tmpl.shape[:-1])
    mat = _materialized(grid, tw, wave, tmpl)
    assert fold.lead_shape == tmpl.shape[:-1]
    _assert_match(fold.trapezoid(), _np_trapezoid(mat, grid, axis=-1))
    weights = np.random.default_rng(11).uniform(-1.0, 1.0, (grid.size, 4))
    _assert_match(fold.dot(weights), mat @ weights)


def test_fold_T_subband_matches_materialized(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    fold = fold_integrand_T(grid, tw, wave, transpose_templates(tmpl), tmpl.shape[:-1])
    mat = _materialized(grid, tw, wave, tmpl)
    denom = _np_trapezoid(tw, grid)
    phi_f, nodes_f = subband_quadrature(grid, tw, fold, denom, N_SUB, EFF_WAVE, 911.76 * 1.8)
    phi_m, nodes_m = subband_quadrature(grid, tw, mat, denom, N_SUB, EFF_WAVE, 911.76 * 1.8)
    _assert_match(phi_f, phi_m)
    _assert_match(nodes_f, nodes_m)


def test_lyc_augmented_fold_T_matches_fold(ssp_like):
    grid, tw = ssp_like["grid"], ssp_like["tw"]
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    fw, ft = ssp_like["filter_wave"], ssp_like["filter_trans"]
    edge = float(wave[60] * 1.0000007 + 0.25)
    grid_a, fold_a, tw_a = lyc_augment_fold(grid, tw, wave, tmpl, fw, ft, edge)
    grid_t, fold_t, tw_t = lyc_augment_fold_T(
        grid, tw, wave, transpose_templates(tmpl), tmpl.shape[:-1], fw, ft, edge
    )
    np.testing.assert_array_equal(grid_t, grid_a)
    np.testing.assert_array_equal(tw_t, tw_a)
    _assert_match(fold_t.trapezoid(), fold_a.trapezoid())
    below_t, above_t = edge_split(fold_t, grid_t, edge)
    below_a, above_a = edge_split(fold_a, grid_a, edge)
    _assert_match(below_t, below_a)
    _assert_match(above_t, above_a)


@pytest.mark.parametrize("lyc_gate", [False, True])
def test_preintegrate_transposed_matches_preintegrate_grid(ssp_like, lyc_gate):
    wave, tmpl = ssp_like["wave_src"], ssp_like["templates"]
    filters = [
        (ssp_like["filter_wave"], ssp_like["filter_trans"]),
        (np.linspace(3000.0, 9000.0, 60), np.linspace(0.2, 0.9, 60)),
    ]
    kwargs = dict(redshift=0.7, dl_cm=1.0, taylor=True, n_subbands=N_SUB, lyc_gate=lyc_gate)
    ref = preintegrate_grid(tmpl, wave, [f[0] for f in filters], [f[1] for f in filters], **kwargs)
    got = preintegrate_transposed(
        transpose_templates(tmpl),
        tmpl.shape[:-1],
        wave,
        [f[0] for f in filters],
        [f[1] for f in filters],
        **kwargs,
    )
    for name in ("phot", "moment", "subband_phot", "subband_waves_rest"):
        _assert_match(np.asarray(getattr(got, name)), np.asarray(getattr(ref, name)))
    if lyc_gate:
        _assert_match(np.asarray(got.lyc_phot), np.asarray(ref.lyc_phot))

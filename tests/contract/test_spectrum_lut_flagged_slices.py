# SPDX-License-Identifier: BSD-3-Clause
"""Static node windows of the flagged pixels and the slice dust transmission (#3413).

A slice of the model grid must carry exactly the dust transmission the full grid carries at
the same nodes: the two-component screen is elementwise in wavelength, so the slice value is
bit-identical to the full-grid value at those nodes.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust._apply import two_component_dust
from tengri.components.stellar.sps.flagged_pixels import (
    flagged_pixel_mask,
    node_window,
    padded_node_slices,
    slice_attenuated_stellar,
)
from tengri.observation.spectrum import lattice_correction_rest, lattice_window
from tengri.utils.physics_constants import C_KM_S


def test_flagged_mask_marks_pixels_whose_padded_range_holds_a_breakpoint():
    edges = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    flags = flagged_pixel_mask(edges, np.array([2500.0]), pad_aa=0.0)
    np.testing.assert_array_equal(flags, [False, True, False])
    # A breakpoint on a shared edge flags both neighbours once padded.
    flags = flagged_pixel_mask(edges, np.array([2000.0]), pad_aa=10.0)
    np.testing.assert_array_equal(flags, [True, True, False])


def test_no_breakpoints_flags_nothing():
    edges = np.array([1.0, 2.0, 3.0])
    assert not flagged_pixel_mask(edges, np.array([]), pad_aa=5.0).any()


def test_padded_windows_address_the_same_nodes():
    wave = np.linspace(1.0, 100.0, 100)
    w0 = node_window(wave, 10.0, 20.0)
    w1 = node_window(wave, 50.0, 53.0)
    idx, mask = padded_node_slices([w0, w1])
    assert idx.shape == mask.shape == (2, max(w0.size, w1.size))
    np.testing.assert_array_equal(idx[0, : w0.size], w0)
    np.testing.assert_array_equal(idx[1, : w1.size], w1)
    assert mask[0].sum() == w0.size and mask[1].sum() == w1.size


def test_slice_transmission_is_bit_identical_to_full_grid():
    wave = np.geomspace(2e4, 2e5, 60)
    ages = jnp.array([1e6, 1e8, 1e10])
    kw = dict(law_bc="calzetti", law_diff="calzetti")
    full = np.asarray(two_component_dust(jnp.asarray(wave), ages, 0.5, 0.3, **kw))
    idx = node_window(wave, 1.9e4, 3.1e4)
    slice_t = np.asarray(two_component_dust(jnp.asarray(wave[idx]), ages, 0.5, 0.3, **kw))
    np.testing.assert_array_equal(slice_t, full[:, idx])


_SIGMA_BOUND_KMS = 2000.0
_PAD_SIGMAS = 8.0


def _window_pad_aa(lam_aa: float, sigma_kms: float) -> float:
    """Half-width of a pixel's node window [Angstrom]: ``_PAD_SIGMAS`` of the bound."""
    return lam_aa * np.expm1(_PAD_SIGMAS * sigma_kms / C_KM_S)


def _pixel_setup():
    wave_full = np.geomspace(1.0e4, 6.0e4, 20000)
    sed_full = 1.0 + 0.5 * np.sin(2.0 * np.pi * wave_full / 2000.0)
    wave_obs = np.linspace(2.9e4, 3.1e4, 75)
    mid = 0.5 * (wave_obs[1:] + wave_obs[:-1])
    half = 0.5 * np.diff(wave_obs)
    edges = np.concatenate([[wave_obs[0] - half[0]], mid, [wave_obs[-1] + half[-1]]])
    return wave_full, sed_full, wave_obs, edges


def _slice_correction(wave_full, sed_full, wave_obs, edges, p, sigma_kms, pad_aa):
    lo, hi = edges[p] - pad_aa, edges[p + 1] + pad_aa
    idx, mask = padded_node_slices([node_window(wave_full, lo, hi)])
    nodes = idx[0][mask[0]]
    return lattice_correction_rest(
        sed_full[nodes],
        wave_full[nodes],
        wave_obs[p - 1 : p + 2],
        sigma_kms,
        span=lattice_window(lo, hi),
    )[1]


def test_slice_attenuated_stellar_matches_full_grid_exact_contraction():
    """The window contraction equals the exact path's per-age contraction at the same nodes."""
    from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig

    dust = DustSEDComponent(config=DustSEDComponentConfig(law_bc="calzetti", law_diff="calzetti"))
    wave = np.geomspace(2e4, 2e5, 60)
    younger = jnp.asarray([[1.0, 0.5, 0.0]])
    rng = np.random.default_rng(3)
    lnu_age = jnp.asarray(rng.uniform(0.1, 1.0, size=(3, wave.size)))
    params = {"dust_tau_bc": 0.5, "dust_tau_diff": 0.3}
    full_t = dust.compute_transmission(params, jnp.asarray(wave), younger)
    full_att = np.asarray(jnp.sum(lnu_age * full_t, axis=0))
    full_int = np.asarray(jnp.sum(lnu_age, axis=0))

    nodes = node_window(wave, 1.9e4, 3.1e4)
    idx, mask = padded_node_slices([nodes])
    att, intr = slice_attenuated_stellar(
        dust, params, jnp.asarray(wave), lnu_age, younger, idx, mask
    )
    np.testing.assert_allclose(np.asarray(att), full_att[nodes], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(intr), full_int[nodes], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("sigma_kms", [150.0, 600.0])
def test_window_correction_matches_full_grid_at_flagged_pixel(sigma_kms):
    wave_full, sed_full, wave_obs, edges = _pixel_setup()
    p = 37
    full = np.asarray(lattice_correction_rest(sed_full, wave_full, wave_obs, sigma_kms))
    pad = _window_pad_aa(wave_obs[p], _SIGMA_BOUND_KMS)
    sliced = float(_slice_correction(wave_full, sed_full, wave_obs, edges, p, sigma_kms, pad))
    # Tolerance on the flux scale: the cumulative sum's float64 roundoff grows with the node
    # count of the window (about 1e-12 of the SED here) and is not a truncation error.
    assert abs(sliced - full[p]) <= 1e-10 * np.max(np.abs(sed_full))


def test_curvature_flag_marks_wide_pixels_and_zero_crossings_only():
    """A curve that bends over a wide pixel flags it; a narrow pixel on the same curve does not."""
    from tengri.components.dust.attenuation import calzetti
    from tengri.components.stellar.sps.flagged_pixels import curvature_flag_mask

    narrow = np.linspace(5000.0, 5100.0, 11)  # 10 A pixels at 5000 A
    assert not curvature_flag_mask(narrow, [calzetti]).any()
    wide = np.geomspace(2e4, 2e5, 61)  # 3.9 % pixels in the mid-IR, curve nearing its zero
    flags = curvature_flag_mask(wide, [calzetti])
    kink = int(np.searchsorted(wide, 31149.76)) - 1
    assert flags[: kink + 1].all()
    assert not flags[kink + 1 :].any()  # above the clip the curve is identically zero


def test_cube_interpolation_matches_model_grid_projection():
    """The window cube is the exact path's interpolation of the library onto the model nodes."""
    from tengri.components.stellar.sps.flagged_pixels import interpolate_cube_onto

    wave_ssp = np.linspace(1000.0, 2000.0, 11)
    cube = np.random.default_rng(0).uniform(size=(2, 3, 11))
    nodes = np.array([900.0, 1000.0, 1234.5, 1500.0, 2000.0, 2100.0])
    got = interpolate_cube_onto(wave_ssp, cube, nodes)
    want = np.stack(
        [[np.interp(nodes, wave_ssp, row, left=0.0, right=0.0) for row in plane] for plane in cube]
    )
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-15)


def test_bin_plan_reproduces_full_lattice_piecewise_kernel_on_a_window():
    """A window convolved with its bin plan equals the full-lattice piecewise kernel there."""
    from tengri.observation.spectrum import (
        _LATTICE_DLNLAMBDA,
        _lattice_instrument,
        _lattice_instrument_planned,
        _lattice_pad_cells,
        lattice_bin_plan,
    )

    h = _LATTICE_DLNLAMBDA
    n_g = 20000
    origin_g = np.log(4000.0)
    lam_c = np.exp(origin_g + h * (np.arange(n_g) + 0.5))
    sigma = 40.0 + 120.0 * np.linspace(0.0, 1.0, n_g)  # km/s, varying along the lattice
    w0, n_w = 9000, 1800
    g_window = np.zeros(n_w)
    g_window[n_w // 2 - 40 : n_w // 2 + 40] = np.hanning(80)
    g_full = np.zeros(n_g)
    g_full[w0 : w0 + n_w] = g_window
    full = np.asarray(
        _lattice_instrument(jnp.asarray(g_full), jnp.asarray(lam_c), jnp.asarray(sigma), 16)
    )
    plan = lattice_bin_plan((origin_g + w0 * h, n_w), (origin_g, n_g), sigma, 16)
    got = np.asarray(_lattice_instrument_planned(jnp.asarray(g_window), *plan))
    inner = slice(n_w // 2 - 300, n_w // 2 + 300)
    assert _lattice_pad_cells() > 300
    np.testing.assert_allclose(got[inner], full[w0:w0 + n_w][inner], rtol=0.0, atol=1e-12)

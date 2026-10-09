# SPDX-License-Identifier: BSD-3-Clause
"""Batched per-galaxy spectroscopic observations: padding, bucketing, pytree contract (#2833)."""

import jax
import numpy as np
import pytest

import tengri  # noqa: F401  (enables float64)
from tengri.observation.banded import (
    BandedMatrix,
    banded_matvec,
    gaussian_resolution_bands,
)
from tengri.observation.batched import (
    GalaxySpectrum,
    SpectroBatch,
    SpectroBatchSpec,
    build_spectro_batches,
    classify_grid,
    pad_banded,
    pad_wave,
    require_spec,
)

pytestmark = pytest.mark.contract


def _linear_grid(n, lo=3600.0, hi=9800.0):
    return np.linspace(lo, hi, n)


def _galaxy(n, *, z=0.1, grid=None, ivar=None, mask=None, resolution=3000.0):
    wave = _linear_grid(n) if grid is None else grid
    flux = np.linspace(1.0, 2.0, n)
    ivar = np.full(n, 4.0) if ivar is None else ivar
    return GalaxySpectrum(
        wave=wave,
        flux=flux,
        ivar=ivar,
        resolution=resolution,
        z=z,
        mask=mask,
    )


def _zero_out_of_range(offsets, data, n):
    """Independent reference: zero entries whose column i + offsets[k] leaves [0, n)."""
    data = np.array(data, dtype=float, copy=True)
    for k, o in enumerate(np.asarray(offsets)):
        rows = np.arange(data.shape[1])
        bad = (rows + o < 0) | (rows + o >= n)
        data[k, bad] = 0.0
    return data


# ── (1) Padding and fills ─────────────────────────────────────────────────


def test_pad_wave_is_strictly_increasing_and_finite():
    wave = _linear_grid(300)
    padded = pad_wave(wave, 360)
    assert padded.shape == (360,)
    assert np.all(np.isfinite(padded))
    assert np.all(np.diff(padded) > 0)
    np.testing.assert_array_equal(padded[:300], wave)
    step = wave[-1] - wave[-2]
    np.testing.assert_allclose(np.diff(padded[299:]), step)


def test_padding_and_bad_pixel_fills():
    n = 300
    ivar = np.full(n, 4.0)
    ivar[10] = 0.0  # ivar <= 0
    mask = np.ones(n, dtype=bool)
    mask[20] = False  # masked but ivar > 0: sigma would be 0.5 without the fill
    gal = _galaxy(n, ivar=ivar, mask=mask)

    [(spec, batch, idx)] = build_spectro_batches([gal], quantum=128)
    assert spec.n_max == 384
    np.testing.assert_array_equal(idx, [0])

    pm = np.asarray(batch.pix_mask[0])
    sig = np.asarray(batch.sigma[0])
    fl = np.asarray(batch.flux[0])

    # Padded pixels.
    assert np.all(pm[n:] == 0.0)
    assert np.all(sig[n:] == 1.0)
    assert np.all(fl[n:] == 0.0)

    # ivar <= 0 and masked real pixels.
    assert pm[10] == 0.0 and sig[10] == 1.0
    assert pm[20] == 0.0 and sig[20] == 1.0
    assert fl[20] == 0.0

    # Good real pixels.
    good = np.ones(n, dtype=bool)
    good[[10, 20]] = False
    assert np.all(pm[:n][good] == 1.0)
    np.testing.assert_allclose(sig[:n][good], 0.5)


# ── (2) Key test: banded matvec ignores padded entries ───────────────────


def test_padded_banded_matvec_ignores_padding():
    n, n_max = 300, 360
    # DESI-like spacing (~0.72 A at 5000 A): the R=3000 LSF is about one pixel
    # wide, so the edge columns carry real weight. A coarse grid would leave
    # them numerically zero and the test would prove nothing.
    bm = gaussian_resolution_bands(_linear_grid(n, lo=5000.0, hi=5215.0), 3000.0)
    offsets = np.asarray(bm.offsets)
    data_raw = np.asarray(bm.data)
    data_real = _zero_out_of_range(offsets, data_raw, n)
    edge_weight = np.abs(data_raw - data_real).max()
    assert edge_weight > 1e-3, "out-of-range band weights must be non-negligible"

    rng = np.random.default_rng(0)
    x_real = rng.uniform(0.5, 2.0, n)
    ref = np.asarray(banded_matvec(offsets, data_real, x_real))

    data_pad = np.asarray(pad_banded(bm, n, n_max))
    assert data_pad.shape == (offsets.shape[0], n_max)
    np.testing.assert_array_equal(data_pad[:, n:], 0.0)

    x_a = np.concatenate([x_real, np.full(n_max - n, 1e6)])
    x_b = np.concatenate([x_real, np.full(n_max - n, -3e5)])
    y_a = np.asarray(banded_matvec(offsets, data_pad, x_a))
    y_b = np.asarray(banded_matvec(offsets, data_pad, x_b))

    np.testing.assert_allclose(y_a[:n], ref, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(y_a[:n], y_b[:n])


# ── (3) Grid classification ──────────────────────────────────────────────


def test_classify_linspace_is_nonuniform():
    assert classify_grid(np.linspace(3600.0, 9800.0, 300)) == "nonuniform"


def test_classify_geomspace_is_log_uniform():
    assert classify_grid(np.geomspace(3600.0, 9800.0, 300)) == "log_uniform"


def test_classify_rejects_non_monotonic_grid():
    w = np.geomspace(3600.0, 9800.0, 300)
    w[[100, 101]] = w[[101, 100]]
    with pytest.raises(ValueError, match="strictly increasing"):
        classify_grid(w)


def test_classify_rejects_non_finite_grid():
    w = np.geomspace(3600.0, 9800.0, 300)
    w[50] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        classify_grid(w)


def test_classify_tolerance_rejects_near_log_uniform_grid():
    # Log spacing jittered by 2e-4 relative: not log-uniform at rtol 1e-6.
    steps = 1.0 + 1e-4 * np.sin(np.arange(299))
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    w = np.exp(np.log(3600.0) + cum * np.log(9800.0 / 3600.0) / cum[-1])
    assert classify_grid(w) == "nonuniform"


def test_concatenated_overlapping_cameras_raise_with_merge_hint():
    blue = np.linspace(3600.0, 5800.0, 200)
    red = np.linspace(5700.0, 7600.0, 150)  # overlaps the blue camera
    with pytest.raises(ValueError, match="merged or sorted"):
        classify_grid(np.concatenate([blue, red]))


# ── (4) Bucketing ────────────────────────────────────────────────────────


def test_buckets_by_exact_n_pix():
    galaxies = [_galaxy(n) for n in [300, 300, 360, 300, 420]]
    buckets = build_spectro_batches(galaxies)
    assert len(buckets) == 3
    by_nmax = {spec.n_max: idx for spec, _, idx in buckets}
    assert set(by_nmax) == {300, 360, 420}
    np.testing.assert_array_equal(by_nmax[300], [0, 1, 3])
    np.testing.assert_array_equal(by_nmax[360], [2])
    np.testing.assert_array_equal(by_nmax[420], [4])
    for spec, batch, idx in buckets:
        assert batch.flux.shape == (len(idx), spec.n_max)


def test_buckets_with_quantum_round_up():
    galaxies = [_galaxy(n) for n in [300, 300, 360, 300, 420]]
    buckets = build_spectro_batches(galaxies, quantum=128)
    by_nmax = {spec.n_max: idx for spec, _, idx in buckets}
    assert set(by_nmax) == {384, 512}
    np.testing.assert_array_equal(by_nmax[384], [0, 1, 2, 3])
    np.testing.assert_array_equal(by_nmax[512], [4])


def test_mixed_offsets_split_into_separate_buckets():
    wave = _linear_grid(300)
    narrow = BandedMatrix(
        offsets=np.arange(-1, 2),
        data=np.full((3, 300), 1.0 / 3.0),
    )
    wide = BandedMatrix(
        offsets=np.arange(-2, 3),
        data=np.full((5, 300), 1.0 / 5.0),
    )
    galaxies = [
        _galaxy(300, grid=wave, resolution=narrow),
        _galaxy(300, grid=wave, resolution=wide),
    ]
    buckets = build_spectro_batches(galaxies)
    assert len(buckets) == 2
    offsets_seen = sorted(spec.offsets for spec, _, _ in buckets)
    assert offsets_seen == [(-2, -1, 0, 1, 2), (-1, 0, 1)]


def test_mixed_resolution_kinds_split_into_separate_buckets():
    wave = _linear_grid(300)
    banded = BandedMatrix(offsets=np.arange(-1, 2), data=np.full((3, 300), 1.0 / 3.0))
    galaxies = [_galaxy(300, grid=wave, resolution=banded), _galaxy(300, grid=wave)]
    buckets = build_spectro_batches(galaxies)
    assert len(buckets) == 2
    kinds = sorted(spec.grid_kind for spec, _, _ in buckets)
    assert kinds == ["banded", "nonuniform"]


def test_banded_bucket_pads_resolution_rows():
    n = 300
    wave = _linear_grid(n)
    bm = gaussian_resolution_bands(wave, 3000.0)
    gal = _galaxy(n, grid=wave, resolution=bm)
    [(spec, batch, _)] = build_spectro_batches([gal], quantum=128)
    assert spec.grid_kind == "banded"
    assert spec.offsets == tuple(int(o) for o in np.asarray(bm.offsets))
    assert batch.r_data.shape == (1, len(spec.offsets), spec.n_max)
    assert batch.resolution_pp.shape == (1, 0)
    np.testing.assert_array_equal(
        np.asarray(batch.r_data[0]), np.asarray(pad_banded(bm, n, spec.n_max))
    )


def test_gaussian_bucket_has_placeholder_banded_rows():
    [(spec, batch, _)] = build_spectro_batches([_galaxy(300)], quantum=128)
    assert spec.offsets is None
    assert spec.grid_kind == "nonuniform"
    assert batch.r_data.shape == (1, 0, spec.n_max)
    assert batch.resolution_pp.shape == (1, spec.n_max)
    np.testing.assert_allclose(np.asarray(batch.resolution_pp[0, :300]), 3000.0)


def test_photometry_width_is_part_of_bucket_key():
    phot = dict(phot_flux=np.array([1.0, 2.0, 3.0]), phot_err=np.array([0.1, 0.1, 0.1]))
    with_phot = GalaxySpectrum(
        wave=_linear_grid(300),
        flux=np.ones(300),
        ivar=np.full(300, 4.0),
        resolution=3000.0,
        z=0.1,
        **phot,
    )
    buckets = build_spectro_batches([with_phot, _galaxy(300)])
    assert len(buckets) == 2
    by_phot = {spec.has_phot: (spec, batch) for spec, batch, _ in buckets}
    spec, batch = by_phot[True]
    assert spec.n_filt == 3
    assert batch.phot_flux.shape == (1, 3)
    assert batch.phot_presence.shape == (1, 3)
    _, batch_none = by_phot[False]
    assert batch_none.phot_flux.shape == (1, 0)


def test_bad_galaxy_error_names_index():
    bad = GalaxySpectrum(
        wave=_linear_grid(300),
        flux=np.ones(299),
        ivar=np.ones(300),
        resolution=3000.0,
        z=0.1,
    )
    with pytest.raises(ValueError, match="galaxy 1"):
        build_spectro_batches([_galaxy(300), bad])


# ── (5) Pytree contract ──────────────────────────────────────────────────


def test_spectro_batch_is_pytree_with_vmap_and_galaxy_slice():
    [(spec, batch, _)] = build_spectro_batches([_galaxy(300), _galaxy(300, z=0.2)])
    assert isinstance(batch, SpectroBatch)

    leaves = jax.tree_util.tree_leaves(batch)
    assert len(leaves) == 12
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    sums = jax.vmap(lambda o: o.flux.sum())(batch)
    assert sums.shape == (2,)
    np.testing.assert_allclose(np.asarray(sums), np.asarray(batch.flux).sum(axis=1))

    one = batch.galaxy(1)
    assert isinstance(one, SpectroBatch)
    assert one.flux.shape == (spec.n_max,)
    assert one.z.shape == ()
    np.testing.assert_array_equal(np.asarray(one.flux), np.asarray(batch.flux[1]))
    np.testing.assert_array_equal(np.asarray(one.wave), np.asarray(batch.wave[1]))
    assert float(one.z) == pytest.approx(0.2)


# ── (6) Spec and range guards ────────────────────────────────────────────


def test_require_spec_rejects_none():
    with pytest.raises(ValueError):
        require_spec(None)


def test_require_spec_rejects_non_spec():
    with pytest.raises(ValueError):
        require_spec({"n_max": 300})


def test_require_spec_returns_valid_spec():
    spec = SpectroBatchSpec(n_max=300, offsets=None, grid_kind="nonuniform")
    assert require_spec(spec) is spec


def test_rest_wave_range_upper_violation_raises():
    gal = _galaxy(300, z=0.5)  # rest-frame upper edge 9800 / 1.5 = 6533
    with pytest.raises(ValueError, match="rest"):
        build_spectro_batches([gal], rest_wave_range=(1000.0, 5000.0))
    buckets = build_spectro_batches([gal], rest_wave_range=(1000.0, 9000.0))
    assert len(buckets) == 1


def test_rest_wave_range_lower_violation_raises():
    gal = _galaxy(300, z=0.5)  # rest-frame lower edge 3600 / 1.5 = 2400
    with pytest.raises(ValueError, match="rest"):
        build_spectro_batches([gal], rest_wave_range=(2500.0, 9000.0))


def test_rest_wave_range_counts_padded_tail():
    gal = _galaxy(300, z=0.5)
    # Real grid ends at 6533 rest; padding to 384 px runs past 7000 rest.
    build_spectro_batches([gal], rest_wave_range=(1000.0, 7000.0))
    with pytest.raises(ValueError, match="rest"):
        build_spectro_batches([gal], quantum=128, rest_wave_range=(1000.0, 7000.0))

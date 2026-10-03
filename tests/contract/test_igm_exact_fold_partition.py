# SPDX-License-Identifier: BSD-3-Clause
"""The exact IGM fold's ratio is taken over the partition of the tensor it multiplies.

:func:`tengri.components.igm.exact_fold.subband_fold` returns a ratio that the
build multiplies into a sub-band tensor built elsewhere. Sub-bands are chunks of
equal *weighted* filter mass, and a model with a live nebular Lyman-continuum
mask adds a forced chunk edge at 912 Å (``lyc_gate``, K -> K + 1 chunks). A
ratio built over another partition has the same shape, so nothing raises: every
chunk is simply scaled by the transmission of a different wavelength range.

Measured on the bare FSPS grid, GALEX NUV (the Lyman limit at 912(1+z) Å lies
inside it for 1 < z < 2.3), K = 3, inoue: with the partition passed through,
``table * ratio`` reproduces the with-IGM quadrature to <= 2.2e-16. With the
partition the fold used to assume (K + 1 equal-mass chunks, no edge), the
band-integrated flux was off by 12.7 % at z = 2.0 and 44 % at z = 2.5, and u by
0.9 %. Every Cue model carries the edge: ``neb_fesc`` is fixed below one by
default.

The helpers are exercised directly: no model build, no JAX compile.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import Photometry
from tengri.components.igm import _subband_cache, exact_fold
from tengri.utils.filter_convention import FilterConvention
from tengri.utils.grid_interp import preintegrate_grid

pytestmark = pytest.mark.contract

#: GALEX NUV contains 912(1+z) at the probe redshift; the other three are
#: redward of Ly-alpha there, so the IGM cannot reach them.
BANDS = ["galex_nuv", "sdss_u", "sdss_r", "sdss_z"]
NUV, U, R, Z = range(4)
PROBE_Z = 2.0
K = 3
IGM_MODEL = "inoue"

#: Measured worst 2.2e-16 across z = 1.5, 2.0, 2.5 and both partitions.
_FOLD_TOL = 1e-12
#: Measured 12.7 % band-integrated error at PROBE_Z under the old partition.
_WRONG_PARTITION_FLOOR = 0.01


@pytest.fixture(scope="module")
def ssp(ssp_data_fsps):
    return ssp_data_fsps


@pytest.fixture(scope="module")
def filters():
    phot = Photometry.from_names(BANDS)
    return tuple(zip(phot.filter_waves, phot.filter_trans, strict=False))


def _quadrature(ssp, filters, z, templates, *, lyc_gate, n_subbands=K):
    return np.asarray(
        preintegrate_grid(
            templates=templates,
            wave_rest=np.asarray(ssp.ssp_wave, dtype=np.float64),
            filter_waves=[np.asarray(fw, dtype=np.float64) for fw, _ in filters],
            filter_trans=[np.asarray(ft, dtype=np.float64) for _, ft in filters],
            redshift=z,
            dl_cm=1.0,
            axes=(np.asarray(ssp.ssp_lgmet), np.asarray(ssp.ssp_lg_age_gyr)),
            taylor=False,
            n_subbands=n_subbands,
            lyc_gate=lyc_gate,
        ).subband_phot,
        dtype=np.float64,
    )


def _ratio(ssp, filters, z, *, lyc_gate, n_subbands=K):
    return exact_fold.subband_fold(
        ssp,
        filters,
        z,
        igm_model=IGM_MODEL,
        n_subbands=n_subbands,
        lyc_gate=lyc_gate,
        convention=FilterConvention.BESSELL,
    ).ratio


def _table_and_target(ssp, filters, z, *, lyc_gate):
    """The bare sub-band tensor and the same quadrature with the IGM inside."""
    templates = np.asarray(ssp.ssp_flux, dtype=np.float64)
    transmission = exact_fold._transmission(IGM_MODEL, np.asarray(ssp.ssp_wave, np.float64), z)
    return (
        _quadrature(ssp, filters, z, templates, lyc_gate=lyc_gate),
        _quadrature(ssp, filters, z, templates * transmission, lyc_gate=lyc_gate),
    )


@pytest.mark.parametrize("lyc_gate", [False, True])
def test_the_folded_table_is_the_quadrature_with_the_igm_inside(ssp, filters, lyc_gate):
    table, target = _table_and_target(ssp, filters, PROBE_Z, lyc_gate=lyc_gate)
    folded = table * _ratio(ssp, filters, PROBE_Z, lyc_gate=lyc_gate)

    assert folded.shape == target.shape
    live = target != 0.0
    worst = np.max(np.abs(folded - target)[live] / np.abs(target[live]))
    assert worst < _FOLD_TOL, f"folded table off the with-IGM quadrature by {worst:.2e}"


def test_the_old_partition_is_wrong_on_a_lyc_live_table(ssp, filters):
    """Anti-vacuity: the partition argument is what the test above depends on.

    Without it, K + 1 equal-mass chunks stand in for the K chunks plus a 912 Å
    edge. The shapes match, so this is the version of the defect that nothing
    else would catch.
    """
    table, target = _table_and_target(ssp, filters, PROBE_Z, lyc_gate=True)
    wrong = table * _ratio(ssp, filters, PROBE_Z, lyc_gate=False, n_subbands=K + 1)

    assert wrong.shape == target.shape
    band, band_target = wrong.sum(-1)[..., NUV], target.sum(-1)[..., NUV]
    live = band_target != 0.0
    worst = np.max(np.abs(band - band_target)[live] / np.abs(band_target[live]))
    assert worst > _WRONG_PARTITION_FLOOR, (
        f"the old partition errs by only {worst:.2e} in NUV at z={PROBE_Z}; this "
        "probe no longer exercises the Lyman-limit edge"
    )


def test_bands_the_igm_cannot_reach_skip_the_quadrature_exactly(ssp, filters):
    """The skip is exact, not an approximation of the brute-force ratio.

    ``T_IGM`` is exactly one redward of Ly-alpha, so a band there has identical
    integrands with and without the IGM. The skip returns 1.0 for it; the
    brute-force ratio, computed over every band, must agree bit for bit.
    """
    templates = np.asarray(ssp.ssp_flux, dtype=np.float64)
    transmission = exact_fold._transmission(
        IGM_MODEL, np.asarray(ssp.ssp_wave, np.float64), PROBE_Z
    )
    bare = _quadrature(ssp, filters, PROBE_Z, templates, lyc_gate=False)
    with_igm = _quadrature(ssp, filters, PROBE_Z, templates * transmission, lyc_gate=False)
    brute = np.where(bare != 0.0, with_igm / np.where(bare != 0.0, bare, 1.0), 0.0)

    skipped = _ratio(ssp, filters, PROBE_Z, lyc_gate=False)

    for band in (R, Z):
        np.testing.assert_array_equal(brute[..., band, :][bare[..., band, :] != 0.0], 1.0)
        np.testing.assert_array_equal(skipped[..., band, :], 1.0)
    np.testing.assert_allclose(skipped[..., NUV, :], brute[..., NUV, :], rtol=_FOLD_TOL, atol=0)
    assert np.min(brute[..., NUV, :][bare[..., NUV, :] != 0.0]) < 0.9, "NUV must be absorbed"


@pytest.fixture
def cache_on(tmp_path, monkeypatch):
    """Turn the precomp cache on (the suite disables it) at a private path."""
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    _subband_cache.clear_memo()
    yield tmp_path
    _subband_cache.clear_memo()


def _table(ssp, filters, z_grid, *, lyc_gate):
    return exact_fold.subband_fold_table(
        ssp,
        filters,
        z_grid,
        igm_model=IGM_MODEL,
        n_subbands=K,
        lyc_gate=lyc_gate,
        convention=FilterConvention.BESSELL,
    )


def test_a_second_build_reads_the_ratio_table_from_disk(ssp, filters, cache_on, monkeypatch):
    z_grid = np.array([1.5, PROBE_Z])
    first = _table(ssp, filters, z_grid, lyc_gate=True)
    assert len(list(cache_on.glob(f"{_subband_cache.EXACT_FOLD_PREFIX}_*.npz"))) == 1

    def _must_not_run(*args, **kwargs):
        raise AssertionError("recomputed a table that was on disk")

    _subband_cache.clear_memo()
    monkeypatch.setattr(exact_fold, "subband_fold", _must_not_run)
    again = _table(ssp, filters, z_grid, lyc_gate=True)
    np.testing.assert_array_equal(again.ratio, first.ratio)
    np.testing.assert_array_equal(again.nodes_rest, first.nodes_rest)

    # A different partition is a different table, even with the same inputs otherwise.
    with pytest.raises(AssertionError, match="recomputed"):
        _table(ssp, filters, z_grid, lyc_gate=False)


def test_the_build_folds_a_cue_model_over_its_lyman_limit_partition(ssp, filters):
    """The build passes the model's own partition: Cue carries the 912 Å edge.

    ``neb_fesc`` is fixed below one by default, so every Cue model has a live
    Lyman-continuum mask and a K + 1 chunk tensor. The helper tests above take
    the partition as an argument; this one pins that ``SEDModel.build`` supplies
    the right one.
    """
    import tengri
    from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp

    model = SEDModel.build(
        ssp_data=ssp,
        observation=tengri.Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(PROBE_Z),
        igm={"type": IGM_MODEL, "all_params": Fixed(DEFAULT)},
        approx=WavePrecomp(igm_fold="exact", n_subbands=K),
    )
    lut = model._build_component_chain()[0]._state.ssp_phot_lut
    bare, folded = np.asarray(lut.ssp_subband_phot), np.asarray(lut.ssp_subband_phot_igm)
    assert bare.shape[-1] == K + 1, "a Cue model should carry the forced Lyman-limit chunk"

    live = bare != 0.0
    np.testing.assert_allclose(
        folded[live] / bare[live],
        _ratio(ssp, filters, PROBE_Z, lyc_gate=True)[live],
        rtol=_FOLD_TOL,
        atol=0,
    )

# SPDX-License-Identifier: BSD-3-Clause
"""On-disk cache contract for the WavePrecomp photometry z-table.

The z-table depends only on (SSP grid, filter set, z grid, quadrature
flags), so it is content-hashed and persisted: the first build pays the
quadrature, later builds load the npz. These tests pin the cache's three
behaviors — hit on identical inputs, miss on changed inputs, and full
bypass via TENGRI_DISABLE_PRECOMP_CACHE.
"""

import numpy as np
import pytest

import tengri.components.stellar.sps.precompute as pc

pytestmark = pytest.mark.contract


@pytest.fixture()
def tophat_filters():
    fw1 = np.linspace(4000.0, 5000.0, 51)
    fw2 = np.linspace(6000.0, 7500.0, 61)
    ft1 = np.where((fw1 > 4100) & (fw1 < 4900), 0.8, 0.0)
    ft2 = np.where((fw2 > 6100) & (fw2 < 7400), 0.6, 0.0)
    return [fw1, fw2], [ft1, ft2]


@pytest.fixture()
def counting_compute(monkeypatch):
    calls = []
    orig = pc._compute_photometry_ztable

    def counting(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(pc, "_compute_photometry_ztable", counting)
    return calls


def _tables_equal(a, b):
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is None and y is None
        elif isinstance(x, int):
            assert x == y
        else:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_second_call_loads_from_disk(
    synthetic_ssp_wide, tophat_filters, tmp_path, monkeypatch, counting_compute
):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    fw, ft = tophat_filters

    t1 = pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=7, z_max=2.0)
    t2 = pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=7, z_max=2.0)

    assert len(counting_compute) == 1, "second identical build must be a cache hit"
    assert len(list(tmp_path.glob("ztable_*.npz"))) == 1
    _tables_equal(t1, t2)


def test_changed_inputs_miss(
    synthetic_ssp_wide, tophat_filters, tmp_path, monkeypatch, counting_compute
):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    fw, ft = tophat_filters

    pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=7, z_max=2.0)
    pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=9, z_max=2.0)
    # `apply_igm` is this function's OWN parameter, not the retired grammar key
    # -- precompute_photometry_ztable has no `igm` argument. The retirement is
    # about parse_groups; internal flags that happen to share the name stay.
    pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=7, z_max=2.0, apply_igm=True)
    pc.precompute_photometry_ztable(synthetic_ssp_wide, fw[:1], ft[:1], n_z=7, z_max=2.0)

    assert len(counting_compute) == 4, "different z grid / flags / filters must recompute"
    assert len(list(tmp_path.glob("ztable_*.npz"))) == 4


def test_disable_env_bypasses_cache(
    synthetic_ssp_wide, tophat_filters, tmp_path, monkeypatch, counting_compute
):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TENGRI_DISABLE_PRECOMP_CACHE", "1")
    fw, ft = tophat_filters

    pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=7, z_max=2.0)
    pc.precompute_photometry_ztable(synthetic_ssp_wide, fw, ft, n_z=7, z_max=2.0)

    assert len(counting_compute) == 2, "disabled cache must recompute every call"
    assert list(tmp_path.glob("ztable_*.npz")) == []

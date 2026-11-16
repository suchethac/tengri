# SPDX-License-Identifier: BSD-3-Clause
"""Every band-table cache key carries ``BandSpec.key()`` (#3505).

A cache key that lists the band-integral choices field by field forgets the next one added. The
keys of the photometry z-table and of the IGM band-factor and exact-fold tables take the spec
and hash its ``key()``; two specs that differ in any field give different keys and different
cached tables, and an identical spec is a hit.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri.components.stellar.sps.precompute as pc
from tengri.components.igm import _subband_cache
from tengri.utils.band_projection import BandSpec
from tengri.utils.filter_convention import FilterConvention

pytestmark = pytest.mark.contract


def _curves(scale=1.0):
    fw1 = np.linspace(4000.0, 5000.0, 51)
    fw2 = np.linspace(6000.0, 7500.0, 61)
    ft1 = np.where((fw1 > 4100) & (fw1 < 4900), 0.8 * scale, 0.0)
    ft2 = np.where((fw2 > 6100) & (fw2 < 7400), 0.6, 0.0)
    return [fw1, fw2], [ft1, ft2]


def _spec(*, scale=1.0, convention=FilterConvention.BESSELL, n_subbands=3, lyc_gate=False):
    fw, ft = _curves(scale)
    return BandSpec(
        filters=(fw, ft), convention=convention, n_subbands=n_subbands, lyc_gate=lyc_gate
    )


#: One variant per field of the spec, each differing from ``_spec()`` in that field alone.
VARIANTS = {
    "filters": dict(scale=0.9),
    "convention": dict(convention=FilterConvention.ENERGY),
    "n_subbands": dict(n_subbands=4),
    "lyc_gate": dict(lyc_gate=True),
}


def test_every_field_of_the_spec_changes_its_key():
    base = _spec().key()
    assert _spec().key() == base
    keys = {name: _spec(**kw).key() for name, kw in VARIANTS.items()}
    assert all(k != base for k in keys.values()), keys
    assert len(set(keys.values())) == len(keys)


def test_none_and_zero_subbands_are_one_scheme():
    assert _spec(n_subbands=None).key() == _spec(n_subbands=0).key()


@pytest.mark.parametrize("field", list(VARIANTS))
def test_igm_cache_keys_follow_the_spec(field, synthetic_ssp_wide):
    wave = np.logspace(2.5, 4.5, 40)
    zs = np.linspace(0.1, 1.0, 4)
    a, b = _spec(), _spec(**VARIANTS[field])
    assert _subband_cache.band_factor_key(
        wave, a, zs, "inoue14"
    ) != _subband_cache.band_factor_key(wave, b, zs, "inoue14")
    assert _subband_cache.exact_fold_key(
        synthetic_ssp_wide, a, zs, igm_model="inoue14"
    ) != _subband_cache.exact_fold_key(synthetic_ssp_wide, b, zs, igm_model="inoue14")


@pytest.mark.parametrize("field", list(VARIANTS))
def test_ztable_cache_separates_specs_and_hits_on_an_identical_one(
    field, synthetic_ssp_wide, tmp_path, monkeypatch
):
    monkeypatch.setenv("TENGRI_PRECOMP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TENGRI_DISABLE_PRECOMP_CACHE", raising=False)
    calls = []
    orig = pc._compute_photometry_ztable

    def counting(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(pc, "_compute_photometry_ztable", counting)

    def build(spec):
        return pc.precompute_photometry_ztable(
            synthetic_ssp_wide,
            list(spec.filters[0]),
            list(spec.filters[1]),
            n_z=5,
            z_max=1.5,
            convention=spec.convention,
            n_subbands=spec.subbands,
            lyc_gate=spec.lyc_gate,
        )

    base, other = _spec(), _spec(**VARIANTS[field])
    t_base = build(base)
    t_other = build(other)
    assert len(calls) == 2, f"a spec differing in {field} must not hit the base entry"
    assert len(list(tmp_path.glob("ztable_*.npz"))) == 2
    t_again = build(_spec())
    assert len(calls) == 2, "an identical spec is a cache hit"
    np.testing.assert_array_equal(
        np.asarray(t_again.ssp_phot_table), np.asarray(t_base.ssp_phot_table)
    )
    differs = (
        not np.array_equal(np.asarray(t_other.ssp_phot_table), np.asarray(t_base.ssp_phot_table))
        or (t_other.ssp_subband_phot_table is None) != (t_base.ssp_subband_phot_table is None)
        or np.shape(t_other.ssp_subband_phot_table) != np.shape(t_base.ssp_subband_phot_table)
        or (t_other.ssp_phot_lyc_table is None) != (t_base.ssp_phot_lyc_table is None)
    )
    assert differs, f"a spec differing in {field} produced the same cached table"

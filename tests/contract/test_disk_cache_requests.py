# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for disk cache request dataclasses and digest stability.

Tests that:
1. Every field of a request dataclass changes the digest when perturbed
2. The field count is consistent across runs
3. Version constants are in place
4. Cosmology is included in ZTableRequest
5. End-to-end digest stability
"""

import dataclasses
import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_ztable_request_all_fields_change_digest():
    """ZTableRequest: every field in dataclasses.fields changes the digest."""
    from tengri._cache_keys import array_key, baked, frozen_dataclass_key
    from tengri.components.stellar.sps.precompute import ZTableRequest
    from tengri.utils.cosmology import DEFAULT_COSMO

    # Baseline instance
    baseline = ZTableRequest(
        version=3,
        ssp_wave=array_key(np.arange(4.0)),
        ssp_flux=array_key(np.arange(4.0)),
        filters=(
            (array_key(np.arange(3.0)), array_key(np.arange(3.0))),
            (array_key(np.arange(2.0)), array_key(np.arange(2.0))),
        ),
        z_grid=array_key(np.arange(5.0)),
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=0,
        cosmology=baked(DEFAULT_COSMO),
        x64=False,
        backend="cpu",
    )

    baseline_digest = frozen_dataclass_key(baseline)

    # Every field must change the digest when perturbed
    for field in dataclasses.fields(ZTableRequest):
        if field.name == "version":
            perturbed = dataclasses.replace(baseline, version=4)
        elif field.name == "ssp_wave":
            perturbed = dataclasses.replace(baseline, ssp_wave=array_key(np.arange(5.0)))
        elif field.name == "ssp_flux":
            perturbed = dataclasses.replace(baseline, ssp_flux=array_key(np.arange(5.0)))
        elif field.name == "filters":
            # Change one filter wave
            perturbed = dataclasses.replace(
                baseline,
                filters=(
                    (array_key(np.arange(4.0)), array_key(np.arange(3.0))),
                    (array_key(np.arange(2.0)), array_key(np.arange(2.0))),
                ),
            )
        elif field.name == "z_grid":
            perturbed = dataclasses.replace(baseline, z_grid=array_key(np.arange(6.0)))
        elif field.name == "apply_igm":
            perturbed = dataclasses.replace(baseline, apply_igm=True)
        elif field.name == "taylor_correction":
            perturbed = dataclasses.replace(baseline, taylor_correction=True)
        elif field.name == "convention":
            perturbed = dataclasses.replace(baseline, convention="vega")
        elif field.name == "n_subbands":
            perturbed = dataclasses.replace(baseline, n_subbands=1)
        elif field.name == "cosmology":
            perturbed = dataclasses.replace(baseline, cosmology=baked({"Om0": 0.3, "h": 0.7}))
        elif field.name == "x64":
            perturbed = dataclasses.replace(baseline, x64=True)
        elif field.name == "backend":
            perturbed = dataclasses.replace(baseline, backend="gpu")
        else:
            pytest.fail(f"Unknown field: {field.name}")

        perturbed_digest = frozen_dataclass_key(perturbed)
        assert baseline_digest != perturbed_digest, f"Field {field.name} did not change the digest"


def test_subband_request_all_fields_change_digest():
    """SubbandRequest: every field in dataclasses.fields changes the digest."""
    from tengri._cache_keys import array_key, frozen_dataclass_key
    from tengri.components.igm._subband_cache import SubbandRequest

    # Baseline instance
    baseline = SubbandRequest(
        version=3,
        waves_rest=array_key(np.arange(4.0)),
        z_grid=array_key(np.arange(5.0)),
        igm_model="inoue",
        igm_patchy=False,
        use_dla=False,
        x64=False,
        backend="cpu",
    )

    baseline_digest = frozen_dataclass_key(baseline)

    # Every field must change the digest when perturbed
    for field in dataclasses.fields(SubbandRequest):
        if field.name == "version":
            perturbed = dataclasses.replace(baseline, version=4)
        elif field.name == "waves_rest":
            perturbed = dataclasses.replace(baseline, waves_rest=array_key(np.arange(5.0)))
        elif field.name == "z_grid":
            perturbed = dataclasses.replace(baseline, z_grid=array_key(np.arange(6.0)))
        elif field.name == "igm_model":
            perturbed = dataclasses.replace(baseline, igm_model="madau")
        elif field.name == "igm_patchy":
            perturbed = dataclasses.replace(baseline, igm_patchy=True)
        elif field.name == "use_dla":
            perturbed = dataclasses.replace(baseline, use_dla=True)
        elif field.name == "x64":
            perturbed = dataclasses.replace(baseline, x64=True)
        elif field.name == "backend":
            perturbed = dataclasses.replace(baseline, backend="gpu")
        else:
            pytest.fail(f"Unknown field: {field.name}")

        perturbed_digest = frozen_dataclass_key(perturbed)
        assert baseline_digest != perturbed_digest, f"Field {field.name} did not change the digest"


def test_subband_band_request_all_fields_change_digest():
    """SubbandBandRequest: every field in dataclasses.fields changes the digest."""
    from tengri._cache_keys import array_key, frozen_dataclass_key
    from tengri.components.igm._subband_cache import SubbandBandRequest

    # Baseline instance
    baseline = SubbandBandRequest(
        version=3,
        waves_rest=array_key(np.arange(4.0)),
        z_grid=array_key(np.arange(5.0)),
        igm_model="inoue",
        igm_patchy=False,
        use_dla=False,
        filters=((array_key(np.arange(3.0)), array_key(np.arange(3.0))),),
        convention="bessell",
        x64=False,
        backend="cpu",
    )

    baseline_digest = frozen_dataclass_key(baseline)

    # Every field must change the digest when perturbed
    for field in dataclasses.fields(SubbandBandRequest):
        if field.name == "version":
            perturbed = dataclasses.replace(baseline, version=4)
        elif field.name == "waves_rest":
            perturbed = dataclasses.replace(baseline, waves_rest=array_key(np.arange(5.0)))
        elif field.name == "z_grid":
            perturbed = dataclasses.replace(baseline, z_grid=array_key(np.arange(6.0)))
        elif field.name == "igm_model":
            perturbed = dataclasses.replace(baseline, igm_model="madau")
        elif field.name == "igm_patchy":
            perturbed = dataclasses.replace(baseline, igm_patchy=True)
        elif field.name == "use_dla":
            perturbed = dataclasses.replace(baseline, use_dla=True)
        elif field.name == "filters":
            perturbed = dataclasses.replace(
                baseline, filters=((array_key(np.arange(4.0)), array_key(np.arange(3.0))),)
            )
        elif field.name == "convention":
            perturbed = dataclasses.replace(baseline, convention="vega")
        elif field.name == "x64":
            perturbed = dataclasses.replace(baseline, x64=True)
        elif field.name == "backend":
            perturbed = dataclasses.replace(baseline, backend="gpu")
        else:
            pytest.fail(f"Unknown field: {field.name}")

        perturbed_digest = frozen_dataclass_key(perturbed)
        assert baseline_digest != perturbed_digest, f"Field {field.name} did not change the digest"


def test_ztable_cache_key_no_schema_literal():
    """_ztable_cache_key source has no 'schema=' literal."""
    from tengri.components.stellar.sps.precompute import _ztable_cache_key

    source = inspect.getsource(_ztable_cache_key)
    assert "schema=" not in source, (
        "schema= literal found in _ztable_cache_key; version constant should be used instead"
    )


def test_precompute_has_one_version_constant():
    """precompute module has exactly one version constant."""
    from tengri.components.stellar.sps import precompute

    version_constants = [n for n in dir(precompute) if "VERSION" in n and n.isupper()]
    assert len(version_constants) == 1, (
        f"Expected one version constant, found {len(version_constants)}: {version_constants}"
    )
    assert version_constants[0] == "_ZTABLE_CACHE_VERSION"


def test_ztable_version_is_3():
    """_ZTABLE_CACHE_VERSION is 3."""
    from tengri.components.stellar.sps.precompute import _ZTABLE_CACHE_VERSION

    assert _ZTABLE_CACHE_VERSION == 3


def test_subband_version_is_3():
    """_CACHE_VERSION in _subband_cache is 3."""
    from tengri.components.igm._subband_cache import _CACHE_VERSION

    assert _CACHE_VERSION == 3


def test_ionspec_version_is_1():
    """_IONSPEC_CACHE_VERSION is 1."""
    from tengri.components.nebular.ionizing_spectrum import _IONSPEC_CACHE_VERSION

    assert _IONSPEC_CACHE_VERSION == 1


def test_ztable_request_has_cosmology_field():
    """ZTableRequest has a cosmology field."""
    from tengri.components.stellar.sps.precompute import ZTableRequest

    field_names = {f.name for f in dataclasses.fields(ZTableRequest)}
    assert "cosmology" in field_names, "cosmology field not found in ZTableRequest"


def test_ztable_cache_key_reads_the_cosmology_default(monkeypatch):
    """The z-table key moves when the module cosmology default moves (#2145 tripwire).

    The integrand resolves ``luminosity_distance(z)`` with no overrides, which
    falls back to ``DEFAULT_COSMO``; a table built under one default must not
    be served under another. Threading a per-model cosmology through this path
    is a feature (#2145 stays open); this test only pins that the key reads it.
    """
    from tengri.components.stellar.sps import precompute
    from tengri.utils.cosmology import CosmoParams

    class FakeSSPData:
        ssp_wave = np.arange(3.0)
        ssp_flux = np.arange(3.0)

    args = (FakeSSPData(), [np.array([0.1, 0.2])], [np.array([0.5, 0.8])], np.array([0.0, 1.0]))
    kwargs = dict(apply_igm=False, taylor_correction=False, convention="bessell", n_subbands=0)
    key_default = precompute._ztable_cache_key(*args, **kwargs)
    monkeypatch.setattr(precompute, "DEFAULT_COSMO", CosmoParams(Om0=0.25, w0=-1.0, wa=0.0, h=0.7))
    key_other = precompute._ztable_cache_key(*args, **kwargs)
    assert key_default != key_other, "a changed cosmology default did not change the z-table key"


def test_ztable_cache_key_stability():
    """_ztable_cache_key returns stable digest strings for identical inputs."""
    from tengri.components.stellar.sps.precompute import _ztable_cache_key

    # Create minimal synthetic SSP data
    class FakeSSPData:
        ssp_wave = np.arange(3.0)
        ssp_flux = np.arange(3.0)

    ssp_data = FakeSSPData()
    filter_waves = [np.array([0.1, 0.2])]
    filter_trans = [np.array([0.5, 0.8])]
    z_grid = np.array([0.0, 1.0])

    # Two identical calls should return the same hash
    key1 = _ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid,
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=0,
    )
    key2 = _ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid,
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=0,
    )

    assert key1 == key2, "Identical calls produced different hashes"
    assert isinstance(key1, str), "Cache key should be a string"


def test_ztable_cache_key_changes_with_n_subbands():
    """_ztable_cache_key changes when n_subbands changes."""
    from tengri.components.stellar.sps.precompute import _ztable_cache_key

    # Create minimal synthetic SSP data
    class FakeSSPData:
        ssp_wave = np.arange(3.0)
        ssp_flux = np.arange(3.0)

    ssp_data = FakeSSPData()
    filter_waves = [np.array([0.1, 0.2])]
    filter_trans = [np.array([0.5, 0.8])]
    z_grid = np.array([0.0, 1.0])

    key_k0 = _ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid,
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=0,
    )
    key_k5 = _ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid,
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=5,
    )

    assert key_k0 != key_k5, "n_subbands change did not change the cache key"


def test_ztable_cache_key_changes_with_filter():
    """_ztable_cache_key changes when filter transmission changes."""
    from tengri.components.stellar.sps.precompute import _ztable_cache_key

    # Create minimal synthetic SSP data
    class FakeSSPData:
        ssp_wave = np.arange(3.0)
        ssp_flux = np.arange(3.0)

    ssp_data = FakeSSPData()
    filter_waves = [np.array([0.1, 0.2])]
    z_grid = np.array([0.0, 1.0])

    filter_trans_1 = [np.array([0.5, 0.8])]
    filter_trans_2 = [np.array([0.5, 0.9])]  # Changed transmission

    key1 = _ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans_1,
        z_grid,
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=0,
    )
    key2 = _ztable_cache_key(
        ssp_data,
        filter_waves,
        filter_trans_2,
        z_grid,
        apply_igm=False,
        taylor_correction=False,
        convention="bessell",
        n_subbands=0,
    )

    assert key1 != key2, "Filter transmission change did not change the cache key"


def test_subband_cache_key_stability():
    """cache_key and band_factor_key return stable digest strings."""
    from tengri.components.igm._subband_cache import band_factor_key, cache_key

    waves_rest = np.array([100.0, 200.0])
    z_grid = np.array([0.0, 1.0])

    # Two identical calls to cache_key should match
    key1 = cache_key(waves_rest, z_grid, "inoue", igm_patchy=False, use_dla=False)
    key2 = cache_key(waves_rest, z_grid, "inoue", igm_patchy=False, use_dla=False)
    assert key1 == key2, "Identical cache_key calls produced different hashes"

    # Two identical calls to band_factor_key should match
    filter_waves = [np.array([100.0, 200.0])]
    filter_trans = [np.array([0.5, 0.8])]

    bkey1 = band_factor_key(waves_rest, filter_waves, filter_trans, z_grid, "inoue", "bessell")
    bkey2 = band_factor_key(waves_rest, filter_waves, filter_trans, z_grid, "inoue", "bessell")
    assert bkey1 == bkey2, "Identical band_factor_key calls produced different hashes"

# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for CatalogFitter and CatalogPosterior.

Uses a stub SEDModel and stub Fitter so SSP data is not required.  The tests
verify:
  - CatalogPosterior container behavior (index, iter, len, repr)
  - CatalogFitter.run() routes native vs sequential methods correctly
  - forward_chunk_size warning for non-native methods
  - _validate_uniform_data raises on mismatched galaxy sizes
  - Sequential fallback runs all galaxies and returns correct n_galaxies
"""

import pytest

pytestmark = pytest.mark.contract

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.catalog_fitter import CatalogFitter, CatalogPosterior

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _FakePosterior:
    def __init__(self, idx):
        self.idx = idx
        self.params = {"x": jnp.array(float(idx))}


# ---------------------------------------------------------------------------
# CatalogPosterior container
# ---------------------------------------------------------------------------


class TestCatalogPosterior:
    def _make(self, n=3):
        posts = [_FakePosterior(i) for i in range(n)]
        return CatalogPosterior(posteriors=posts, method="test", wall_time_s=1.0, n_galaxies=n)

    def test_len(self):
        cp = self._make(5)
        assert len(cp) == 5

    def test_getitem(self):
        cp = self._make(3)
        assert cp[1].idx == 1

    def test_iter(self):
        cp = self._make(4)
        idxs = [p.idx for p in cp]
        assert idxs == list(range(4))

    def test_repr_contains_n_galaxies(self):
        cp = self._make(7)
        r = repr(cp)
        assert "7" in r
        assert "test" in r

    def test_empty(self):
        cp = CatalogPosterior(posteriors=[], n_galaxies=0)
        assert len(cp) == 0
        assert list(cp) == []


# ---------------------------------------------------------------------------
# _validate_uniform_data
# ---------------------------------------------------------------------------


class TestValidateUniformData:
    def _make_fitter(self, sizes):
        galaxies = [{"flux_obs": jnp.ones(s), "noise": jnp.ones(s)} for s in sizes]
        # SEDModel is only needed inside _run_native; pass None for unit tests
        return CatalogFitter(None, galaxies)

    def test_uniform_ok(self):
        cf = self._make_fitter([5, 5, 5])
        n = cf._validate_uniform_data()
        assert n == 5

    def test_nonuniform_raises(self):
        cf = self._make_fitter([5, 5, 6])
        with pytest.raises(ValueError, match="same number of data points"):
            cf._validate_uniform_data()

    def test_single_galaxy_ok(self):
        cf = self._make_fitter([8])
        assert cf._validate_uniform_data() == 8


# ---------------------------------------------------------------------------
# Sequential fallback: forward_chunk_size warning
# ---------------------------------------------------------------------------


class TestChunkSizeWarning:
    def test_warns_for_non_native_with_chunk_size(self, monkeypatch):
        """forward_chunk_size > 1 with a non-native method should emit UserWarning."""
        galaxies = [{"flux_obs": jnp.ones(4), "noise": jnp.ones(4)} for _ in range(2)]
        cf = CatalogFitter(None, galaxies)

        # Patch _run_sequential so we don't need a real model
        calls = []

        def _fake_sequential(method, *, key, **kwargs):
            calls.append(method)
            return CatalogPosterior(posteriors=[], method=method, n_galaxies=0)

        monkeypatch.setattr(cf, "_run_sequential", _fake_sequential)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cf.run("map", key=jax.random.PRNGKey(0), forward_chunk_size=4)

        assert any("forward_chunk_size" in str(w.message) for w in caught)
        assert calls == ["map"]

    def test_no_warning_when_chunk_size_1(self, monkeypatch):
        galaxies = [{"flux_obs": jnp.ones(4), "noise": jnp.ones(4)}]
        cf = CatalogFitter(None, galaxies)

        def _fake_sequential(method, *, key, **kwargs):
            return CatalogPosterior(posteriors=[], method=method, n_galaxies=0)

        monkeypatch.setattr(cf, "_run_sequential", _fake_sequential)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cf.run("map", key=jax.random.PRNGKey(0), forward_chunk_size=1)

        chunk_warns = [w for w in caught if "forward_chunk_size" in str(w.message)]
        assert chunk_warns == []


# ---------------------------------------------------------------------------
# Sequential fallback: runs all galaxies
# ---------------------------------------------------------------------------


class TestSequentialFallback:
    def test_run_sequential_returns_correct_n_galaxies(self, monkeypatch):
        n = 4
        galaxies = [{"flux_obs": jnp.ones(3), "noise": jnp.ones(3)} for _ in range(n)]
        cf = CatalogFitter(None, galaxies)

        fit_calls = []

        class _FakeFitterInstance:
            def run(self, method, *, key, verbose=False, **kwargs):
                fit_calls.append(method)
                return _FakePosterior(len(fit_calls) - 1)

        import tengri.inference.fitter as _fitter_mod

        monkeypatch.setattr(_fitter_mod, "Fitter", lambda *a, **kw: _FakeFitterInstance())

        result = cf._run_sequential("map", key=jax.random.PRNGKey(0))
        assert isinstance(result, CatalogPosterior)
        assert result.n_galaxies == n
        assert len(result) == n
        assert len(fit_calls) == n

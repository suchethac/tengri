# SPDX-License-Identifier: BSD-3-Clause
"""The batched catalog sampler compiles once per bucket and reuses the program (#3252).

``fit_spectra_catalog_mcmc`` used to build a fresh per-bucket sampler closure and wrap
it in a new ``jax.jit`` inside ``map_with_shared`` on every call, so a repeated call on
a batch of the same spec recompiled the sampling program. The driver now caches the
per-bucket sampler per (model, bucket spec, sampler options), and ``map_with_shared``
reuses the jitted map of a cached sampler. The batch, the start points, the keys and
the template tables stay arguments, so a second call with new data of the same spec
compiles nothing and draws the same samples as a freshly built program.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri.inference import batched_spectro
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_map_compiles_once import _dpl_template, _shifted
from tests.contract.test_batched_spectro_loglik import _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded

pytestmark = pytest.mark.contract

_SHIFT = 0.3
_KW = dict(
    sampler="nuts",
    n_warmup=5,
    n_burnin=0,
    n_samples=5,
    init="map",
    map_steps=5,
    batch_size=None,
)


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _dpl_template(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


@pytest.fixture(scope="module")
def batch_a(mock, template):
    return build_spectro_batches([_record(g) for g in mock], quantum=420, model=template)


@pytest.fixture(scope="module")
def batch_b(mock, template):
    records = [_shifted(_record(g), _SHIFT, g["sigma"]) for g in mock]
    return build_spectro_batches(records, quantum=420, model=template)


def test_second_call_with_same_spec_compiles_nothing(template, batch_a, batch_b):
    """A repeated catalog call on a new batch of the same spec adds no compilation-cache miss."""
    [(spec_a, _, _)] = batch_a
    [(spec_b, _, _)] = batch_b
    assert spec_a == spec_b, "the two batches must share one bucket spec"

    batched_spectro.fit_spectra_catalog_mcmc(template, batch_a, seed=0, **_KW)
    with jtu.count_jit_compilation_cache_miss() as misses:
        batched_spectro.fit_spectra_catalog_mcmc(template, batch_b, seed=0, **_KW)
    assert misses() == 0, f"second call recompiled {misses()} programs"


def test_cached_draws_match_a_freshly_built_program(template, batch_a):
    """Draws from the cached program equal those of a program built from scratch."""
    cached = batched_spectro.fit_spectra_catalog_mcmc(template, batch_a, seed=3, **_KW)
    again = batched_spectro.fit_spectra_catalog_mcmc(template, batch_a, seed=3, **_KW)
    batched_spectro._CATALOG_PROGRAMS.clear()
    fresh = batched_spectro.fit_spectra_catalog_mcmc(template, batch_a, seed=3, **_KW)
    for result in (again, fresh):
        assert set(result["draws"]) == set(cached["draws"])
        for name, values in cached["draws"].items():
            np.testing.assert_allclose(
                np.asarray(result["draws"][name]), np.asarray(values), rtol=1e-10, atol=0.0
            )

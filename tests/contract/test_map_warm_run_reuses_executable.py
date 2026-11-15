# SPDX-License-Identifier: BSD-3-Clause
"""A warm MAP run reuses the in-memory executables of the run before it (#3530).

The certificate's compiled value-and-gradient used to be keyed on the identity of a
flat objective that ``_run_map_scipy`` rebuilt on every call. A warm ``Fitter.run("map")``
therefore re-traced and re-lowered the program, and reloaded its executable from the
persistent compile cache (about 0.6 s on the quickstart). The contract pinned here:

* a second ``run("map")`` on the same Fitter compiles nothing;
* a fresh Fitter on the same model and data shape compiles nothing either.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax._src import test_util as jtu

from tengri import Fixed
from tengri.inference.fitter import Fitter
from tests.contract.test_batched_map_lbfgs import _dpl_template
from tests.contract.test_batched_spectro_loglik import _mock_catalog
from tests.contract.test_batched_spectrum_forward import _banded

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def galaxy(synthetic_ssp):
    g = _mock_catalog(synthetic_ssp)[0]
    template = _dpl_template(synthetic_ssp, g["wave"], _banded(g["wave"]), redshift=Fixed(g["z"]))
    return template, g


def _fitter(template, g):
    return Fitter(
        template,
        data=jnp.asarray(g["flux"]),
        noise=jnp.asarray(g["sigma"]),
        data_type="spectroscopy",
        approx=None,
        profile_mass=False,
    )


def test_second_map_run_on_the_same_fitter_compiles_nothing(galaxy):
    template, g = galaxy
    fitter = _fitter(template, g)
    fitter.run("map", tol=1e-3)
    with jtu.count_jit_compilation_cache_miss() as misses:
        post = fitter.run("map", tol=1e-3)
    assert post.diagnostics["map_converged"] is True
    assert misses() == 0, f"warm MAP run recompiled {misses()} programs"


def test_fresh_fitter_on_the_same_signature_compiles_nothing(galaxy):
    template, g = galaxy
    _fitter(template, g).run("map", tol=1e-3)
    with jtu.count_jit_compilation_cache_miss() as misses:
        _fitter(template, g).run("map", tol=1e-3)
    assert misses() == 0, f"fresh Fitter MAP run recompiled {misses()} programs"


def test_certificate_on_new_data_certifies_that_data(galaxy, monkeypatch):
    """The certificate takes the data as an argument: a reused program certifies new data.

    scipy's stop is forced to a failure, so the certificate runs on both fits. The second
    dataset is rescaled, so its optimum differs; a certificate that kept the first
    dataset's objective would not find a stationary point of the second.
    """
    import scipy.optimize as so

    from tengri.inference.map_stationarity import MAP_DECREMENT_BAR

    template, g = galaxy
    real = so.minimize

    def failed(*args, **kwargs):
        result = real(*args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(so, "minimize", failed)
    _fitter(template, g).run("map", tol=1e-3)
    rescaled = dict(g, flux=g["flux"] * 1.1)
    post = _fitter(template, rescaled).run("map", tol=1e-3)
    assert post.diagnostics["map_converged"] is True
    assert post.diagnostics["map_decrement"] < MAP_DECREMENT_BAR

# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: a deprecated shim warns about **itself**, and nothing else.

#1049's acceptance criterion is *"``git grep`` shows no ``src/tengri`` caller of
any deprecated name."* That check has a hole, and this test exists because the
hole was live on main.

``Posterior.derived`` (deprecated) drove its batch with::

    jax.vmap(self._model.predict_sfh_quantities)(self.samples)

``predict_sfh_quantities`` is itself deprecated — but the method is passed as a
**reference**, not called, so ``grep 'predict_sfh_quantities('`` never sees it.
The user then got two warnings: the one they earned, and one naming a method
they never touched.

A syntactic check cannot see a semantic property. So this test does not grep —
it **runs** the surfaces with ``DeprecationWarning`` recorded, and asserts:

* every **recommended** surface warns **zero** times (the library must never warn
  at its own users);
* every **deprecated** surface warns **exactly once** — about itself.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


@pytest.fixture
def posterior(model):
    """A FRESH Posterior per test — deliberately not module-scoped.

    ``Posterior.derived`` is a ``functools.cached_property``, so it warns exactly
    once per instance (good: no spam on repeated reads). A shared fixture would
    therefore let the first test to touch it consume the only warning, and every
    later assertion about warnings would silently pass against a cached value.
    """
    samples = {
        "sfh_dpl_log_total_mass": jax.random.uniform(
            jax.random.PRNGKey(0), (64,), minval=9.0, maxval=11.0
        )
    }
    return Posterior(
        samples=samples,
        params={"sfh_dpl_log_total_mass": jnp.asarray(10.0)},
        method="test",
        wall_time_s=0.0,
        diagnostics={},
        _model=model,
    )


def _deprecations(fn):
    """Run ``fn`` and return the DeprecationWarning messages it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
        return [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]


def test_the_recommended_surfaces_never_warn(model, posterior):
    """The library must not warn at its own users on the path it tells them to use."""
    params = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}

    got = _deprecations(
        lambda: (
            model.predict(params).properties["stellar_mass"],
            model.predict(params).photometry(),
            model.predict(params).rest_sed,
            posterior.properties["stellar_mass"],
            posterior.observables(n_draws=4),
        )
    )
    assert got == [], f"a recommended surface emitted DeprecationWarnings: {got}"


def test_posterior_derived_warns_about_itself_and_nothing_else(posterior):
    """The regression: ``derived`` leaked a SECOND warning naming an internal method.

    A shim must warn about the thing the user called. Naming
    ``predict_sfh_quantities`` — which they never touched — sends them chasing a
    method that is not in their code.
    """
    got = _deprecations(lambda: posterior.derived)

    assert len(got) == 1, (
        f"Posterior.derived emitted {len(got)} DeprecationWarnings, expected 1 "
        f"(its own). It is driving its batch with a deprecated method: {got}"
    )
    assert "Posterior.derived" in got[0]
    assert "predict_sfh_quantities" not in got[0], (
        "the warning names an internal method the user never called"
    )


def test_the_check_is_not_vacuous(posterior):
    """If ``derived`` stopped warning at all, the test above would pass for the wrong reason."""
    assert len(_deprecations(lambda: posterior.derived)) >= 1, (
        "Posterior.derived emits NO DeprecationWarning — it is supposed to be "
        "deprecated, so the assertions above would be vacuously satisfied"
    )

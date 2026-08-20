# SPDX-License-Identifier: BSD-3-Clause
"""BlackJAX posterior draw: one compiled program, reused — not re-accreted per call.

``Fitter._draw_blackjax_samples`` rebuilt, on every call, a ``@jax.jit logdensity_fn``
closing over ``self._data_args`` (which carries the SSP grid), a fresh
``window_adaptation``, a fresh NUTS kernel and a fresh ``@jax.jit one_step``. Fresh
function identities miss JAX's compilation cache, so each call compiled and *retained*
another set of executables — a measured **~72 MB/call** leak (#1249). A catalog VI run
with ``posterior_method="blackjax"`` draws samples per galaxy, so it accrued that per
galaxy: ~6.6 GB over 100 galaxies.

Memoizing only the log-density is not sufficient (measured 72 -> 55 MB/call): the
adaptation-dependent kernel must live inside the cached program too. It cannot be cached
on its own because it is constructed from the *runtime* warmup output — caching that
would freeze one call's step size, which is the bug #1234 fixed for the batch path.

So warmup and sampling now compile once, inside a single memoized ``jax.jit`` that takes
the data as a traced argument. Measured after: **+0.0 MB/call**, flat.

This pins:

* repeated same-config draws reuse one compiled program (memo hit — no re-accretion),
* a structurally different config keys a distinct program,
* a caller-supplied likelihood disables the memo, since an arbitrary callable cannot be
  fingerprinted and a stale kernel must never be served,
* samples stay finite and reproducible for a fixed key.

Numerics note: merging the previously separate jits into one program changes XLA's
fusion boundaries, so draws are **not bit-identical** to the pre-fix path — they agree
to ~1e-12 relative (``allclose(rtol=1e-9)``), which is float reassociation, not a
physics change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

_CACHE = "_blackjax_draw_kernel_cache"


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def fitter(synthetic_ssp_wide):
    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(8.0, 12.0)},
        dust={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    rng = np.random.default_rng(0)
    flux = np.abs(rng.normal(1.0, 0.1, size=3))
    f = Fitter(model, jnp.asarray(flux), jnp.asarray(0.1 * flux), data_type="photometry")
    return f, f._initialize_unbounded(jax.random.PRNGKey(0))


def _draw(f, init, *, n_samples=8, seed=0, likelihood=None):
    return f._draw_posterior_samples(
        likelihood,
        init,
        jax.random.PRNGKey(seed),
        n_samples,
        [],
        method="blackjax",
        verbose=False,
    )


def test_draw_kernel_compiled_once_and_reused(fitter):
    """Repeated same-config draws reuse one compiled program (no per-call accretion)."""
    f, init = fitter
    f.__dict__.pop(_CACHE, None)

    for i in range(3):
        _draw(f, init, seed=i)

    cache = getattr(f, _CACHE, None)
    assert cache is not None, "the blackjax draw kernel must be memoized on the Fitter"
    assert len(cache) == 1, (
        "three same-config draws must reuse ONE compiled program — a fresh program per "
        f"call is the leak; got {len(cache)} cached kernels"
    )


def test_distinct_config_keys_a_distinct_kernel(fitter):
    """A different n_samples must not reuse the cached program."""
    f, init = fitter
    f.__dict__.pop(_CACHE, None)

    _draw(f, init, n_samples=8)
    _draw(f, init, n_samples=12)
    assert len(getattr(f, _CACHE, {})) == 2, "distinct n_samples must key distinct kernels"


def test_custom_likelihood_disables_the_memo(fitter):
    """An arbitrary callable cannot be fingerprinted, so it must never be served cached."""
    f, init = fitter
    f.__dict__.pop(_CACHE, None)

    def _likelihood(x):
        return 0.5 * sum(jnp.sum(v**2) for v in x.values())

    out = _draw(f, init, likelihood=_likelihood)
    assert len(out) == 8
    assert not getattr(f, _CACHE, {}), (
        "a caller-supplied likelihood must disable the memo (key=None), never cache a "
        "kernel that could be served to a different likelihood"
    )


def test_samples_are_finite_and_reproducible(fitter):
    """Same key gives the same draws; the memo hit must not perturb them."""
    f, init = fitter
    f.__dict__.pop(_CACHE, None)

    a = _draw(f, init, seed=7)
    b = _draw(f, init, seed=7)
    for name in a[0]:
        va = np.asarray([s[name] for s in a], dtype=np.float64)
        vb = np.asarray([s[name] for s in b], dtype=np.float64)
        assert np.all(np.isfinite(va)), f"{name}: non-finite draws"
        np.testing.assert_array_equal(va, vb)

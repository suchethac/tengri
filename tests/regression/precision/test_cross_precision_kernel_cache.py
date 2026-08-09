# SPDX-License-Identifier: BSD-3-Clause
r"""The structural-kernel cache must not serve a float64 kernel to a float32 model (#1392).

``SEDModel._get_or_build_predict_observables_jit`` memoizes the JIT'd observables
closure in a **process-global** cache keyed on :meth:`SEDModel.compile_signature`.
That closure captured ``self``, so any model whose signature collides is served the
*other* model's kernel — including its wavelength grid.

Precision was missing from that key. ``forward_dtype`` is in the signature, but it
stays ``"float64"`` in a **pure** float32 run (entered with
``jax.enable_x64(False)``, not by setting that knob), so it could not tell a float64
model from a float32 one — and being inert it never could have (#1433: it casts
nothing, and the two settings give bit-identical results). The consequence was
order-dependent and silent:

1. compute any gradient in float64 — this populates the cache;
2. build a fresh float32 model and take its gradient — it is handed the float64
   kernel, whose ``state.wave`` is a float64 array;
3. every float32 gate downstream keys on a dtype (``wave.dtype == jnp.float32``),
   reads float64, and **switches itself off**. In the AGN block that disables the
   reference-L_bol evaluation, so the disc is evaluated at the true
   ``agn_log_lbol`` (L_bol ~ 1e44 erg/s, past the float32 max 3.4e38) and the
   reverse pass returns NaN.

The forward pass stayed finite throughout, and nothing raised — the fit simply
produced NaN gradients, and only when an f64 gradient had run earlier in the same
process. No cache-clearing helped, because this cache is neither a ``functools``
cache nor one of the ``jit_engine._SHARED_*`` family.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.context import InferenceContext

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def shared_obs():
    """One ``Observation``, reused by both precisions — the realistic usage.

    This sharing is load-bearing, not incidental. ``compile_signature`` already
    contains ``filter_trans_dtype``, so an ``Observation`` constructed *inside* a
    ``jax.enable_x64(False)`` block carries float32 filter curves and makes the two
    signatures differ for a reason that has nothing to do with #1392 — which would
    make the guard below pass whether or not the bug is present (verified: the test
    still passed with the fix reverted). Users build filters once and models later,
    so the shared object is both the honest case and the one that breaks.
    """
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w3"]))


def _build(ssp, obs):
    """Smallest model that exhibits #1392: stellar + AGN (disc whose shape needs L_bol)."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        # Pinned explicitly: a model with no ``dust`` group still declares
        # ``dust_tau_diff`` / ``dust_tau_bc`` free, which would leave the free set
        # out of step with the truth dict below.
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
        redshift=Fixed(0.1),
    )


def test_compile_signature_separates_precisions(ssp_bare, shared_obs):
    """A float64 and a float32 build must not share a structural-kernel cache key.

    This is the root-cause guard: the signature is the cache key, so if the two
    precisions agree here, one model's compiled kernel *will* be handed to the
    other and every dtype-keyed float32 gate downstream fails open.
    """
    with jax.enable_x64(True):
        sig64 = _build(ssp_bare, shared_obs).compile_signature()
    with jax.enable_x64(False):
        sig32 = _build(ssp_bare, shared_obs).compile_signature()

    assert sig64 != sig32, (
        "SEDModel.compile_signature() is identical in float64 and pure float32, so "
        "the process-global structural-kernel cache will serve one precision's "
        "compiled kernel (and its wave grid) to the other (#1392)"
    )


def test_shared_fn_caches_separate_the_two_precisions(ssp_bare, shared_obs):
    """A float32 fitter must not be served the float64 fitter's compiled loss function.

    ``jit_engine.get_or_build_cached`` keys on ``Fitter.compile_signature()``, which is
    ``(model_signature, _engine_cache_key())``. Only the model half carries precision
    (via ``build_precision``), so this cache's correctness rests entirely on that —
    ``_engine_cache_key()`` has none of its own. Before the model half was fixed, the
    loss cache measurably held **one** entry across both precisions, i.e. the float32
    fitter hit the float64 one.

    Asserted on entry *count* rather than on gradient values so the failure names the
    cache rather than a downstream symptom.
    """
    from tengri.inference import jit_engine as je

    loss_cache, loss_lock = je._SHARED_CACHES["loss"]
    with loss_lock:
        loss_cache.clear()

    _gradient(ssp_bare, shared_obs, True, jnp.float64)
    after_f64 = len(loss_cache)
    _gradient(ssp_bare, shared_obs, False, jnp.float32)
    after_f32 = len(loss_cache)

    assert after_f64 == 1, f"expected the float64 fit to populate one loss entry, got {after_f64}"
    assert after_f32 == 2, (
        f"the float32 fitter reused the float64 compiled loss function (cache went "
        f"{after_f64} -> {after_f32}, expected 2): Fitter.compile_signature() no longer "
        "distinguishes the precisions (#1392/#1412)"
    )


def _gradient(ssp, obs, x64, dtype):
    """Negative-log-posterior gradient at the origin of standardized space."""
    with jax.enable_x64(x64):
        model = _build(ssp, obs)
        truth = {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}
        # The mock must be built at the same precision as the model that fits it.
        mock = model.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
        flux, noise = mock.flux_obs, mock.noise
        ctx = InferenceContext.from_target(Fitter(model, flux, noise))
        data_args = ctx.data_args
        keys = sorted(ctx.initial_params(jax.random.PRNGKey(1)))
        point = {k: jnp.asarray(0.0, dtype=dtype) for k in keys}
        grad = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, data_args))(point)
        return np.array([float(np.asarray(grad[k])) for k in keys])


def test_float64_gradient_does_not_poison_a_later_float32_gradient(ssp_bare, shared_obs):
    """An f64 gradient must leave a later f32 gradient in the same process untouched.

    **The order is load-bearing.** float64 has to go first, so that it is the float64
    kernel sitting in the structural-kernel cache when the float32 model asks for one.
    Measuring a "clean" float32 gradient first caches the *float32* kernel, the later
    float64 call hits that, and the bug cannot appear — verified: written that way the
    test passed with the fix reverted.

    Costs ~9 s (three gradients across two precisions), which buys the end-to-end
    signature of the bug rather than just its cause, so it runs on every PR.

    The cache is cleared *first* for the same reason it is cleared in the middle:
    "float64 has to go first" is a statement about the cache, and this test cannot
    assert it while inheriting whatever its file-mates left there. Measured — run
    alone it passed, run after the two tests above it failed, on this commit and
    on the pre-merge one; clearing at entry makes the whole file pass.
    """
    from tengri.inference._model_cache import clear_structural_kernel_cache

    clear_structural_kernel_cache()
    _gradient(ssp_bare, shared_obs, True, jnp.float64)
    served = _gradient(ssp_bare, shared_obs, False, jnp.float32)

    # The reference: the same float32 gradient, but with the cache emptied first, so
    # this model is guaranteed its OWN kernel. Both sides are float32, so they must
    # agree bit-for-bit — no tolerance to choose, and the float32-vs-float64 gradient
    # accuracy question (a separate matter) never enters.
    clear_structural_kernel_cache()
    own = _gradient(ssp_bare, shared_obs, False, jnp.float32)

    assert np.all(np.isfinite(served)), (
        "float32 gradient is non-finite after an unrelated float64 gradient ran "
        "earlier in the same process — the structural-kernel cache served the "
        "float64 kernel, whose wave grid is float64, which switches off every "
        "dtype-keyed float32 gate downstream (#1392)"
    )
    np.testing.assert_array_equal(
        served,
        own,
        err_msg=(
            "the float32 gradient depends on whether a float64 gradient ran earlier, "
            "so the structural-kernel cache is serving a kernel built at the wrong "
            "precision (#1392)"
        ),
    )

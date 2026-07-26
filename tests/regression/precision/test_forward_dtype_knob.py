# SPDX-License-Identifier: BSD-3-Clause
r"""``forward_dtype="float32"`` must actually compute in float32 (#1433).

The kwarg is documented as a performance knob — it used to cast the SSP grid, the
dust weights and the effective wavelengths, and the three largest exact-path
intermediates. Those casts lived in ``forward/_kernels/`` and went out with
``1e57d973d`` (2026-05-20, "Phase 6 — delete forward/_kernels/"). The kwarg, its
docstring, its :class:`SEDModelState` field, its ``compile_signature`` entry and its
``config/display.py`` line all survived the refactor. The casts did not, and
``state.forward_dtype`` has had **zero** readers since.

So the knob is inert: identical results, plus a second compile (it is still part of
the cache key). Worse, it is *silently* inert — a user who asks for float32 gets
float64 arithmetic and no error.

The two tests here are a matched pair, and the pairing is the point. A lone xfail
saying "float32 does not differ from float64" is not evidence of a bug: it is
equally consistent with "this model is insensitive to precision" or "this
comparison cannot resolve a precision change". The control rules that out by
showing the *same* comparison, on the *same* model, does detect the precision change
that pure float32 makes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

#: Parameters held identical across every build below; only precision varies.
_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1"]))


def _build(ssp, obs, forward_dtype):
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
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        },
        redshift=Fixed(0.1),
        forward_dtype=forward_dtype,
    )


def _photometry(ssp, obs, *, x64, forward_dtype, dtype):
    """Photometry [erg/s/cm^2/Hz] as float64, for comparison across precisions."""
    with jax.enable_x64(x64):
        model = _build(ssp, obs, forward_dtype)
        params = {k: jnp.asarray(v, dtype=dtype) for k, v in _TRUTH.items()}
        return np.asarray(model.predict_photometry(params), dtype=np.float64)


def test_pure_float32_changes_the_computation(ssp_bare, obs):
    """Control: the comparison below CAN see a precision change.

    Pure float32 — a ``jax.enable_x64(False)`` context, the mechanism that actually
    works — must move the photometry off the float64 answer, and by an amount
    consistent with float32 (order 1e-7..1e-5 relative, not 1e-16 and not 1e-1).

    Without this, the strict xfail that follows would be uninterpretable.
    """
    ref = _photometry(ssp_bare, obs, x64=True, forward_dtype="float64", dtype=jnp.float64)
    pure = _photometry(ssp_bare, obs, x64=False, forward_dtype="float64", dtype=jnp.float32)

    rel = np.max(np.abs(pure - ref) / np.abs(ref))
    assert not np.array_equal(pure, ref), (
        "pure float32 produced bit-identical photometry to float64, so this "
        "comparison cannot detect a precision change and proves nothing about "
        "forward_dtype below"
    )
    assert 1e-9 < rel < 1e-3, (
        f"pure float32 differs from float64 by {rel:.3e}, which is not the size of a "
        "float32 rounding difference — the control model is not exercising float32 "
        "arithmetic the way this file assumes"
    )


@pytest.mark.xfail(
    reason="#1433: forward_dtype casts nothing. The casts were deleted with "
    "forward/_kernels/ in 1e57d973d (2026-05-20) and state.forward_dtype has had no "
    "readers since, so 'float32' returns bit-identical float64 results — while still "
    "entering compile_signature, so it costs an extra compile for nothing. When the "
    "knob is wired (or retired) this XPASSes: update the forward_dtype docstring in "
    "forward/sed_model.py at the same time, since it is what tells users the truth.",
    strict=True,
)
def test_forward_dtype_float32_actually_computes_in_float32(ssp_bare, obs):
    """``forward_dtype="float32"`` must change the numbers, on the path it names.

    Compared bit-for-bit rather than with a tolerance: any genuine float32
    arithmetic anywhere in the forward model perturbs the last bits of a float64
    result. Bit-identity is therefore proof that no float32 arithmetic happened at
    all — a much sharper statement than "agrees within 0.1%", which is what the
    knob's own test file checks (of a private replica kernel, not of this kwarg).
    """
    ref = _photometry(ssp_bare, obs, x64=True, forward_dtype="float64", dtype=jnp.float64)
    knob = _photometry(ssp_bare, obs, x64=True, forward_dtype="float32", dtype=jnp.float64)

    assert not np.array_equal(knob, ref), (
        f"forward_dtype='float32' returned photometry bit-identical to "
        f"forward_dtype='float64' ({knob[0]:.17e}), so nothing was computed in "
        "float32 (#1433)"
    )

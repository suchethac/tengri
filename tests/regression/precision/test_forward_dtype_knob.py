# SPDX-License-Identifier: BSD-3-Clause
r"""``forward_dtype`` is retired, and must say so rather than pretend (#1433).

The kwarg was documented as a performance knob — it used to cast the SSP grid, the
dust weights and the effective wavelengths, and the three largest exact-path
intermediates. Those casts lived in ``forward/_kernels/`` and went out with
``1e57d973d`` (2026-05-20, "Phase 6 — delete forward/_kernels/"). The kwarg, its
docstring, its :class:`SEDModelState` field, its ``compile_signature`` entry and its
``config/display.py`` line all survived the refactor. The casts did not, and
``state.forward_dtype`` had **zero** readers for the two months that followed.

The damage was never the missing speedup. It was that the knob was *silently*
inert: a caller who asked for float32 got float64 arithmetic and no signal, and the
tests that were supposed to catch that exercised a private replica kernel and a
``.astype`` tautology instead of the kwarg.

**Retired rather than wired.** Pure float32 — ``jax.enable_x64(False)`` — is the
mode the range protections in ``components/`` gate on and the one #1206 delivers;
reviving a second float32 path would mean maintaining distinct gate semantics and
re-earning a speed claim nobody had re-measured. Passing anything but the default
now warns, and the knob no longer enters the compile key, so it no longer costs the
redundant compile it used to.

The control test below is retained deliberately. It shows that this comparison — the
same models, the same photometry — *does* detect the change pure float32 makes. Without
it, "float32 equals float64" would be equally consistent with "this model cannot
resolve a precision difference", and the inertness assertion would prove nothing.
"""

import warnings

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
            "law": "calzetti",
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


def test_forward_dtype_is_retired_and_says_so(ssp_bare, obs):
    """Asking for ``float32`` must warn, and must not silently pretend to comply.

    The knob was retired rather than wired (#1433): it had cast nothing since
    ``1e57d973d`` (2026-05-20), and pure float32 — ``jax.enable_x64(False)`` — is
    both the mode the range protections in ``components/`` gate on and the one
    #1206 delivers. What made it dangerous was never the missing speedup; it was
    that a caller asking for float32 got float64 arithmetic and no signal.

    So the contract this file pins has changed. It used to be "float32 must change
    the numbers" (a strict xfail). It is now "float32 must *say* it does nothing".
    """
    with jax.enable_x64(True):
        with pytest.warns(DeprecationWarning, match="forward_dtype"):
            _build(ssp_bare, obs, "float32")

        # The default must stay silent — a warning on every model build would be
        # noise, and would train users to filter the one that matters.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _build(ssp_bare, obs, "float64")


def test_forward_dtype_no_longer_forces_a_second_compile(ssp_bare, obs):
    """Two builds differing only in ``forward_dtype`` must share a cache key.

    While the knob was live in :meth:`compile_signature` it bought a second
    compile of a kernel that computes bit-identical results — pure cost. Now that
    it is retired the signatures must agree, and the results must still be
    identical (which is what makes sharing the kernel correct rather than a bug).

    If someone wires the knob later, this test fails — correctly, and in the same
    change that would need to restore the signature entry.
    """
    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sig64 = _build(ssp_bare, obs, "float64").compile_signature()
        sig32 = _build(ssp_bare, obs, "float32").compile_signature()

    assert sig64 == sig32, (
        "forward_dtype still separates two compile signatures, so it still costs a "
        "second compile of an identical kernel (#1433)"
    )

    ref = _photometry(ssp_bare, obs, x64=True, forward_dtype="float64", dtype=jnp.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        knob = _photometry(ssp_bare, obs, x64=True, forward_dtype="float32", dtype=jnp.float64)
    np.testing.assert_array_equal(
        knob,
        ref,
        err_msg=(
            "forward_dtype='float32' now changes the result, so it is no longer inert "
            "and must go back into compile_signature (#1433)"
        ),
    )

# SPDX-License-Identifier: BSD-3-Clause
r"""One contract for every log-domain magnitude producer (#1527).

Tier A/B replaced multiplicative scale seams with log10 offsets, and each new
``log_*`` producer re-derived the same guard by hand::

    positive = value > 0
    safe = jnp.where(positive, value, 1.0)
    return jnp.where(positive, jnp.log10(safe), -jnp.inf)

``value > 0`` is False for ``NaN``. So every one of them mapped a corrupt input
onto the *zero* sentinel, and ``pow10(-inf)`` is exactly ``0.0`` — a silent,
plausible-looking answer. Five sites had it independently:

=========================================  ==========================================
site                                       what silently became zero
=========================================  ==========================================
``forward.energy_balance``                 the whole dust IR budget
``dust.energy_balance_precompute``         the stellar half, under ``WavePrecomp``
``dust.emission...energy_balance_split``   the split IR normalization
``stellar._integrate_nion_log10``          **all nebular emission** (Q_H -> 0)
``utils.scale.log10_add``                  either term of a summed pair
=========================================  ==========================================

``log10_add`` is the sharpest one: its own comment argues that folding a
non-finite term into the zero sentinel "would report an overflowed term as
exactly zero — a fail-open on precisely the axis this module exists to close".
It then tested ``isposinf(larger)``, which catches ``+inf`` and misses ``NaN``
entirely.

**The contract.** ``-inf`` means the quantity is exactly zero and is a value.
``+inf`` means no answer exists and is not a value. They are not
interchangeable, and no input may turn the second into the first.

This file tests the *rule* against every producer rather than re-testing each
fix, so a sixth site cannot be added with the old hand-rolled guard and stay
green. :func:`tengri.utils.scale.log10_magnitude` is the one spelling; new
producers should call it rather than re-deriving.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import log10_add, log10_magnitude, pow10

pytestmark = pytest.mark.regression_bug

#: Corrupt inputs for a producer whose argument is a **linear** quantity. All
#: three non-finites are failures there: a luminosity is never legitimately
#: ``-inf``.
CORRUPT_LINEAR = [np.nan, np.inf, -np.inf]

#: Corrupt inputs for a producer whose argument is already **log**-domain.
#: ``-inf`` is excluded deliberately — in log space it is the legitimate "this
#: term is exactly zero" sentinel, so feeding it is a valid call, not a fault.
#: The two alphabets are genuinely different and conflating them makes the rule
#: untestable: the same value cannot be both the zero case and a corrupt case.
CORRUPT_LOG = [np.nan, np.inf]


def _energy_balance_log(value):
    """``forward.energy_balance.bolometric_absorbed_log10``, poisoned at one pixel."""
    from tengri.forward.energy_balance import bolometric_absorbed_log10

    wave = np.logspace(3.0, 6.0, 256)
    intrinsic = np.full_like(wave, 1e28)
    intrinsic[100] = value
    return bolometric_absorbed_log10(
        jnp.asarray(intrinsic),
        jnp.asarray(np.full_like(wave, 0.4e28)),
        jnp.asarray(C_AA / wave),
        wave=jnp.asarray(wave),
    )[0]


def _nion_log(value):
    """``stellar._integrate_nion_log10``, poisoned inside the ionizing range."""
    from tengri.components.stellar.component import _integrate_nion_log10

    wave = jnp.asarray(np.logspace(1.0, 4.0, 400))
    sed = jnp.asarray(np.full(400, 1e28)).at[5].set(value)
    return _integrate_nion_log10(sed, wave)


def _split_log(value):
    """``dust.emission.analytic.energy_balance_split._log10_nonneg``."""
    from tengri.components.dust.emission.analytic.energy_balance_split import _log10_nonneg

    return _log10_nonneg(jnp.asarray(value))


def _lut_log(value, monkeypatch=None):
    """``dust.energy_balance_precompute.lut_l_absorbed_stellar_log10``."""
    import tengri.components.dust.energy_balance_precompute as ebp

    original = ebp._lut_contract
    try:
        ebp._lut_contract = lambda *a, **k: jnp.asarray(value)
        return ebp.lut_l_absorbed_stellar_log10(
            object(), jnp.ones((2, 3)), jnp.asarray(10.0), jnp.asarray(0.5), jnp.asarray(0.3)
        )[0]
    finally:
        ebp._lut_contract = original


def _add_log(value):
    """``utils.scale.log10_add`` with one corrupt term beside a healthy one."""
    return log10_add(jnp.asarray(43.0), jnp.asarray(value))


#: (name, callable, clean input, "nothing here" input, corrupt alphabet). The
#: clean and zero columns are what stop this from passing on a producer that
#: returns +inf unconditionally.
PRODUCERS = [
    ("energy_balance.bolometric_absorbed_log10", _energy_balance_log, 1e28, None, CORRUPT_LINEAR),
    ("stellar._integrate_nion_log10", _nion_log, 1e28, None, CORRUPT_LINEAR),
    ("energy_balance_split._log10_nonneg", _split_log, 1e43, 0.0, CORRUPT_LINEAR),
    ("energy_balance_precompute.lut_...log10", _lut_log, 1.5, 0.0, CORRUPT_LINEAR),
    # zero=None: ``log10_add``'s zero case is *both* terms being -inf, which this
    # one-argument shape cannot express — ``_add_log(-inf)`` is 43.0 + 0, i.e.
    # 43.0, and correctly so. ``TestLog10AddPreservesBothSentinels`` covers it.
    ("scale.log10_add", _add_log, 42.0, None, CORRUPT_LOG),
]
_IDS = [p[0] for p in PRODUCERS]


@pytest.mark.parametrize("name,fn,clean,zero,corrupt", PRODUCERS, ids=_IDS)
def test_corrupt_input_is_never_the_zero_sentinel(name, fn, clean, zero, corrupt):
    """The rule. A non-finite input must not come back as ``-inf``.

    ``-inf`` powers back through ``pow10`` to exactly ``0.0``, so returning it
    for a corrupt input is a silent wrong answer — and a plausible one, which is
    what makes it dangerous.
    """
    for bad in corrupt:
        _assert_not_the_zero_sentinel(name, fn, bad)


def _assert_not_the_zero_sentinel(name, fn, bad):
    out = float(fn(bad))
    assert not np.isneginf(out), (
        f"{name} mapped a {bad} input onto -inf, the 'exactly zero' sentinel. "
        f"pow10(-inf) is {float(pow10(jnp.asarray(out)))}, so this silently reports the "
        "quantity as absent. Use tengri.utils.scale.log10_magnitude rather than a "
        "hand-rolled `value > 0` guard, which is False for NaN"
    )
    assert np.isposinf(out) or np.isnan(out), (
        f"{name} answered a {bad} input with the finite value {out}, which is worse than "
        "either sentinel — a corrupt computation must not produce a usable-looking number"
    )


@pytest.mark.parametrize("name,fn,clean,zero,corrupt", PRODUCERS, ids=_IDS)
def test_clean_input_is_still_finite(name, fn, clean, zero, corrupt):
    """Negative control for the rule above.

    Without this, every producer could return ``+inf`` unconditionally and the
    corrupt test would pass while the model computed nothing at all.
    """
    out = float(fn(clean))
    assert np.isfinite(out), f"{name} gave {out} on healthy input {clean}"
    assert np.any(out != 0.0), (
        "`out` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


@pytest.mark.parametrize("name,fn,clean,zero,corrupt", PRODUCERS, ids=_IDS)
def test_a_genuine_zero_is_still_minus_infinity(name, fn, clean, zero, corrupt):
    """The other negative control: the ``-inf`` sentinel must still work.

    The fix must separate corrupt from zero, not abolish zero. Producers whose
    zero case cannot be reached through this entry point declare ``None``.
    """
    if zero is None:
        pytest.skip("no single-argument spelling of the zero case for this producer")
    out = float(fn(zero))
    assert np.isneginf(out), (
        f"{name} gave {out} for a genuinely zero input; -inf is the value that powers "
        "back to exactly 0.0 and it must survive"
    )
    assert float(pow10(jnp.asarray(out))) == 0.0


class TestThePrimitiveItself:
    """``log10_magnitude`` is the one spelling the producers delegate to."""

    def test_the_three_branches(self):
        assert float(log10_magnitude(jnp.asarray(1e43))) == pytest.approx(43.0, rel=1e-12)
        assert np.isneginf(float(log10_magnitude(jnp.asarray(0.0))))
        for bad in CORRUPT_LINEAR:
            assert np.isposinf(float(log10_magnitude(jnp.asarray(bad)))), (
                f"log10_magnitude({bad}) must be +inf — it is the definition the "
                "producers are checked against, so a hole here hides holes everywhere"
            )

    def test_it_takes_the_magnitude_of_a_signed_input(self):
        """Producers feed it signed integrals whose sign only tracks grid orientation."""
        assert float(log10_magnitude(jnp.asarray(-1e43))) == pytest.approx(43.0, rel=1e-12)

    def test_the_zero_branch_has_no_nan_gradient(self):
        """The where-dummy exists for this; losing it would NaN the backward pass."""
        grad = float(jax.grad(lambda v: log10_magnitude(v))(jnp.asarray(0.0)))
        # grad-assert: finite-only — the zero branch is the subject of this test
        assert not np.isnan(grad), "log10_magnitude has a NaN gradient at exactly zero"

    def test_it_is_jittable(self):
        assert np.isposinf(float(jax.jit(log10_magnitude)(jnp.asarray(jnp.nan))))


class TestLog10AddPreservesBothSentinels:
    """The summation seam has to carry the contract through, not flatten it."""

    def test_a_corrupt_term_is_not_absorbed_by_a_healthy_one(self):
        """The original defect: ``maximum(43.0, nan)`` is NaN, ``isposinf(nan)`` is False."""
        assert np.isposinf(float(log10_add(jnp.asarray(43.0), jnp.asarray(jnp.nan))))
        assert np.isposinf(float(log10_add(jnp.asarray(jnp.nan), jnp.asarray(43.0))))

    def test_a_corrupt_sign_is_caught_too(self):
        """A NaN sign with a finite magnitude poisons the sum the same way."""
        out = float(log10_add(jnp.asarray(43.0), jnp.asarray(42.0), sign_b=jnp.asarray(jnp.nan)))
        assert np.isposinf(out), f"a NaN sign gave {out}, not +inf"

    def test_a_zero_term_still_contributes_nothing(self):
        """``-inf`` must keep meaning "this term is zero" and not trip the guard."""
        out = float(log10_add(jnp.asarray(43.0), jnp.asarray(-jnp.inf)))
        assert out == pytest.approx(43.0, rel=1e-12)

    def test_both_zero_is_still_zero(self):
        assert np.isneginf(float(log10_add(jnp.asarray(-jnp.inf), jnp.asarray(-jnp.inf))))

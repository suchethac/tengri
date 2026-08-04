# SPDX-License-Identifier: BSD-3-Clause
r"""A corrupt energy-balance integrand must not be reported as zero absorption (#1527).

Before this, ``_peak_factored_trapezoid`` collapsed two unrelated situations into
one flag: an all-zero integrand (nothing absorbed — a true zero) and a non-finite
one (something upstream produced Inf or NaN). Both became ``-inf``, and
``pow10(-inf)`` is ``0.0``, so a single corrupt pixel zeroed the entire IR budget
and the model emitted a plausible dust-free galaxy — a wrong answer wearing the
shape of a right one.

**The two forms deliberately answer it differently now.**

===============================  =====================  ==================
form                             live callers in src/   corrupt integrand
===============================  =====================  ==================
``bolometric_absorbed_log10``    4                      ``+inf``
``bolometric_absorbed``          0                      ``0.0`` (clamped)
===============================  =====================  ==================

The split is what lets both be right. The linear clamp is inherited from #922 and
protects a real artifact class (Inf·0 from extreme-metallicity SSP fluxes,
BUG-NSS-02); it is pinned by ``TestFiniteGuard`` in
``tests/physics/conservation/test_lyc_mask_energy_balance.py``, which must keep
passing untouched. But that test guards the function *nothing calls* — measured
2026-08-04, every production call site uses the log form. So tightening the live
path costs the linear contract nothing, and an earlier attempt that changed the
shared flag for both was reverted precisely because it did not respect that.

``+inf`` was chosen over NaN because :func:`tengri.utils.scale.log10_add` already
propagates it deliberately (its ``isposinf`` branch), so the two seams agree
without further plumbing.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import CorruptEnergyBalanceWarning
from tengri.forward.energy_balance import (
    bolometric_absorbed,
    bolometric_absorbed_log10,
    warn_if_corrupt,
)
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import log10_add, pow10

pytestmark = pytest.mark.regression_bug

_WAVE = np.logspace(np.log10(1000.0), np.log10(1e6), 512)
_NU = C_AA / _WAVE


def _seds(scale=1.0e28):
    """A plain absorbing case: attenuated sits below intrinsic everywhere."""
    intrinsic = np.full_like(_WAVE, scale)
    attenuated = intrinsic * 0.4
    return intrinsic, attenuated


def _corrupt(bad, where):
    """The same SEDs with one pixel poisoned."""
    intrinsic, attenuated = _seds()
    if where == "intrinsic":
        intrinsic = intrinsic.copy()
        intrinsic[100] = bad
    else:
        attenuated = attenuated.copy()
        attenuated[100] = bad
    return intrinsic, attenuated


def _log_form(intrinsic, attenuated):
    return bolometric_absorbed_log10(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )


def test_setup_a_clean_sed_absorbs_something():
    """Guard the guard: the clean case must be finite and positive.

    Without this, every assertion below could pass on a reference that was
    already broken.
    """
    log_l, _ = _log_form(*_seds())
    assert np.isfinite(float(log_l)), "clean SED gives a non-finite log absorbed luminosity"
    assert float(pow10(log_l)) > 0.0, "clean SED absorbs nothing"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["intrinsic", "attenuated"])
def test_a_corrupt_pixel_reports_positive_infinity_not_zero(bad, where):
    """The fix itself: corrupt must be distinguishable from "nothing absorbed"."""
    log_l, sign = _log_form(*_corrupt(bad, where))

    assert np.isposinf(float(log_l)), (
        f"a {bad} in the {where} SED gave log_L_absorbed={float(log_l)}, not +inf. If this "
        "is -inf the fail-open is back: pow10(-inf) is exactly 0.0, so the fit would "
        "silently report a dust-free galaxy"
    )
    assert np.isposinf(float(pow10(log_l))), "the +inf sentinel must survive pow10"
    assert np.isnan(float(sign)), (
        f"corrupt input gave sign={float(sign)}; 0.0 already means 'nothing absorbed', so "
        "an uncomputable integral must not reuse it"
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_the_linear_form_deliberately_still_clamps(bad):
    """The other half of the split — and the reason it is a split at all.

    ``bolometric_absorbed`` keeps the #922 clamp. If this ever starts returning
    inf, ``TestFiniteGuard`` will fail too, and the two must be changed together.
    """
    intrinsic, attenuated = _corrupt(bad, "intrinsic")
    linear = bolometric_absorbed(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )
    assert float(linear) == 0.0, (
        f"the linear form returned {float(linear)} rather than the clamped 0.0. That clamp is "
        "BUG-NSS-02 behavior carried forward by #922 and pinned by TestFiniteGuard — if it "
        "was changed on purpose, update both files together"
    )


def test_a_genuinely_zero_integrand_is_still_exactly_zero():
    """The case the ``-inf`` sentinel is *right* for, now genuinely separated.

    This is what the old code could not distinguish from the corrupt case. It is
    the whole point of the change, so it gets its own test rather than riding
    along on another.
    """
    intrinsic, _ = _seds()
    log_l, sign = _log_form(intrinsic, intrinsic)

    assert np.isneginf(float(log_l)), f"unattenuated SED gave log_L_absorbed={float(log_l)}"
    assert float(pow10(log_l)) == 0.0
    assert float(sign) == 0.0, "a true zero must keep sign 0.0, not the corrupt NaN"


def test_zero_and_corrupt_are_now_different_answers():
    """States the property directly, so it cannot regress by halves.

    Both previous behaviors were ``-inf``. A test that only checked one of them
    would pass on a partial revert.
    """
    zero_log, _ = _log_form(*(lambda s: (s, s))(_seds()[0]))
    corrupt_log, _ = _log_form(*_corrupt(np.nan, "intrinsic"))
    assert float(zero_log) != float(corrupt_log), (
        "an all-zero integrand and a corrupt one produce the same value again — the "
        "distinction #1527 added has collapsed"
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_the_plus_inf_survives_log10_add(bad):
    """The seam the log form feeds in the LUT branch of ``two_component``.

    ``+inf`` was chosen over NaN because ``log10_add`` already treats it as an
    overflow sentinel. If that stopped holding, a corrupt nebular term would be
    swallowed while combining it with the stellar one.
    """
    corrupt_log, corrupt_sign = _log_form(*_corrupt(bad, "intrinsic"))
    clean_log, clean_sign = _log_form(*_seds())

    combined = log10_add(clean_log, corrupt_log, sign_a=clean_sign, sign_b=corrupt_sign)
    assert np.isposinf(float(combined)), (
        f"log10_add folded a +inf term into {float(combined)} — the corrupt half of a "
        "stellar+nebular sum is being swallowed"
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_the_lut_stellar_producer_carries_the_same_split(monkeypatch, bad):
    """The stellar half of the ``approx=WavePrecomp`` path, which is easy to miss.

    ``lut_l_absorbed_stellar_log10`` reaches the same ``log10_add`` as the exact
    form. Its ``positive = magnitude > 0`` test is False for NaN, so before #1527
    a corrupt contraction silently became ``-inf`` — zero absorption — on the
    configuration most fits actually run.

    The contraction is stubbed rather than driven through a real LUT: the point
    is the guard on its *output*, and building a corrupt LUT would test the
    builder instead.
    """
    import tengri.components.dust.energy_balance_precompute as ebp

    monkeypatch.setattr(ebp, "_lut_contract", lambda *a, **k: jnp.asarray(bad))
    log_mag, sign = ebp.lut_l_absorbed_stellar_log10(
        object(), jnp.ones((2, 3)), jnp.asarray(10.0), jnp.asarray(0.5), jnp.asarray(0.3)
    )

    assert np.isposinf(float(log_mag)), (
        f"the LUT stellar producer gave {float(log_mag)} on a {bad} contraction. -inf here "
        "means the stellar absorbed luminosity silently vanishes on the WavePrecomp path"
    )
    assert np.isnan(float(sign))


def test_the_warning_names_the_component():
    """``+inf`` is loud but not diagnosable on its own — this is the other half.

    A user seeing a NaN fit has nothing pointing at a corrupt intrinsic SED
    unless something says so.
    """
    with pytest.warns(CorruptEnergyBalanceWarning, match="two_component"):
        warn_if_corrupt(jnp.asarray(jnp.inf), component="two_component")


def test_the_warning_stays_quiet_on_healthy_and_zero_values():
    """The negative control: it must not fire for a normal fit or a true zero.

    Without this the warning could be unconditional and the test above would
    still pass.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorruptEnergyBalanceWarning)
        warn_if_corrupt(jnp.asarray(43.5), component="two_component")
        warn_if_corrupt(jnp.asarray(-jnp.inf), component="two_component")


def test_the_warning_is_silent_under_jit():
    """Inference explores corrupt draws routinely; a per-sample warning is unusable.

    ``float()`` raises ``ConcretizationTypeError`` while tracing, which the helper
    swallows. If that stopped working, ``warn_if_corrupt`` would raise inside every
    jitted forward pass rather than warn.
    """
    import jax

    @jax.jit
    def traced(x):
        warn_if_corrupt(x, component="two_component")
        return x * 2.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", CorruptEnergyBalanceWarning)
        out = traced(jnp.asarray(jnp.inf))
    assert np.isposinf(float(out)), "the value itself must still travel through the jit"


def test_the_log_and_linear_forms_agree_on_a_clean_sed():
    """The two spellings of one integral must not drift apart on healthy input.

    They differ only on corrupt input; everywhere else they are the same number.
    Both are *signed* and follow grid orientation: ``_WAVE`` ascends, so ``nu``
    descends and ``trapezoid`` returns a negative value.
    """
    intrinsic, attenuated = _seds()
    log_l, sign = _log_form(intrinsic, attenuated)
    linear = float(
        bolometric_absorbed(
            jnp.asarray(intrinsic),
            jnp.asarray(attenuated),
            jnp.asarray(_NU),
            wave=jnp.asarray(_WAVE),
        )
    )

    assert float(sign) == np.sign(linear), (
        f"log form reports sign {float(sign)} while the linear form is {linear:.3e}"
    )
    rel = abs(float(pow10(log_l)) - abs(linear)) / abs(linear)
    assert rel < 1e-12, f"log and linear forms disagree by {rel:.3e}"

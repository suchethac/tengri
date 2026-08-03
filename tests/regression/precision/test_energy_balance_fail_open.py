# SPDX-License-Identifier: BSD-3-Clause
r"""The log energy-balance contract on degenerate input — currently a fail-open.

``bolometric_absorbed_log10`` is the float32-safe spelling of the absorbed-
luminosity integral (#1206). Its behavior on a *corrupt* integrand was pinned
nowhere: ``TestFiniteGuard`` in
``tests/physics/conservation/test_lyc_mask_energy_balance.py`` covers only the
linear :func:`bolometric_absorbed`. These tests close that gap.

**What is pinned here is the current behavior, and it is a fail-open.** A single
non-finite pixel makes ``peak`` non-finite, which clears ``ok``, which returns
``-inf`` — and ``pow10(-inf)`` is ``0.0``, i.e. *no dust absorption*. The entire
IR budget is silently zeroed and the model emits a plausible dust-free galaxy
rather than an error.

That is inherited, not chosen. #922's table lists the finite-guard as a property
of the retired compositional kernel, carried through the consolidation to avoid
changing behavior, and it does protect against a real artifact class (Inf·0 from
extreme-metallicity SSP fluxes, BUG-NSS-02).

It nonetheless contradicts :func:`tengri.utils.scale.log10_add`, which reports an
overflowed term as ``+inf`` precisely so it cannot be mistaken for zero. Both
conventions ship. Changing this one alters dust IR for corrupt inputs and is a
policy call, tracked in **#1527** — so these tests assert what the code does
today and will fail loudly when that is deliberately changed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.forward.energy_balance import bolometric_absorbed, bolometric_absorbed_log10
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import pow10

pytestmark = pytest.mark.regression_bug

_WAVE = np.logspace(np.log10(1000.0), np.log10(1e6), 512)
_NU = C_AA / _WAVE


def _seds(scale=1.0e28):
    """A plain absorbing case: attenuated sits below intrinsic everywhere."""
    intrinsic = np.full_like(_WAVE, scale)
    attenuated = intrinsic * 0.4
    return intrinsic, attenuated


def test_setup_a_clean_sed_absorbs_something():
    """Guard the guard: the clean case must be finite and positive.

    Without this, every assertion below could pass on a reference that was
    already broken.
    """
    intrinsic, attenuated = _seds()
    log_l, _ = bolometric_absorbed_log10(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )
    assert np.isfinite(float(log_l)), "clean SED gives a non-finite log absorbed luminosity"
    assert float(pow10(log_l)) > 0.0, "clean SED absorbs nothing"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["intrinsic", "attenuated"])
def test_a_corrupt_pixel_currently_reports_zero_absorption(bad, where):
    """Pins the fail-open on the log path, which nothing else covered.

    Deliberately asserts the *undesirable* behavior. The point is that it is now
    visible and versioned: if #1527 changes the convention this test fails, and
    whoever changes it is told here, in one place, what the old contract was and
    which other test (``TestFiniteGuard``) encodes the linear half of it.
    """
    intrinsic, attenuated = _seds()
    if where == "intrinsic":
        intrinsic = intrinsic.copy()
        intrinsic[100] = bad
    else:
        attenuated = attenuated.copy()
        attenuated[100] = bad

    log_l, sign = bolometric_absorbed_log10(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )

    assert np.isneginf(float(log_l)), (
        f"a {bad} in the {where} SED no longer returns -inf but {float(log_l)}. If this "
        "is #1527 landing, update this file and TestFiniteGuard together — they are "
        "the two halves of one convention"
    )
    assert float(pow10(log_l)) == 0.0, "the -inf sentinel no longer exponentiates to zero"
    assert float(sign) == 0.0


def test_a_genuinely_zero_integrand_is_also_exactly_zero():
    """The case the sentinel is *right* for — and why the two are hard to separate.

    An unattenuated galaxy absorbs nothing, and ``-inf`` -> ``0.0`` is the
    correct answer. It is indistinguishable in the output from the corrupt case
    above, which is the whole substance of #1527.
    """
    intrinsic, _ = _seds()
    log_l, sign = bolometric_absorbed_log10(
        jnp.asarray(intrinsic), jnp.asarray(intrinsic), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )
    linear = bolometric_absorbed(
        jnp.asarray(intrinsic), jnp.asarray(intrinsic), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )

    assert np.isneginf(float(log_l)), f"unattenuated SED gave log_L_absorbed={float(log_l)}"
    assert float(pow10(log_l)) == 0.0
    assert float(linear) == 0.0
    assert float(sign) == 0.0


def test_the_log_and_linear_forms_agree_on_a_clean_sed():
    """The two spellings of one integral must not drift apart.

    Both are *signed* and follow grid orientation: ``_WAVE`` ascends, so ``nu``
    descends and ``trapezoid`` returns a negative value. That sign is a property
    of the axis, not the physics, so magnitudes are what must match — and the
    two forms must agree with each other about the sign.
    """
    intrinsic, attenuated = _seds()
    log_l, sign = bolometric_absorbed_log10(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )
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

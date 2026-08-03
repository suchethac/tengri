# SPDX-License-Identifier: BSD-3-Clause
r"""A corrupt SED must not be reported as "no dust absorption" (#1206).

``_peak_factored_trapezoid`` used to return a single degenerate flag, so two
different situations produced the same answer:

* the integrand is identically zero — nothing was absorbed, a true ``0.0``;
* the integrand contains a NaN or ``inf`` — something upstream is broken.

Both mapped to ``ok=False``, and both callers turned that into ``0.0`` (linear)
or ``-inf`` (log, which exponentiates to ``0.0``). So a single non-finite pixel
anywhere in the intrinsic SED silently zeroed the whole IR budget, and the model
went on to emit no dust IR at all — a plausible-looking galaxy, not an error.

Two other places in the same codebase already argue the opposite convention:
:func:`tengri.utils.scale.log10_add` reports an overflowed term as ``+inf``
because folding it into the zero sentinel would be "a fail-open on precisely the
axis this module exists to close", and :func:`bolometric_absorbed`'s own
docstring describes this failure mode as something to avoid. These tests pin the
agreement.
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
def test_a_corrupt_pixel_is_not_reported_as_zero_absorption(bad, where):
    """One bad pixel must be loud, not a silent zero IR budget."""
    intrinsic, attenuated = _seds()
    if where == "intrinsic":
        intrinsic = intrinsic.copy()
        intrinsic[100] = bad
    else:
        attenuated = attenuated.copy()
        attenuated[100] = bad

    log_l, _ = bolometric_absorbed_log10(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )
    linear = bolometric_absorbed(
        jnp.asarray(intrinsic), jnp.asarray(attenuated), jnp.asarray(_NU), wave=jnp.asarray(_WAVE)
    )

    assert not np.isneginf(float(log_l)), (
        f"a {bad} in the {where} SED returned log_L_absorbed = -inf, which is "
        "pow10 -> 0.0, i.e. 'no dust absorption' — the fail-open this test exists "
        "to prevent"
    )
    assert float(pow10(log_l)) != 0.0, "corrupt integrand still exponentiates to zero"
    # NaN satisfies this too, which is the point: anything but a clean 0.0.
    assert float(linear) != 0.0, (
        f"a {bad} in the {where} SED returned exactly 0.0 erg/s absorbed — "
        "indistinguishable from a genuinely unattenuated galaxy"
    )


def test_a_genuinely_zero_integrand_is_still_exactly_zero():
    """The other half: a true zero must NOT become loud.

    Making corruption loud is only correct if it does not also make the
    legitimate "nothing absorbed" case loud. Unattenuated means zero, and zero
    is the right answer.
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


def test_the_clean_path_is_numerically_unchanged():
    """The split must be invisible whenever the integrand is well-behaved.

    Compares the log form against the linear one, which is the independent
    spelling of the same integral — so this is a statement about the two
    contracts agreeing, not about the previous implementation.

    Both are *signed* and follow grid orientation: ``_WAVE`` ascends, so ``nu``
    descends and ``trapezoid`` returns a negative value. That sign is a property
    of the axis, not of the physics, so the magnitudes are what must match — and
    the two forms must at least agree with each other about it.
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

# SPDX-License-Identifier: BSD-3-Clause
r"""A saturating exponent bound must be representable in the dtype it saturates to.

The ceiling-side mirror of ``representable_floor`` (#1492), and the more dangerous
half. A floor written below float32's smallest normal merely stops protecting --
it silently becomes ``0.0`` and the guard is inert. A *ceiling* written above
float32's largest value is worse: the guarded expression returns ``inf`` for every
input the guard fires on, so the defense **manufactures the failure it exists to
prevent**.

Measured (#1206): both Cue emission paths clipped their base-10 exponent to
``±50`` dex, with the comment *"the clip is the only defense against NaN/inf
poisoning a JAX gradient"*. ``10**50`` is ``inf`` in float32, whose ceiling is
``10**38.53``. In pure float32 that single constant was the reason the entire
forward state went non-finite: the Cue SED poisoned the dust energy balance, so
``L_absorbed``, ``L_ir``, ``log_L_ir`` and every gradient through them were NaN --
on a model whose float64 result is perfectly well behaved.

The three tests below are deliberately of different kinds, because each alone is
weak:

* the **unit** property (float64 untouched, float32 capped, ``10**cap`` finite),
* the **end-to-end** consequence (a pure-float32 Cue forward is finite), which is
  what actually regressed and what a future refactor would break,
* the **census** (no other call site reintroduces a float64-era exponent bound),
  because every guard in this repository that was keyed on a hand-written list has
  eventually been green on the one item nobody thought to add.
"""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.scale import representable_exponent

pytestmark = pytest.mark.regression_bug

#: Source root, resolved from this file so the sweep cannot silently scan nothing.
_SRC = Path(__file__).resolve().parents[3] / "src" / "tengri"

#: ``10.0 ** jnp.clip(x, lo, HI)`` / ``pow10(jnp.clip(x, lo, HI))`` -- the shape that
#: bit. Captures the upper bound so the sweep can compare it against the dtype.
_POW10_CLIP = re.compile(
    r"(?:10\.0\s*\*\*|pow10\()\s*(?:jnp\.)?clip\([^,]+,\s*[^,]+,\s*([0-9]+\.?[0-9]*)\s*\)"
)


def test_float64_is_untouched():
    """The cap must be a no-op in float64, so no existing result can move."""
    with jax.enable_x64(True):
        for value in (10.0, 38.0, 50.0, 100.0, 300.0):
            assert representable_exponent(value) == value, (
                f"representable_exponent({value}) changed a float64 bound; float64's "
                "ceiling is 10**308.25 and must return the literal unchanged"
            )


def test_float32_caps_below_its_ceiling_and_the_power_is_finite():
    """In float32 the bound is lowered, and ``10**bound`` is actually representable.

    Asserting only ``cap < 50`` would pass for a cap that still overflows. The
    load-bearing assertion is the second one: ``log10(finfo.max)`` is itself
    unusable, because ``10**log10(max)`` rounds *up* to ``inf`` in the last bits.
    """
    with jax.enable_x64(False):
        cap = representable_exponent(50.0)
        assert cap < 50.0, f"float32 bound was not lowered: {cap}"
        assert cap <= float(np.log10(np.finfo(np.float32).max)), cap
        power = 10.0 ** jnp.asarray(cap, jnp.float32)
        assert np.isfinite(float(power)), (
            f"10**{cap} is {float(power)} in float32 -- the cap is not representable, "
            "which is the whole defect this function exists to fix"
        )


def test_setup_the_unfixed_bound_really_overflows():
    """Guard the guard: if ``10**50`` were finite in float32 this suite is vacuous."""
    with jax.enable_x64(False):
        assert not np.isfinite(float(10.0 ** jnp.asarray(50.0, jnp.float32))), (
            "10**50 is finite in float32 here, so the defect this module regresses "
            "cannot be reproduced and every assertion below proves nothing"
        )


def test_cue_forward_is_finite_in_pure_float32(ssp_bare):
    """The end-to-end consequence: a pure-float32 Cue model must not go non-finite.

    This is the assertion that would catch a refactor reverting the call sites while
    leaving :func:`representable_exponent` in place -- a unit test of the helper
    alone cannot, because the helper would still be correct and unused.
    """
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

    params = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}

    def build():
        return SEDModel.build(
            ssp_data=ssp_bare,
            observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
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
            neb={"type": "cue", "all_params": FIXED},
            redshift=Fixed(0.1),
            approx=None,
        )

    with jax.enable_x64(False):
        phot = np.asarray(build().predict_photometry(params), dtype=np.float64)
    assert np.isfinite(phot).all(), (
        f"pure-float32 Cue photometry is non-finite: {phot}. The ±50 dex exponent "
        "clip is back, or another float64-era bound has been introduced."
    )


def test_no_call_site_clips_a_base_ten_exponent_above_the_float32_ceiling():
    """Census: nobody may reintroduce a literal exponent bound float32 cannot hold.

    Swept from source rather than from a list of known sites, for the reason the
    module docstring gives. A new ``10.0 ** jnp.clip(x, lo, 50.0)`` anywhere in the
    package fails here, whether or not anyone remembers this issue.
    """
    with jax.enable_x64(False):
        f32_ceiling = float(np.log10(np.finfo(np.float32).max))

    files = sorted(_SRC.rglob("*.py"))
    assert files, f"the sweep found no source files under {_SRC} -- it is vacuous"

    offenders = []
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            match = _POW10_CLIP.search(line)
            if match and float(match.group(1)) > f32_ceiling:
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "a base-10 exponent is clipped to a bound float32 cannot represent "
        f"(its ceiling is 10**{f32_ceiling:.2f}), so the clip returns inf for every "
        "input it fires on. Wrap the bound in representable_exponent():\n  "
        + "\n  ".join(offenders)
    )

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

#: ``jnp.exp(jnp.clip(x, lo, HI))`` -- the natural-log twin, and the reason this sweep
#: covers two patterns instead of one. The first version of this guard swept only the
#: ``10**`` form and was green while three qsogen Planck terms carried a literal
#: ``500.0`` against float32's ``e**88.72`` ceiling: a guard is only as wide as its
#: census, including this one.
#:
#: Only the **positive** form is swept. ``exp(-clip(x, 0, 500))`` is correct as
#: written -- ``exp(-500)`` underflows to ``0.0``, which is the right answer with a
#: ``-0.0`` gradient -- and capping its magnitude to 88.72 would return 3e-39 instead,
#: i.e. turn a correct zero into a wrong non-zero.
_EXP_CLIP = re.compile(
    r"(?:jnp\.)?exp\(\s*(?:jnp\.)?clip\([^,]+,\s*[^,]+,\s*([0-9]+\.?[0-9]*)\s*\)"
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


@pytest.mark.parametrize(
    ("pattern", "base_name", "ceiling", "call"),
    [
        (
            "_POW10_CLIP",
            "10",
            float(np.log10(np.finfo(np.float32).max)),
            "representable_exponent(HI)",
        ),
        (
            "_EXP_CLIP",
            "e",
            float(np.log(np.finfo(np.float32).max)),
            "representable_exponent(HI, base=math.e)",
        ),
    ],
)
def test_no_call_site_clips_an_exponent_above_the_float32_ceiling(
    pattern, base_name, ceiling, call
):
    """Census: nobody may reintroduce a literal exponent bound float32 cannot hold.

    Swept from source rather than from a list of known sites, for the reason the
    module docstring gives. A new ``10.0 ** jnp.clip(x, lo, 50.0)`` or
    ``jnp.exp(jnp.clip(x, lo, 500.0))`` anywhere in the package fails here, whether
    or not anyone remembers this issue.

    Both bases are swept because the first version of this guard swept only ``10**``
    and was green while three qsogen Planck terms carried a literal ``500.0``. The
    ``exp`` form is the worse of the two: ``1/(exp(inf)-1)`` is ``0.0``, which is the
    *correct* value for a Planck tail, so the forward pass is bit-identical and only
    the gradient goes NaN.
    """
    regex = {"_POW10_CLIP": _POW10_CLIP, "_EXP_CLIP": _EXP_CLIP}[pattern]

    files = sorted(_SRC.rglob("*.py"))
    assert files, f"the sweep found no source files under {_SRC} -- it is vacuous"

    offenders = []
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            # Prose describing the pattern is not an instance of it. Docstrings that
            # explain this very defect quote the broken form in RST literals, and a
            # comment above a fixed call site may still name the old bound. Both are
            # documentation; only executable code can overflow.
            if "``" in line or line.lstrip().startswith("#"):
                continue
            match = regex.search(line)
            if match and float(match.group(1)) > ceiling:
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {line.strip()}")

    assert not offenders, (
        f"a base-{base_name} exponent is clipped to a bound float32 cannot represent "
        f"(its ceiling is {base_name}**{ceiling:.2f}), so the clip returns inf for "
        f"every input it fires on. Wrap the bound in {call}:\n  " + "\n  ".join(offenders)
    )


def test_the_exp_form_poisons_only_the_gradient(ssp_bare):
    """The ``exp`` twin is silent: identical forward value, NaN gradient.

    Pinned as its own test because it is the reason the census sweeps two patterns.
    A forward-value comparison -- the check most float32 guards make -- cannot see
    this defect at all: measured, ``1/(exp(clip(x, 0, 500)) - 1)`` returns the *same*
    number with the broken and the fixed bound, because ``1/(inf - 1)`` is ``0.0``
    and ``0.0`` is the physically correct Planck tail.
    """
    from tengri.components.agn.qsogen import _hot_dust_blackbody

    wave = np.logspace(2.7, 5.0, 400)

    def gradient(x64, dtype):
        with jax.enable_x64(x64):
            w = jnp.asarray(wave, dtype)
            cont = jnp.ones_like(w)

            def total(tbb):
                sed = _hot_dust_blackbody(w, cont, tbb=tbb, bbnorm=jnp.asarray(1.0, dtype))
                return jnp.sum(jnp.asarray(sed, dtype))

            return float(jax.grad(total)(jnp.asarray(1200.0, dtype)))

    g64 = gradient(True, jnp.float64)
    g32 = gradient(False, jnp.float32)
    assert np.isfinite(g32), (
        f"d(hot dust blackbody)/d(tbb) is {g32} in pure float32 -- the exp() bound is "
        "back above float32's e**88.72 ceiling, so exp() saturates to inf and the "
        "reverse pass forms 0 * inf"
    )
    assert abs(g32 - g64) / abs(g64) < 1e-4, f"float32 {g32} vs float64 {g64}"

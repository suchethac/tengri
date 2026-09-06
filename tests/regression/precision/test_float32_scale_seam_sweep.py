# SPDX-License-Identifier: BSD-3-Clause
r"""Every enumerated scale seam, swept across its own declared prior (#2178, #1436).

Four bugs have come out of one shape: a large physical constant multiplies a
parameter as a **standalone scalar factor**, leaves float32's range, and the
product then meets a zero or a tiny partner, so ``inf * 0`` is ``nan`` or the
underflow flushes to zero. #1388 (``apply_log10_scale``), #1439
(``multicolor_disc``'s bolometric renorm, first misdiagnosed as an unreachable
cancellation when it was a grouping bug), #2100 (``_mass_scale_lnu``'s reverse
pass) and #2178 (the *same* product from the forward pass) are the four.

The fixes were per-site, and the per-site strategy is what this module exists to
replace. #2100 pinned the reverse pass and #2178 was still exposed through the
forward one; nothing said so, because nothing was enumerating the seams.

    *A float32 result established on one model configuration says nothing about
    a configuration with a different scale seam ... coverage has to be
    enumerated by seam, not by "a representative model".* -- #1436

So the inventory here is not written down twice. It is **read from**
``tools/check_float32_scale_seams.py``, which enumerates the seams from the AST
and sizes each against the range its parameter declares in the registry.
:func:`test_every_over_range_seam_family_has_a_behavioral_sweep` then refuses to pass
if that enumeration grows a seam this module does not sweep -- which is the
mechanism, and the only part of this file that closes the class rather than one
of its instances.

**Both halves of the assertion, every time.** ``nan != 0.0`` is ``True`` and
``0.0`` is finite, so "finite" alone admits a dead gradient and "non-zero" alone
admits a ``nan``. #2100's hole was the first spelling and #2178's was the second.
Every sweep point asserts finite **and** non-zero.

**Which of these bite on the installed jaxlib.** The #2178 forward defect
reproduces on jaxlib >= 0.11.1 and not on 0.11.0, on byte-identical optimized
HLO. On a 0.11.0 machine the float32 sweeps below therefore pass with *or*
without the fix: they are behavioral tests of a behavior that jaxlib cannot
exhibit there. They bite in CI and on any 0.11.1 box.
:func:`test_the_scale_seam_enumeration_is_wired_into_the_gate` and
:func:`test_every_over_range_seam_family_has_a_behavioral_sweep` are **structural** and
bite everywhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, SpectrumPrecomp, Uniform
from tengri.parameters.registry import registry
from tengri.utils.scale import loss_scaled_grad

pytestmark = pytest.mark.regression_bug

_ROOT = Path(__file__).resolve().parents[3]


def _load_tool(name: str):
    """Import a ``tools/`` script by path -- they are not an installed package."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TOOL = _load_tool("check_float32_scale_seams")

#: ``seam family -> sweep id``. The families are the tool's, not this file's
#: opinion of them:
#: :func:`test_every_over_range_seam_family_has_a_behavioral_sweep` reads the
#: enumeration and fails if a family it reports over-range is
#: missing here. One sweep per PRODUCT, not per call site -- thirty-eight AGN
#: blocks renormalize by the same ``L_sun * 10**agn_log_lbol``, and what a sweep
#: can prove about that product is the same in all thirty-eight.
_SWEEP_OF_FAMILY: dict[str, str] = {
    "stellar_mass_scale": "stellar_mass_scale",
    "agn_bolometric_renorm": "agn_bolometric",
    "agn_black_hole_mass": "agn_black_hole_mass",
    "xrb_mass_scale": "xrb_mass_scale",
}

#: Sweep resolution. Nine points across the declared prior, endpoints included:
#: the endpoints are where the product is largest and smallest, and the interior
#: points are what says the failure is not a single-point artifact. #1439's fix
#: was verified across "the whole declared ``agn_log_lbol`` prior" and this is
#: that standard generalized.
_N_SWEEP = 9

_DUST = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_diff": 0.3,
    "tau_bc": 0.0,
}

_SPEC_WAVE = np.linspace(4000.0, 9000.0, 128)


def _declared_range(pattern: str) -> tuple[str, float, float]:
    """``(name, lo, hi)`` of the widest declared prior matching *pattern*.

    Read from the registry, never transcribed from a grid axis or a fixture: commit
    45741f4cd is the case where a *grid axis* range was used as a prior and the
    two shared no point at all.
    """
    import re

    best = None
    for name, record in registry().items():
        if not re.match(pattern, name):
            continue
        bounds = _TOOL._prior_bounds(record)
        if bounds is None:
            continue
        lo, hi = bounds
        if best is None or (hi - lo) > (best[2] - best[1]):
            best = (name, lo, hi)
    assert best is not None, f"no registry parameter matches {pattern!r}"
    return best


def _sweep_points(lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, _N_SWEEP)


def _reference_point(sed) -> dict[str, float]:
    """Every free parameter at the center of its own standardized prior."""
    return {
        n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
        for n in sed.spec.free_params
    }


# --------------------------------------------------------------------------------------
# The sweeps
# --------------------------------------------------------------------------------------


def _stellar_mass_scale(ssp, dtype):
    """``_mass_scale_lnu`` on the SpectrumPrecomp path -- the #2178 seam itself.

    The spectrum LUT is the projector on which the forward product went ``nan``:
    ages beyond the galaxy's age carry an exactly-zero SFH weight, so an
    ``inf`` scalar factor reaches the age reduction as ``inf * 0``.
    """
    name, lo, hi = _declared_range(r"^sfh_delayed_log_total_mass$")
    with jax.enable_x64(dtype is jnp.float64):
        from tengri.observation.spectroscopy import Spectroscopy

        sed = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(spectroscopy=Spectroscopy(wave_obs=jnp.asarray(_SPEC_WAVE))),
            approx=SpectrumPrecomp(n_z=16, z_min=0.05, z_max=1.0),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(lo, hi),
                "tau_gyr": 1.0,
                "age_gyr": 5.0,
            },
            redshift=Fixed(0.1),
            dust_attenuation=_DUST,
        )
        base = _reference_point(sed)

        def objective(x):
            return jnp.sum(sed.predict_spectrum({**base, name: x}))

        return name, _evaluate(objective, lo, hi, dtype)


def _agn_bolometric(ssp, dtype):
    """``L_sun * 10**agn_log_lbol`` -- #1439's seam, across the whole prior.

    At the top of the declared prior the product is ~3.8e47, five orders past
    float32's 3.4e38. #1439 was first read as an unreachable cancellation; it
    was a grouping bug, and it is reachable everywhere above ``log_lbol ~ 4.9``.
    """
    name, lo, hi = _declared_range(r"^agn_log_lbol$")
    with jax.enable_x64(dtype is jnp.float64):
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(["sdss_r", "wise_w1", "wise_w4"])
            ),
            approx=None,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "tau_gyr": 1.0, "age_gyr": 5.0},
            redshift=Fixed(0.1),
            dust_attenuation=_DUST,
            agn={
                "type": "composable",
                "all_params": Fixed(DEFAULT),
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "norm": "cigale_joint",
                "log_lbol": Uniform(lo, hi),
                "fracAGN": 0.1,
            },
        )
        base = _reference_point(sed)

        def objective(x):
            return jnp.sum(sed.predict_photometry({**base, name: x}))

        return name, _evaluate(objective, lo, hi, dtype)


def _agn_black_hole_mass(ssp, dtype):
    """``M_sun * 10**agn_log_mbh`` in the Kubota & Done disc, across its prior."""
    name, lo, hi = _declared_range(r"^agn_log_mbh$")
    with jax.enable_x64(dtype is jnp.float64):
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(["sdss_r", "wise_w1", "wise_w4"])
            ),
            approx=None,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "tau_gyr": 1.0, "age_gyr": 5.0},
            redshift=Fixed(0.1),
            dust_attenuation=_DUST,
            agn={
                "type": "composable",
                "all_params": Fixed(DEFAULT),
                "disc": {
                    "type": "kubota_done",
                    "all_params": Fixed(DEFAULT),
                    "log_mbh": Uniform(lo, hi),
                },
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "norm": "cigale_joint",
                "log_lbol": Fixed(12.0),
                "fracAGN": 0.1,
            },
        )
        base = _reference_point(sed)

        def objective(x):
            return jnp.sum(sed.predict_photometry({**base, name: x}))

        return name, _evaluate(objective, lo, hi, dtype)


def _xrb_mass_scale(ssp, dtype):
    """The XRB mass term (#722), read through its float32 path, across the prior.

    Swept on ``log_l_x_xrb``, not ``l_x_xrb``. The linear form is not evaluable
    in float32 *at all* -- the HMXB coefficient ``2.6e39`` is past float32's
    ceiling before it is multiplied by anything, so the sum overflows even at
    zero SFR -- and ``utils.sed_quantities.compute_log_l_x_xrb`` is the
    documented companion that carries both coefficients in log space. Sweeping
    the linear form would measure a documented impossibility; sweeping the
    companion measures whether the seam's actual float32 path holds across the
    whole declared mass prior, which is the claim ``_HANDLED`` records.
    """
    name, lo, hi = _declared_range(r"^sfh_delayed_log_total_mass$")
    with jax.enable_x64(dtype is jnp.float64):
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"])),
            approx=None,
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(lo, hi),
                "tau_gyr": 1.0,
                "age_gyr": 5.0,
            },
            redshift=Fixed(0.1),
            dust_attenuation=_DUST,
            xray={"type": "simple"},
        )
        base = _reference_point(sed)

        def objective(x):
            return jnp.asarray(sed.predict({**base, name: x}).properties["log_l_x_xrb"]).sum()

        return name, _evaluate(objective, lo, hi, dtype)


_SWEEPS = {
    "stellar_mass_scale": _stellar_mass_scale,
    "agn_bolometric": _agn_bolometric,
    "agn_black_hole_mass": _agn_black_hole_mass,
    "xrb_mass_scale": _xrb_mass_scale,
}


def _evaluate(objective, lo, hi, dtype):
    """``[(x, value, gradient, dtype), ...]`` across the declared prior.

    The gradient is taken with :func:`~tengri.utils.scale.loss_scaled_grad`, not
    bare ``jax.grad``. These objectives are **unweighted** observables (F_nu
    ~1e-28, L_nu the same), and #2100 established that a bare reverse-mode
    cotangent chain at that scale runs among the float32 subnormals and flushes
    to *exactly zero* -- measured again here on all three model sweeps. That is a
    different defect with its own guards in
    ``test_float32_fitting_path_seams.py``; taking it again here would mean every
    sweep reported the underflow instead of the seam it was written for. The
    ``2**70`` boost is a power of two, so it is exact in both precisions and
    changes nothing about whether a scale seam overflows: ``inf * 0`` is still
    ``nan`` under any cotangent scale.
    """
    grad_fn = loss_scaled_grad(objective)
    out = []
    for x in _sweep_points(lo, hi):
        xa = jnp.asarray(x, dtype=dtype)
        value = np.asarray(objective(xa))
        gradient = np.asarray(grad_fn(xa))
        out.append((float(x), value, gradient, str(gradient.dtype)))
    return out


@pytest.fixture(scope="module")
def swept(ssp_bare, request):
    """``(sweep_id, float32 rows, float64 rows)`` for one enumerated seam."""
    sweep_id = request.param
    fn = _SWEEPS[sweep_id]
    name32, rows32 = fn(ssp_bare, jnp.float32)
    name64, rows64 = fn(ssp_bare, jnp.float64)
    assert name32 == name64
    return sweep_id, name32, rows32, rows64


def _parametrize(fn):
    return pytest.mark.parametrize("swept", sorted(_SWEEPS), indirect=True)(fn)


# --------------------------------------------------------------------------------------
# The mechanism: the enumeration and this module cannot drift apart
# --------------------------------------------------------------------------------------


def test_the_scale_seam_enumeration_finds_the_four_measured_instances():
    """The enumerator sees the seams the four bugs came out of.

    Structural, and the premise of everything below: an enumeration that missed
    ``_mass_scale_lnu`` would report a clean inventory and mean nothing.
    """
    seams = {s.key for s in _TOOL._scan()}
    for expected in (
        "tengri.components.stellar.component:_mass_scale_lnu",
        "tengri.components.agn.disc:multicolor_disc",
    ):
        assert expected in seams, (
            f"{expected} is not in the enumeration ({len(seams)} seams found); the "
            f"scan no longer sees a seam that has already produced a bug"
        )


def test_every_over_range_seam_family_has_a_behavioral_sweep():
    """A new over-range seam cannot be registered without being swept.

    This is the class mechanism. ``tools/check_float32_scale_seams.py`` will
    accept a new seam once it carries a recorded reason; that alone would let
    the reason be the only evidence. This test requires the reason to be backed
    by a float32 sweep across the parameter's own declared prior.
    """
    over = {s.key for s in _TOOL._scan() if s.over_range}
    unregistered = sorted(over - set(_TOOL._FAMILY_OF))
    assert not unregistered, (
        "these scale seams are over float32 range within their own declared prior "
        f"and carry no recorded grouping: {unregistered}"
    )
    families = {_TOOL._FAMILY_OF[key] for key in over}
    missing = sorted(families - set(_SWEEP_OF_FAMILY))
    assert not missing, (
        f"these seam families are over float32 range and no sweep in this module "
        f"covers them: {missing}. A recorded reason on its own is the tool's "
        f"evidence; this module is what makes it a measurement."
    )
    stale = sorted(set(_SWEEP_OF_FAMILY) - families)
    assert not stale, (
        f"_SWEEP_OF_FAMILY names families the enumeration no longer reports as over "
        f"range: {stale}. Delete them rather than leave a sweep pinned to nothing."
    )
    assert set(_SWEEP_OF_FAMILY.values()) <= set(_SWEEPS), (
        "_SWEEP_OF_FAMILY refers to a sweep id that does not exist"
    )


# --------------------------------------------------------------------------------------
# The behavioral sweeps
# --------------------------------------------------------------------------------------


@_parametrize
def test_the_swept_arms_really_ran_at_the_dtypes_they_claim(swept):
    """Proven on the gradient array's dtype, never on the config flag (#1840).

    ``import tengri`` re-enables x64, so a module that believes it is in float32
    can be in float64 and every precision claim below would be void.
    """
    sweep_id, _name, rows32, rows64 = swept
    assert {r[3] for r in rows32} == {"float32"}, (
        f"the float32 arm of {sweep_id} produced {sorted({r[3] for r in rows32})}"
    )
    assert {r[3] for r in rows64} == {"float64"}, (
        f"the float64 arm of {sweep_id} produced {sorted({r[3] for r in rows64})}"
    )


@_parametrize
def test_the_float32_gradient_is_finite_and_nonzero_across_the_declared_prior(swept):
    """Both halves, at every point in the parameter's own declared range.

    ``nan != 0.0`` is ``True``, so a non-zero assertion admits the #2178 nan;
    ``0.0`` is finite, so a finiteness assertion admits the #2100 dead
    gradient. Neither half is coverage on its own.
    """
    sweep_id, name, rows32, _ = swept
    for x, value, gradient, _dtype in rows32:
        assert np.all(np.isfinite(gradient)), (
            f"{sweep_id}: d/d{name} is NON-FINITE at {name}={x:.4g}, inside the "
            f"declared prior: {gradient}"
        )
        assert np.any(gradient != 0.0), (
            f"{sweep_id}: d/d{name} is exactly zero at {name}={x:.4g}, inside the "
            f"declared prior -- the parameter is dead there in float32"
        )
        assert np.all(np.isfinite(value)), (
            f"{sweep_id}: the forward itself is non-finite at {name}={x:.4g}"
        )


@_parametrize
def test_the_float64_arm_is_finite_and_nonzero_too(swept):
    """The reference the float32 arm is judged against has to exist.

    Without this a float32 sweep could pass by comparing two undefined numbers,
    which is the failure mode ``_skip_if_lut_forward_is_broken`` was added for.
    """
    sweep_id, name, _, rows64 = swept
    for x, value, gradient, _dtype in rows64:
        assert np.all(np.isfinite(gradient)) and np.any(gradient != 0.0), (
            f"{sweep_id}: the float64 reference is unusable at {name}={x:.4g}: {gradient}"
        )
        assert np.all(np.isfinite(value))

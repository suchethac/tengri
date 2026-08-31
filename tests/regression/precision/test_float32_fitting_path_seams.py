# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 posterior gradients on the path a real fit takes (#1415, #1436, #1671).

``test_float32_grad_bolometric_seams.py`` enumerates four **model** seams — stellar+dust,
dust IR (+44.5 dex), AGN (+34.6 dex), panchromatic — and shows
``grad(neg_log_posterior_fn)`` tracking float64 to ≤5.3e-04 in pure float32. That is a
statement about the *model*. It is not yet a statement about the *fit*, because a fit
also chooses a **projector** and may **free the redshift**, and #1436's rule applies to
those axes exactly as it applies to components:

    *A float32 result established on one model configuration says nothing about a
    configuration with a different scale seam.*

This module adds the axes a real fit crosses:

======================  ====================================================
axis                    what it adds
======================  ====================================================
``WavePrecomp``         the SSP x filter LUT — ``band_integration="quadrature"``
                        at the default ``n_subbands=5`` and at ``n_subbands=8``
free redshift           ``z`` reaches ``log10_flux_scale`` itself, not only the
                        array argument, and under the LUT also the ztable
CUDA                    run this file with no ``JAX_PLATFORMS`` on a CUDA box;
                        PR #2100 found the cotangent boost wrong by 0.7-18 % on
                        CPU while right to 1e-06 on CUDA, so a backend is an axis
======================  ====================================================

**Which projector a fit actually uses is not the one it was built with.**
``Fitter``'s default ``approx="auto"`` *re-resolves* the build-time knob
(``sed_model.py``: *"``Fitter(approx="auto")`` (the default) re-resolves the build-time
knob, so fit arms that differ only in ``SEDModel.build(approx=...)`` can be one
configuration wearing three labels"*). A model built with ``approx=None`` is therefore
**fitted under the LUT** unless the *Fitter* is also given ``approx=None`` — which is
pinned below, and which means the exact-projector rows are the ones that were missing,
not the LUT rows.

**The reference is float64 autodiff, not same-precision finite differences.** With
chi-squared ~1e4 a float32 central difference subtracts two nearly-equal ~1e4 numbers and
its own noise floor reaches 17 %, larger than the error being looked for. The float64
reference is itself checked against float64 central differences at the same points, per
seam, so a verdict is never taken from an unvalidated instrument.
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.inference.context import InferenceContext
from tengri.utils.scale import DEFAULT_COTANGENT_BOOST, loss_scaled_grad

#: Not marked ``slow``: the default ``addopts`` deselects that mark, and a seam
#: inventory that does not run in the default suite is not a guard. The module-scoped
#: ``measured`` fixture keeps the cost to one model build per precision per seam.
pytestmark = pytest.mark.regression_bug

# --------------------------------------------------------------------------------------
# Model groups — identical to test_float32_grad_bolometric_seams.py, so a row here is
# directly comparable to the published PR #2100 number for the same model.
# --------------------------------------------------------------------------------------

_DUST_FREE = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": FIXED,
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}
_DUST_FIXED = dict(_DUST_FREE, tau_diff=0.3)

_AGN = {
    "type": "composable",
    "all_params": FIXED,
    "disc": {"type": "multicolor", "all_params": FIXED},
    "torus": {"type": "skirtor", "all_params": FIXED},
    "norm": "cigale_joint",
    "log_lbol": Fixed(10.5),  # #2069: pinned to break the flat direction
    "fracAGN": 0.1,
}

_MODELS = {
    "stellar_dust": dict(dust_attenuation=_DUST_FREE),
    "dust_ir": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014", "all_params": FIXED},
    ),
    "agn": dict(dust_attenuation=_DUST_FIXED, agn=_AGN),
    "panchromatic": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014_cigale", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED},
        agn=_AGN,
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        xray={"type": "simple"},
        shock={"frac": 0.1},
    ),
}

_Z_LO, _Z_HI = 0.05, 1.0

#: Deliberately coarse. ``n_z`` sets the ztable's *own* interpolation bias, which is
#: identical in both precisions and cancels out of every float32-vs-float64 comparison
#: here, while the default ``n_z=250`` costs ~20 s of build per model per precision.
_N_Z = 48


def _wp(**kw):
    return WavePrecomp(n_z=_N_Z, z_min=_Z_LO, z_max=_Z_HI, **kw)


#: ``path -> (build_approx_factory, fit_approx, redshift_factory)``.
#: ``fit_approx`` is what reaches ``Fitter(approx=...)`` and it is the one that decides.
_PATHS = {
    "exact_fixedz": (lambda: None, None, lambda: Fixed(0.1)),
    "lut_fixedz": (_wp, "auto", lambda: Fixed(0.1)),
    "lut_quad8_fixedz": (
        lambda: _wp(band_integration="quadrature", n_subbands=8),
        "auto",
        lambda: Fixed(0.1),
    ),
    "exact_freez": (lambda: None, None, lambda: Uniform(_Z_LO, _Z_HI)),
    "lut_freez": (_wp, "auto", lambda: Uniform(_Z_LO, _Z_HI)),
    "lut_quad8_freez": (
        lambda: _wp(band_integration="quadrature", n_subbands=8),
        "auto",
        lambda: Uniform(_Z_LO, _Z_HI),
    ),
}

#: The inventory. Every model on the default fit projector; the two extra projector /
#: redshift variants on the cheapest model and on the kitchen sink, which is where a
#: defect at one seam could compound with (or cancel against) another.
_SEAMS = [
    "stellar_dust/exact_fixedz",
    "stellar_dust/lut_fixedz",
    "dust_ir/lut_fixedz",
    "agn/lut_fixedz",
    "panchromatic/lut_fixedz",
    "stellar_dust/lut_quad8_fixedz",
    "panchromatic/lut_quad8_fixedz",
    "stellar_dust/exact_freez",
    "dust_ir/exact_freez",
    "stellar_dust/lut_freez",
    "dust_ir/lut_freez",
    "panchromatic/lut_freez",
    # The worst cell measured on 2026-08-31 — every axis at once (kitchen sink, LUT at
    # K=8, free redshift). Kept precisely because it is the worst: a bar that is never
    # approached is not a bar.
    "panchromatic/lut_quad8_freez",
]

#: Standardized-space evaluation points. The origin is where the residuals are smallest
#: and the float32 cancellation is worst; 0.5 sigma is a generic interior point.
_POINTS = {"origin": 0.0, "half_sigma": 0.5}

#: Central-difference step in standardized units, for the float64 soundness check only.
_FD_H = 1e-4

_SNR = 30.0


def _base(zspec):
    return dict(
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        redshift=zspec,
    )


@pytest.fixture(scope="module")
def obs():
    # herschel_250 is load-bearing for the dust-IR seam: without a far-IR band the
    # component barely reaches the photometry at all.
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1", "herschel_250"])
    )


def _build(ssp, obs, model, approx, zspec):
    return SEDModel.build(
        ssp_data=ssp, observation=obs, approx=approx, **_base(zspec), **_MODELS[model]
    )


def _gradients(ssp, obs, model, build_approx, fit_approx, zspec, flux, noise, *, x64, dtype):
    """Every gradient of one seam at one precision, from one model build.

    Keys per point: ``grad`` (``jax.grad``), ``boosted`` (:func:`loss_scaled_grad`),
    ``fd`` (central differences at the same precision), ``dtype`` (the dtypes the
    gradient arrays actually came back as — the only admissible proof of precision,
    #1840/#2097: a config flag can say float32 while the arrays are float64).
    """
    with jax.enable_x64(x64):
        sed = _build(ssp, obs, model, build_approx, zspec)
        ctx = InferenceContext.from_target(
            Fitter(
                sed,
                jnp.asarray(flux, dtype=dtype),
                jnp.asarray(noise, dtype=dtype),
                approx=fit_approx,
            )
        )
        data_args = ctx.data_args
        names = sorted(ctx.initial_params(jax.random.PRNGKey(1)))

        def nlp(vals):
            return ctx.neg_log_posterior_fn({k: vals[i] for i, k in enumerate(names)}, data_args)

        def as_array(g):
            return np.array([float(np.asarray(x)) for x in g])

        out = {
            "names": names,
            "approx_state": str(getattr(ctx.fitter.model, "approx", None)),
        }
        for label, offset in _POINTS.items():
            point = [jnp.asarray(offset, dtype=dtype) for _ in names]
            g = jax.grad(nlp)(point)
            fd = []
            for i in range(len(names)):
                p, m = list(point), list(point)
                p[i] = jnp.asarray(offset + _FD_H, dtype=dtype)
                m[i] = jnp.asarray(offset - _FD_H, dtype=dtype)
                fd.append(float((np.asarray(nlp(p)) - np.asarray(nlp(m))) / (2 * _FD_H)))
            out[label] = {
                "grad": as_array(g),
                "boosted": as_array(loss_scaled_grad(nlp)(point)),
                "fd": np.array(fd),
                "dtype": sorted({str(np.asarray(x).dtype) for x in g}),
            }
        return out


def _mock(ssp, obs, model, zspec, snr):
    """One float64 mock from the **exact** projector, so every arm fits identical data."""
    with jax.enable_x64(True):
        sed = _build(ssp, obs, model, None, zspec)
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        mock = sed.mock(truth, snr=snr, key=jax.random.PRNGKey(0))
        return (
            np.asarray(mock.flux_obs, dtype=np.float64),
            np.asarray(mock.noise, dtype=np.float64),
        )


@pytest.fixture(scope="module")
def measured(ssp_bare, obs, request):
    """``(seam, f64, f32)`` for one seam id, computed once and shared by every assertion."""
    seam = request.param
    model, path = seam.split("/")
    build_approx, fit_approx, zfac = _PATHS[path]
    flux, noise = _mock(ssp_bare, obs, model, zfac(), _SNR)

    # Thirteen seams, each building two models at two precisions, accumulate a large
    # number of distinct compiled programs in one process. Left alone this module
    # segfaults in XLA's CPU backend around the twelfth seam (observed 2026-08-31,
    # 48 GiB of system memory still free, so it is JIT-cache churn rather than RSS).
    # Dropping the caches between seams is what keeps the inventory runnable at this
    # width; it costs a recompile per seam, which the module was paying anyway.
    jax.clear_caches()
    gc.collect()

    kw = dict(model=model, zspec=zfac(), flux=flux, noise=noise)
    f64 = _gradients(
        ssp_bare,
        obs,
        build_approx=build_approx(),
        fit_approx=fit_approx,
        x64=True,
        dtype=jnp.float64,
        **kw,
    )
    f32 = _gradients(
        ssp_bare,
        obs,
        build_approx=build_approx(),
        fit_approx=fit_approx,
        x64=False,
        dtype=jnp.float32,
        **kw,
    )
    return seam, f64, f32


def _parametrize(fn):
    return pytest.mark.parametrize("measured", _SEAMS, indirect=True)(fn)


def _rel_norm(a, b):
    """Relative deviation in the 2-norm — the metric a sampler actually feels.

    Componentwise relative error is unbounded on a component that happens to pass
    through zero at the evaluation point, and this inventory has such points (the
    ``agn_log_lbol`` direction is nearly flat by construction, #2069). A gradient is
    consumed as a *vector* by every sampler and optimizer in the library, so the norm
    is the honest primary metric; the componentwise bar below keeps it from hiding a
    single badly-wrong direction inside a large norm.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _where():
    return f"backend={jax.default_backend()} devices={[str(d) for d in jax.devices()]}"


# --------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------


@_parametrize
def test_the_precision_of_each_arm_is_proven_on_the_gradient_dtype(measured):
    """Never on ``jax.config.jax_enable_x64``, which can disagree with the arrays.

    #1840: ``tengri/__init__.py`` re-enables x64 on import unless ``JAX_ENABLE_X64`` is
    already in the environment, and PR #2097 records a benchmark whose
    ``jax.config.update`` inside ``main()`` silently produced float64 and reported it as
    float32. The flag is a request; the dtype is the answer. Every other assertion in
    this module is meaningless if this one does not hold, so it is asserted first and
    per seam rather than assumed once.
    """
    seam, f64, f32 = measured
    for label in _POINTS:
        assert f64[label]["dtype"] == ["float64"], (
            f"the float64 arm of the {seam} seam produced {f64[label]['dtype']} gradients "
            f"at {label} ({_where()})"
        )
        assert f32[label]["dtype"] == ["float32"], (
            f"the float32 arm of the {seam} seam produced {f32[label]['dtype']} gradients "
            f"at {label} — the run is not in the precision it claims ({_where()})"
        )


@_parametrize
def test_float64_autodiff_agrees_with_float64_finite_differences(measured):
    """The instrument, per seam, before anything is concluded from it.

    If float64 autodiff did not reproduce float64 central differences at these points,
    the float64 column below would be an unvalidated reference and no float32 verdict
    could be attributed to precision rather than to the model.
    """
    seam, f64, _ = measured
    for label in _POINTS:
        rel = _rel_norm(f64[label]["grad"], f64[label]["fd"])
        assert rel < 5e-3, (
            f"float64 autodiff disagrees with float64 central differences by "
            f"{rel:.2e} in norm at {label} on the {seam} seam (names={f64['names']}, "
            f"auto={f64[label]['grad']}, fd={f64[label]['fd']}) — the reference is unsound. "
            f"The bar is 5e-3 rather than the 1e-4 of the fixed-z-only modules because a "
            f"free redshift makes the objective stiffer in that direction and a "
            f"second-order central difference at h={_FD_H} carries its own truncation "
            f"error there (measured 7.4e-04 on the AGN seam); it is still two orders "
            f"below any defect this file could be asked to adjudicate"
        )


@_parametrize
def test_float32_posterior_gradient_tracks_float64(measured):
    """The result this module exists for, stated per seam.

    1e-2 is the bar ``test_float32_grad_bolometric_seams.py`` sets on the model axis: two
    orders below the 0.30 defect #1436 recorded and an order above what the fixes
    achieve. Keeping the same bar is what makes a row here comparable to a row there.
    It is applied twice, to two different quantities, because they answer different
    questions and this inventory has a seam where they disagree:

    * **the 2-norm**, at 1e-2 — what a sampler or optimizer consumes. Worst measured
      across all 24 CPU cells and all 24 CUDA cells: **1.4e-03**, on
      ``panchromatic/lut_quad8_freez``.
    * **componentwise**, at 2e-2 relative plus a floor of 1e-3 of the largest component.
      The floor is there because a direction whose float64 gradient passes through zero
      would otherwise manufacture a failure. The **2e-2** is the honest number, not 1e-2:
      the worst single direction measured anywhere here is ``d/d redshift`` on
      ``panchromatic/lut_quad8_freez``, at **1.30e-02** (float64 −4.4488, float32
      −4.5065). That direction is stated rather than tightened away because on the *same*
      component the LUT's own bias against the exact projector is **1.98e-01** — fifteen
      times larger — so a bar drawn tightly enough to fail float32 there would be
      policing the smaller of the two errors. See
      ``test_the_lut_bias_dominates_the_float32_error_on_every_lut_seam``.
    """
    seam, f64, f32 = measured
    for label in _POINTS:
        g32, g64 = f32[label]["grad"], f64[label]["grad"]
        assert np.all(np.isfinite(g32)), (
            f"float32 posterior gradient is non-finite at {label} on the {seam} seam: "
            f"{g32} ({_where()})"
        )
        rel_norm = _rel_norm(g32, g64)
        assert rel_norm < 1e-2, (
            f"float32 posterior gradient disagrees with float64 by {rel_norm:.2e} in norm "
            f"at {label} on the {seam} seam (names={f64['names']}, f32={g32}, f64={g64}, "
            f"approx={f32['approx_state']}, {_where()})"
        )
        g64a = np.asarray(g64, dtype=np.float64)
        excess = np.abs(np.asarray(g32) - g64a) - (
            2e-2 * np.abs(g64a) + 1e-3 * np.max(np.abs(g64a))
        )
        assert excess.max() <= 0.0, (
            f"one component of the float32 posterior gradient is outside the "
            f"componentwise bar at {label} on the {seam} seam (names={f64['names']}, "
            f"f32={g32}, f64={g64}, excess={excess}, approx={f32['approx_state']}, "
            f"{_where()})"
        )


#: How far the cotangent boost may move a float64 gradient before it is a defect rather
#: than a reassociation, **relative**. Measured worst on CUDA: **1.4e-14** (118 ulp) on
#: ``stellar_dust/lut_freez`` at 0.5 sigma. Stated as a relative bound rather than in ulp
#: because ulp is not comparable across components of different magnitude — the same
#: 3.4e-12 absolute wobble reads 118 ulp on a gradient of 246 and 1 ulp on one of 2e4.
#: 1e-12 leaves ~70x head-room and still sits nine orders below the ~1e-3 scale at which
#: a real defect in the boost would show.
_BOOST_REL_BUDGET = 1e-12


@_parametrize
def test_the_cotangent_boost_does_not_move_the_float64_gradient(measured):
    """Bit-identical on CPU; **not** bit-identical on CUDA, by 1-18 ulp.

    ``DEFAULT_COTANGENT_BOOST`` is a power of two, so multiplying the objective by it and
    dividing the gradient back shifts binary exponents and leaves every mantissa bit
    alone — *in exact arithmetic*. PR #2100 asserted that as ``np.array_equal`` on three
    photometry seams, on CPU. It holds on CPU here too, on all thirteen fitting-path
    seams, LUT and free-redshift arithmetic included.

    **It does not hold on CUDA.** Measured 2026-08-31 on an RTX 3060 with
    ``JAX_DEFAULT_MATMUL_PRECISION=highest``: 9 of the 13 seams move, by up to
    **1.4e-14 relative** (118 ulp on the worst component, ``stellar_dust/lut_freez`` at
    0.5 sigma; 1-2 ulp on the fixed-*z* stellar+dust seams — the free-redshift LUT rows
    are the loose ones). The boost is exact, but the *graph* is not the same graph — the
    scaled
    objective gives XLA a different fusion and reduction problem, and GPU reductions are
    order-dependent — so a few ulp is what comes back. It is also not stable within a
    backend: a standalone script that takes only the two gradients reproduces exact
    equality on the same seam that fails here, where the module also takes finite
    differences in between. That is the signature of compile-time choice, not of
    arithmetic.

    So the assertion is split, deliberately, rather than loosened to the weaker of the
    two everywhere:

    * on **CPU**, exact equality, which is the claim PR #2100 made and the one that
      would catch the boost becoming a non-power-of-two;
    * on **any** backend, at most :data:`_BOOST_REL_BUDGET` relative, which catches a
      boost that actually changes the answer.

    **Nothing shipped is affected.** ``loss_scaled_grad`` is documented as unnecessary
    for a fit — a Gaussian likelihood's ``1/sigma**2`` supplies the same lift for free —
    and no fitting code path applies it, so no float64 fit result moves. What is
    corrected here is the *scope* of the bit-identity claim: it is CPU-specific, and
    "float64 is bit-identical" should not be carried onto CUDA without this caveat.
    """
    seam, f64, _ = measured
    assert float(np.log2(DEFAULT_COTANGENT_BOOST)).is_integer(), (
        f"DEFAULT_COTANGENT_BOOST = {DEFAULT_COTANGENT_BOOST!r} is not a power of two"
    )
    for label in _POINTS:
        boosted = np.asarray(f64[label]["boosted"], dtype=np.float64)
        plain = np.asarray(f64[label]["grad"], dtype=np.float64)
        rel = _rel_norm(boosted, plain)
        ulp = float(np.max(np.abs(boosted - plain) / np.spacing(np.abs(plain))))
        assert rel <= _BOOST_REL_BUDGET, (
            f"loss_scaled_grad moved the float64 posterior gradient by {rel:.2e} "
            f"relative ({ulp:.0f} ulp worst component) at {label} on the {seam} seam, "
            f"past the {_BOOST_REL_BUDGET:.0e} budget (names={f64['names']}, "
            f"boosted={boosted}, plain={plain}, {_where()}) — that is no longer a "
            f"reassociation"
        )
        if jax.default_backend() == "cpu":
            assert np.array_equal(boosted, plain), (
                f"loss_scaled_grad changed the float64 posterior gradient at {label} on "
                f"the {seam} seam ON CPU, where it has always been exact (names="
                f"{f64['names']}, boosted={boosted}, plain={plain}, {rel:.2e} relative, "
                f"{ulp:.0f} ulp). On CUDA a few ulp is expected and is bounded above; on "
                f"CPU it is a defect"
            )


# --------------------------------------------------------------------------------------
# The projector a fit actually runs
# --------------------------------------------------------------------------------------


def test_a_model_built_exact_is_fitted_under_the_lut_by_default(ssp_bare, obs):
    """``Fitter``'s default ``approx="auto"`` re-resolves the build-time knob.

    This is why the exact-projector rows above had to be taken with ``approx=None`` on
    the **Fitter**, and it is load-bearing for reading PR #2100: its likelihood-gradient
    measurements built with ``approx=None`` and then constructed ``Fitter(model, flux,
    noise)``, so the projector under test was ``WavePrecomp``, not the exact path its
    report names. The two resolutions are pinned here so the distinction cannot quietly
    disappear again.
    """
    with jax.enable_x64(True):
        sed = _build(ssp_bare, obs, "stellar_dust", None, Fixed(0.1))
        assert not sed.approx.wave_precomp, (
            f"SEDModel.build(approx=None) is no longer the exact path: {sed.approx}"
        )
        flux = jnp.asarray(
            np.asarray(
                sed.predict_photometry({"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}),
                dtype=np.float64,
            )
        )
        auto = Fitter(sed, flux, flux / _SNR).model.approx
        exact = Fitter(sed, flux, flux / _SNR, approx=None).model.approx
    assert auto.wave_precomp, (
        f"Fitter(approx='auto') no longer resolves a photometry fit to the LUT ({auto}); "
        "if that is deliberate, the exact/LUT rows in this module and the correction in "
        "docs/dev/float32-tier-b-boundary.md need re-reading"
    )
    assert not exact.wave_precomp, (
        f"Fitter(approx=None) no longer forces the exact projector ({exact}) — the "
        "exact-path rows in this module would then be LUT rows wearing another label"
    )


# --------------------------------------------------------------------------------------
# Which of the two errors is the big one
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lut_bias_freez(ssp_bare, obs, request):
    """Exact-f64, LUT-f64 and LUT-f32 gradients for one model at free redshift.

    Three arms so the two errors can be separated: ``lut64 - exact64`` is the
    approximation's own bias and ``lut32 - lut64`` is precision. Free redshift because
    that is where the ztable puts the LUT's largest error, and fixing *z* is what let
    every previous measurement of this path look tame.
    """
    model = request.param
    zfac = _PATHS["lut_freez"][2]
    flux, noise = _mock(ssp_bare, obs, model, zfac(), _SNR)
    kw = dict(model=model, zspec=zfac(), flux=flux, noise=noise)
    return model, {
        "exact64": _gradients(
            ssp_bare, obs, build_approx=None, fit_approx=None, x64=True, dtype=jnp.float64, **kw
        ),
        "lut64": _gradients(
            ssp_bare, obs, build_approx=_wp(), fit_approx="auto", x64=True, dtype=jnp.float64, **kw
        ),
        "lut32": _gradients(
            ssp_bare,
            obs,
            build_approx=_wp(),
            fit_approx="auto",
            x64=False,
            dtype=jnp.float32,
            **kw,
        ),
    }


@pytest.mark.parametrize("lut_bias_freez", ["stellar_dust", "panchromatic"], indirect=True)
def test_the_lut_bias_dominates_the_float32_error_on_every_lut_seam(lut_bias_freez):
    """The error that limits a default fit is the approximation's, not the precision's.

    This is the finding the tolerances above are calibrated against, so it is asserted
    rather than left in a report. On the default fit path (``approx="auto"`` →
    ``WavePrecomp``) with a free redshift, the ``d/d redshift`` component of the float64
    posterior gradient differs from the exact projector's by **6.5e-02** on stellar+dust
    and **1.8e-01** on the panchromatic model, at SNR 30 — while the float32-vs-float64
    error on the same component is 2.9e-04 and 6.6e-03.

    Two consequences, and the second is the one that gets forgotten:

    1. reaching for float64 to fix a WavePrecomp fit's gradient buys nothing; the fix is
       ``approx=None``, or a finer ``n_z``;
    2. a tolerance drawn tight enough to fail float32 on this path would be policing the
       smaller error while the larger one passes unremarked.
    """
    model, arms = lut_bias_freez
    names = arms["exact64"]["names"]
    assert "redshift" in names, f"{model} at free z has no redshift parameter: {names}"
    for label in _POINTS:
        gx = np.asarray(arms["exact64"][label]["grad"], dtype=np.float64)
        g64 = np.asarray(arms["lut64"][label]["grad"], dtype=np.float64)
        g32 = np.asarray(arms["lut32"][label]["grad"], dtype=np.float64)
        bias = np.abs(g64 - gx) / np.maximum(np.abs(gx), 1e-300)
        prec = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
        j = int(np.argmax(bias))
        assert bias[j] > prec[j], (
            f"on the {model} LUT free-z seam at {label} the float32 error "
            f"({prec[j]:.2e}) has overtaken the LUT's own bias ({bias[j]:.2e}) in the "
            f"{names[j]!r} direction (exact64={gx[j]:.6g}, lut64={g64[j]:.6g}, "
            f"lut32={g32[j]:.6g}, {_where()}) — the ordering this module's tolerances "
            f"assume no longer holds, so re-derive them rather than adjusting them"
        )


# --------------------------------------------------------------------------------------
# The SNR interaction (#1671)
# --------------------------------------------------------------------------------------

#: Two SNRs an order apart. #1671 measured ~5 % relative gradient error at SNR 30 and
#: ~50 % at SNR 300 on a 4-band reference model, and the mechanism — a *constant* forward
#: bias entering the posterior gradient multiplied by ``1/sigma``, hence by SNR —
#: predicts the ratio, not the absolute size, which is model-dependent.
_SNR_LOW, _SNR_HIGH = 30.0, 300.0


@pytest.fixture(scope="module")
def snr_interaction(ssp_bare, obs):
    """Exact-f64, LUT-f64 and LUT-f32 posterior gradients at two SNRs.

    Three arms, because the question has three terms and conflating any two of them is
    how #1671 gets misread as a precision problem:

    * ``exact64`` — the gradient the model actually implies;
    * ``lut64`` — what the LUT gives at full precision, so ``lut64 - exact64`` isolates
      the approximation's own bias, the term #1671 says grows with SNR;
    * ``lut32`` — the same LUT in pure float32, so ``lut32 - lut64`` isolates precision.
    """
    out = {}
    for snr in (_SNR_LOW, _SNR_HIGH):
        flux, noise = _mock(ssp_bare, obs, "stellar_dust", Fixed(0.1), snr)
        kw = dict(model="stellar_dust", zspec=Fixed(0.1), flux=flux, noise=noise)
        out[snr] = {
            "exact64": _gradients(
                ssp_bare,
                obs,
                build_approx=None,
                fit_approx=None,
                x64=True,
                dtype=jnp.float64,
                **kw,
            ),
            "lut64": _gradients(
                ssp_bare,
                obs,
                build_approx=_wp(),
                fit_approx="auto",
                x64=True,
                dtype=jnp.float64,
                **kw,
            ),
            "lut32": _gradients(
                ssp_bare,
                obs,
                build_approx=_wp(),
                fit_approx="auto",
                x64=False,
                dtype=jnp.float32,
                **kw,
            ),
        }
    return out


def _bias(rec, label="origin"):
    return _rel_norm(rec["lut64"][label]["grad"], rec["exact64"][label]["grad"])


def _f32_error(rec, label="origin"):
    return _rel_norm(rec["lut32"][label]["grad"], rec["lut64"][label]["grad"])


def test_the_lut_gradient_bias_is_amplified_by_snr(snr_interaction):
    """#1671 on the posterior gradient, reproduced here rather than cited.

    The LUT's forward photometry bias is *constant* in SNR, so no forward check can see
    it; the posterior gradient carries a factor ``1/sigma``, so the same bias arrives
    multiplied by SNR. A tenfold SNR must therefore buy a materially larger gradient
    error. The bar is 3x rather than 10x because the residual at the evaluation point is
    not purely the bias — the noise realization differs between the two mocks.
    """
    lo, hi = snr_interaction[_SNR_LOW], snr_interaction[_SNR_HIGH]
    bias_lo, bias_hi = _bias(lo), _bias(hi)
    assert bias_lo > 0.0, "the LUT reproduced the exact gradient exactly, so this test is inert"
    assert bias_hi / bias_lo > 3.0, (
        f"the WavePrecomp gradient bias did not grow with SNR: {bias_lo:.2e} at SNR "
        f"{_SNR_LOW:.0f} vs {bias_hi:.2e} at SNR {_SNR_HIGH:.0f}. Either #1671's "
        f"amplification has been fixed — in which case say so and delete this — or the "
        f"fixture stopped being sensitive to it"
    )


def test_float32_is_amplified_by_snr_exactly_as_the_lut_bias_is(snr_interaction):
    """float32 does not make #1671 worse — it *is* #1671, with a smaller coefficient.

    The measured answer, and it is not the intuitive one. The float32 error is **not**
    independent of SNR: it grows linearly, 5.2e-04 at SNR 30 to 5.6e-03 at SNR 300 on
    this fixture — the same tenfold as the LUT bias over the same tenfold in SNR. That is
    the same mechanism, not a coincidence. A *relative* forward-model error ``eps``,
    constant in SNR, reaches the posterior gradient multiplied by ``1/sigma`` and hence
    by SNR. ``WavePrecomp``'s ``eps`` is its LUT bias (~1e-3); float32's is its forward
    rounding (~1e-6). Both gradient errors therefore scale identically and **their ratio
    is what stays constant**, which is what this asserts.

    Consequences, and the second is the one that gets forgotten:

    1. float32 is not what limits a ``WavePrecomp`` fit at any SNR in this range — the
       LUT leads by more than an order of magnitude throughout;
    2. **float64 is not the fix for a high-SNR LUT fit** either. The bias survives at
       full precision; ``approx=None`` (or a finer LUT) is what removes it.

    float32 does have an SNR ceiling of its own, and it follows directly: extrapolating
    the linear scaling, the 1e-2 bar this module holds is crossed somewhere between SNR
    300 and SNR 1000 (measured: 2.3e-02 at SNR 1000 on the same arm). Nothing here
    licenses float32 at SNR 1000.
    """
    lo, hi = snr_interaction[_SNR_LOW], snr_interaction[_SNR_HIGH]
    f32_lo, f32_hi = _f32_error(lo), _f32_error(hi)
    bias_lo, bias_hi = _bias(lo), _bias(hi)
    assert f32_hi < 1e-2, (
        f"the float32 posterior gradient error at SNR {_SNR_HIGH:.0f} is {f32_hi:.2e}, "
        f"outside the 1e-2 bar the rest of this module holds ({_where()})"
    )
    assert f32_hi < 0.1 * bias_hi, (
        f"at SNR {_SNR_HIGH:.0f} the float32 error ({f32_hi:.2e}) is no longer small "
        f"beside the LUT's own bias ({bias_hi:.2e}), so precision has become a comparable "
        f"term and the report's conclusion needs re-measuring"
    )
    # The ratio, not either error on its own. Both grow ~linearly in SNR, so a *changing*
    # ratio would mean the shared mechanism above is wrong. A factor of 3 either way is
    # loose enough for one mock's noise realization and tight enough to catch a genuine
    # divergence in scaling.
    ratio_lo = bias_lo / max(f32_lo, 1e-300)
    ratio_hi = bias_hi / max(f32_hi, 1e-300)
    assert 1 / 3 < ratio_hi / ratio_lo < 3, (
        f"the LUT-bias-to-float32 ratio moved with SNR: {ratio_lo:.1f}x at SNR "
        f"{_SNR_LOW:.0f} vs {ratio_hi:.1f}x at SNR {_SNR_HIGH:.0f} (float32 "
        f"{f32_lo:.2e} -> {f32_hi:.2e}, LUT bias {bias_lo:.2e} -> {bias_hi:.2e}). Both "
        f"errors are supposed to be a constant relative forward error times SNR, so a "
        f"moving ratio means one of them is not"
    )

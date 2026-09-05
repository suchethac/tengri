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

from tengri import DEFAULT, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
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
    "all_params": Fixed(DEFAULT),
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}
_DUST_FIXED = dict(_DUST_FREE, tau_diff=0.3)

_AGN = {
    "type": "composable",
    "all_params": Fixed(DEFAULT),
    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
    "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
    "norm": "cigale_joint",
    "log_lbol": Fixed(10.5),  # #2069: pinned to break the flat direction
    "fracAGN": 0.1,
}

_MODELS = {
    "stellar_dust": dict(dust_attenuation=_DUST_FREE),
    "dust_ir": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
    ),
    "agn": dict(dust_attenuation=_DUST_FIXED, agn=_AGN),
    "panchromatic": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
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
            "all_params": Fixed(DEFAULT),
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
        # Finite AND non-zero, together. Either alone admits exactly the value that
        # breaks the other: PR #2100's guard asserted *finite* and zero is finite;
        # a *non-zero* guard admits NaN. See Finding 8 of
        # bench/reports/2026-09-05_float32_spectroscopy_lines.md.
        assert np.all(np.isfinite(g32)), (
            f"float32 posterior gradient is non-finite at {label} on the {seam} seam: "
            f"{g32} ({_where()})"
        )
        assert np.any(g32 != 0.0), (
            f"float32 posterior gradient is identically zero at {label} on the {seam} "
            f"seam ({g32}) — finite, and useless to every sampler ({_where()})"
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


# ======================================================================================
# The ``data_type`` axis — spectroscopy and the two emission-line channels
# ======================================================================================
#
# PR #2104 closed the projector, free-redshift and CUDA axes **on photometry** and named
# these two as the first things a real fit crosses that it had not reached:
#
#     **Spectroscopy.** ``predict_spectrum`` applies the same projection, and
#     ``SpectrumPrecomp`` is the LUT's spectroscopy sibling with its own measured
#     ~1-sigma posterior shift (#1688). [...] **Emission-line fluxes.**
#     ``line_measurement.py`` applies its own combined ``log10_conv -
#     log10_four_pi_dl2`` offset, and ``FeaturePrecomp`` serves the line channel from a
#     table. Neither is measured here — and note #1770's lesson that a
#     photometry-surface measurement says nothing about the line channel.
#
# #1770's lesson is why these are their own rows rather than an extrapolation from the
# photometry rows above: that issue cost a measured 4.77x because a photometry-surface
# FLOP count was used to answer a line-channel question. A green photometry *precision*
# result is subject to the same rule — the channels share no arithmetic.
#
# The line channel is **two** seams, not one, because ``loss_functions.py`` picks
# between two operators on ``model._has_line_catalog()``:
#
# ``lines_cue``   a backend publishing a discrete catalog (Cue) → ``predict_line_fluxes``.
# ``lines_meas``  a baked-in (wNE) SSP publishes no catalog → ``measure_line_fluxes``,
#                 the ``line_measurement.py`` window path that applies the combined
#                 ``log10_conv - log10_four_pi_dl2`` offset through ``apply_log10_scale``.
#
# That offset is the hazard the channel exists to test. A line luminosity is ~1e40 erg/s
# against a float32 max of 3.4e38 and ``4*pi*d_L**2`` is ~1e57, so #1859 records that the
# naive linear spelling was ``inf/inf`` — ``nan`` at every redshift — and #1415 then had
# to hold the factorization peak under ``stop_gradient`` because in float32 the two
# autodiff paths do not cancel, which was gradients exactly **2x** too large.

#: Baked-in nebular (wNE) grid: the lines live *inside* the SSP templates, which is what
#: makes ``_has_line_catalog()`` False and routes the loss through ``line_measurement``.
_SSP_WNE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

#: Optical spectrum: 256 pixels over 4000-9000 A — wide enough to carry the Balmer
#: decrement and the 4000-A break, small enough that a seam builds in seconds.
_SPEC_WAVE = np.linspace(4000.0, 9000.0, 256)

#: Four strong optical lines, rest-frame vacuum [A].
_LINE_NAMES = ("Hbeta", "OIII_5007", "Halpha", "NII_6584")
_LINE_WAVES = np.array([4862.68, 5008.24, 6564.61, 6585.28])

#: A line channel rides *alongside* photometry: ``Fitter`` requires ``data``/``noise``,
#: so a lines-only ``Observation`` cannot be fitted through that constructor at all
#: (it reports ``data_type="spectroscopy"`` and then fails inside ``predict_photometry``).
#: These two bands are the continuum backbone the line term rides on.
_PHOT2 = ["sdss_g", "sdss_r"]

_CHANNEL_MODELS = dict(_MODELS)
_CHANNEL_MODELS["neb_cue"] = dict(
    dust_attenuation=_DUST_FREE, neb={"type": "cue", "all_params": Fixed(DEFAULT)}
)
#: Baked-in: no nebular *component* at all, the lines are in the templates.
_CHANNEL_MODELS["bakedin"] = dict(dust_attenuation=_DUST_FREE, neb={"type": "none"})

_CHANNEL_SSP = {"bakedin": _SSP_WNE}


def _channel_obs(channel, line_data=None):
    """The ``Observation`` for one channel id."""
    from tengri.observation.spectroscopy import Spectroscopy

    if channel == "spec":
        return Observation(spectroscopy=Spectroscopy(wave_obs=jnp.asarray(_SPEC_WAVE)))
    if channel == "phot2":
        return Observation(photometry=Photometry.from_names(_PHOT2))
    if channel in ("lines_cue", "lines_meas"):
        return Observation(photometry=Photometry.from_names(_PHOT2), line_fluxes=line_data)
    raise ValueError(f"unknown channel {channel!r}")


def _channel_approx(channel, kind):
    """Build-time ``approx=`` for one channel and projector ``kind``.

    ``"auto"`` is deliberately not spelled here: it is what ``Fitter(approx="auto")``
    resolves to, and PR #2104's Finding 0 is that the *fitter's* knob is the one that
    decides. A photometry channel resolves to ``WavePrecomp``, a spectroscopy channel to
    ``SpectrumPrecomp``, and either gains ``FeaturePrecomp`` when lines are fit.
    """
    from tengri import SpectrumPrecomp

    if kind == "exact":
        return None
    if channel == "spec":
        return SpectrumPrecomp(n_z=_N_Z, z_min=_Z_LO, z_max=_Z_HI)
    return _wp()


def _channel_build(ssp, channel, model, kind, zspec, line_data=None):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_channel_obs(channel, line_data),
        approx=_channel_approx(channel, kind),
        **_base(zspec),
        **_CHANNEL_MODELS[model],
    )


#: Per-line 1-sigma is set by the *strongest* line, floored, rather than per line.
#: ``sigma_i = |pred_i| / snr`` — the spelling the repo's line fixtures use — hands a
#: line the model predicts near zero (the baked-in NII_6584 comes out ~3e-18 against
#: Halpha's ~1e-15) a ~1e-19 error bar, and ``sigma**2`` then **underflows float32**
#: (1e-38 is the smallest normal). That is a property of the fixture, not of the
#: arithmetic under test, so the realistic flux-limited convention is used here.
_LINE_SIGMA_FLOOR = 0.05


def _line_mock(ssp, channel, model, zspec, snr, seed=0):
    """``(phot_flux, phot_noise, LineFluxData, truth)`` for a line-channel seam.

    The observed line fluxes are drawn through **whichever operator the loss will
    use** — ``predict_line_fluxes`` when the backend publishes a discrete catalog,
    ``measure_line_fluxes`` when it does not — so the mock is self-consistent with the
    likelihood rather than merely plausible.
    """
    from tengri.observation.line_flux_data import LineFluxData
    from tengri.observation.line_measurement import default_line_defs

    key = jax.random.PRNGKey(seed)
    with jax.enable_x64(True):
        placeholder = LineFluxData(
            names=_LINE_NAMES,
            fluxes=np.ones(len(_LINE_NAMES)),
            errors=np.ones(len(_LINE_NAMES)),
            wavelengths=_LINE_WAVES,
        )
        sed = _channel_build(ssp, channel, model, "exact", zspec, placeholder)
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        if sed._has_line_catalog():
            pred = np.asarray(
                sed.predict_line_fluxes(truth, target_wavelengths=jnp.asarray(_LINE_WAVES)),
                dtype=np.float64,
            )
        else:
            pred = np.asarray(
                sed.measure_line_fluxes(
                    truth, default_line_defs(_LINE_WAVES, _LINE_NAMES), approx=False
                ),
                dtype=np.float64,
            )
        a = np.abs(pred)
        lsig = np.maximum(a, _LINE_SIGMA_FLOOR * np.max(a)) / snr
        k1, k2 = jax.random.split(key)
        draw = np.asarray(
            jax.random.normal(k1, (len(_LINE_NAMES),), dtype=jnp.float64), dtype=np.float64
        )
        line_data = LineFluxData(
            names=_LINE_NAMES,
            fluxes=pred + draw * lsig,
            errors=lsig,
            wavelengths=_LINE_WAVES,
        )
        sed2 = _channel_build(ssp, channel, model, "exact", zspec, line_data)
        mock = sed2.mock(truth, snr=snr, key=k2)
        return (
            np.asarray(mock.flux_obs, dtype=np.float64),
            np.asarray(mock.noise, dtype=np.float64),
            line_data,
            truth,
        )


def _channel_mock(ssp, channel, model, zspec, snr):
    """``(flux, noise, line_data)`` from the **exact** projector at float64."""
    if channel in ("lines_cue", "lines_meas"):
        flux, noise, line_data, _ = _line_mock(ssp, channel, model, zspec, snr)
        return flux, noise, line_data
    with jax.enable_x64(True):
        sed = _channel_build(ssp, channel, model, "exact", zspec)
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        mock = (
            sed.mock_spectrum(truth, jnp.asarray(_SPEC_WAVE), snr=snr, key=jax.random.PRNGKey(0))
            if channel == "spec"
            else sed.mock(truth, snr=snr, key=jax.random.PRNGKey(0))
        )
        return (
            np.asarray(mock.flux_obs, dtype=np.float64),
            np.asarray(mock.noise, dtype=np.float64),
            None,
        )


def _channel_gradients(
    ssp, channel, model, kind, fit_approx, zspec, flux, noise, line_data, *, x64, dtype
):
    """One precision's gradients for one channel seam, from one model build."""
    with jax.enable_x64(x64):
        sed = _channel_build(ssp, channel, model, kind, zspec, line_data)
        fitter = Fitter(
            sed,
            jnp.asarray(flux, dtype=dtype),
            jnp.asarray(noise, dtype=dtype),
            approx=fit_approx,
        )
        ctx = InferenceContext.from_target(fitter)
        data_args = ctx.data_args
        names = sorted(ctx.initial_params(jax.random.PRNGKey(1)))

        def nlp(vals):
            return ctx.neg_log_posterior_fn({k: vals[i] for i, k in enumerate(names)}, data_args)

        out = {
            "names": names,
            "approx_state": str(getattr(ctx.fitter.model, "approx", None)),
            "data_type": str(getattr(fitter, "data_type", None)),
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
                "grad": np.array([float(np.asarray(x)) for x in g]),
                "fd": np.array(fd),
                "dtype": sorted({str(np.asarray(x).dtype) for x in g}),
            }
        return out


def _env() -> str:
    """Everything about this environment that could change a float32 verdict."""
    return (
        f"backend={jax.default_backend()} devices={[str(d) for d in jax.devices()]} "
        f"jax={jax.__version__} x64={jax.config.jax_enable_x64} "
        f"matmul={jax.config.jax_default_matmul_precision}"
    )


def _lut_forward_finite(ssp, channel, model, zspec, line_data=None):
    """``(is_finite, approx_state)`` for this channel's LUT forward at **float64**.

    A precision comparison is a statement about two numbers. If the *float64* forward
    under the LUT is already non-finite in this environment then there is no number to
    compare against, and any float32-vs-float64 verdict taken here would be measuring
    a NaN rather than a rounding error. This asks the question directly and answers it
    from the **returned array**, which is the same standard #1840 imposes on precision:
    read the array, never a flag, a version or a platform string.

    Measured 2026-09-05: finite on this box (jax 0.11.0, Ryzen 9 5900X, CPU) and
    **non-finite on the GitHub runner** (jax 0.11.1, ubuntu-24.04, also CPU) — see
    run 33958554553, where ``spec/*/auto_*`` raised ``Max |prediction| = nan`` from
    ``_check_channel_scales`` (#1495) on six seams and ``spec/lut`` returned
    ``[nan nan]`` from the boosted unweighted gradient.
    """
    with jax.enable_x64(True):
        sed = _channel_build(ssp, channel, model, "lut", zspec, line_data)
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        arr = np.asarray(
            sed.predict_spectrum(truth) if channel == "spec" else sed.predict_photometry(truth)
        )
        return bool(np.all(np.isfinite(arr))), str(getattr(sed, "approx", None))


def _skip_if_lut_forward_is_broken(ssp, channel, model, zspec, line_data=None):
    """Refuse to take a precision verdict on a projector that is already NaN here.

    This is **not** a relaxation of any bar. The float32-vs-float64 comparison is
    simply undefined when the float64 arm is NaN, and reporting it as agreement or
    disagreement would be a false statement either way. The skip is loud and names the
    environment, so an environment where the LUT is broken is visible as a broken LUT
    rather than as a passing precision test.
    """
    finite, approx_state = _lut_forward_finite(ssp, channel, model, zspec, line_data)
    if not finite:
        pytest.skip(
            f"the {channel} LUT forward is NON-FINITE at float64 in this environment "
            f"({approx_state}, {_env()}), so there is no float64 reference to compare "
            f"float32 against on this seam. This is an environment-dependent defect in "
            f"the LUT itself, not a precision result: the same forward is finite on the "
            f"2026-09-05 reference box (jax 0.11.0, CPU) and non-finite on the GitHub "
            f"runner (jax 0.11.1, CPU). See Finding 8 of "
            f"bench/reports/2026-09-05_float32_spectroscopy_lines.md."
        )


#: ``channel/model/path``. Deliberately narrow: ``bakedin`` and ``stellar_dust`` build in
#: seconds, and the Cue seams that would widen it are the ones the module cannot fit at
#: all in float32 (see ``test_the_discrete_line_catalog_operator_survives_float32``).
_CHANNEL_SEAMS = [
    "spec/stellar_dust/exact_fixedz",
    "spec/stellar_dust/auto_fixedz",
    "spec/stellar_dust/auto_freez",
    "lines_meas/bakedin/exact_fixedz",
    "lines_meas/bakedin/exact_freez",
    # ``lines_meas/bakedin/auto_*`` is deliberately absent: the FeaturePrecomp-served
    # line path is non-finite in float32 and cannot produce a gradient at all. It is
    # pinned by ``test_the_feature_precomp_line_path_survives_float32`` instead, so the
    # gap is recorded as a defect rather than hidden by dropping the row.
    # The photometry-only control on the *same* model and bands: the line rows above
    # differ from it only by the line term, which is what makes the line channel's
    # contribution attributable rather than merely present (#1770).
    "phot2/bakedin/auto_fixedz",
]

_CHANNEL_PATHS = {
    "exact_fixedz": ("exact", None, lambda: Fixed(0.1)),
    "auto_fixedz": ("exact", "auto", lambda: Fixed(0.1)),
    "exact_freez": ("exact", None, lambda: Uniform(_Z_LO, _Z_HI)),
    "auto_freez": ("exact", "auto", lambda: Uniform(_Z_LO, _Z_HI)),
}


@pytest.fixture(scope="module")
def channel_measured(ssp_bare, request):
    """``(seam, f64, f32)`` for one ``channel/model/path`` seam."""
    seam = request.param
    channel, model, path = seam.split("/")
    kind, fit_approx, zfac = _CHANNEL_PATHS[path]

    ssp = ssp_bare
    if model in _CHANNEL_SSP:
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        ssp = load_ssp_data(_CHANNEL_SSP[model])

    flux, noise, line_data = _channel_mock(ssp, channel, model, zfac(), _SNR)

    # Before measuring precision, check that this environment can produce a float64
    # reference at all on the LUT arms. See _skip_if_lut_forward_is_broken.
    if fit_approx == "auto":
        _skip_if_lut_forward_is_broken(ssp, channel, model, zfac(), line_data)

    # Same reason as the photometry fixture above: this module builds many distinct
    # compiled programs in one process and XLA's CPU backend segfaults without this.
    jax.clear_caches()
    gc.collect()

    kw = dict(
        channel=channel,
        model=model,
        kind=kind,
        fit_approx=fit_approx,
        zspec=zfac(),
        flux=flux,
        noise=noise,
        line_data=line_data,
    )
    f64 = _channel_gradients(ssp, x64=True, dtype=jnp.float64, **kw)
    f32 = _channel_gradients(ssp, x64=False, dtype=jnp.float32, **kw)
    return seam, f64, f32


def _parametrize_channel(fn):
    return pytest.mark.parametrize("channel_measured", _CHANNEL_SEAMS, indirect=True)(fn)


@_parametrize_channel
def test_the_precision_of_each_channel_arm_is_proven_on_the_gradient_dtype(channel_measured):
    """As above, and for the same reason: a config flag can disagree with the arrays.

    Asserted first on this axis too, because every other assertion below is void if the
    "float32" arm silently produced float64 (#1840, PR #2097 Finding 0).
    """
    seam, f64, f32 = channel_measured
    for label in _POINTS:
        assert f64[label]["dtype"] == ["float64"], (
            f"the float64 arm of {seam} produced {f64[label]['dtype']} at {label} ({_where()})"
        )
        assert f32[label]["dtype"] == ["float32"], (
            f"the float32 arm of {seam} produced {f32[label]['dtype']} at {label} — "
            f"every precision claim in this module is void for this seam ({_where()})"
        )


#: Per-seam budget for the float64-autodiff-vs-float64-finite-difference soundness
#: check. The default is the 5e-2 the photometry axis uses.
#:
#: **Free redshift on a spectrum is the exception, and it is truncation error rather
#: than a defect.** Moving ``z`` slides every spectral feature across a fixed 256-pixel
#: grid, so the objective's curvature along that one direction is enormous compared with
#: the smooth continuum directions, and a central difference at ``h = 1e-4`` in
#: standardized units carries an O(h**2 f''') error to match: measured 1.3e-01 at
#: SNR 30 and 2.2e-01 at SNR 300. Three things say it is the difference quotient that is
#: wrong and not the autodiff graph — the exact and LUT projectors agree with each other
#: to the digit (1.29e-01 vs 1.30e-01), the same seam at *fixed* z checks out at 2.0e-05,
#: and the residual shrinks with ``h`` (``test_the_spectroscopic_redshift_finite_
#: difference_is_truncation_not_defect``). Photometry never shows it because a bandpass
#: integrates the feature motion away.
#: Measured 1.30e-01 on ``spec/stellar_dust/auto_freez`` at SNR 30. The line channel's
#: own free-redshift seam needs no exception (1.86e-05), which is itself the evidence
#: that this is about a *pixel grid*, not about freeing ``z``.
_CHANNEL_FD_BUDGET = {
    "spec/stellar_dust/auto_freez": 3.0e-1,
}
_CHANNEL_FD_DEFAULT = 5e-2


@_parametrize_channel
def test_float64_autodiff_agrees_with_float64_finite_differences_on_every_channel(
    channel_measured,
):
    """The reference is checked before it is used as one, per seam.

    A central difference at ``h = 1e-4`` in standardized units carries its own O(h**2)
    truncation error, so this is a soundness check on the autodiff graph, not a
    precision measurement. See ``_CHANNEL_FD_BUDGET`` for the one direction where that
    truncation error is large and why it is not a defect.
    """
    seam, f64, _ = channel_measured
    budget = _CHANNEL_FD_BUDGET.get(seam, _CHANNEL_FD_DEFAULT)
    for label in _POINTS:
        rel = _rel_norm(f64[label]["grad"], f64[label]["fd"])
        assert rel < budget, (
            f"float64 autodiff disagrees with float64 central differences by {rel:.2e} "
            f"(budget {budget:.1e}) at {label} on the {seam} seam — the reference itself "
            f"is suspect, so no float32 verdict can be taken from it "
            f"(autodiff={f64[label]['grad']}, fd={f64[label]['fd']}, {_where()})"
        )


@_parametrize_channel
def test_float32_posterior_gradient_tracks_float64_on_every_channel(channel_measured):
    """The result this section exists for, stated per channel seam.

    The bar is **2e-2 in norm**, twice the 1e-2 the photometry axis uses, and the
    loosening is the finding rather than a convenience. Measured at SNR 30 on CPU:

    ==================================  ==========  ==================================
    seam                                f32 vs f64  note
    ==================================  ==========  ==================================
    ``spec/stellar_dust/exact_fixedz``  5.2e-03     the worst cell on this axis
    ``spec/stellar_dust/auto_fixedz``   2.9e-03
    ``spec/stellar_dust/auto_freez``    4.3e-03     (7.9e-04 on CUDA — backend-split)
    ``lines_meas/bakedin/exact_fixedz`` 1.4e-04
    ``lines_meas/bakedin/exact_freez``  4.5e-04
    ``phot2/bakedin/auto_fixedz``       2.0e-04     the photometry control
    ==================================  ==========  ==================================

    **Spectroscopy is ~26x the photometry control on the same box and seed**, and PR
    #2104's photometry worst across all 48 of its cells was 1.4e-03. The line channel,
    where it runs at all, is *not* worse than photometry — 1.4e-04 against the control's
    2.0e-04 — so the bar is set by spectroscopy, and the headroom is what a free
    redshift plus a higher SNR needs.

    The cause on spectroscopy is cancellation, not magnitude: a spectrum pixel residual
    is a small difference on a steep continuum, while a photometric band integrates over
    a bandpass and cancels far less. That is why #1770's rule applies to precision
    exactly as it applies to FLOPs — this is a different objective, not the photometry
    one at a different scale.

    Note the bar is **not** what makes the line channel interesting; two of its three
    operators do not produce a number at all. See
    ``test_the_discrete_line_catalog_operator_survives_float32`` and
    ``test_the_feature_precomp_line_path_survives_float32``.
    """
    seam, f64, f32 = channel_measured
    for label in _POINTS:
        g32, g64 = f32[label]["grad"], f64[label]["grad"]
        assert np.all(np.isfinite(g32)), (
            f"float32 posterior gradient is non-finite at {label} on the {seam} seam: "
            f"{g32} ({_where()})"
        )
        # Zero is finite. PR #2100 found the bare observable gradient identically zero
        # on both backends while a guard that pinned it *finite* stayed green, so the
        # non-zero assertion is made explicitly wherever a gradient is checked.
        assert np.any(g32 != 0.0), (
            f"float32 posterior gradient is identically zero at {label} on the {seam} "
            f"seam — finite, and useless to every sampler ({_where()})"
        )
        rel_norm = _rel_norm(g32, g64)
        assert rel_norm < 2e-2, (
            f"float32 posterior gradient disagrees with float64 by {rel_norm:.2e} in "
            f"norm at {label} on the {seam} seam (names={f64['names']}, f32={g32}, "
            f"f64={g64}, approx={f32['approx_state']}, "
            f"data_type={f32['data_type']}, {_where()})"
        )


def test_the_discrete_line_catalog_operator_survives_float32(ssp_bare):
    """``predict_line_fluxes`` returns ``nan`` in float32 — #1859's fix is one-sided.

    ``loss_functions.py`` routes a line-flux fit to one of two operators on
    ``model._has_line_catalog()``:

    * **no catalog** (baked-in / wNE) → ``measure_line_fluxes``, the
      ``line_measurement.py`` window path. #1859 grouped the ``erg/s`` line luminosity
      (~1e40, against a float32 max of 3.4e38) and ``4*pi*d_L**2`` (~1e57) into a single
      ~-45 dex exponent applied to an O(1e28) mean, so **no intermediate is
      materialized**. Measured here: finite in float32 and tracking float64 to ~2e-04.
    * **a catalog** (Cue and every other line-publishing backend) →
      ``predict_line_fluxes``, which does materialize the luminosity. In float32 that
      overflows to ``inf`` and the subsequent division by the distance gives ``nan`` —
      on every line, at every redshift, which is exactly the failure #1859 describes.

    Measured on this tree, stellar+dust+Cue at z = 0.1, standardized origin::

        [cue/f64] L_Halpha = 4.60e+40
                  predict_line_fluxes = [4.81e-16 5.88e-16 1.70e-15 5.31e-16]
        [cue/f32] L_Halpha = inf
                  predict_line_fluxes = [nan nan nan nan]

    So a float32 fit with a line-publishing nebular backend cannot run. It does not run
    *wrongly* — ``_check_channel_scales`` (#1495) raises at construction on the
    non-finite log-prob, so this is a loud failure, not a silent one — but the channel
    is unavailable in float32 and the report says so.

    Marked ``xfail(strict=True)`` because finite is the behavior that is wanted, not
    the behavior that exists: extending #1859's grouping to this operator will flip
    this test to XPASS and the report's Finding 2 will then need re-taking.
    """
    # PR #2104's hazard, met again: this module accumulates a large number of
    # distinct compiled programs, and XLA's CPU backend dies part-way through the
    # inventory with tens of GiB still free. Observed 2026-09-05 killing the process
    # outright at the 19th test of this file. Dropping the caches before each heavy
    # standalone build is what keeps the module runnable in one process.
    jax.clear_caches()
    gc.collect()

    with jax.enable_x64(False):
        sed = SEDModel.build(
            ssp_data=ssp_bare,
            observation=Observation(photometry=Photometry.from_names(_PHOT2)),
            approx=None,
            **_base(Fixed(0.1)),
            **_CHANNEL_MODELS["neb_cue"],
        )
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        fluxes = np.asarray(
            sed.predict_line_fluxes(truth, target_wavelengths=jnp.asarray(_LINE_WAVES))
        )
    assert np.any(np.asarray(fluxes) != 0.0), (
        f"predict_line_fluxes is identically zero in float32: {fluxes} — finite is not "
        f"enough, a zeroed line catalog is as unusable as a NaN one ({_where()})"
    )
    assert np.all(np.isfinite(fluxes)), (
        f"predict_line_fluxes is non-finite in float32: {fluxes}. The line luminosity "
        f"overflows float32 before the distance division (#1859 fixed this on "
        f"measure_line_fluxes only). {_where()}"
    )


# Applied after definition so the docstring above reads as the statement of intent
# rather than as a decorator argument.
test_the_discrete_line_catalog_operator_survives_float32 = pytest.mark.xfail(
    strict=True,
    reason=(
        "#1859's log-offset grouping is applied in line_measurement.py only; "
        "predict_line_fluxes still materializes a ~1e40 erg/s luminosity, which "
        "overflows float32 to inf and then to nan. Flipping to XPASS means the fix "
        "was extended — re-take bench/reports/2026-09-05_float32_spectroscopy_lines.md."
    ),
)(test_the_discrete_line_catalog_operator_survives_float32)


# --------------------------------------------------------------------------------------
# The unweighted-observable path, under the LUT this time
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unweighted(ssp_bare, request):
    """``grad(sum(predict_*))`` at both precisions for one ``channel/kind``.

    The **unweighted** observable — no ``1/sigma**2`` from a Gaussian likelihood, so the
    cotangent chain runs at the raw scale of the observable (~1e-28 for a photometric
    F_nu) instead of being lifted ~1e32 by the residual weighting. That is the regime
    PR #2100 found the ``2**70`` cotangent boost necessary in, and it measured it on the
    **exact** projector only; PR #2104 left re-taking it under the LUT as the one item
    from the original list still open.
    """
    channel, kind = request.param.split("/")
    from tengri.utils.scale import loss_scaled_grad

    if kind == "lut":
        _skip_if_lut_forward_is_broken(ssp_bare, channel, "stellar_dust", Fixed(0.1))

    out = {}
    for x64, dtype, tag in ((True, jnp.float64, "f64"), (False, jnp.float32, "f32")):
        with jax.enable_x64(x64):
            sed = _channel_build(ssp_bare, channel, "stellar_dust", kind, Fixed(0.1))
            names = sorted(sed.spec.free_params)
            truth = {
                n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0))) for n in names
            }

            def observable(vals, _sed=sed, _names=names, _truth=truth, _channel=channel):
                p = dict(_truth)
                p.update({k: vals[i] for i, k in enumerate(_names)})
                # The LEAN surfaces, called with NO explicit wavelength grid: wave_obs=None
                # falls back to observation.spectroscopy.wave_obs, which is the grid
                # SpectrumPrecomp is built for. Passing the same array explicitly bypasses
                # the LUT — measured, and caught by
                # test_the_lut_actually_reaches_the_unweighted_observable_graph.
                # ``model.predict(p).photometry()`` does NOT take the
                # LUT — an exact and a wave_precomp model give bit-identical float64
                # gradients through the rich accessor, #1748's signature of a config that
                # never reaches the graph — so measuring there would make the "LUT" arm
                # the exact arm under another name.
                return jnp.sum(
                    _sed.predict_spectrum(p) if _channel == "spec" else _sed.predict_photometry(p)
                )

            point = [jnp.asarray(float(truth[k]), dtype=dtype) for k in names]
            out[tag] = {
                "names": names,
                "grad": np.array([float(np.asarray(x)) for x in jax.grad(observable)(point)]),
                "boosted": np.array(
                    [float(np.asarray(x)) for x in loss_scaled_grad(observable)(point)]
                ),
                "dtype": sorted({str(np.asarray(x).dtype) for x in jax.grad(observable)(point)}),
                "approx_state": str(getattr(sed, "approx", None)),
            }
    jax.clear_caches()
    gc.collect()
    return request.param, out


#: Both projectors on both channels. The LUT arms are the ones PR #2100 never took.
_UNWEIGHTED_SEAMS = ["phot2/exact", "phot2/lut", "spec/exact", "spec/lut"]


@pytest.mark.parametrize("unweighted", _UNWEIGHTED_SEAMS, indirect=True)
def test_the_boosted_unweighted_observable_gradient_is_nonzero_in_float32(unweighted):
    """The path that works, asserted against **zero** first and magnitude second.

    The bare ``sum(predict_photometry)`` gradient was **identically zero** in float32 on
    both CPU and GPU, silently, and ``test_inference_grad_float32.py`` pinned it
    *finite*. Zero is finite, so that guard could never have caught it. Measured again
    here, it is still zero — on both channels and on **both** projectors, which is the
    part PR #2100 never took (see
    ``test_the_bare_unweighted_observable_gradient_is_nonzero_in_float32``).

    What this test pins is the remedy: with ``loss_scaled_grad``'s ``2**70`` cotangent
    boost the same gradient is finite, non-zero and tracks float64. That is the only
    working spelling of an unweighted float32 gradient in this library, so it is the one
    that must not regress. A fit never needs it — a Gaussian likelihood's
    ``1/sigma**2`` ~1e32 is the same lift arriving free — but a forward-model benchmark,
    a mock-generation loop or a sensitivity study takes exactly this gradient.
    """
    seam, out = unweighted
    b32 = out["f32"]["boosted"]
    assert out["f32"]["dtype"] == ["float32"], (
        f"the float32 arm of {seam} produced {out['f32']['dtype']} ({_where()})"
    )
    assert np.all(np.isfinite(b32)), (
        f"boosted unweighted float32 gradient is non-finite on {seam}: {b32} ({_where()})"
    )
    assert np.any(b32 != 0.0), (
        f"boosted unweighted float32 gradient is IDENTICALLY ZERO on {seam} ({b32}) — "
        f"the cotangent boost is the one thing that made this path usable and it no "
        f"longer does. approx={out['f32']['approx_state']} ({_where()})"
    )
    rel = _rel_norm(b32, out["f64"]["grad"])
    assert rel < 1e-2, (
        f"boosted unweighted float32 gradient disagrees with float64 by {rel:.2e} in "
        f"norm on {seam} (f32={b32}, f64={out['f64']['grad']}, "
        f"approx={out['f32']['approx_state']}, {_where()})"
    )


@pytest.mark.parametrize("unweighted", _UNWEIGHTED_SEAMS, indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The bare (unboosted) unweighted observable gradient is identically zero in "
        "float32 — measured on 2026-09-05 on BOTH channels and BOTH projectors "
        "(exact and WavePrecomp/SpectrumPrecomp). PR #2100 measured only the exact "
        "projector; the LUT arms are new and behave identically. loss_scaled_grad is "
        "the remedy and is pinned by the test above. The assertion requires the "
        "gradient to be BOTH finite AND non-zero, so a NaN cannot masquerade as a "
        "fix. XPASS therefore means the underflow was genuinely repaired — re-take "
        "bench/reports/2026-09-05_float32_spectroscopy_lines.md."
    ),
)
def test_the_bare_unweighted_observable_gradient_is_nonzero_in_float32(unweighted):
    """The defect itself, pinned so that fixing it is visible.

    ``predict_photometry`` returns F_nu ~1e-28 and ``predict_spectrum`` the same, so the
    reverse-mode cotangent chain runs among the float32 subnormals and flushes to zero.
    Under a Gaussian likelihood the residual weighting lifts it ~1e32 and the problem
    does not arise, which is why the *fitting* path measured elsewhere in this module is
    healthy while this one is not.

    Recorded as ``xfail(strict=True)`` rather than deleted: a gradient that is silently
    zero is the failure mode this whole module exists to make visible, and an absent
    test would hide it again exactly as a finite-only assertion did.
    """
    seam, out = unweighted
    g32 = out["f32"]["grad"]
    # BOTH halves, and the order of the two assertions is not the point — having both
    # is. "Non-zero" alone is satisfied by NaN (``nan != 0.0`` is True), so a NaN
    # gradient would report XPASS and be read as "the underflow was fixed at source".
    # That is the exact dual of the trap this test exists for: PR #2100's guard asserted
    # *finite* and zero is finite; asserting only *non-zero* lets NaN through instead.
    # CI caught this on 2026-09-05 (run 33958554553), where ``spec/lut`` XPASSed with a
    # NaN rather than a recovered gradient.
    assert np.all(np.isfinite(g32)), (
        f"unweighted float32 gradient is NON-FINITE on {seam} ({g32}) — this is not the "
        f"#2100 underflow being fixed, it is a different defect. "
        f"approx={out['f32']['approx_state']} ({_where()})"
    )
    assert np.any(g32 != 0.0), (
        f"unweighted float32 gradient is IDENTICALLY ZERO on {seam} ({g32}) — this is "
        f"the #2100 defect, and a finite-only guard cannot see it. "
        f"approx={out['f32']['approx_state']} ({_where()})"
    )


@pytest.mark.parametrize("unweighted", _UNWEIGHTED_SEAMS, indirect=True)
def test_the_cotangent_boost_does_not_move_the_float64_unweighted_gradient(unweighted):
    """Bit-identical on CPU; a small relative budget on any backend.

    PR #2104 qualified this claim and the qualification is carried rather than
    inherited: ``DEFAULT_COTANGENT_BOOST`` is a power of two, so the rescale is exact and
    CPU reproduces bit equality, but on CUDA the *graph* differs — the scaled objective
    hands XLA a different fusion and reduction problem — and float64 moved by up to
    1.4e-14 relative on 9 of 13 fitting-path seams. 1e-12 leaves ~70x head-room over
    that and still sits nine orders below the scale at which a real defect would show.
    """
    seam, out = unweighted
    g64, b64 = out["f64"]["grad"], out["f64"]["boosted"]
    if jax.default_backend() == "cpu":
        assert np.array_equal(g64, b64), (
            f"the cotangent boost moved the float64 unweighted gradient on CPU for "
            f"{seam}, where the power-of-two rescale must be exact: "
            f"plain={g64} boosted={b64} ({_where()})"
        )
    rel = _rel_norm(b64, g64)
    assert rel < 1e-12, (
        f"the cotangent boost moved the float64 unweighted gradient by {rel:.2e} "
        f"relative on {seam} — beyond XLA reassociation ({_where()})"
    )


def test_the_feature_precomp_line_path_survives_float32(ssp_bare):
    """A **default-configuration** line fit cannot be built in float32 — the *second* gap.

    This is deliberately asserted at the level a user meets it: build the model, hand it
    to ``Fitter`` with the default ``approx="auto"``, and take one gradient. That is the
    whole of what a float32 line fit is, and it raises.

    ``Fitter(approx="auto")`` — the default for ``Fitter``, ``PopulationFitter`` and
    ``CatalogFitter`` — **appends ``FeaturePrecomp`` whenever a line channel is fit**, so
    the resolved state on this seam is
    ``ApproxState(wave_precomp=True, feature_precomp=True, n_subbands=5)``. In float32 the
    LUT-served line values are non-finite, and ``_check_channel_scales`` (#1495) then
    raises at construction with ``log_prob = nan`` on the ``line_flux_constraint``
    channel. The failure is **loud**, which is the one piece of good news: a float32 line
    fit does not sample a corrupted posterior, it refuses to start.

    Three line paths, and only one of them works:

    ===============================  ==============================================
    operator                         float32
    ===============================  ==============================================
    ``measure_line_fluxes(False)``   finite, tracks float64 to ~2e-04 (the #1859 fix)
    ``measure_line_fluxes(True)``    **non-finite** — this test, via the default fit
    ``predict_line_fluxes``          **non-finite** — the Cue/catalog operator
    ===============================  ==============================================

    So the arithmetic that survives is reachable only by passing ``approx=None``
    explicitly, which is also the slowest of the three (CLAUDE.md prices the
    line-channel LUT at 4.77x on the #1477 fixture).

    ``xfail(strict=True)``: a runnable float32 line fit is what is wanted, so a fix flips
    this to XPASS and the report's Finding 3 needs re-taking.
    """
    # PR #2104's hazard, met again: this module accumulates a large number of
    # distinct compiled programs, and XLA's CPU backend dies part-way through the
    # inventory with tens of GiB still free. Observed 2026-09-05 killing the process
    # outright at the 19th test of this file. Dropping the caches before each heavy
    # standalone build is what keeps the module runnable in one process.
    jax.clear_caches()
    gc.collect()

    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    ssp = load_ssp_data(_SSP_WNE)
    flux, noise, line_data = _channel_mock(ssp, "lines_meas", "bakedin", Fixed(0.1), _SNR)
    out = _channel_gradients(
        ssp,
        "lines_meas",
        "bakedin",
        "exact",
        "auto",
        Fixed(0.1),
        flux,
        noise,
        line_data,
        x64=False,
        dtype=jnp.float32,
    )
    grad = out["origin"]["grad"]
    assert np.any(grad != 0.0), (
        f"the default (approx='auto') float32 line fit produced an identically zero "
        f"gradient: {grad} (approx={out['approx_state']}, {_where()})"
    )
    assert np.all(np.isfinite(grad)), (
        f"the default (approx='auto') float32 line fit produced a non-finite gradient: "
        f"{grad} (approx={out['approx_state']}, {_where()})"
    )


test_the_feature_precomp_line_path_survives_float32 = pytest.mark.xfail(
    strict=True,
    raises=(ValueError, AssertionError),
    reason=(
        "The FeaturePrecomp-served line path is non-finite in float32, so the DEFAULT "
        "line fit (Fitter(approx='auto') appends FeaturePrecomp whenever lines are fit) "
        "cannot be constructed: _check_channel_scales (#1495) raises with "
        "log_prob = nan on the line_flux_constraint channel. XPASS means fixed — "
        "re-take bench/reports/2026-09-05_float32_spectroscopy_lines.md."
    ),
)(test_the_feature_precomp_line_path_survives_float32)


def test_the_spectroscopic_redshift_finite_difference_is_truncation_not_defect(ssp_bare):
    """Why ``spec/*/…_freez`` gets a loosened finite-difference budget.

    A central difference validates the autodiff reference everywhere else in this module
    to ~1e-05. On a **spectrum with a free redshift** it reads 1.3e-01, and the honest
    question is whether the difference quotient is wrong or the gradient is. Moving
    ``z`` slides every spectral feature across a fixed pixel grid, so the objective's
    third derivative along that direction is large and a central difference carries an
    O(h**2 f''') error to match.

    The discriminating evidence is that the disagreement **shrinks with h**: a defect in
    the autodiff graph would be a constant offset that no step size removes, whereas
    truncation error falls. This test asserts that ordering rather than any particular
    value, so it stays meaningful if the fixture's conditioning changes.
    """
    # PR #2104's hazard, met again: this module accumulates a large number of
    # distinct compiled programs, and XLA's CPU backend dies part-way through the
    # inventory with tens of GiB still free. Observed 2026-09-05 killing the process
    # outright at the 19th test of this file. Dropping the caches before each heavy
    # standalone build is what keeps the module runnable in one process.
    jax.clear_caches()
    gc.collect()

    from tengri.observation.spectroscopy import Spectroscopy

    with jax.enable_x64(True):
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.asarray(_SPEC_WAVE)))
        zfac = Uniform(_Z_LO, _Z_HI)
        sed = SEDModel.build(
            ssp_data=ssp_bare,
            observation=obs,
            approx=None,
            **_base(zfac),
            **_CHANNEL_MODELS["stellar_dust"],
        )
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        mock = sed.mock_spectrum(
            truth, jnp.asarray(_SPEC_WAVE), snr=_SNR, key=jax.random.PRNGKey(0)
        )
        ctx = InferenceContext.from_target(Fitter(sed, mock.flux_obs, mock.noise, approx=None))
        names = sorted(ctx.initial_params(jax.random.PRNGKey(1)))
        data_args = ctx.data_args

        def nlp(vals):
            return ctx.neg_log_posterior_fn({k: vals[i] for i, k in enumerate(names)}, data_args)

        origin = [jnp.asarray(0.0)] * len(names)
        grad = np.array([float(np.asarray(x)) for x in jax.grad(nlp)(origin)])
        errs = {}
        for h in (1e-3, 1e-4, 1e-5):
            fd = []
            for i in range(len(names)):
                p = [jnp.asarray(0.0)] * len(names)
                m = [jnp.asarray(0.0)] * len(names)
                p[i] = jnp.asarray(h)
                m[i] = jnp.asarray(-h)
                fd.append(float((np.asarray(nlp(p)) - np.asarray(nlp(m))) / (2 * h)))
            errs[h] = _rel_norm(np.array(fd), grad)

    assert errs[1e-5] < errs[1e-3], (
        f"the float64 finite-difference disagreement on the spectroscopic free-redshift "
        f"seam does not shrink with the step size ({errs}) — that is the signature of a "
        f"defect in the autodiff graph rather than truncation error, and the loosened "
        f"_CHANNEL_FD_BUDGET entry for this seam would no longer be justified ({_where()})"
    )


@pytest.fixture(scope="module")
def unweighted_pair(ssp_bare, request):
    """Both projectors' float64 unweighted gradients for one channel, to compare."""
    channel = request.param
    out = {}
    for kind in ("exact", "lut"):
        with jax.enable_x64(True):
            sed = _channel_build(ssp_bare, channel, "stellar_dust", kind, Fixed(0.1))
            names = sorted(sed.spec.free_params)
            truth = {
                n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0))) for n in names
            }

            def observable(vals, _sed=sed, _names=names, _truth=truth, _channel=channel):
                p = dict(_truth)
                p.update({k: vals[i] for i, k in enumerate(_names)})
                return jnp.sum(
                    _sed.predict_spectrum(p) if _channel == "spec" else _sed.predict_photometry(p)
                )

            point = [jnp.asarray(float(truth[k])) for k in names]
            out[kind] = np.array([float(np.asarray(x)) for x in jax.grad(observable)(point)])
    jax.clear_caches()
    gc.collect()
    return channel, out


@pytest.mark.parametrize("unweighted_pair", ["phot2", "spec"], indirect=True)
def test_the_lut_actually_reaches_the_unweighted_observable_graph(unweighted_pair):
    """Without this the "LUT" arm above is the exact arm wearing another label.

    This is PR #2104's Finding 0 one surface further out. There, the trap was that
    ``Fitter(approx="auto")`` re-resolves the build-time knob, so an arm *labeled*
    exact was fitted under the LUT. Here it is the mirror image: a model built
    ``approx=WavePrecomp(...)`` carries ``wave_precomp=True`` in its ``ApproxState``, but
    the **rich** ``model.predict(params).photometry()`` accessor does not route through
    the LUT at all — measured 2026-09-05, bit-identical float64 values *and* gradients
    between the two projectors, which CLAUDE.md names as the signature of "a config that
    never reaches the graph".

    The lean ``predict_photometry`` / ``predict_spectrum`` surfaces are the ones
    ``WavePrecomp`` / ``SpectrumPrecomp`` route, and are what PR #2100 measured. This
    test pins that the seam under test is the LUT one, so that a future refactor cannot
    silently turn the unweighted LUT rows back into duplicated exact rows.
    """
    channel, out = unweighted_pair
    assert not np.array_equal(out["exact"], out["lut"]), (
        f"the {channel} unweighted observable gives BIT-IDENTICAL float64 gradients on "
        f"the exact and LUT projectors ({out['exact']}) — the LUT is not reaching the "
        f"graph, so any 'under the LUT' claim taken on this surface is the exact "
        f"measurement relabeled ({_where()})"
    )

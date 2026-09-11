# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 parity for the two line-flux operators #1206 repairs (regression pins).

``test_float32_fitting_path_seams.py`` establishes that the two operators run at all
in pure float32 (``test_the_discrete_line_catalog_operator_survives_float32``,
``test_the_feature_precomp_line_path_survives_float32``, both flipped from strict
``xfail`` to plain passing tests by the same change this module pins). This module
adds the **accuracy** and **float64-regression** bar those tests do not: how close
float32 tracks float64 on each operator's flux and gradient, and that float64 itself
has not moved.

Two operators, matching the two channel families a line fit can select
(``loss_functions.py`` dispatches on ``model._has_line_catalog()``):

* **Cue** (a backend publishing a discrete line catalog) -> ``predict_line_fluxes``,
  the exact operator of ``test_the_discrete_line_catalog_operator_survives_float32``.
* **wNE baked-in** (lines baked into the SSP templates, no catalog) ->
  ``measure_line_fluxes(approx=True)``, the FeaturePrecomp window-LUT operator
  ``Fitter(approx="auto")`` resolves to and that
  ``test_the_feature_precomp_line_path_survives_float32`` exercises through a full
  fit. This module calls the operator directly rather than through ``Fitter``, so a
  failure here localizes to the operator itself.

**float64 reference values.** ``_CUE_F64_REF`` and ``_WNE_F64_REF`` are
``predict_line_fluxes`` / ``measure_line_fluxes(approx=True)`` on the exact models
below, computed on ``origin/main`` at commit ``18cf9fb9ec73e5b2010b278f21b7c473431ae800``
(2026-09-11, before the #1206 line-channel fix -- commits ``97a1e892a`` /
``d67a46cf5`` / ``f0fc015d8`` / ``e24d44a6d``, cherry-picked from
``float32-line-channel-fix`` onto this branch), with

.. code-block:: bash

    PYTHONPATH=src taskset -c 0-3 env JAX_PLATFORMS=cpu \
        /home/suchetha/Projects/tengri/.venv/bin/python /tmp/f1206_ab_probe.py \
        /tmp/f1206_ab_main.npz

hardcoded to 12 significant digits. A cross-tree float64 A/B of 8 representative
line-channel arrays (this pair plus the discrete catalog, the fast nebular grid, the
exact wNE window path, and the dormant ``line_precompute`` table) between that
capture and this branch found **4 of 8 bit-identical** by ``np.array_equal`` and a
**maximum relative movement of 3.14e-14** on the rest -- both operators measured here
were among the moved four (``predict_line_fluxes`` 7.17e-15,
``measure_line_fluxes(approx=True)`` bit-identical), comfortably inside the
``rtol <= 1e-12`` no-behavioral-change bar (#1206) and consistent with commit
``97a1e892a``'s own published cross-tree A/B (71/116 bit-identical, <= 3.8e-14
elsewhere): a peak-normalized log-space regrouping is not bit-exact with the linear
form it replaces even in float64, only equal to within rounding.
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.observation import Observation, Photometry
from tengri.observation.line_measurement import default_line_defs
from tengri.parameters.resolve import resolve_fixed_params

pytestmark = pytest.mark.regression_bug

_PHOT2 = ["sdss_g", "sdss_r"]
_LINE_WAVES = np.array([4862.68, 5008.24, 6564.61, 6585.28])
_LINE_NAMES = ("Hbeta", "OIII_5007", "Halpha", "NII_6584")
_SSP_WNE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

_DUST_FREE = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}

#: ``predict_line_fluxes`` on the Cue model, float64, origin/main @ 18cf9fb9e (see
#: module docstring). 12 significant digits.
_CUE_F64_REF = np.array(
    [4.81137256894e-16, 5.88262869721e-16, 1.69818364890e-15, 5.31329869121e-16]
)

#: ``measure_line_fluxes(..., approx=True)`` on the wNE model, float64, same capture.
_WNE_F64_REF = np.array(
    [2.76902836687e-16, 4.08501587720e-16, 1.34668648728e-15, 3.21861572920e-18]
)

#: Relative sigma for the synthetic chi-square target the gradient checks use: not a
#: mock (no noise draw), just an offset target so the chi-square has non-trivial
#: curvature at the evaluation point (a target equal to the prediction gives an
#: identically-zero gradient there by construction, which tests nothing).
_CHI2_OFFSET = 1.1
_CHI2_SIGMA_FRAC = 0.1


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


def _truth(sed):
    """Standardized-origin truth dict, with Fixed params resolved in (#1206).

    The fast nebular grid path's no-state fallback (``_compute_log_nion``) reads
    ``params["redshift"]`` directly rather than through a :class:`ForwardState`, so
    a truth dict that omits a ``Fixed`` redshift raises ``KeyError`` there even
    though every other operator in this module tolerates it.
    """
    free = {
        n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
        for n in sed.spec.free_params
    }
    return resolve_fixed_params(sed, free)


def _cue_model(ssp_bare):
    return SEDModel.build(
        ssp_data=ssp_bare,
        observation=Observation(photometry=Photometry.from_names(_PHOT2)),
        approx=None,
        **_base(Fixed(0.1)),
        dust_attenuation=_DUST_FREE,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
    )


def _wne_model(ssp_wne):
    return SEDModel.build(
        ssp_data=ssp_wne,
        observation=Observation(photometry=Photometry.from_names(_PHOT2)),
        approx=None,
        **_base(Fixed(0.1)),
        dust_attenuation=_DUST_FREE,
        neb={"type": "none"},
    )


@pytest.fixture(scope="module")
def ssp_bare():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


@pytest.fixture(scope="module")
def ssp_wne():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(_SSP_WNE)


def _cue_chi2(log_mass, sed, truth, obs, sigma):
    p = dict(truth, sfh_delayed_log_total_mass=log_mass)
    pred = sed.predict_line_fluxes(p, target_wavelengths=jnp.asarray(_LINE_WAVES))
    return jnp.sum(((pred - obs) / sigma) ** 2)


def _wne_chi2(log_mass, sed, truth, line_defs, obs, sigma):
    p = dict(truth, sfh_delayed_log_total_mass=log_mass)
    pred = sed.measure_line_fluxes(p, line_defs, approx=True)
    return jnp.sum(((pred - obs) / sigma) ** 2)


def test_predict_line_fluxes_cue_float32_tracks_float64(ssp_bare):
    """The Cue ``predict_line_fluxes`` operator: float64 unchanged, float32 usable.

    ``test_the_discrete_line_catalog_operator_survives_float32`` establishes finite
    and non-zero at one seed; this pins float64 against the pre-fix origin/main
    capture and adds a numeric accuracy bar (rtol <= 3e-3), the same class of bound
    ``test_float32_fitting_path_seams.py`` uses elsewhere in this tree.

    Measured on this tree: componentwise relative error
    ``[3.53e-05 2.39e-05 1.06e-05 9.82e-06]``, worst case 3.53e-05, two orders of
    magnitude inside the 3e-3 bar.
    """
    jax.clear_caches()
    gc.collect()

    with jax.enable_x64(True):
        sed64 = _cue_model(ssp_bare)
        truth64 = _truth(sed64)
        f64 = sed64.predict_line_fluxes(truth64, target_wavelengths=jnp.asarray(_LINE_WAVES))
        dtype64 = f64.dtype
        f64 = np.asarray(f64, dtype=np.float64)

    np.testing.assert_allclose(
        f64,
        _CUE_F64_REF,
        rtol=1e-9,
        err_msg=(
            "float64 predict_line_fluxes (Cue) moved vs the origin/main reference "
            "captured before the #1206 line-channel fix"
        ),
    )

    with jax.enable_x64(False):
        sed32 = _cue_model(ssp_bare)
        truth32 = _truth(sed32)
        f32 = sed32.predict_line_fluxes(truth32, target_wavelengths=jnp.asarray(_LINE_WAVES))
        dtype32 = f32.dtype
        f32 = np.asarray(f32, dtype=np.float64)

    # #1840: tengri/__init__.py re-enables x64 on import, so the config flag lies;
    # the returned array's dtype is the only admissible proof of precision.
    assert dtype32 == jnp.float32 and dtype64 == jnp.float64, (
        f"precision not established from the returned arrays: f32 arm gave {dtype32}, "
        f"f64 arm gave {dtype64}"
    )
    assert np.all(np.isfinite(f32)), f"float32 predict_line_fluxes is non-finite: {f32}"
    assert np.all(f32 != 0.0), f"float32 predict_line_fluxes is identically zero: {f32}"
    rel = np.abs(f32 - f64) / np.abs(f64)
    assert np.all(rel <= 3e-3), (
        f"float32 predict_line_fluxes disagrees with float64 by up to {rel.max():.2e} "
        f"(f32={f32}, f64={f64})"
    )


def test_predict_line_fluxes_cue_grad_float32_tracks_float64(ssp_bare):
    """Gradient of the Cue line-channel chi-square: finite, float32 tracks float64.

    Measured on this tree: ``d(chi2)/d(log_total_mass)`` = -184.2068 (float64),
    -184.2024 (float32), relative error 2.40e-05.
    """
    jax.clear_caches()
    gc.collect()

    with jax.enable_x64(True):
        sed64 = _cue_model(ssp_bare)
        truth64 = _truth(sed64)
        f64 = np.asarray(
            sed64.predict_line_fluxes(truth64, target_wavelengths=jnp.asarray(_LINE_WAVES)),
            dtype=np.float64,
        )
        obs64 = jnp.asarray(f64 * _CHI2_OFFSET)
        sigma64 = jnp.asarray(_CHI2_SIGMA_FRAC * np.abs(f64))
        mass0 = float(truth64["sfh_delayed_log_total_mass"])
        g64 = float(jax.grad(_cue_chi2)(jnp.asarray(mass0), sed64, truth64, obs64, sigma64))

    with jax.enable_x64(False):
        sed32 = _cue_model(ssp_bare)
        truth32 = _truth(sed32)
        obs32 = jnp.asarray(obs64, dtype=jnp.float32)
        sigma32 = jnp.asarray(sigma64, dtype=jnp.float32)
        g32 = jax.grad(_cue_chi2)(
            jnp.asarray(mass0, dtype=jnp.float32), sed32, truth32, obs32, sigma32
        )
        dtype32 = g32.dtype
        g32 = float(g32)

    assert dtype32 == jnp.float32, f"gradient did not run in float32: {dtype32}"
    assert np.isfinite(g32), f"float32 chi-square gradient is non-finite: {g32}"
    assert g32 != 0.0, "float32 chi-square gradient is identically zero"
    rel = abs(g32 - g64) / max(abs(g64), 1e-300)
    assert rel <= 1e-2, (
        f"float32 chi-square gradient disagrees with float64 by {rel:.2e} (f32={g32}, f64={g64})"
    )


def test_measure_line_fluxes_approx_wne_float32_tracks_float64(ssp_wne):
    """The wNE ``measure_line_fluxes(approx=True)`` operator: float64 unchanged,
    float32 usable.

    ``test_the_feature_precomp_line_path_survives_float32`` establishes this through
    a full default (``approx="auto"``) ``Fitter`` gradient; this calls the operator
    directly so a failure localizes to it rather than to the fit machinery around it.

    Measured on this tree: componentwise relative error
    ``[8.09e-06 7.28e-06 9.89e-06 1.70e-03]``, worst case 1.70e-03 (the faint
    NII_6584 line, the smallest flux of the four), inside the 3e-3 bar.
    """
    jax.clear_caches()
    gc.collect()

    line_defs = default_line_defs(_LINE_WAVES, _LINE_NAMES)

    with jax.enable_x64(True):
        sed64 = _wne_model(ssp_wne)
        truth64 = _truth(sed64)
        f64 = sed64.measure_line_fluxes(truth64, line_defs, approx=True)
        dtype64 = f64.dtype
        f64 = np.asarray(f64, dtype=np.float64)

    np.testing.assert_allclose(
        f64,
        _WNE_F64_REF,
        rtol=1e-9,
        err_msg=(
            "float64 measure_line_fluxes(approx=True) (wNE) moved vs the origin/main "
            "reference captured before the #1206 line-channel fix"
        ),
    )

    with jax.enable_x64(False):
        sed32 = _wne_model(ssp_wne)
        truth32 = _truth(sed32)
        f32 = sed32.measure_line_fluxes(truth32, line_defs, approx=True)
        dtype32 = f32.dtype
        f32 = np.asarray(f32, dtype=np.float64)

    assert dtype32 == jnp.float32 and dtype64 == jnp.float64, (
        f"precision not established from the returned arrays: f32 arm gave {dtype32}, "
        f"f64 arm gave {dtype64}"
    )
    assert np.all(np.isfinite(f32)), (
        f"float32 measure_line_fluxes(approx=True) is non-finite: {f32}"
    )
    assert np.all(f32 != 0.0), (
        f"float32 measure_line_fluxes(approx=True) is identically zero: {f32}"
    )
    rel = np.abs(f32 - f64) / np.abs(f64)
    assert np.all(rel <= 3e-3), (
        f"float32 measure_line_fluxes(approx=True) disagrees with float64 by up to "
        f"{rel.max():.2e} (f32={f32}, f64={f64})"
    )


def test_measure_line_fluxes_approx_wne_grad_float32_tracks_float64(ssp_wne):
    """Gradient of the wNE FeaturePrecomp line-channel chi-square: finite, tracks.

    Measured on this tree: ``d(chi2)/d(log_total_mass)`` = -184.2068 (float64),
    -183.5021 (float32), relative error 3.83e-03, inside the 1e-2 bar and the
    largest of the four gradient checks in this module -- consistent with this
    being the operator whose overflow (``total_mass * L_sun`` in the window LUT,
    Defect 2) sits deepest in the chain the chi-square gradient differentiates
    through.
    """
    jax.clear_caches()
    gc.collect()

    line_defs = default_line_defs(_LINE_WAVES, _LINE_NAMES)

    with jax.enable_x64(True):
        sed64 = _wne_model(ssp_wne)
        truth64 = _truth(sed64)
        f64 = np.asarray(sed64.measure_line_fluxes(truth64, line_defs, approx=True))
        obs64 = jnp.asarray(f64 * _CHI2_OFFSET)
        sigma64 = jnp.asarray(_CHI2_SIGMA_FRAC * np.abs(f64))
        mass0 = float(truth64["sfh_delayed_log_total_mass"])
        g64 = float(
            jax.grad(_wne_chi2)(jnp.asarray(mass0), sed64, truth64, line_defs, obs64, sigma64)
        )

    with jax.enable_x64(False):
        sed32 = _wne_model(ssp_wne)
        truth32 = _truth(sed32)
        obs32 = jnp.asarray(obs64, dtype=jnp.float32)
        sigma32 = jnp.asarray(sigma64, dtype=jnp.float32)
        g32 = jax.grad(_wne_chi2)(
            jnp.asarray(mass0, dtype=jnp.float32), sed32, truth32, line_defs, obs32, sigma32
        )
        dtype32 = g32.dtype
        g32 = float(g32)

    assert dtype32 == jnp.float32, f"gradient did not run in float32: {dtype32}"
    assert np.isfinite(g32), f"float32 chi-square gradient is non-finite: {g32}"
    assert g32 != 0.0, "float32 chi-square gradient is identically zero"
    rel = abs(g32 - g64) / max(abs(g64), 1e-300)
    assert rel <= 1e-2, (
        f"float32 chi-square gradient disagrees with float64 by {rel:.2e} (f32={g32}, f64={g64})"
    )

# SPDX-License-Identifier: BSD-3-Clause
"""Float32-safe rest-frame photometry, spectra and emission-line SEDs (#3361).

The z=0, d_L = 10 pc reprojection carried the factor ``1/(4 pi (10 pc)^2)`` ~ 8.4e-41
as a linear scalar. That is subnormal in float32 and flushes to zero in the product,
so ``phot_rest_fnu`` and ``spec_rest_fnu`` were exactly 0 and ``mag_absolute`` read
46.2 for every band. The fix applies the factor as a log10 offset through
``apply_log10_scale``, the same way the observed-frame flux is projected.

``lines_to_sed`` formed ``L_lambda * L_SUN * lambda^2 * ANGSTROM_CM / C_LIGHT`` left to
right, so the intermediate ``L_SUN * lambda^2`` (~1e41) overflowed float32 before the
final ~1e22 result.

Each test measures float32 (``jax.enable_x64(False)``) against float64 with the same
inputs. A finiteness check alone would pass the flushed-to-zero case only if it also
checked for zero, so every test asserts finite, nonzero and agreement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import Observation, Photometry, SEDModel, recipes
from tengri.components.agn._phys import lines_to_sed

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def key():
    """PRNG key for the sampled parameters, built inside each test rather than at import."""
    return jax.random.PRNGKey(0)


PHOT_BANDS = ["galex_fuv", "galex_nuv", "sdss_u", "sdss_r", "wise_w1", "wise_w4", "herschel_250"]
REL_TOL = 1e-4
MAG_TOL = 1e-3


def _phot_model(ssp):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(PHOT_BANDS)),
        **recipes.star_forming_photometry(),
    )


def _float32_arm(fn):
    with jax.enable_x64(False):
        return fn()


def _assert_f32_matches_f64(f32, f64, rtol=REL_TOL):
    f32 = np.asarray(f32, dtype=np.float64)
    f64 = np.asarray(f64, dtype=np.float64)
    assert np.all(np.isfinite(f32)), f"float32 non-finite: {f32}"
    assert np.all(f32 != 0.0), f"float32 flushed to zero: {f32}"
    assert_allclose(f32, f64, rtol=rtol)


# ── Item 1: rest-frame photometry and absolute magnitudes ─────────────────────


def test_phot_rest_fnu_float32_matches_float64(ssp_bare, key):
    model = _phot_model(ssp_bare)
    params = dict(model.spec.sample(key))

    ref = model.predict_observables(params)
    f32 = _float32_arm(lambda: model.predict_observables(params))

    _assert_f32_matches_f64(f32.phot_rest_fnu, ref.phot_rest_fnu)
    mag64 = np.asarray(ref.mag_absolute, dtype=np.float64)
    mag32 = np.asarray(f32.mag_absolute, dtype=np.float64)
    assert np.all(np.isfinite(mag32)), f"float32 mag_absolute non-finite: {mag32}"
    assert np.all(mag32 != 0.0), f"float32 mag_absolute flushed to zero: {mag32}"
    assert np.max(np.abs(mag32 - mag64)) < MAG_TOL, (mag32, mag64)


# ── Item 2: lines_to_sed intermediate overflow ────────────────────────────────


def _f32_exact(a):
    """Round to float32 and back, so the float64 reference sees the same inputs."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def test_lines_to_sed_float32_matches_float64():
    wave_obs = _f32_exact(np.linspace(4000.0, 7000.0, 500))
    line_wave = _f32_exact([4862.68, 6564.61])
    line_lum = np.array([1.0, 2.86])  # Lsun

    def run():
        return lines_to_sed(
            jnp.asarray(line_wave),
            jnp.asarray(line_lum),
            jnp.asarray(wave_obs),
            fwhm_kms=200.0,
        )

    ref = np.asarray(run(), dtype=np.float64)
    f32 = np.asarray(_float32_arm(run), dtype=np.float64)
    assert np.all(np.isfinite(f32)), "float32 lines_to_sed non-finite"
    # Far wings of a Gaussian legitimately underflow to 0 in both precisions; the
    # claim is on the line core, where the answer is representable.
    core = ref > 1e-6 * ref.max()
    assert core.any()
    assert np.all(np.isfinite(f32[core])), "float32 line core non-finite"
    assert np.all(f32[core] != 0.0), "float32 core flushed to zero"
    assert_allclose(f32[core], ref[core], rtol=REL_TOL)


# ── Item 3 (AGN luminosity seams): FIR-radio q_TIR in float32 ────────────────


@pytest.mark.parametrize(("l_tir_lsun", "l_1p4ghz"), [(2e12, 1e25), (1e11, 1e23), (3e10, 2e22)])
def test_q_ir_float32_matches_float64(l_tir_lsun, l_1p4ghz):
    """q_TIR forms L_TIR [W] = L_TIR[Lsun] * L_SUN * 1e-7, which overflows float32 above
    ~9e4 Lsun (3e10 Lsun gives inf before the log). The log form is finite and agrees
    with float64 to rtol 1e-4 (measured 2e-7)."""
    from tengri.utils.sed_quantities import compute_q_ir

    def run():
        return compute_q_ir(jnp.asarray(l_tir_lsun), jnp.asarray(l_1p4ghz))

    ref = float(run())
    f32 = float(_float32_arm(run))
    assert np.isfinite(f32), f"float32 q_TIR non-finite: {f32}"
    assert f32 != 0.0, "float32 q_TIR flushed to zero"
    assert_allclose(f32, ref, rtol=REL_TOL)


def test_q_ir_gradient_finite_and_nonzero_in_float32():
    """Gradient w.r.t. L_TIR is finite and nonzero in float32 (positive input), and
    finite at L_TIR = 0, where the floor applies."""
    from tengri.utils.sed_quantities import compute_q_ir

    def loss(x):
        return compute_q_ir(x, jnp.asarray(1e23))

    def grad_at(v):
        return jax.grad(loss)(jnp.asarray(v))

    g_pos = float(_float32_arm(lambda: grad_at(1e10)))
    g_zero = float(_float32_arm(lambda: grad_at(0.0)))
    assert np.isfinite(g_pos) and g_pos != 0.0, g_pos
    # grad-assert: finite-only — at L_TIR = 0 the floor binds, so the exact derivative is 0
    assert np.isfinite(g_zero), g_zero

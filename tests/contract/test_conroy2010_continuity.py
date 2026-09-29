# SPDX-License-Identifier: BSD-3-Clause
"""``conroy2010`` reproduces FSPS's CCM89 (dust_type=1) continuity behavior.

FSPS's ``attn_curve.f90`` (dtype=1, lines 43-102) adds a continuity term
(``hack``) that smooths the step a variable UV-bump strength (``uvb``) creates
at the optical/near-UV junction (x=3.3 um^-1). Segment boundaries come from
``sps_setup.f90`` lines 1277-1282 (``mwdindex``): x >= 0.1, 1.1, 3.3, 5.9, 8.0,
x > 12. Critically, ``hack`` is added on ``tmp(mwdindex(4):mwdindex(3))`` only
-- the near-UV segment 3.3 <= x < 5.9 -- and NOT on the adjoining mid-UV
segment 5.9 <= x < 8.0. Widening that range (as a previous version of
``conroy2010`` did) makes the curve disagree with FSPS by up to ~2e-3 in the
mid-UV.

``_fsps_ccm89_reference`` below is an independent float64 numpy transcription
of the Fortran, used only for parity checks -- it must never call
``conroy2010`` itself, or a test built on it would tautologically pass.

Scope note (far-UV domain): tengri's ``cardelli``/``conroy2010`` clip the
far-UV branch to CCM89's own published domain, x in [8, 10] ("CCM89 Table 4"),
rather than continuing FSPS's unclamped cubic out to x=12 before FSPS's own
constant-extrapolation cap. That clip is a pre-existing design choice shared
by ``cardelli`` and predates this fix; it is reproduced (not altered) by the
reference below and is unrelated to the near-UV continuity term this file
tests. Consequently the reference below matches tengri's actual domain
(x <= 10, held constant beyond) rather than FSPS's raw x=12 cap -- there is no
wavelength in the 912 Angstrom - 3 micron range (x <= 10.96) where tengri's
own x=12 behavior could be exercised in the first place.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import conroy2010

pytestmark = pytest.mark.contract

# ── Independent float64 numpy transcription of attn_curve.f90 (dust_type=1) ─

#: mwdindex segment boundaries in x = 1e4/lambda_AA [um^-1] (sps_setup.f90:1277-1282).
_X_IR_OPT = 1.1
_X_OPT_NUV = 3.3
_X_NUV_MUV = 5.9
_X_MUV_FUV = 8.0
_X_FUV_CAP = 10.0  # tengri's own far-UV domain cap (CCM89 Table 4); see module docstring.


def _k5500_ccm89(rv: float) -> float:
    """k(5500 Angstrom) from the optical polynomial alone (bump-independent)."""
    x5 = 1.0 / 0.55
    y5 = x5 - 1.82
    a5 = (
        1.0
        + 0.17699 * y5
        - 0.50447 * y5**2
        - 0.02427 * y5**3
        + 0.72085 * y5**4
        + 0.01979 * y5**5
        - 0.77530 * y5**6
        + 0.32999 * y5**7
    )
    b5 = (
        1.41338 * y5
        + 2.28305 * y5**2
        + 1.07233 * y5**3
        - 5.38434 * y5**4
        - 0.62251 * y5**5
        + 5.30260 * y5**6
        - 2.09002 * y5**7
    )
    return a5 + b5 / rv


def _hack_amplitude(uvb: float, rv: float) -> float:
    """k_opt(3.3) - k_nuv(3.3): the mismatch the continuity term corrects."""
    xb = _X_OPT_NUV
    y_b = xb - 1.82
    a_opt_b = (
        1.0
        + 0.17699 * y_b
        - 0.50447 * y_b**2
        - 0.02427 * y_b**3
        + 0.72085 * y_b**4
        + 0.01979 * y_b**5
        - 0.77530 * y_b**6
        + 0.32999 * y_b**7
    )
    b_opt_b = (
        1.41338 * y_b
        + 2.28305 * y_b**2
        + 1.07233 * y_b**3
        - 5.38434 * y_b**4
        - 0.62251 * y_b**5
        + 5.30260 * y_b**6
        - 2.09002 * y_b**7
    )
    k_opt_b = a_opt_b + b_opt_b / rv
    a_uv_b = 1.752 - 0.316 * xb - 0.104 * uvb / ((xb - 4.67) ** 2 + 0.341)
    b_uv_b = -3.09 + 1.825 * xb + 1.206 * uvb / ((xb - 4.62) ** 2 + 0.263)
    k_uv_b = a_uv_b + b_uv_b / rv
    return k_opt_b - k_uv_b


def _fsps_ccm89_reference(wavelength: np.ndarray, uvb: float, rv: float) -> np.ndarray:
    """Float64 numpy transcription of FSPS ``attn_curve.f90`` (dust_type=1).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Angstrom]
    uvb : float
        UV-bump scale (FSPS ``pset%uvb``; tengri's ``dust_bump_strength``).
    rv : float
        Total-to-selective extinction ratio (FSPS ``pset%mwr``; tengri's ``dust_Rv``).

    Returns
    -------
    ndarray, shape (n_wave,)
        k(lambda), normalized so k(5500 Angstrom) = 1.

    Notes
    -----
    Independent of ``tengri.components.dust.attenuation`` -- never calls it.
    The continuity ``hack`` term is added only on 3.3 <= x < 5.9 (attn_curve.f90
    ``tmp(mwdindex(4):mwdindex(3))``); the far-UV branch is clipped to x in
    [8, 10] rather than FSPS's own x=12 cap (see module docstring scope note).
    """
    wave_aa = np.asarray(wavelength, dtype=np.float64)
    x = 1.0e4 / wave_aa

    def a_uv_base(xx):
        return 1.752 - 0.316 * xx - 0.104 * uvb / ((xx - 4.67) ** 2 + 0.341)

    def b_uv_base(xx):
        return -3.09 + 1.825 * xx + 1.206 * uvb / ((xx - 4.62) ** 2 + 0.263)

    hack_amp = _hack_amplitude(uvb, rv)

    # IR: x < 1.1
    a_ir = 0.574 * x**1.61
    b_ir = -0.527 * x**1.61

    # Optical: 1.1 <= x < 3.3
    y = x - 1.82
    a_opt = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b_opt = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    # Near-UV (3.3 <= x < 5.9) + mid-UV (5.9 <= x < 8.0): shared Drude form.
    fa = np.where(x >= _X_NUV_MUV, -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3, 0.0)
    fb = np.where(x >= _X_NUV_MUV, 0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3, 0.0)
    a_uv = a_uv_base(x) + fa
    b_uv = b_uv_base(x) + fb

    # Continuity term: near-UV segment only (3.3 <= x < 5.9).
    hack = (_X_OPT_NUV / np.clip(x, _X_OPT_NUV, np.inf)) ** 6 * hack_amp

    # Far-UV: clipped to x in [8, 10] -- tengri's domain (see module docstring).
    y_fuv = np.clip(x, _X_MUV_FUV, _X_FUV_CAP) - _X_MUV_FUV
    a_fuv = -1.073 - 0.628 * y_fuv + 0.137 * y_fuv**2 - 0.070 * y_fuv**3
    b_fuv = 13.67 + 4.257 * y_fuv - 0.42 * y_fuv**2 + 0.374 * y_fuv**3

    a = np.where(
        x < _X_IR_OPT, a_ir, np.where(x < _X_OPT_NUV, a_opt, np.where(x < _X_MUV_FUV, a_uv, a_fuv))
    )
    b = np.where(
        x < _X_IR_OPT, b_ir, np.where(x < _X_OPT_NUV, b_opt, np.where(x < _X_MUV_FUV, b_uv, b_fuv))
    )

    k = a + b / rv
    k = k + np.where((x >= _X_OPT_NUV) & (x < _X_NUV_MUV), hack, 0.0)
    return k / _k5500_ccm89(rv)


def _analytic_step_at_5p9(uvb: float, rv: float) -> float:
    """k(x=5.9+) - k(x=5.9-), analytic: the near-UV hack term switching off."""
    return -((_X_OPT_NUV / _X_NUV_MUV) ** 6) * _hack_amplitude(uvb, rv) / _k5500_ccm89(rv)


def _analytic_step_at_8p0(uvb: float, rv: float) -> float:
    """k(x=8+) - k(x=8-), analytic: mid-UV Drude tail vs. the far-UV polynomial."""
    x = _X_MUV_FUV
    fa = -0.04473 * (x - 5.9) ** 2 - 0.009779 * (x - 5.9) ** 3
    fb = 0.2130 * (x - 5.9) ** 2 + 0.1207 * (x - 5.9) ** 3
    a_uv = 1.752 - 0.316 * x - 0.104 * uvb / ((x - 4.67) ** 2 + 0.341) + fa
    b_uv = -3.09 + 1.825 * x + 1.206 * uvb / ((x - 4.62) ** 2 + 0.263) + fb
    mid_uv_at_8 = a_uv + b_uv / rv
    far_uv_at_8 = -1.073 + 13.67 / rv  # y_fuv = 0
    return (far_uv_at_8 - mid_uv_at_8) / _k5500_ccm89(rv)


def _analytic_drude_delta_at(x: float, rv: float) -> float:
    """[a_uv(s=1,x)-a_uv(s=0,x)] + [b_uv(s=1,x)-b_uv(s=0,x)]/rv for x < 5.9."""
    da = -0.104 / ((x - 4.67) ** 2 + 0.341)
    db = 1.206 / ((x - 4.62) ** 2 + 0.263)
    return da + db / rv


# ── Wavelength grid: 912 Angstrom - 3 micron, bracketing every reachable ──
# ── segment boundary (x = 1.1, 3.3, 5.9, 8.0, and tengri's own far-UV cap ──
# ── at x = 10; FSPS's x = 12 cap is not reachable -- see module docstring) ─

_BOUNDARIES_X = (1.1, 3.3, 5.9, 8.0, 10.0)
_BASE_WAVE_AA = np.geomspace(912.0, 30000.0, 50)
_BRACKET_WAVE_AA = np.concatenate(
    [[1.0e4 / (xb - 0.02), 1.0e4 / (xb + 0.02)] for xb in _BOUNDARIES_X]
)
WAVELENGTHS_AA = np.unique(np.concatenate([_BASE_WAVE_AA, _BRACKET_WAVE_AA]))
assert WAVELENGTHS_AA.size >= 60

BUMP_STRENGTHS = (0.0, 0.5, 1.0, 2.0)
RV_VALUES = (2.0, 3.1, 5.0)


@pytest.mark.parametrize("rv", RV_VALUES)
@pytest.mark.parametrize("bump", BUMP_STRENGTHS)
def test_conroy2010_matches_fsps_reference_across_grid(bump, rv):
    """``conroy2010`` matches the independent FSPS numpy transcription to rtol 1e-7."""
    k_tengri = np.asarray(
        conroy2010(jnp.array(WAVELENGTHS_AA), dust_Rv=rv, dust_bump_strength=bump)
    )
    k_ref = _fsps_ccm89_reference(WAVELENGTHS_AA, uvb=bump, rv=rv)
    np.testing.assert_allclose(k_tengri, k_ref, rtol=1e-7)


@pytest.mark.parametrize("rv", RV_VALUES)
@pytest.mark.parametrize("bump", BUMP_STRENGTHS)
def test_conroy2010_is_continuous_at_x_3p3(bump, rv):
    """k(lambda) is continuous across x=3.3 um^-1 for every (bump, R_V) cell.

    FSPS's ``hack`` term exists precisely to make this true even though the
    CCM89 optical polynomial and near-UV Drude profile do not naturally meet
    at the boundary.
    """
    x_b = _X_OPT_NUV
    eps = 1e-7
    # x = 1e4/lambda: a *larger* x needs a *shorter* wavelength.
    lam_at_x_below = 1.0e4 / (x_b * (1.0 - eps))
    lam_at_x_above = 1.0e4 / (x_b * (1.0 + eps))
    k = np.asarray(
        conroy2010(
            jnp.array([lam_at_x_below, lam_at_x_above]), dust_Rv=rv, dust_bump_strength=bump
        )
    )
    rel = abs(k[1] - k[0]) / abs(k[0])
    assert rel < 1e-5, f"bump={bump} Rv={rv}: |Delta k|/k = {rel:.3e} at x=3.3"


@pytest.mark.parametrize("rv", RV_VALUES)
@pytest.mark.parametrize("bump", BUMP_STRENGTHS)
def test_conroy2010_residual_step_at_x_5p9_matches_analytic_size(bump, rv):
    """The (deliberate, published) step at x=5.9 matches its closed form.

    The continuity term is defined only on 3.3 <= x < 5.9, so dropping out of
    it at x=5.9 leaves a residual step of size ``(3.3/5.9)^6 * hack_amplitude``
    whenever ``bump != 1``. Pinning this catches a silent change of the
    continuity term's range (the #2522/#2523 defect this file guards).
    """
    x_b = _X_NUV_MUV
    eps = 1e-9
    lam_at_x_below = 1.0e4 / (x_b * (1.0 - eps))
    lam_at_x_above = 1.0e4 / (x_b * (1.0 + eps))
    k = np.asarray(
        conroy2010(
            jnp.array([lam_at_x_below, lam_at_x_above]), dust_Rv=rv, dust_bump_strength=bump
        )
    )
    measured_step = float(k[1] - k[0])
    predicted_step = _analytic_step_at_5p9(bump, rv)
    np.testing.assert_allclose(measured_step, predicted_step, rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("rv", RV_VALUES)
@pytest.mark.parametrize("bump", BUMP_STRENGTHS)
def test_conroy2010_residual_step_at_x_8p0_matches_analytic_size(bump, rv):
    """The (deliberate, published) step at x=8.0 matches its closed form.

    The far-UV CCM89 polynomial carries no bump-strength scaling of its own
    (an implicit bump=1 continuation), so its junction with the (scaled)
    mid-UV segment steps whenever ``bump != 1`` -- about 0.7% of k for
    ``bump=0`` at R_V=3.1.
    """
    x_b = _X_MUV_FUV
    eps = 1e-9
    lam_at_x_below = 1.0e4 / (x_b * (1.0 - eps))
    lam_at_x_above = 1.0e4 / (x_b * (1.0 + eps))
    k = np.asarray(
        conroy2010(
            jnp.array([lam_at_x_below, lam_at_x_above]), dust_Rv=rv, dust_bump_strength=bump
        )
    )
    measured_step = float(k[1] - k[0])
    predicted_step = _analytic_step_at_8p0(bump, rv)
    np.testing.assert_allclose(measured_step, predicted_step, rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("rv", RV_VALUES)
def test_conroy2010_bump_zero_removes_2175_angstrom_feature(rv):
    """``bump=0`` minus ``bump=1`` at 2175 Angstrom equals the analytic Drude delta.

    Regression pin: if the UV-bump scaling factor (0.104, 1.206) or the
    continuity term stopped tracking ``dust_bump_strength`` correctly, this
    closed-form comparison -- independent of ``conroy2010`` itself -- would
    catch it.
    """
    wave_2175 = jnp.array([2175.0])
    k1 = np.asarray(conroy2010(wave_2175, dust_Rv=rv, dust_bump_strength=1.0))[0]
    k0 = np.asarray(conroy2010(wave_2175, dust_Rv=rv, dust_bump_strength=0.0))[0]
    measured = k1 - k0

    x = 1.0e4 / 2175.0  # 4.598 um^-1
    delta_x = _analytic_drude_delta_at(x, rv)
    delta_xb = _analytic_drude_delta_at(_X_OPT_NUV, rv)
    hack_delta = -((_X_OPT_NUV / x) ** 6) * delta_xb
    predicted = (delta_x + hack_delta) / _k5500_ccm89(rv)

    np.testing.assert_allclose(measured, predicted, rtol=1e-6)
    assert measured > 0, "bump=1 should attenuate more than bump=0 at the 2175 A feature"

# SPDX-License-Identifier: BSD-3-Clause
r"""``agn_lum_ratio = 0`` gives an exact zero SED with an exact, finite gradient (#2767).

``kubota_done_disc`` and ``adaf_spectrum`` used to fold ``log10(agn_lum_ratio)`` into the
normalization exponent. At a ratio of 0 that logarithm is ``-inf``, and the reverse pass
forms ``0 * inf = NaN``, which also reached the ``agn_log_lbol`` gradient.

The fix normalizes at the UNIT ratio (the exponent carries ``agn_log_lbol`` only) and
multiplies the spectrum by ``agn_lum_ratio`` linearly. The ratio then enters without a
logarithm, so at 0 the SED is exactly 0 and the derivative with respect to the ratio is the
SED at unit ratio, which is the true value.

For the kubota_done, multicolor_disc and ADAF models, in float32 and float64, this checks:

(a) the SED at ratio 0 is exactly 0;
(b) ``jax.grad`` with respect to ``agn_lum_ratio`` and ``agn_log_lbol`` at ratio 0 is finite;
(c) the ratio derivative at 0 matches a central finite difference, and equals the SED at
    unit ratio. The step is 1e-3 in float64 and 1e-2 in float32 (the float32 rounding of the
    difference is O(eps / h), so the larger step keeps it well below the 1e-3 tolerance the
    neighboring float32 gradient guards use);
(d) at ratio 0.5 the float64 SED equals the pre-change SED to 1e-12. The golden values were
    recorded from the code before the change.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(200.0, 1.0e7, 3000)
_LOG_LBOL = 11.0
_LOG_MBH = 8.0
_MODELS = ("kubota_done", "multicolor", "adaf")
_DTYPES = (jnp.float32, jnp.float64)

#: Central-difference step and the rtol it must meet, per dtype.
_STEP = {jnp.float64: 1e-3, jnp.float32: 1e-2}
_FD_RTOL = {jnp.float64: 1e-6, jnp.float32: 1e-3}

#: Pre-change SED at ratio 0.5, log L_bol = 11, log M_BH = 8, float64, on ``_WAVE``.
_GOLDEN_IDX = (0, 137, 500, 1001, 1500, 2002, 2500, 2999)
_GOLDEN = {
    "kubota_done": {
        "sum": 3.6481063752490684e31,
        "pts": (
            1.0378988736612377e27,
            1.8701281245645517e27,
            1.0156740462657394e28,
            5.876731883861794e28,
            5.76310031548177e27,
            1.8893992069754992e26,
            5.367659904144678e24,
            1.4736381577320894e23,
        ),
    },
    "multicolor": {
        "sum": 5.958577707758368e31,
        "pts": (
            1.4235669566768029e26,
            3.8925519395876026e27,
            6.1891594078450125e28,
            5.657572510354477e28,
            4.87558154880109e27,
            1.5664276642823378e26,
            4.437008816797985e24,
            1.2175515723209214e23,
        ),
    },
    "adaf": {
        "sum": 4.6983704894797534e29,
        "pts": (
            2.5059166008550745e25,
            2.975558425020228e25,
            4.690389250045613e25,
            8.789468095369126e25,
            1.6428804934065007e26,
            3.0815496566429876e26,
            2.395516133889615e26,
            1.1526534599627982e26,
        ),
    },
}


def _x64(dtype):
    return jax.enable_x64(dtype == jnp.float64)


def _sed(name, ratio, log_lbol, dtype):
    """The model's L_nu on ``_WAVE`` at the given ratio and log L_bol, in ``dtype``."""
    if name == "kubota_done":
        from tengri.components.agn.disc import kubota_done_disc as model
    elif name == "multicolor":
        from tengri.components.agn.disc import multicolor_disc as model
    else:
        from tengri.components.agn.adaf import adaf_spectrum as model
    w = jnp.asarray(_WAVE, dtype=dtype)
    return model(
        w,
        agn_log_lbol=jnp.asarray(log_lbol, dtype=dtype),
        agn_lum_ratio=jnp.asarray(ratio, dtype=dtype),
        agn_log_mbh=jnp.asarray(_LOG_MBH, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("name", _MODELS)
def test_sed_at_zero_ratio_is_exactly_zero(name, dtype):
    with _x64(dtype):
        sed = np.asarray(_sed(name, 0.0, _LOG_LBOL, dtype))
    assert sed.dtype == np.dtype(dtype)
    assert np.all(sed == 0.0), "SED at agn_lum_ratio = 0 must be exactly zero"


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("name", _MODELS)
def test_gradient_at_zero_ratio_is_finite(name, dtype):
    def total(ratio, log_lbol):
        return jnp.sum(_sed(name, ratio, log_lbol, dtype))

    with _x64(dtype):
        g_ratio, g_lbol = jax.grad(total, argnums=(0, 1))(
            jnp.asarray(0.0, dtype=dtype), jnp.asarray(_LOG_LBOL, dtype=dtype)
        )
    assert np.isfinite(float(g_ratio)), f"d/d(agn_lum_ratio) at 0 is {float(g_ratio)}"
    assert float(g_ratio) != 0.0, "d/d(agn_lum_ratio) at 0 collapsed to zero"
    # grad-assert: finite-only — at ratio 0 the SED is identically zero for every log L_bol, so
    # d/d(agn_log_lbol) is exactly 0 by construction; the exact-zero check below pins that value.
    assert np.isfinite(float(g_lbol)), f"d/d(agn_log_lbol) at ratio 0 is {float(g_lbol)}"
    assert float(g_lbol) == 0.0, f"d/d(agn_log_lbol) at ratio 0 must be exactly 0, got {g_lbol}"


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("name", _MODELS)
def test_ratio_derivative_at_zero_matches_finite_difference(name, dtype):
    def total(ratio):
        return jnp.sum(_sed(name, ratio, _LOG_LBOL, dtype))

    h = _STEP[dtype]
    with _x64(dtype):
        r0 = jnp.asarray(0.0, dtype=dtype)
        analytic = float(jax.grad(total)(r0))
        fd = (float(total(r0 + h)) - float(total(r0 - h))) / (2.0 * h)
        unit = float(jnp.sum(_sed(name, 1.0, _LOG_LBOL, dtype)))
    assert np.isfinite(analytic) and analytic != 0.0, f"analytic derivative is {analytic}"
    rtol = _FD_RTOL[dtype]
    assert abs(analytic - fd) <= rtol * abs(fd), (
        f"{name} {dtype.__name__}: analytic {analytic!r} vs central FD {fd!r} (h={h})"
    )
    # Linear in the ratio, so the derivative at 0 is the SED at unit ratio (the true value).
    assert abs(analytic - unit) <= rtol * abs(unit), (
        f"{name} {dtype.__name__}: derivative {analytic!r} vs SED at unit ratio {unit!r}"
    )


@pytest.mark.parametrize("name", _MODELS)
def test_half_ratio_sed_matches_pre_change_golden_in_float64(name):
    with _x64(jnp.float64):
        sed = np.asarray(_sed(name, 0.5, _LOG_LBOL, jnp.float64))
    golden = _GOLDEN[name]
    np.testing.assert_allclose(
        sed[list(_GOLDEN_IDX)], np.asarray(golden["pts"]), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(float(sed.sum()), golden["sum"], rtol=1e-12, atol=0.0)

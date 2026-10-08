# SPDX-License-Identifier: BSD-3-Clause
r"""#2767: the kubota_done SED and its gradient are finite and accurate in pure float32.

Two defects met here. The Planck prefactor ``2 h nu (nu/c)^2`` (~4e11 at 0.1 A) times an
outer-disc ring area (~6e31 cm^2) is ~2e43, past the float32 maximum, while the Boltzmann
factor ``exp(-h nu / k T)`` makes the product's true value zero; XLA forms
``(prefactor * area) * exp(-x)`` and the forward SED is ``inf * 0 = NaN``. Separately,
``exp(-500)`` underflows to exactly 0 in float32, so the corona shape formed as a product of
two underflowing exponentials went ``0 / 0`` in reverse mode.

The reproducer evaluates the rest-frame SED over the black-hole-mass range and the corners of
the declared prior box, and differentiates it with respect to ``agn_log_mbh`` and
``agn_log_lbol`` under ``jit(grad)``, once in float32 and once in float64. It requires the
float32 values to be finite and to agree with float64, and the gradients to be finite,
non-trivial and to agree with float64 to 1e-3 (the tolerance the neighbouring float32 gradient
guards use).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_SFH = {
    "type": "delayed",
    "all_params": Fixed(DEFAULT),
    "log_total_mass": Uniform(9.0, 11.0),
    "tau_gyr": 1.0,
    "age_gyr": 5.0,
}
_DUST = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_diff": 0.3,
    "tau_bc": 0.0,
}
_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0, "agn_log_mbh": 8.0}


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"]))


def _groups():
    return dict(
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {
                "type": "kubota_done",
                "all_params": Fixed(DEFAULT),
                "log_mbh": Uniform(7.0, 9.5),
            },
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
    )


_MBH_POINTS = (6.0, 7.5, 8.0, 8.5, 10.0)
_PRIOR_CORNERS = ((7.0, 9.0), (7.0, 12.0), (9.5, 9.0), (9.5, 12.0))
_GRAD_NAMES = ("agn_log_lbol", "agn_log_mbh")


def _evaluate(ssp, obs, *, x64, dtype):
    """Forward values at every probe point and the ``jit(grad)`` at the reproducer point.

    Returns ``(values, grad)``: ``values`` maps ``(log_mbh, log_lbol)`` to the summed
    rest-frame SED, and ``grad`` is the gradient of that sum with respect to
    ``(agn_log_lbol, agn_log_mbh)`` at the truth.
    """
    with jax.enable_x64(x64):
        model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **_groups())
        names = sorted(n for n in model.spec.free_params if n in _TRUTH)
        assert set(_GRAD_NAMES) <= set(names), names

        def sed_sum(log_lbol, log_mbh):
            params = {
                k: {"agn_log_lbol": log_lbol, "agn_log_mbh": log_mbh}.get(
                    k, jnp.asarray(_TRUTH[k], dtype=dtype)
                )
                for k in names
            }
            return jnp.sum(model.predict(params).rest_sed())

        def scalar(x):
            return jnp.asarray(x, dtype=dtype)

        forward = jax.jit(sed_sum)
        points = [(m, _TRUTH["agn_log_lbol"]) for m in _MBH_POINTS] + [
            (m, lb) for m, lb in _PRIOR_CORNERS
        ]
        values = {(m, lb): float(np.asarray(forward(scalar(lb), scalar(m)))) for m, lb in points}
        grad = jax.jit(jax.grad(sed_sum, argnums=(0, 1)))(
            scalar(_TRUTH["agn_log_lbol"]), scalar(_TRUTH["agn_log_mbh"])
        )
        return values, np.asarray([float(g) for g in grad])


@pytest.fixture(scope="module")
def float64_reference(ssp_bare, obs):
    return _evaluate(ssp_bare, obs, x64=True, dtype=jnp.float64)


@pytest.fixture(scope="module")
def float32_result(ssp_bare, obs):
    return _evaluate(ssp_bare, obs, x64=False, dtype=jnp.float32)


def test_kubota_done_forward_is_finite_and_accurate_in_float32(float64_reference, float32_result):
    """The float32 SED is finite and matches float64 over the mass range and the prior corners."""
    ref, _ = float64_reference
    got, _ = float32_result
    for point, v64 in ref.items():
        v32 = got[point]
        assert np.isfinite(v32), f"float32 forward value is non-finite at {point} (f64={v64})"
        rel = abs(v32 - v64) / abs(v64)
        assert rel < 1e-3, f"float32 forward disagrees with float64 by {rel:.2e} at {point}"


def test_kubota_done_gradient_is_finite_and_accurate_in_float32(float64_reference, float32_result):
    """Gradients with respect to ``agn_log_mbh`` and ``agn_log_lbol`` match float64 to 1e-3."""
    _, g64 = float64_reference
    _, g32 = float32_result
    assert np.all(np.isfinite(g32)), f"float32 gradient is non-finite (f32={g32}, f64={g64})"
    assert np.all(g32 != 0.0), f"float32 gradient collapsed to zero (f32={g32}, f64={g64})"
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 gradient disagrees with float64 by {rel.max():.2e} ({g32} vs {g64})"
    )

# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 gradients of the SED itself, for seams no filter can see (#1439).

Every other float32 gradient guard differentiates ``predict_photometry``. That is the
inference path and the right default — but it cannot cover a component no filter
reaches, and it is a different kernel from ``predict(p).rest_sed()``.

**No filter in the library reaches radio or X-ray.** Measured, on the widest band set
available (``galex_fuv`` ~1500 Å to ``herschel_250``): adding the radio component
changes photometry by ``0.000e+00``, and X-ray by ``1.83e-16``. So a float32 claim about
either, made through a photometry likelihood, is vacuous — it passes because the
component contributes nothing, not because its arithmetic is sound. This module measures
them where they are visible: the rest-frame SED, restricted to the wavelength range each
component actually emits in.

Doing so found #1439 — with any AGN present, ``d(sum rest_sed)/d(agn_log_lbol)`` is NaN
in pure float32, while the forward value and the mass gradient are both exact.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_SFH = {
    "type": "delayed",
    "all_params": FIXED,
    "log_total_mass": Uniform(9.0, 11.0),
    "tau_gyr": 1.0,
    "age_gyr": 5.0,
}
_DUST = {
    "type": "two_component",
    "law_bc": "calzetti",
    "all_params": FIXED,
    "tau_diff": 0.3,
    "tau_bc": 0.0,
}
_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"]))


def _band_gradient(ssp, obs, groups, *, lo, hi, x64, dtype):
    """(names, band L_nu, gradient) of the rest-frame SED summed over [lo, hi) Å."""
    with jax.enable_x64(x64):
        model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)
        names = sorted(n for n in model.spec.free_params if n in _TRUTH)
        wave = np.asarray(model._rest_wavelength, dtype=np.float64)
        mask = jnp.asarray((wave >= lo) & (wave < hi))

        def band_sum(values):
            params = {k: values[i] for i, k in enumerate(names)}
            return jnp.sum(jnp.where(mask, model.predict(params).rest_sed(), 0.0))

        base = [jnp.asarray(_TRUTH[k], dtype=dtype) for k in names]
        value = float(np.asarray(band_sum(base)))
        grad = np.array([float(np.asarray(g)) for g in jax.grad(band_sum)(base)])
        return names, int(mask.sum()), value, grad


def test_radio_sed_gradient_is_accurate_in_float32(ssp_bare, obs):
    """The radio seam, measured where it is actually visible.

    Asserts its own setup first: if no grid point carries radio emission, the
    comparison below would pass for the wrong reason. That is not hypothetical — the
    photometry version of this check is vacuous, because no filter reaches cm
    wavelengths.
    """
    groups = dict(sfh=_SFH, dust=_DUST, radio={"type": "condon92"})
    kw = dict(lo=1e7, hi=1e12)

    _, n_in, v64, g64 = _band_gradient(ssp_bare, obs, groups, x64=True, dtype=jnp.float64, **kw)
    _, _, _, g32 = _band_gradient(ssp_bare, obs, groups, x64=False, dtype=jnp.float32, **kw)

    assert n_in > 0 and v64 > 0.0, (
        f"no rest-frame grid point in [1e7, 1e12) Å carries emission (n={n_in}, "
        f"L_nu={v64:.3e}); this test cannot say anything about the radio seam"
    )
    assert np.all(np.isfinite(g32)), f"float32 radio-band SED gradient is non-finite: {g32}"
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 radio-band SED gradient disagrees with float64 by {rel.max():.2e} "
        f"(f32={g32}, f64={g64})"
    )


@pytest.mark.xfail(
    reason="#1439: with any AGN present, d(sum rest_sed)/d(agn_log_lbol) is NaN in pure "
    "float32 — while the forward value is exact (2.2891e+32, identical to float64) and "
    "the mass gradient is exact (3.2755e+32). Reproduces for the multicolor disc alone, "
    "the SKIRTOR torus alone, and both together with or without cigale_joint, at every "
    "wavelength band including the optical, and with no mask involved. "
    "predict_photometry gradients are unaffected, so inference through photometry is "
    "not blocked. Mechanism not yet pinned: the apply_log10_scale reference offset is "
    "ruled out (34.583 dex, Jacobian ~3.8e34, inside float32 range). One confirmed "
    "hazard in the area is that the 36 `1e-100` floors in src/ are exact no-ops in "
    "float32 (1e-100 -> 0.0, log10 -> -inf), the #1404 zero-hiding clamp class.",
    strict=True,
)
def test_agn_sed_gradient_is_finite_in_float32(ssp_bare, obs):
    """Differentiating the SED w.r.t. ``agn_log_lbol`` must not produce NaN in float32.

    Deliberately asserts only **finiteness**, not accuracy. A NaN is not a precision
    question — float64 gets a perfectly ordinary answer here and the float32 forward
    pass reproduces it exactly, so there is nothing about this quantity that float32
    cannot represent.
    """
    groups = dict(
        sfh=_SFH,
        dust=_DUST,
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
    )
    kw = dict(lo=0.0, hi=1e12)

    names, _, _, g64 = _band_gradient(ssp_bare, obs, groups, x64=True, dtype=jnp.float64, **kw)
    _, _, v32, g32 = _band_gradient(ssp_bare, obs, groups, x64=False, dtype=jnp.float32, **kw)

    assert np.all(np.isfinite(g64)), (
        f"float64 gradient is already non-finite ({g64}) — the setup is wrong"
    )
    assert np.isfinite(v32), (
        f"float32 forward value is non-finite ({v32}) — a different bug from #1439"
    )
    assert np.all(np.isfinite(g32)), (
        f"float32 SED gradient is non-finite (names={names}, f32={g32}, f64={g64}) "
        "while the forward pass is exact — #1439"
    )

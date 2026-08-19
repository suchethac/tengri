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
_DUST = {"law_diff": 'calzetti', 
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


def _forward_mode_lbol_gradient(ssp, obs, groups, *, lo, hi, x64, dtype, with_value=False):
    """``d(band sum)/d(agn_log_lbol)`` by ``jvp`` — the same quantity, other mode.

    Deliberately a separate helper rather than a flag on :func:`_band_gradient`:
    the two differ only in which autodiff mode runs, and keeping the model build
    and masking identical is what makes the comparison attributable.
    """
    with jax.enable_x64(x64):
        model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)
        names = sorted(n for n in model.spec.free_params if n in _TRUTH)
        wave = np.asarray(model._rest_wavelength, dtype=np.float64)
        mask = jnp.asarray((wave >= lo) & (wave < hi))
        idx = names.index("agn_log_lbol")

        def band_sum(lbol):
            params = {
                k: (lbol if i == idx else jnp.asarray(_TRUTH[k], dtype=dtype))
                for i, k in enumerate(names)
            }
            return jnp.sum(jnp.where(mask, model.predict(params).rest_sed(), 0.0))

        base = jnp.asarray(_TRUTH["agn_log_lbol"], dtype=dtype)
        value, tangent = jax.jvp(band_sum, (base,), (jnp.ones_like(base),))
        if with_value:
            return float(np.asarray(value)), float(np.asarray(tangent))
        return float(np.asarray(tangent))


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
    reason="#1439 residual, now narrowed to REVERSE MODE on the SHAPE-CLASS discs. "
    "d(sum rest_sed)/d(agn_log_lbol) is NaN in pure float32 for 'multicolor' and "
    "'kubota_done' — the discs that take agn_log_lbol_shape, so the true L_bol drives "
    "the shape while the magnitude rides the reference. That creates two paths through "
    "the log-space renormalization, and reverse mode fails to cancel them. It is not a "
    "single bad point: on multicolor_disc alone the reverse gradient degrades SMOOTHLY "
    "with luminosity — 1.02x at log L_bol 9, 1.54x at 10, 2.14x at 11, 2.33x at 12.5 — "
    "finite and plausible the whole way, across the entire realistic AGN range, and it "
    "becomes NaN once the full chain runs. Degrading with magnitude separation is the "
    "#1388 class (a reverse-mode cancellation no local rule reaches), which is why the "
    "fix is the scaled-SED contract and not another regrouping here. Three things are "
    "NOT the cause, "
    "each ruled out by A/B measurement: the Planck reciprocal (identical to the last "
    "digit with and without its custom_jvp), the EUV tail (all four options give 2.13-"
    "2.14x), and the ring count (n_radii=8 gives 2.33x, so it is not accumulation). "
    "Scope: forward mode is EXACT (1.0000 vs float64) for every disc, and the "
    "shape-invariant 'richards2006' disc is exact in BOTH modes — see the two passing "
    "companions below. predict_photometry gradients are unaffected, so inference "
    "through photometry is not blocked.",
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


def _agn_groups(disc):
    return dict(
        sfh=_SFH,
        dust=_DUST,
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": disc, "all_params": FIXED},
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
    )


def test_agn_sed_forward_mode_gradient_is_exact_in_float32(ssp_bare, obs):
    """Forward mode gets the AGN SED gradient right where reverse mode does not.

    This is what makes the xfail above a statement about **reverse mode** rather
    than about float32: the quantity is representable and float32 computes it.

    It is also the guard that the forward *value* stays finite. Until the rest
    grid was made to follow the session precision (#1206/#1439), a composable AGN
    with no torus left the grid float64, every ``wave.dtype == jnp.float32`` gate
    in components/ fell through to its float64 branch, and ``sed_agn`` was NaN at
    all 5994 points. The strict xfail above absorbed that silently — it was still
    "failing", just for a different reason than its text claimed. A passing test
    is what keeps the forward path honest.
    """
    groups = _agn_groups("multicolor")
    kw = dict(lo=0.0, hi=1e12)

    with jax.enable_x64(True):
        _, _, v64, _ = _band_gradient(ssp_bare, obs, groups, x64=True, dtype=jnp.float64, **kw)
    g64 = _forward_mode_lbol_gradient(ssp_bare, obs, groups, x64=True, dtype=jnp.float64, **kw)
    v32_g32 = _forward_mode_lbol_gradient(
        ssp_bare, obs, groups, x64=False, dtype=jnp.float32, with_value=True, **kw
    )
    v32, g32 = v32_g32

    assert np.isfinite(v64) and np.isfinite(g64), f"setup: float64 is {v64}, {g64}"
    assert np.isfinite(v32), (
        f"float32 forward VALUE is non-finite ({v32}) — a dtype gate fell through "
        "to its float64 branch (#1439)"
    )
    assert abs(v32 - v64) / abs(v64) < 1e-3, f"float32 value {v32:.6e} vs {v64:.6e}"
    assert np.isfinite(g32), f"float32 forward-mode gradient is non-finite ({g32})"
    assert abs(g32 - g64) / abs(g64) < 1e-3, (
        f"float32 forward-mode d/d(agn_log_lbol) {g32:.6e} vs float64 {g64:.6e}"
    )


def test_shape_invariant_disc_sed_gradient_is_exact_in_float32(ssp_bare, obs):
    """A shape-invariant disc is exact in BOTH modes — the xfail's other boundary.

    ``richards2006`` does not take ``agn_log_lbol_shape``, so its magnitude rides the
    reference evaluation with no second path through the renormalization. That it
    passes in reverse mode is what localizes the residual defect to the
    shape-class discs rather than to the AGN path as a whole.
    """
    groups = _agn_groups("richards2006")
    kw = dict(lo=0.0, hi=1e12)

    names, _, v64, g64 = _band_gradient(ssp_bare, obs, groups, x64=True, dtype=jnp.float64, **kw)
    _, _, v32, g32 = _band_gradient(ssp_bare, obs, groups, x64=False, dtype=jnp.float32, **kw)

    assert np.all(np.isfinite(g64)), f"setup: float64 gradient is {g64}"
    assert np.isfinite(v32), f"float32 forward value is non-finite ({v32})"
    assert abs(v32 - v64) / abs(v64) < 1e-3, f"float32 value {v32:.6e} vs {v64:.6e}"
    assert np.all(np.isfinite(g32)), (
        f"float32 reverse-mode gradient is non-finite (names={names}, f32={g32})"
    )
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 reverse-mode SED gradient disagrees with float64 by {rel.max():.2e} "
        f"(names={names}, f32={g32}, f64={g64})"
    )

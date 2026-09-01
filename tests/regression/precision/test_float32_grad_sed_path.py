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

On ``multicolor`` that is now **fixed**: the disc's float32 renormalization returned
``l_nu * scale``, and transposing that product forms ``sum(g * l_nu)`` — ~1e64 with the
cotangent the AGN reference offset supplies, so ``inf``, against a partner term that
underflows to ``0``. Returning the L1-normalized SED against the correspondingly
inflated scale is the same number with both factors in range. ``kubota_done`` is a
different defect at the same call site and is still open; see its strict xfail.
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
_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"]))


def _band_gradient(ssp, obs, groups, *, lo, hi, x64, dtype, at=None):
    """(names, band L_nu, gradient) of the rest-frame SED summed over [lo, hi) Å.

    ``at`` overrides individual entries of :data:`_TRUTH` — used to sweep
    ``agn_log_lbol`` across its declared prior, where the reverse-mode defect of
    #1439 varied smoothly with luminosity rather than failing at one point.
    """
    truth = {**_TRUTH, **(at or {})}
    with jax.enable_x64(x64):
        model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)
        names = sorted(n for n in model.spec.free_params if n in truth)
        wave = np.asarray(model._rest_wavelength, dtype=np.float64)
        mask = jnp.asarray((wave >= lo) & (wave < hi))

        def band_sum(values):
            params = {k: values[i] for i, k in enumerate(names)}
            return jnp.sum(jnp.where(mask, model.predict(params).rest_sed(), 0.0))

        base = [jnp.asarray(truth[k], dtype=dtype) for k in names]
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
    radio_spec = {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}
    groups = dict(sfh=_SFH, dust_attenuation=_DUST, radio=radio_spec)
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


@pytest.mark.parametrize("log_lbol", [9.0, 11.0, 12.0])
def test_multicolor_agn_sed_gradient_is_accurate_in_float32(ssp_bare, obs, log_lbol):
    """``d(sum rest_sed)/d(agn_log_lbol)`` must track float64 across the whole prior.

    This was #1439's strict xfail: NaN in pure float32 while the forward pass and
    ``jacfwd`` were both exact. The cause was **not** a cancellation no local rule
    could reach — it was the grouping of ``multicolor_disc``'s float32
    renormalization. Transposing ``arr * scale`` makes JAX form ``sum(g * arr)``, and
    with the raw ~1e28 disc SED against the ~10**34.6 cotangent the AGN reference
    offset hands back, that inner product is ~1e64: ``inf`` in float32, while its
    partner ``d scale/d arr`` ~1e-64 flushes to 0, and ``inf * 0`` is NaN. Returning
    the L1-normalized SED times the correspondingly inflated scale — algebraically the
    same number — keeps both factors in range.

    Swept across the declared ``Uniform(9, 12)`` rather than measured at one point,
    because the defect varied smoothly with luminosity (the recorded 1.02x at 9,
    2.14x at 11) before it became NaN: a single-point check could have landed where
    the error was small. Measured after the fix: 1.000002 at every point here.

    Not covered: ``kubota_done``, which takes the same ``agn_log_lbol_shape`` hand-off
    and is still wrong — see the strict xfail below.
    """
    groups = dict(
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
    )
    kw = dict(lo=0.0, hi=1e12, at={"agn_log_lbol": log_lbol})

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
        "while the forward pass is exact — multicolor_disc's float32 renormalization "
        "lost its L1 factorization (#1439)"
    )
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 SED gradient disagrees with float64 by {rel.max():.2e} at "
        f"log L_bol = {log_lbol} (names={names}, f32={g32}, f64={g64})"
    )


@pytest.mark.xfail(
    reason="#1439 residual, now narrowed to 'kubota_done' alone — 'multicolor' is "
    "fixed and pinned by the sweep above. This is NOT the same defect: it is not a "
    "range problem at all. With an O(1) cotangent, where nothing can overflow, "
    "d(sum L_nu)/d(agn_log_lbol) in pure float32 is -0.034x float64 — SIGN FLIPPED — "
    "and it becomes NaN only once the cotangent passes ~1e10. Localized by A/B "
    "measurement: agn_f_hard=0.0 (no hot corona) restores float32/float64 agreement "
    "to 3e-04, and every nonzero agn_f_hard reproduces the -0.034x exactly, so the "
    "defect is in the hot-corona zone (_hot_corona_lnu / the nthcomp custom_jvp of "
    "#1822), not in the disc renormalization. Regrouping the renormalization the way "
    "multicolor_disc's is regrouped was written and measured here: it does not close "
    "this, and was reverted rather than shipped unverified.",
    strict=True,
)
def test_kubota_done_agn_sed_gradient_is_accurate_in_float32(ssp_bare, obs):
    """The other shape-class disc, pinned so its state cannot change silently."""
    groups = _agn_groups("kubota_done")
    kw = dict(lo=0.0, hi=1e12)

    names, _, _, g64 = _band_gradient(ssp_bare, obs, groups, x64=True, dtype=jnp.float64, **kw)
    _, _, v32, g32 = _band_gradient(ssp_bare, obs, groups, x64=False, dtype=jnp.float32, **kw)

    assert np.isfinite(v32), f"float32 forward value is non-finite ({v32})"
    assert np.all(np.isfinite(g32)), (
        f"float32 SED gradient is non-finite (names={names}, f32={g32}, f64={g64})"
    )
    rel = np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-300)
    assert rel.max() < 1e-3, (
        f"float32 SED gradient disagrees with float64 by {rel.max():.2e} "
        f"(names={names}, f32={g32}, f64={g64})"
    )


def _agn_groups(disc):
    return dict(
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": disc, "all_params": Fixed(DEFAULT)},
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
    )


def test_agn_sed_forward_mode_gradient_is_exact_in_float32(ssp_bare, obs):
    """Forward mode gets the AGN SED gradient right where reverse mode did not.

    This is what made #1439 a statement about **reverse mode** rather than about
    float32: the quantity is representable and float32 computes it in either mode.
    Reverse mode now agrees (the sweep above); keeping the forward-mode measurement
    is what would separate a future regression in the *transpose* from one in the
    disc physics, which is how the multicolor defect was localized in the first place.

    It is also the guard that the forward *value* stays finite. Until the rest
    grid was made to follow the session precision (#1206/#1439), a composable AGN
    with no torus left the grid float64, every ``wave.dtype == jnp.float32`` gate
    in components/ fell through to its float64 branch, and ``sed_agn`` was NaN at
    all 5994 points. The strict xfail that used to stand here absorbed that
    silently — it was still "failing", just for a different reason than its text
    claimed. A passing test is what keeps the forward path honest.
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
    """A shape-invariant disc is exact in BOTH modes — the other boundary of #1439.

    ``richards2006`` does not take ``agn_log_lbol_shape``, so its magnitude rides the
    reference evaluation with no second path through the renormalization. That it
    passed in reverse mode while the shape-class discs did not is what localized the
    defect to them rather than to the AGN path as a whole.
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

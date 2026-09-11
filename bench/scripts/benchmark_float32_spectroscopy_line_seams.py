# SPDX-License-Identifier: BSD-3-Clause
"""Float32 posterior-gradient accuracy on the **spectroscopy** and **line** channels.

PR #2104 (``bench/reports/2026-08-31_float32_fitting_path.md``) closed the projector,
free-redshift and CUDA axes **on photometry** and said, in its own "what is still NOT
covered" section, what it had not reached:

    **Spectroscopy.** ``predict_spectrum`` applies the same projection, and
    ``SpectrumPrecomp`` is the LUT's spectroscopy sibling with its own measured
    ~1-sigma posterior shift (#1688). [...] **Emission-line fluxes.**
    ``line_measurement.py`` applies its own combined ``log10_conv -
    log10_four_pi_dl2`` offset, and ``FeaturePrecomp`` serves the line channel from
    a table. Neither is measured here --- and note #1770's lesson that a
    photometry-surface measurement says nothing about the line channel.

This script adds the ``data_type`` axis. It is the photometry script's method, not a
new one: float32 against **float64 autodiff**, the float64 reference itself checked
against float64 central differences at the same points, enumerated **per seam**
(#1436) rather than in aggregate, at **two SNRs** because the LUT bias enters the
gradient multiplied by SNR (#1671), and with precision proven on the **dtype of the
gradient array that came back** (#1840 --- ``tengri/__init__.py`` re-enables x64 on
import, so the config flag lies).

The line channel is **two** seams, not one, because tengri has two line operators and
they share no arithmetic:

``lines_cue``
    A backend publishing a discrete line catalog (Cue). ``loss_functions.py`` routes
    to ``model.predict_line_fluxes``; ``FeaturePrecomp`` replaces the *emulator call*
    with a grid over the ionization axes.
``lines_meas``
    A baked-in (wNE) SSP publishes no catalog, so ``model._has_line_catalog()`` is
    False and the same loss routes to ``model.measure_line_fluxes`` --- the
    ``line_measurement.py`` window path that applies the combined
    ``log10_conv - log10_four_pi_dl2`` offset through ``apply_log10_scale``.
    ``FeaturePrecomp`` here is a per-line **window LUT** of SSP integrals instead.

That offset is the reason this channel cannot be extrapolated from photometry. A line
luminosity is ~1e40 erg/s (float32 max 3.4e38) and 4*pi*d_L^2 is ~1e57; #1859 keeps
both out of the arithmetic by grouping them into one ~-45 dex exponent, and #1415
holds the factorization peak under ``stop_gradient`` because in float32 the two
autodiff paths do not cancel --- that defect was gradients exactly **2x** too large.
Whether the *combined* offset is as safe as the photometry one is what this measures.

Usage
-----
::

    JAX_PLATFORMS=cpu python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
        --snr 30 300 --out bench/results/2026-09-05_float32_spec_lines_cpu.json

    JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
        --snr 30 300 --out bench/results/2026-09-05_float32_spec_lines_cuda.json

    # the unweighted-observable path under the LUT (PR #2100 measured it exact-only)
    JAX_PLATFORMS=cpu python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
        --mode unweighted --out bench/results/2026-09-05_float32_unweighted_lut.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from tengri import (
    DEFAULT,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
)
from tengri.inference.context import InferenceContext
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.spectroscopy import Spectroscopy

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

#: Bare-stellar grid --- the same library PR #2104 used, so a ``phot`` control row here
#: is directly comparable to a published row there.
SSP_BARE = "data/fsps_prsc_miles_chabrier.h5"
#: Baked-in nebular (wNE) grid. Its lines live *inside* the SSP templates, which is what
#: makes ``_has_line_catalog()`` False and routes the loss through ``line_measurement``.
SSP_WNE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

_DUST_FREE = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}

_AGN = {
    "type": "composable",
    "all_params": Fixed(DEFAULT),
    "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
    "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
    "norm": "cigale_joint",
    "log_lbol": Fixed(10.5),
    "fracAGN": 0.1,
}

#: Model groups. ``stellar_dust`` and ``panchromatic`` are PR #2104's, unchanged, so the
#: spectroscopy rows can be read against the photometry rows for the same model.
MODELS = {
    "stellar_dust": dict(dust_attenuation=_DUST_FREE),
    "neb_cue": dict(
        dust_attenuation=_DUST_FREE,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
    ),
    "panchromatic": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        agn=_AGN,
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        xray={"type": "simple"},
        shock={"frac": 0.1},
    ),
    #: Baked-in: no nebular *component* at all, the lines are in the templates.
    "bakedin": dict(dust_attenuation=_DUST_FREE, neb={"type": "none"}),
}

#: Which SSP library each model needs.
MODEL_SSP = {
    "stellar_dust": SSP_BARE,
    "neb_cue": SSP_BARE,
    "panchromatic": SSP_BARE,
    "bakedin": SSP_WNE,
}

Z_LO, Z_HI = 0.05, 1.0
#: Coarse on purpose, exactly as PR #2104: ``n_z`` sets the LUT's *own* redshift
#: interpolation bias, which is common to both precisions and cancels out of the
#: float32-vs-float64 comparison while costing ~20 s of build per model per precision.
N_Z = 64

#: Optical spectrum: 256 pixels over 4000-9000 A. Wide enough to carry the Balmer
#: decrement and the 4000-A break, small enough that a seam builds in seconds.
SPEC_WAVE = np.linspace(4000.0, 9000.0, 256)

#: Four strong optical lines, rest-frame vacuum [A] (NIST / FastSpecFit convention).
LINE_NAMES = ("Hbeta", "OIII_5007", "Halpha", "NII_6584")
LINE_WAVES = np.array([4862.68, 5008.24, 6564.61, 6585.28])

#: How the per-line 1-sigma is set. This is **not** cosmetic: it decides how stiff the
#: line channel is, and the two conventions do not measure the same thing.
#:
#: ``"per_line"``   ``sigma_i = |pred_i| / snr`` --- every line gets the same *fractional*
#:                  precision. It is the spelling the repo's own line fixtures use, and
#:                  on a line the model predicts near zero (the baked-in NII_6584 comes
#:                  out at ~3e-18 against Halpha's ~1e-15) it hands that line a ~1e-19
#:                  error bar, so one nearly-empty channel acquires most of the
#:                  chi-squared curvature. No instrument produces that.
#: ``"floored"``    ``sigma_i = max(|pred_i|, floor * max|pred|) / snr`` --- a
#:                  flux-limited convention where the weak lines are not measured to a
#:                  part in 1e19. This is the default because it is the realistic one.
#:
#: Both are measured and both are reported: the gap between them is a property of the
#: *fixture*, and quoting a float32 verdict without saying which one was used would be
#: quoting the fixture rather than the arithmetic.
LINE_SIGMA_FLOOR = 0.05


def line_sigma(pred, snr, convention):
    """Per-line 1-sigma [erg/s/cm^2] under one of the two conventions above."""
    a = np.abs(np.asarray(pred, dtype=np.float64))
    if convention == "per_line":
        return a / snr
    if convention == "floored":
        return np.maximum(a, LINE_SIGMA_FLOOR * np.max(a)) / snr
    raise ValueError(f"unknown line sigma convention {convention!r}")


def base_params(zspec):
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


# --------------------------------------------------------------------------------------
# Channels --- the axis this script adds
# --------------------------------------------------------------------------------------

_PHOT4 = ["sdss_g", "sdss_r", "wise_w1", "herschel_250"]
_PHOT2 = ["sdss_g", "sdss_r"]


def make_obs(channel, line_data=None):
    """The ``Observation`` for one channel.

    ``line_data`` carries the *measured* line fluxes; it is None on the first pass that
    generates them and a :class:`LineFluxData` on the second.
    """
    if channel == "phot":
        return Observation(photometry=Photometry.from_names(_PHOT4))
    if channel == "spec":
        return Observation(spectroscopy=Spectroscopy(wave_obs=jnp.asarray(SPEC_WAVE)))
    if channel == "joint":
        return Observation(
            photometry=Photometry.from_names(_PHOT4),
            spectroscopy=Spectroscopy(wave_obs=jnp.asarray(SPEC_WAVE)),
        )
    if channel in ("lines_cue", "lines_meas"):
        # A line channel rides *alongside* photometry: `Fitter` requires data and noise,
        # so a lines-only Observation cannot be fitted through this constructor. The
        # ``phot2`` control below is the same model and bands with the line channel
        # removed, which is what makes the line term's contribution attributable.
        return Observation(
            photometry=Photometry.from_names(_PHOT2),
            line_fluxes=line_data,
        )
    if channel == "phot2":
        return Observation(photometry=Photometry.from_names(_PHOT2))
    raise ValueError(f"unknown channel {channel!r}")


#: ``channel -> (models, needs_line_data)``.
CHANNEL_MODELS = {
    "phot": ("stellar_dust", "neb_cue", "panchromatic"),
    "spec": ("stellar_dust", "neb_cue", "panchromatic"),
    "joint": ("neb_cue",),
    "lines_cue": ("neb_cue",),
    "lines_meas": ("bakedin",),
    "phot2": ("neb_cue", "bakedin"),
}


def approx_for(channel, kind, n_z=N_Z):
    """Build-time ``approx=`` for one channel and one projector ``kind``.

    ``kind`` is ``"exact"`` (no LUT anywhere) or ``"lut"`` (the channel's own LUT named
    explicitly). ``"auto"`` is not spelled here: it is what ``Fitter(approx="auto")``
    resolves to, and the whole point of PR #2104's Finding 0 is that the *fitter's*
    knob is the one that decides.
    """
    if kind == "exact":
        return None
    wp = WavePrecomp(n_z=n_z, z_min=Z_LO, z_max=Z_HI)
    sp = SpectrumPrecomp(n_z=n_z, z_min=Z_LO, z_max=Z_HI)
    if channel in ("phot", "phot2", "lines_cue", "lines_meas"):
        return wp
    if channel == "spec":
        return sp
    if channel == "joint":
        return (wp, sp)
    raise ValueError(channel)


#: ``path -> (approx_kind, fit_approx, redshift_factory)``. ``fit_approx`` is what
#: reaches ``Fitter(approx=...)`` and it is the one that decides (PR #2104 Finding 0).
PATHS = {
    "exact_fixedz": ("exact", None, lambda: Fixed(0.1)),
    "auto_fixedz": ("exact", "auto", lambda: Fixed(0.1)),
    "exact_freez": ("exact", None, lambda: Uniform(Z_LO, Z_HI)),
    "auto_freez": ("exact", "auto", lambda: Uniform(Z_LO, Z_HI)),
}

DEFAULT_PATHS = ("exact_fixedz", "auto_fixedz", "exact_freez", "auto_freez")

#: Evaluation points in standardized (unbounded) space. The origin is where residuals
#: are smallest and float32 cancellation is worst; 0.5 sigma is a generic interior point.
POINTS = {"origin": 0.0, "half_sigma": 0.5}

#: Central-difference step in standardized units, for the float64 soundness check only.
FD_H = 1e-4


# --------------------------------------------------------------------------------------
# Model / data construction
# --------------------------------------------------------------------------------------


def build(ssp, obs, model, approx, zspec):
    return SEDModel.build(
        ssp_data=ssp, observation=obs, approx=approx, **base_params(zspec), **MODELS[model]
    )


def _truth(sed):
    return {
        n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
        for n in sed.spec.free_params
    }


def make_mock(ssp, channel, model, zspec, snr, seed=0, sigma_convention="floored"):
    """One float64 mock from the **exact** projector, so every arm fits identical data.

    Returns ``(flux, noise, line_data, truth)``. ``line_data`` is None off the line
    channels. Every observable is drawn as ``prediction + N(0, |prediction|/snr)``,
    which is what ``SEDModel.mock`` does for photometry; the line channel gets the same
    treatment through whichever operator the loss will use for it, so the mock is
    self-consistent with the likelihood rather than merely plausible.
    """
    key = jax.random.PRNGKey(seed)
    with jax.enable_x64(True):
        if channel in ("lines_cue", "lines_meas"):
            # Pass 1: a placeholder line schema, only so the model will build.
            ph = LineFluxData(
                names=LINE_NAMES,
                fluxes=np.ones(len(LINE_NAMES)),
                errors=np.ones(len(LINE_NAMES)),
                wavelengths=LINE_WAVES,
            )
            sed = build(ssp, make_obs(channel, ph), model, None, zspec)
            truth = _truth(sed)
            if sed._has_line_catalog():
                pred = np.asarray(
                    sed.predict_line_fluxes(truth, target_wavelengths=jnp.asarray(LINE_WAVES)),
                    dtype=np.float64,
                )
            else:
                from tengri.observation.line_measurement import default_line_defs

                pred = np.asarray(
                    sed.measure_line_fluxes(
                        truth, default_line_defs(LINE_WAVES, LINE_NAMES), approx=False
                    ),
                    dtype=np.float64,
                )
            lsig = line_sigma(pred, snr, sigma_convention)
            k1, k2 = jax.random.split(key)
            lnoise = np.asarray(
                jax.random.normal(k1, (len(LINE_NAMES),), dtype=jnp.float64), dtype=np.float64
            )
            line_data = LineFluxData(
                names=LINE_NAMES,
                fluxes=pred + lnoise * lsig,
                errors=lsig,
                wavelengths=LINE_WAVES,
            )
            # The photometry riding alongside, from the same model at the same SNR.
            sed2 = build(ssp, make_obs(channel, line_data), model, None, zspec)
            m = sed2.mock(truth, snr=snr, key=k2)
            return (
                np.asarray(m.flux_obs, dtype=np.float64),
                np.asarray(m.noise, dtype=np.float64),
                line_data,
                truth,
            )

        sed = build(ssp, make_obs(channel), model, None, zspec)
        truth = _truth(sed)
        if channel == "spec":
            m = sed.mock_spectrum(truth, jnp.asarray(SPEC_WAVE), snr=snr, key=key)
        else:
            m = sed.mock(truth, snr=snr, key=key)
        return (
            np.asarray(m.flux_obs, dtype=np.float64),
            np.asarray(m.noise, dtype=np.float64),
            None,
            truth,
        )


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def grad_at(
    ssp,
    channel,
    model,
    approx_kind,
    fit_approx,
    zspec,
    flux,
    noise,
    line_data,
    *,
    x64,
    dtype,
    n_z=N_Z,
    fd=False,
):
    """One precision's gradients of ``neg_log_posterior_fn`` at every point in POINTS."""
    with jax.enable_x64(x64):
        obs = make_obs(channel, line_data)
        sed = build(ssp, obs, model, approx_for(channel, approx_kind, n_z), zspec)
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
            "has_line_catalog": bool(sed._has_line_catalog()),
            "n_data": int(np.asarray(flux).size),
        }
        for label, offset in POINTS.items():
            point = [jnp.asarray(offset, dtype=dtype) for _ in names]
            g = jax.grad(nlp)(point)
            rec = {
                "grad": [float(np.asarray(x)) for x in g],
                # The precision proof: an output array's dtype, never the config flag.
                "grad_dtype": sorted({str(np.asarray(x).dtype) for x in g}),
                "value": float(np.asarray(nlp(point))),
            }
            if fd:
                cd = []
                for i in range(len(names)):
                    p, m = list(point), list(point)
                    p[i] = jnp.asarray(offset + FD_H, dtype=dtype)
                    m[i] = jnp.asarray(offset - FD_H, dtype=dtype)
                    cd.append(float((np.asarray(nlp(p)) - np.asarray(nlp(m))) / (2 * FD_H)))
                rec["fd"] = cd
            out[label] = rec
        return out


def rel_norm(a, b):
    """Relative deviation in the 2-norm --- what a sampler actually consumes.

    Componentwise relative error is unbounded on a direction whose float64 gradient
    passes through zero, and this inventory has such directions.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def rel_max(a, b):
    """Max componentwise relative deviation, floored on the largest component."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1e-3 * max(np.max(np.abs(b)), 1e-300))
    return float(np.max(np.abs(a - b) / denom))


def run_seam(ssp_cache, channel, model, path, snr, n_z, sigma_convention="floored"):
    approx_kind, fit_approx, zfac = PATHS[path]
    ssp = ssp_cache(MODEL_SSP[model])
    flux, noise, line_data, truth = make_mock(
        ssp, channel, model, zfac(), snr, sigma_convention=sigma_convention
    )

    t0 = time.time()
    kw = dict(
        channel=channel,
        model=model,
        zspec=zfac(),
        flux=flux,
        noise=noise,
        line_data=line_data,
        n_z=n_z,
    )
    f64 = grad_at(
        ssp,
        approx_kind=approx_kind,
        fit_approx=fit_approx,
        x64=True,
        dtype=jnp.float64,
        fd=True,
        **kw,
    )
    f32 = grad_at(
        ssp, approx_kind=approx_kind, fit_approx=fit_approx, x64=False, dtype=jnp.float32, **kw
    )
    # Truly-exact float64 reference, to separate the LUT's own bias from float32's error.
    exact64 = (
        f64
        if fit_approx is None
        else grad_at(ssp, approx_kind="exact", fit_approx=None, x64=True, dtype=jnp.float64, **kw)
    )

    rec = {
        "channel": channel,
        "model": model,
        "path": path,
        "snr": snr,
        "line_sigma_convention": sigma_convention if channel.startswith("lines") else None,
        "names": f64["names"],
        "data_type": f64["data_type"],
        "has_line_catalog": f64["has_line_catalog"],
        "n_data": f64["n_data"],
        "approx_state_f64": f64["approx_state"],
        "approx_state_f32": f32["approx_state"],
        "truth": truth,
        "seconds": round(time.time() - t0, 2),
        "points": {},
    }
    for label in POINTS:
        a64, a32, ax = f64[label], f32[label], exact64[label]
        rec["points"][label] = {
            "grad_f64": a64["grad"],
            "grad_f32": a32["grad"],
            "grad_exact_f64": ax["grad"],
            "dtype_f64": a64["grad_dtype"],
            "dtype_f32": a32["grad_dtype"],
            "value_f64": a64["value"],
            "f64_vs_fd64": rel_norm(a64["grad"], a64["fd"]),
            "f32_vs_f64": rel_norm(a32["grad"], a64["grad"]),
            "f32_vs_f64_cw": rel_max(a32["grad"], a64["grad"]),
            "lut_f64_vs_exact_f64": rel_norm(a64["grad"], ax["grad"]),
            "f32_vs_exact_f64": rel_norm(a32["grad"], ax["grad"]),
            "f32_finite": bool(np.all(np.isfinite(a32["grad"]))),
            "f32_nonzero": bool(np.any(np.asarray(a32["grad"]) != 0.0)),
        }
    return rec


# --------------------------------------------------------------------------------------
# Mode 2: the unweighted-observable path under the LUT
# --------------------------------------------------------------------------------------


def run_unweighted(ssp_cache, channel, model, kind, n_z):
    """``grad(sum(predict_*))`` in float32 --- the path with no ``1/sigma**2`` lift.

    PR #2100 measured this on the **exact** projector only. The trap it recorded is why
    it is worth re-taking under the LUT: the bare float32 gradient was once
    **identically zero** on both CPU and GPU, silently, and the guard that existed
    pinned it *finite* --- and zero is finite. So the assertion here is **non-zero**,
    not merely finite.

    **The surface matters and the obvious one is wrong.** ``model.predict_photometry`` /
    ``model.predict_spectrum`` are the lean JIT/vmap surfaces, and they are what
    ``WavePrecomp`` / ``SpectrumPrecomp`` actually route (CLAUDE.md: opting in "routes
    ``predict_photometry`` through ``observation.predict_via_precomp``"). The rich
    ``model.predict(params).photometry()`` accessor does **not** take the LUT: measured
    on 2026-09-05, an ``exact`` and a ``wave_precomp=True`` model built from the same
    spec gave **bit-identical** float64 values and gradients through that accessor, which
    is #1748's stated signature of "a config that never reaches the graph". A first pass
    of this function used the rich accessor and its "LUT" column was therefore the exact
    column wearing another label --- the same class of mistake as PR #2104's Finding 0,
    one surface further out. ``lut_engaged`` below records the check rather than assuming
    it.
    """
    from tengri.utils.scale import DEFAULT_COTANGENT_BOOST, loss_scaled_grad

    ssp = ssp_cache(MODEL_SSP[model])
    out = {"channel": channel, "model": model, "approx_kind": kind}
    for x64, dtype, tag in ((True, jnp.float64, "f64"), (False, jnp.float32, "f32")):
        with jax.enable_x64(x64):
            obs = make_obs(channel)
            sed = build(ssp, obs, model, approx_for(channel, kind, n_z), Fixed(0.1))
            names = sorted(sed.spec.free_params)
            truth = _truth(sed)

            def observable(vals, _sed=sed, _names=names, _truth=truth, _channel=channel):
                p = dict(_truth)
                p.update({k: vals[i] for i, k in enumerate(_names)})
                # The LEAN surfaces, which are the ones the LUT routes, and with NO
                # explicit wavelength grid (wave_obs=None -> the observation's own grid,
                # which is what SpectrumPrecomp is built for; passing it explicitly
                # bypasses the LUT). Not
                # ``_sed.predict(p).photometry()`` --- see the docstring.
                arr = (
                    _sed.predict_spectrum(p) if _channel == "spec" else _sed.predict_photometry(p)
                )
                return jnp.sum(arr)

            point = [jnp.asarray(float(truth[k]), dtype=dtype) for k in names]
            g = jax.grad(observable)(point)
            gb = loss_scaled_grad(observable)(point)
            out[tag] = {
                "names": names,
                "grad": [float(np.asarray(x)) for x in g],
                "grad_boosted": [float(np.asarray(x)) for x in gb],
                "grad_dtype": sorted({str(np.asarray(x).dtype) for x in g}),
                "approx_state": str(getattr(sed, "approx", None)),
                "value": float(np.asarray(observable(point))),
                "boost": float(DEFAULT_COTANGENT_BOOST),
            }
    g32 = np.asarray(out["f32"]["grad"], dtype=np.float64)
    g64 = np.asarray(out["f64"]["grad"], dtype=np.float64)
    b32 = np.asarray(out["f32"]["grad_boosted"], dtype=np.float64)
    out["f32_all_zero"] = bool(np.all(g32 == 0.0))
    out["f32_boosted_all_zero"] = bool(np.all(b32 == 0.0))
    out["f32_finite"] = bool(np.all(np.isfinite(g32)))
    out["f32_vs_f64"] = rel_norm(g32, g64)
    out["f32_boosted_vs_f64"] = rel_norm(b32, g64)
    out["f64_boost_moves_f64"] = rel_norm(
        np.asarray(out["f64"]["grad_boosted"], dtype=np.float64), g64
    )
    out["f64_boost_bit_identical"] = bool(
        np.array_equal(np.asarray(out["f64"]["grad_boosted"], dtype=np.float64), g64)
    )
    return out


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _ssp_cache():
    cache = {}

    def get(path):
        if path not in cache:
            from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

            cache[path] = load_ssp_data(path)
        return cache[path]

    return get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("seams", "unweighted"), default="seams")
    ap.add_argument(
        "--channels", nargs="+", default=["spec", "joint", "lines_cue", "lines_meas", "phot2"]
    )
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    ap.add_argument("--snr", nargs="+", type=float, default=[30.0, 300.0])
    ap.add_argument("--n-z", type=int, default=N_Z)
    ap.add_argument(
        "--line-sigma",
        choices=("floored", "per_line"),
        default="floored",
        help="per-line 1-sigma convention; see LINE_SIGMA_FLOOR",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    ssp_cache = _ssp_cache()

    meta = {
        "platform": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax": jax.__version__,
        "env": {
            k: os.environ.get(k)
            for k in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "NVIDIA_TF32_OVERRIDE",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "TENGRI_DISABLE_JAX_CACHE",
            )
        },
        "n_z": args.n_z,
        "points": POINTS,
        "spec_pixels": int(SPEC_WAVE.size),
        "spec_wave_range": [float(SPEC_WAVE[0]), float(SPEC_WAVE[-1])],
        "lines": list(LINE_NAMES),
        "mode": args.mode,
        "line_sigma_convention": args.line_sigma,
        "line_sigma_floor": LINE_SIGMA_FLOOR,
    }
    print(json.dumps(meta, indent=2), flush=True)

    rows = []
    if args.mode == "unweighted":
        for channel in args.channels:
            models = args.models or CHANNEL_MODELS[channel]
            for model in models:
                by_kind = {}
                for kind in ("exact", "lut"):
                    try:
                        rec = run_unweighted(ssp_cache, channel, model, kind, args.n_z)
                        by_kind[kind] = rec
                    except Exception as exc:
                        rec = {
                            "channel": channel,
                            "model": model,
                            "approx_kind": kind,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(rec)
                    jax.clear_caches()
                    gc.collect()
                    # Did the LUT actually reach the graph? Bit-identical float64
                    # gradients between the two projectors mean it did not (#1748's
                    # signature), and a "LUT" row would then be the exact row relabeled.
                    if kind == "lut" and "exact" in by_kind and "error" not in rec:
                        ge = np.asarray(by_kind["exact"]["f64"]["grad"], dtype=np.float64)
                        gl = np.asarray(rec["f64"]["grad"], dtype=np.float64)
                        rec["lut_engaged"] = not bool(np.array_equal(ge, gl))
                        rec["lut_f64_vs_exact_f64"] = rel_norm(gl, ge)
                    if "error" in rec:
                        print(f"{channel:11s} {model:13s} {kind:6s} ERROR {rec['error'][:140]}")
                    else:
                        print(
                            f"{channel:11s} {model:13s} {kind:6s} "
                            f"f32_all_zero={rec['f32_all_zero']!s:5s} "
                            f"boosted_all_zero={rec['f32_boosted_all_zero']!s:5s} "
                            f"f32/f64={rec['f32_vs_f64']:.2e} "
                            f"boost64_moves={rec['f64_boost_moves_f64']:.2e} "
                            f"lut_engaged={rec.get('lut_engaged')} "
                            f"| {rec['f32']['approx_state']}",
                            flush=True,
                        )
    else:
        for snr in args.snr:
            for channel in args.channels:
                models = args.models or CHANNEL_MODELS[channel]
                for model in models:
                    for path in args.paths:
                        try:
                            rec = run_seam(
                                ssp_cache,
                                channel,
                                model,
                                path,
                                snr,
                                args.n_z,
                                sigma_convention=args.line_sigma,
                            )
                        except Exception as exc:  # a seam that cannot be built is a result
                            rec = {
                                "channel": channel,
                                "model": model,
                                "path": path,
                                "snr": snr,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        rows.append(rec)
                        # PR #2104: this module builds up to three models at two
                        # precisions per seam and segfaults in XLA's CPU backend around
                        # the twelfth without dropping the caches between seams.
                        jax.clear_caches()
                        gc.collect()
                        if "error" in rec:
                            print(
                                f"{channel:11s} {model:13s} {path:14s} snr={snr:5.0f}  "
                                f"ERROR {rec['error'][:130]}",
                                flush=True,
                            )
                        else:
                            o, h = rec["points"]["origin"], rec["points"]["half_sigma"]
                            print(
                                f"{channel:11s} {model:13s} {path:14s} snr={snr:5.0f}  "
                                f"f32/f64 {o['f32_vs_f64']:.2e} {h['f32_vs_f64']:.2e} | "
                                f"lut64/exact64 {o['lut_f64_vs_exact_f64']:.2e} | "
                                f"f64/fd {o['f64_vs_fd64']:.2e} | "
                                f"dt32={o['dtype_f32']} nz={o['f32_nonzero']} "
                                f"| {rec['approx_state_f32']}",
                                flush=True,
                            )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

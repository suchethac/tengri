# SPDX-License-Identifier: BSD-3-Clause
"""Float32 posterior-gradient accuracy on the path a real fit takes.

PR #2100 measured ``grad(neg_log_posterior_fn)`` in pure float32 against float64 on
four *model* seams, at **fixed redshift** and (bar one row) only on **CPU**, and its
report says the measurements were on the **exact** projector. This script extends the
seam inventory along the axes a real fit actually crosses --- ``WavePrecomp``, free
redshift, CUDA --- and along the way re-measures whether the "exact projector" claim
was true (it was not: ``Fitter``'s default ``approx="auto"`` re-resolves the
build-time knob, so a model built with ``approx=None`` is *fitted* under the LUT).

Every arm records the **dtype of the gradient array it produced**, never the value of
``jax.config.jax_enable_x64`` --- #1840/#2097: a config flag can say float32 while the
arrays are float64.

Usage
-----
::

    JAX_PLATFORMS=cpu python bench/scripts/benchmark_float32_fitting_seams.py \
        --snr 30 --out bench/results/2026-08-31_float32_fitting_seams_cpu.json

    JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python bench/scripts/benchmark_float32_fitting_seams.py \
        --snr 30 --out bench/results/2026-08-31_float32_fitting_seams_cuda.json
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

from tengri import FIXED, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.inference.context import InferenceContext

# --------------------------------------------------------------------------------------
# Seam inventory
# --------------------------------------------------------------------------------------

_DUST_FREE = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": FIXED,
    "tau_diff": Uniform(0.0, 1.5),
    "tau_bc": 0.0,
}
_DUST_FIXED = dict(_DUST_FREE, tau_diff=0.3)

#: Model groups, identical to ``test_float32_grad_bolometric_seams.py`` so the rows here
#: are directly comparable to PR #2100's published numbers.
MODELS = {
    "stellar_dust": dict(dust_attenuation=_DUST_FREE),
    "dust_ir": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014", "all_params": FIXED},
    ),
    "agn": dict(
        dust_attenuation=_DUST_FIXED,
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Fixed(10.5),
            "fracAGN": 0.1,
        },
    ),
    "panchromatic": dict(
        dust_attenuation=_DUST_FREE,
        dust_emission={"type": "dale2014_cigale", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED},
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Fixed(10.5),
            "fracAGN": 0.1,
        },
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        xray={"type": "simple"},
        shock={"frac": 0.1},
    ),
}

Z_LO, Z_HI = 0.05, 1.0
#: Coarse ztable on purpose: ``n_z`` sets the LUT's *own* redshift-interpolation bias,
#: which is common to both precisions and therefore cancels out of the float32-vs-float64
#: comparison, while ``n_z=250`` costs ~20 s of build per model per precision.
N_Z = 64


def paths(n_z: int = N_Z):
    """``(build_approx_factory, fit_approx, redshift_factory)`` per seam id.

    ``fit_approx`` is what goes to ``Fitter(approx=...)``. It matters: the default
    ``"auto"`` **re-resolves the build-time knob**, so ``approx=None`` at build time is
    not the exact path once you fit with it.
    """
    fz = lambda: Fixed(0.1)  # noqa: E731
    uz = lambda: Uniform(Z_LO, Z_HI)  # noqa: E731
    wp = lambda: WavePrecomp(n_z=n_z, z_min=Z_LO, z_max=Z_HI)  # noqa: E731
    wp8 = lambda: WavePrecomp(  # noqa: E731
        band_integration="quadrature", n_subbands=8, n_z=n_z, z_min=Z_LO, z_max=Z_HI
    )
    return {
        # Genuinely exact: build exact AND tell the fitter not to re-resolve.
        "exact_fixedz": (lambda: None, None, fz),
        # What every default single-galaxy / catalog fit runs.
        "auto_fixedz": (lambda: None, "auto", fz),
        # The same LUT asked for at build time (#1683 path). Measured bit-identical to
        # ``auto_*`` on every model, which is the point of keeping it available.
        "lutbuild_fixedz": (wp, "auto", fz),
        # Quadrature named explicitly at a different order, so a verdict cannot be an
        # artifact of one K.
        "quad8_fixedz": (wp8, "auto", fz),
        "exact_freez": (lambda: None, None, uz),
        "auto_freez": (lambda: None, "auto", uz),
        "lutbuild_freez": (wp, "auto", uz),
        "quad8_freez": (wp8, "auto", uz),
    }


#: The matrix run for the report: ``lutbuild_*`` is dropped after one confirmation that
#: it reproduces ``auto_*`` bit-for-bit, because it doubles the cost of every row.
DEFAULT_PATHS = (
    "exact_fixedz",
    "auto_fixedz",
    "quad8_fixedz",
    "exact_freez",
    "auto_freez",
    "quad8_freez",
)


def base(zspec):
    return dict(
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        redshift=zspec,
    )


def make_obs():
    # herschel_250 is load-bearing for the dust-IR seam.
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1", "herschel_250"])
    )


def build(ssp, obs, model, approx, zspec):
    return SEDModel.build(
        ssp_data=ssp, observation=obs, approx=approx, **base(zspec), **MODELS[model]
    )


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------

#: Evaluation points in standardized (unbounded) space. The origin is where residuals are
#: smallest and float32 cancellation is worst; 0.5 sigma is a generic interior point.
POINTS = {"origin": 0.0, "half_sigma": 0.5}

#: Central-difference step in standardized units, for the float64 soundness check only.
FD_H = 1e-4


def _probe_lut_bias(fitter):
    """``max(forward bias x SNR)`` for this fit, or ``None`` if it is not on the LUT.

    This is exactly the estimate ``_warn_if_lut_bias_amplified`` renders into the
    :class:`~tengri.config.exceptions.PrecompBiasWarning`. It is invoked here rather than
    caught as a warning because the warn site lives in ``Fitter.run``: a script that only
    takes gradients never reaches it, and a ``None`` column would then read as "the
    advisory did not fire" when the truth is "the advisory was never asked".
    """
    from tengri.inference.fitter import _lut_forward_bias

    pre = getattr(fitter, "_pre_approx_model", None)
    if pre is None:
        return None
    try:
        bias = np.asarray(_lut_forward_bias(pre, fitter.model, fitter.data_type), dtype=float)
        data = np.asarray(fitter.data, dtype=float).reshape(-1)
        noise = np.asarray(fitter.noise, dtype=float).reshape(-1)
        n = int(bias.shape[0])
        if n == 0 or data.size == 0 or data.size % n != 0:
            return None
        snr = np.abs(data) / np.maximum(noise, np.finfo(float).tiny)
        return float(np.nanmax(bias[None, :] * snr.reshape(-1, n)))
    except Exception:
        return None


def grad_at(
    ssp, obs, model, build_approx, fit_approx, zspec, flux, noise, *, x64, dtype, fd=False
):
    """One precision's gradients of ``neg_log_posterior_fn`` at every point in POINTS."""
    with jax.enable_x64(x64):
        sed = build(ssp, obs, model, build_approx, zspec)
        fitter = Fitter(
            sed,
            jnp.asarray(flux, dtype=dtype),
            jnp.asarray(noise, dtype=dtype),
            approx=fit_approx,
        )
        # #1671's runtime price. Note it fires from ``Fitter.run``, not from the
        # constructor, so a gradient-only measurement like this one never triggers it —
        # the probe is invoked directly instead of being caught as a warning.
        lut_bias_warning = _probe_lut_bias(fitter)
        ctx = InferenceContext.from_target(fitter)
        data_args = ctx.data_args
        names = sorted(ctx.initial_params(jax.random.PRNGKey(1)))

        def nlp(vals):
            return ctx.neg_log_posterior_fn({k: vals[i] for i, k in enumerate(names)}, data_args)

        out = {
            "names": names,
            "approx_state": str(getattr(ctx.fitter.model, "approx", None)),
            "lut_bias_warning": lut_bias_warning,
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


def rel(a, b):
    """Max relative deviation of ``a`` from reference ``b``, componentwise."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def make_mock(ssp, obs, model, zspec, snr, seed=0):
    """One float64 mock from the **exact** projector, so every arm fits identical data."""
    with jax.enable_x64(True):
        sed = build(ssp, obs, model, None, zspec)
        truth = {
            n: float(sed.spec._distributions[n].unstandardize(jnp.asarray(0.0)))
            for n in sed.spec.free_params
        }
        m = sed.mock(truth, snr=snr, key=jax.random.PRNGKey(seed))
        return (
            np.asarray(m.flux_obs, dtype=np.float64),
            np.asarray(m.noise, dtype=np.float64),
            truth,
        )


def run_seam(ssp, obs, model, path, snr, path_table):
    build_approx, fit_approx, zfac = path_table[path]
    flux, noise, truth = make_mock(ssp, obs, model, zfac(), snr)

    t0 = time.time()
    kw = dict(model=model, zspec=zfac(), flux=flux, noise=noise)
    f64 = grad_at(
        ssp,
        obs,
        build_approx=build_approx(),
        fit_approx=fit_approx,
        x64=True,
        dtype=jnp.float64,
        fd=True,
        **kw,
    )
    f32 = grad_at(
        ssp,
        obs,
        build_approx=build_approx(),
        fit_approx=fit_approx,
        x64=False,
        dtype=jnp.float32,
        **kw,
    )
    # Truly-exact float64 reference, to separate the LUT's own bias from float32's error.
    exact64 = (
        f64
        if path.startswith("exact")
        else grad_at(
            ssp, obs, build_approx=None, fit_approx=None, x64=True, dtype=jnp.float64, **kw
        )
    )

    rec = {
        "model": model,
        "path": path,
        "snr": snr,
        "names": f64["names"],
        "approx_state_f64": f64["approx_state"],
        "approx_state_f32": f32["approx_state"],
        "lut_bias_warning_f64": f64["lut_bias_warning"],
        "lut_bias_warning_f32": f32["lut_bias_warning"],
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
            "f64_vs_fd64": rel(a64["grad"], a64["fd"]),
            "f32_vs_f64": rel(a32["grad"], a64["grad"]),
            "lut_f64_vs_exact_f64": rel(a64["grad"], ax["grad"]),
            "f32_vs_exact_f64": rel(a32["grad"], ax["grad"]),
            "f32_finite": bool(np.all(np.isfinite(a32["grad"]))),
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=sorted(MODELS))
    ap.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    ap.add_argument("--snr", nargs="+", type=float, default=[30.0])
    ap.add_argument("--n-z", type=int, default=N_Z)
    ap.add_argument("--ssp", default="data/fsps_prsc_miles_chabrier.h5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    ssp = load_ssp_data(args.ssp)
    obs = make_obs()
    path_table = paths(args.n_z)

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
            )
        },
        "n_z": args.n_z,
        "points": POINTS,
    }
    print(json.dumps(meta, indent=2), flush=True)

    rows = []
    for snr in args.snr:
        for model in args.models:
            for path in args.paths:
                try:
                    rec = run_seam(ssp, obs, model, path, snr, path_table)
                except Exception as exc:  # a seam that cannot be built is a result
                    rec = {
                        "model": model,
                        "path": path,
                        "snr": snr,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(rec)
                # This box is shared and each seam builds up to three models at two
                # precisions; without this the CPU arm exhausts LLVM's allocator
                # part-way through the matrix.
                jax.clear_caches()
                gc.collect()
                if "error" in rec:
                    print(f"{model:14s} {path:16s} snr={snr:5.0f}  ERROR {rec['error'][:150]}")
                else:
                    o, h = rec["points"]["origin"], rec["points"]["half_sigma"]
                    print(
                        f"{model:14s} {path:16s} snr={snr:5.0f}  "
                        f"f32/f64 {o['f32_vs_f64']:.2e} {h['f32_vs_f64']:.2e} | "
                        f"lut64/exact64 {o['lut_f64_vs_exact_f64']:.2e} "
                        f"{h['lut_f64_vs_exact_f64']:.2e} | "
                        f"f64/fd {o['f64_vs_fd64']:.2e} | dt32={o['dtype_f32']} "
                        f"| {rec['approx_state_f32']}",
                        flush=True,
                    )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

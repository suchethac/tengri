"""Mock recovery validation for issue #231.

Runs every registered inference backend on two cheap models:
- DPL SFH photometry  (~6 free params)
- DenseBasis SFH photometry  (~7 free params)

For each (backend, model) we measure:
- First-call wall time   (compile + run, what a user actually feels)
- Second-call wall time  (warm — should drop to inference cost)
- Peak RSS               (parent process only — coarse but useful)
- Recovery quality       (|posterior_mean - truth| / posterior_std) per param

No IFT field SFH. No NUTS at D > 8. Watchdog (tools/oom_watchdog.sh) is
assumed to be running externally.

Results go to scripts/_backend_validation_results.json so we can write
short_doc / tier updates from the file rather than from a flaky run.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    builders,
    load_filter_set,
    load_ssp_data,
)
from tengri.inference._backend_registry import _BACKENDS

DATA = Path(__file__).resolve().parents[1] / "data"
SSP_FILE = DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

# Backends we explicitly skip (or run with small budgets).
SKIP = {
    # NUTS auto-promotes to raytrace at D > threshold; we run mcmc_nuts
    # directly on D <= 8 only.
}

# Per-backend kwargs. Conservative — we want signal not statistics.
KW = {
    "map": dict(n_steps=400, verbose=False),
    "laplace": dict(verbose=False),
    "vi": dict(n_iterations=8, n_samples=3, verbose=False),
    "vi_nonlinear": dict(n_iterations=8, n_samples=3, verbose=False),
    "vi_nonlinear_fast": dict(n_iterations=8, n_samples=3, verbose=False),
    "vi_linear": dict(n_iterations=8, n_samples=3, verbose=False),
    "vi_linear_fast": dict(n_iterations=8, n_samples=3, verbose=False),
    "native_vi_nonlinear": dict(n_iterations=20, n_samples=3, n_seeds=1, verbose=False),
    "native_vi_linear": dict(n_iterations=20, n_samples=3, n_seeds=1, verbose=False),
    "mcmc": dict(verbose=False),
    "mcmc_nuts": dict(n_warmup=100, n_samples=200, verbose=False, dense_mass_matrix=False),
    "mcmc_hmc": dict(n_warmup=100, n_samples=200, verbose=False, dense_mass_matrix=False),
    "mcmc_dynamic_hmc": dict(n_warmup=100, n_samples=200, verbose=False),
    "mcmc_ghmc": dict(n_warmup=100, n_samples=200, verbose=False),
    "mcmc_mclmc": dict(n_samples=300, verbose=False),
    "mcmc_adjusted_mclmc": dict(n_samples=300, verbose=False),
    "mcmc_ess": dict(n_samples=300, verbose=False),
    "mcmc_raytrace": dict(n_steps=300, n_burnin=50, verbose=False),
    "nss": dict(verbose=False),
    "pathfinder": dict(verbose=False),
}

# 8 SDSS+2MASS filters.
FILTERS_NAMES = [
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
]


def rss_gb() -> float:
    """Self RSS in GB (macOS reports bytes, Linux kB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if r > 1e9:  # macOS bytes
        return r / 1024**3
    return r / 1024**2  # Linux kB


def build_model(variant: str):
    """variant in {'dpl', 'dense_basis'}."""
    ssp = load_ssp_data(str(SSP_FILE))
    _, _, filters = load_filter_set(FILTERS_NAMES)
    obs = Observation(photometry=Photometry(filters=tuple(filters)))

    if variant == "dpl":
        sfh = builders.sfh.dpl(defaults=FREE)
    elif variant == "dense_basis":
        sfh = builders.sfh.dense_basis(defaults=FREE)
    else:
        raise ValueError(variant)

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=sfh,
        dust_attenuation={
            "type": "two_component",
            "*": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )
    return model


def make_mock(model, seed=42, snr=20.0):
    """Centre-of-prior truth → mock photometry."""
    key = jr.PRNGKey(seed)
    free = model.spec.free_params
    truth = {}
    for name in free:
        prior = model.spec.get_distribution(name)
        if hasattr(prior, "low") and hasattr(prior, "high"):
            truth[name] = float(0.5 * (prior.low + prior.high))
        elif hasattr(prior, "mean"):
            truth[name] = float(prior.mean)
        elif hasattr(prior, "sample"):
            truth[name] = float(prior.sample(jr.PRNGKey(0)))
        else:
            truth[name] = 0.0
    obs = model.mock(truth, snr=snr, key=key)
    return obs, truth


def recovery_quality(posterior, truth: dict) -> dict:
    """Per-param bias in σ units. Returns dict of param_name -> bias_sigma."""
    out = {}
    samples = getattr(posterior, "samples", None)
    if samples is None:
        # MAP / Laplace: use posterior.params with no uncertainty.
        params = getattr(posterior, "params", None)
        if params is None:
            return {"_no_samples_no_params": True}
        for k, v_true in truth.items():
            if k in params:
                out[k] = {"mean": float(params[k]), "truth": v_true, "bias_sigma": None}
        return out
    for k, v_true in truth.items():
        if k not in samples:
            continue
        s = jnp.asarray(samples[k])
        mu = float(jnp.mean(s))
        sd = float(jnp.std(s))
        bias = (mu - v_true) / sd if sd > 0 else None
        out[k] = {"mean": mu, "std": sd, "truth": v_true, "bias_sigma": bias}
    return out


def run_one(backend: str, model, obs, truth) -> dict:
    """Single (backend, model) trial. Returns a JSON-serialisable dict."""
    entry = _BACKENDS[backend]
    kw = KW.get(backend, {})

    fitter = Fitter(model, obs.flux_obs, obs.noise)
    key = jr.PRNGKey(0)

    rec = {
        "backend": backend,
        "tier_before": entry.tier,
        "n_free": model.spec.n_free,
        "kwargs": kw,
    }

    gc.collect()
    rss_before = rss_gb()

    # Cold call.
    t0 = time.perf_counter()
    try:
        post = entry.runner(fitter, key=key, **kw)
        jax.block_until_ready(jnp.asarray(0.0))  # nudge dispatch
        t_cold = time.perf_counter() - t0
        rec["cold_s"] = t_cold
        rec["rss_gb_after_cold"] = rss_gb()
        rec["status"] = "ok"
    except Exception as e:
        rec["cold_s"] = time.perf_counter() - t0
        rec["status"] = "failed"
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = str(e)[:400]
        rec["traceback"] = traceback.format_exc()[-800:]
        return rec

    # Warm call (cache reuse).
    try:
        t0 = time.perf_counter()
        post = entry.runner(fitter, key=jr.PRNGKey(1), **kw)
        jax.block_until_ready(jnp.asarray(0.0))
        rec["warm_s"] = time.perf_counter() - t0
    except Exception as e:
        rec["warm_s"] = None
        rec["warm_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    try:
        rec["recovery"] = recovery_quality(post, truth)
    except Exception as e:
        rec["recovery_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    rec["rss_gb_peak"] = rss_gb()
    return rec


def _child_run(backend: str, variant: str, out_json: str) -> None:
    """Subprocess entry: run one (backend, variant), append JSON result."""
    model = build_model(variant)
    obs, truth = make_mock(model)
    r = run_one(backend, model, obs, truth)
    r["variant"] = variant
    Path(out_json).write_text(json.dumps(r, indent=2, default=str))


def main():
    import subprocess
    import sys

    out_path = Path(__file__).parent / "_backend_validation_results.json"
    tmp_dir = Path("/tmp/validate_231_results")
    tmp_dir.mkdir(exist_ok=True)
    results: list[dict] = []

    # Filter order: cheap first. Pathfinder last (known fragile in BlackJAX).
    order = [
        "map",
        "laplace",
        "native_vi_linear",
        "native_vi_nonlinear",
        "vi_nonlinear_fast",
        "vi_nonlinear",
        "vi",
        "vi_linear_fast",
        "vi_linear",
        "mcmc_hmc",
        "mcmc_nuts",
        "mcmc_dynamic_hmc",
        "mcmc_ghmc",
        "mcmc_mclmc",
        "mcmc_adjusted_mclmc",
        "mcmc_ess",
        "mcmc_raytrace",
        "mcmc",
        "nss",
        "pathfinder",
    ]
    order = [b for b in order if b in _BACKENDS]

    # Per-backend wall-clock cap (seconds). Anything slower we declare unfit.
    TIMEOUT = {
        "map": 60,
        "laplace": 60,
        "native_vi_linear": 180,
        "native_vi_nonlinear": 180,
        "vi": 300,
        "vi_nonlinear": 300,
        "vi_linear": 300,
        "vi_nonlinear_fast": 300,
        "vi_linear_fast": 300,
        "mcmc_hmc": 300,
        "mcmc_nuts": 300,
        "mcmc_dynamic_hmc": 300,
        "mcmc_ghmc": 300,
        "mcmc_mclmc": 300,
        "mcmc_adjusted_mclmc": 300,
        "mcmc_ess": 300,
        "mcmc_raytrace": 300,
        "mcmc": 300,
        "nss": 600,
        "pathfinder": 300,
    }

    for variant in ("dpl", "dense_basis"):
        print(f"\n=== variant={variant} ===", flush=True)
        # Print D once via a quick in-process build.
        m = build_model(variant)
        print(f"  D={m.spec.n_free}  free={m.spec.free_params}", flush=True)
        del m
        gc.collect()

        for backend in order:
            print(f"  -> {backend:25s}", end="", flush=True)
            out_json = tmp_dir / f"{backend}_{variant}.json"
            if out_json.exists():
                out_json.unlink()
            cmd = [sys.executable, __file__, "--child", backend, variant, str(out_json)]
            t0 = time.perf_counter()
            try:
                subprocess.run(cmd, timeout=TIMEOUT[backend], check=False)
                wall = time.perf_counter() - t0
                if out_json.exists():
                    r = json.loads(out_json.read_text())
                else:
                    r = {
                        "backend": backend,
                        "variant": variant,
                        "status": "crashed_no_output",
                        "wall_s": wall,
                        "error_type": "SegfaultOrAbort",
                        "error_msg": "child died without writing JSON",
                    }
            except subprocess.TimeoutExpired:
                r = {
                    "backend": backend,
                    "variant": variant,
                    "status": "timeout",
                    "wall_s": TIMEOUT[backend],
                    "error_type": "TimeoutExpired",
                    "error_msg": f"exceeded {TIMEOUT[backend]}s budget",
                }

            results.append(r)
            if r["status"] == "ok":
                w = r.get("warm_s")
                print(
                    f"   cold={r['cold_s']:6.1f}s  warm={('%.1f' % w) if w else 'NA':>6}s  "
                    f"rss={r['rss_gb_peak']:.2f}GB",
                    flush=True,
                )
            else:
                print(
                    f"   FAIL[{r['status']}]: {r.get('error_type', '?')} "
                    f"{(r.get('error_msg') or '')[:80]}",
                    flush=True,
                )
            out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nResults written to {out_path}", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        _child_run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()

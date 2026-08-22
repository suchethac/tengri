"""Convergence-checked recovery pass for the promoted MCMC backends.

Follow-up to scripts/validate_backends_231.py — runs full-length chains
with adequate warmup so R-hat, ESS, and N>5τ can be meaningfully checked.

Targets:
- mcmc_hmc, mcmc_dynamic_hmc, mcmc_ghmc, mcmc_mclmc
- on DPL (D=6) and dense_basis (D=7)

Settings per backend:
- n_warmup=500, n_samples=2000  (D≤7 is fine for these budgets)
- For R-hat: split each chain in half and use split-R-hat (Vehtari 2021)
- Convergence verdict: split-R-hat < 1.01 AND ESS > 400 AND N > 5τ

Subprocess-per-(backend, variant) so a single crash doesn't kill the rest.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    builders,
    load_filter_set,
    load_ssp_data,
)
from tengri.analysis.diagnostics.autocorrelation import split_rhat
from tengri.inference._backend_registry import _BACKENDS

DATA = Path(__file__).resolve().parents[1] / "data"
SSP_FILE = DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
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

# Promoted MCMC backends (laplace is a Gaussian approximation, not a chain;
# skipped here — it has no R-hat / ESS concept).
PROMOTED = ["mcmc_hmc", "mcmc_dynamic_hmc", "mcmc_ghmc", "mcmc_mclmc"]

# Per-variant kwargs. Dense mass matrix is fine at D=6-7; the OOM concern
# from the speed sweep was about NUTS specifically, not HMC.
KW: dict[tuple[str, str], dict] = {
    # DPL D=6 — dense mass matrix captures age-dust-met correlations
    ("mcmc_hmc", "dpl"): dict(
        n_warmup=1000,
        n_burnin=200,
        n_samples=2000,
        verbose=False,
        dense_mass_matrix=True,
        n_leapfrog_steps=20,
    ),
    ("mcmc_dynamic_hmc", "dpl"): dict(n_warmup=1000, n_burnin=200, n_samples=2000, verbose=False),
    ("mcmc_ghmc", "dpl"): dict(n_warmup=1000, n_burnin=200, n_samples=2000, verbose=False),
    ("mcmc_mclmc", "dpl"): dict(n_samples=4000, verbose=False),
    # dense_basis D=7 — same, dense is OK here too
    ("mcmc_hmc", "dense_basis"): dict(
        n_warmup=1000,
        n_burnin=200,
        n_samples=2000,
        verbose=False,
        dense_mass_matrix=True,
        n_leapfrog_steps=20,
    ),
    ("mcmc_dynamic_hmc", "dense_basis"): dict(
        n_warmup=1000, n_burnin=200, n_samples=2000, verbose=False
    ),
    ("mcmc_ghmc", "dense_basis"): dict(n_warmup=1000, n_burnin=200, n_samples=2000, verbose=False),
    ("mcmc_mclmc", "dense_basis"): dict(n_samples=4000, verbose=False),
}


def build_model(variant: str):
    ssp = load_ssp_data(str(SSP_FILE))
    _, _, filters = load_filter_set(FILTERS_NAMES)
    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    if variant == "dpl":
        sfh = builders.sfh.dpl(all_params=FREE)
    elif variant == "dense_basis":
        sfh = builders.sfh.dense_basis(all_params=FREE)
    else:
        raise ValueError(variant)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=sfh,
        dust_attenuation={
            "type": "two_component",
            "all_params": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def make_mock(model, seed=42, snr=20.0):
    key = jr.PRNGKey(seed)
    truth = {}
    for name in model.spec.free_params:
        prior = model.spec.get_distribution(name)
        if hasattr(prior, "low") and hasattr(prior, "high"):
            truth[name] = float(0.5 * (prior.low + prior.high))
        else:
            truth[name] = float(prior.sample(jr.PRNGKey(0)))
    obs = model.mock(truth, snr=snr, key=key)
    return obs, truth


def diagnose(samples: dict, truth: dict) -> dict:
    """Compute split-R-hat, ESS, N>5τ for every parameter."""
    from tengri.analysis.diagnostics.autocorrelation import (
        check_chain_length,
        effective_sample_size,
    )

    samples_np = {k: np.asarray(v) for k, v in samples.items()}

    rhat = {k: float(split_rhat(samples_np[k])) for k in samples_np}
    ess_info = effective_sample_size(samples_np)
    ess = {k: float(info["ess"]) for k, info in ess_info.items()}
    tau = {k: float(info["tau_max"]) for k, info in ess_info.items()}

    n_per_chain = next(iter(samples_np.values())).shape[0]
    check = check_chain_length(samples_np, verbose=False)

    # Per-param recovery (bias in σ)
    recovery = {}
    for k, v_true in truth.items():
        if k in samples_np:
            mu = float(np.mean(samples_np[k]))
            sd = float(np.std(samples_np[k]))
            recovery[k] = {
                "truth": v_true,
                "mean": mu,
                "std": sd,
                "bias_sigma": (mu - v_true) / sd if sd > 0 else None,
            }

    # Verdict: split-R-hat < 1.01, ESS > 400, N > 5τ
    converged = (
        all(r < 1.01 for r in rhat.values())
        and all(e > 400 for e in ess.values())
        and check.get("all_converged", False)
    )
    return {
        "n_samples": n_per_chain,
        "rhat": rhat,
        "ess": ess,
        "tau": tau,
        "n_gt_5tau": check.get("all_converged", False),
        "rhat_max": float(max(rhat.values())) if rhat else None,
        "ess_min": float(min(ess.values())) if ess else None,
        "converged": converged,
        "recovery": recovery,
    }


def child_run(backend: str, variant: str, out_json: str) -> None:
    model = build_model(variant)
    obs, truth = make_mock(model)
    fitter = Fitter(model, obs.flux_obs, obs.noise)
    entry = _BACKENDS[backend]
    kw = KW[(backend, variant)]

    rec = {"backend": backend, "variant": variant, "n_free": model.spec.n_free}
    t0 = time.perf_counter()
    try:
        post = entry.runner(fitter, key=jr.PRNGKey(0), **kw)
        jax.block_until_ready(jnp.asarray(0.0))
        rec["wall_s"] = time.perf_counter() - t0
        rec["status"] = "ok"
        rec.update(diagnose(post.samples, truth))
        # Surface backend diagnostics if any
        if hasattr(post, "diagnostics") and post.diagnostics:
            rec["backend_diag"] = {
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in post.diagnostics.items()
                if not isinstance(v, dict)
            }
    except Exception as e:
        rec["wall_s"] = time.perf_counter() - t0
        rec["status"] = "failed"
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = str(e)[:400]
        rec["traceback"] = traceback.format_exc()[-1200:]

    Path(out_json).write_text(json.dumps(rec, indent=2, default=str))


def main():
    out_path = Path(__file__).parent / "_backend_convergence_results.json"
    tmp = Path("/tmp/validate_231_convergence")
    tmp.mkdir(exist_ok=True)
    TIMEOUT = 900  # 15 min per backend per variant
    results = []

    for variant in ("dpl", "dense_basis"):
        print(f"\n=== variant={variant} ===", flush=True)
        m = build_model(variant)
        print(f"  D={m.spec.n_free} free={m.spec.free_params}", flush=True)
        del m

        for backend in PROMOTED:
            print(f"  -> {backend:20s}", end="", flush=True)
            j = tmp / f"{backend}_{variant}.json"
            if j.exists():
                j.unlink()
            cmd = [sys.executable, __file__, "--child", backend, variant, str(j)]
            t0 = time.perf_counter()
            try:
                subprocess.run(cmd, timeout=TIMEOUT, check=False)
                wall = time.perf_counter() - t0
                if j.exists():
                    r = json.loads(j.read_text())
                else:
                    r = {
                        "backend": backend,
                        "variant": variant,
                        "status": "crashed_no_output",
                        "wall_s": wall,
                    }
            except subprocess.TimeoutExpired:
                r = {
                    "backend": backend,
                    "variant": variant,
                    "status": "timeout",
                    "wall_s": TIMEOUT,
                }
            results.append(r)
            if r["status"] == "ok":
                conv = "✓" if r["converged"] else "✗"
                rh = r.get("rhat_max", 0) or 0
                es = r.get("ess_min", 0) or 0
                print(
                    f"   wall={r['wall_s']:5.0f}s  Rhat_max={rh:.3f}  "
                    f"ESS_min={es:.0f}  converged={conv}",
                    flush=True,
                )
            else:
                print(
                    f"   FAIL[{r['status']}] {r.get('error_type', '')} "
                    f"{(r.get('error_msg') or '')[:60]}",
                    flush=True,
                )
            out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nResults written to {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        child_run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()

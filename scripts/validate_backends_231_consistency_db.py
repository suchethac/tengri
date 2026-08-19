"""Cross-method posterior consistency on the dense_basis SFH mock.

Mirrors validate_backends_231_consistency.py but with:
- variant = "dense_basis" (D=7 instead of 6)
- mcmc_hmc gets 5000 samples (the 2000-sample DPL run gave ESS_min=17
  on dense_basis — need ~3-5× more to get ESS>400)
- nss timeout bumped to 1800s (DPL ran 234s; D=7 will be slower).

Same single mock + key for every backend so means / stds are directly
comparable.
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
from tengri.forward.sed_model import WavePrecomp
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
MOCK_SEED = 42


def build_model_and_mock():
    ssp = load_ssp_data(str(SSP_FILE))
    _, _, filters = load_filter_set(FILTERS_NAMES)
    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=builders.sfh.dense_basis(defaults=FREE),
        dust={
            "type": "two_component",
            "*": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=WavePrecomp(),  # photometry-only LUT — ~18× speedup
    )
    truth = {}
    for name in model.spec.free_params:
        prior = model.spec.get_distribution(name)
        lo = getattr(prior, "low", getattr(prior, "lo", None))
        hi = getattr(prior, "high", getattr(prior, "hi", None))
        if lo is not None and hi is not None:
            truth[name] = float(0.5 * (lo + hi))
        else:
            truth[name] = float(prior.sample(jr.PRNGKey(0)))
    obs_mock = model.mock(truth, snr=20.0, key=jr.PRNGKey(MOCK_SEED))
    return model, obs_mock, truth


# Budget: <20s warm per inference. Settings tuned from interactive profile —
# see PR #256 for the cache/timing breakdown that justifies these.
KW = {
    "map": dict(n_steps=500, verbose=False),
    "laplace": dict(verbose=False),
    "mcmc_hmc": dict(
        n_warmup=300,
        n_burnin=50,
        n_samples=1000,
        n_leapfrog_steps=10,
        dense_mass_matrix=True,
        verbose=False,
    ),
    "nss": dict(
        verbose=False
    ),  # NSS is inherently slow; out of budget but kept for consistency check
}


def child_run(backend: str, out_json: str) -> None:
    model, mock, truth = build_model_and_mock()
    fitter = Fitter(model, mock.flux_obs, mock.noise)
    entry = _BACKENDS[backend]
    kw = KW[backend]

    rec = {
        "backend": backend,
        "variant": "dense_basis",
        "n_free": model.spec.n_free,
        "truth": truth,
    }
    t0 = time.perf_counter()
    try:
        post = entry.runner(fitter, key=jr.PRNGKey(MOCK_SEED), **kw)
        jax.block_until_ready(jnp.asarray(0.0))
        rec["wall_s"] = time.perf_counter() - t0
        rec["status"] = "ok"

        if getattr(post, "samples", None) is not None:
            summary = {}
            for k, s in post.samples.items():
                s = np.asarray(s)
                summary[k] = {
                    "mean": float(np.mean(s)),
                    "std": float(np.std(s)),
                    "p16": float(np.percentile(s, 16)),
                    "p50": float(np.percentile(s, 50)),
                    "p84": float(np.percentile(s, 84)),
                }
            rec["posterior"] = summary
        elif getattr(post, "params", None) is not None:
            rec["posterior"] = {
                k: {"mean": float(v), "std": None, "p16": None, "p50": float(v), "p84": None}
                for k, v in post.params.items()
            }
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
    out_dir = Path("/tmp/validate_231_consistency_db")
    out_dir.mkdir(exist_ok=True)
    out_path = Path(__file__).parent / "_backend_consistency_db_results.json"
    summary: dict = {}

    print("=== dense_basis cross-method consistency ===", flush=True)
    model, mock, truth = build_model_and_mock()
    print(f"  D={model.spec.n_free} free={model.spec.free_params}", flush=True)
    summary["truth"] = truth

    runs = {}
    TIMEOUT = {"map": 180, "laplace": 180, "mcmc_hmc": 1800, "nss": 1800}
    for backend in ["map", "laplace", "mcmc_hmc", "nss"]:
        print(f"  -> {backend:12s}", end="", flush=True)
        j = out_dir / f"{backend}.json"
        if j.exists():
            j.unlink()
        cmd = [sys.executable, __file__, "--child", backend, str(j)]
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, timeout=TIMEOUT[backend], check=False)
            wall = time.perf_counter() - t0
            if j.exists():
                r = json.loads(j.read_text())
            else:
                r = {"backend": backend, "status": "crashed_no_output", "wall_s": wall}
        except subprocess.TimeoutExpired:
            r = {"backend": backend, "status": "timeout", "wall_s": TIMEOUT[backend]}
        runs[backend] = r
        if r["status"] == "ok":
            print(f"   wall={r['wall_s']:5.0f}s", flush=True)
        else:
            print(f"   FAIL[{r['status']}] {r.get('error_type', '')}", flush=True)
        summary["runs"] = runs
        out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== cross-method posterior means (truth → method mean ± std) ===", flush=True)
    ok = {b: r for b, r in runs.items() if r.get("status") == "ok" and "posterior" in r}
    if ok:
        free = sorted(truth.keys())
        header = f"{'param':<28s} {'truth':>9s} " + " ".join(f"{b:>20s}" for b in ok)
        print(header, flush=True)
        for p in free:
            row = f"{p:<28s} {truth[p]:>9.3f} "
            for b in ok:
                post = ok[b]["posterior"].get(p, {})
                mu = post.get("mean")
                sd = post.get("std")
                if mu is None:
                    row += f"{'—':>20s} "
                elif sd is None:
                    row += f"{mu:>10.3f} (point)  "
                else:
                    row += f"  {mu:>7.3f}±{sd:>5.3f}     "
            print(row, flush=True)

    print(f"\nWritten to {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        child_run(sys.argv[2], sys.argv[3])
    else:
        main()

"""Three follow-up checks for issue #231.

(a) **Gradient flow**: compute ∇loss at the prior midpoint and at a
    perturbation. All entries must be finite and non-zero.

(b) **Cross-method posterior consistency** on a single DPL mock with
    the SAME seed: run map, laplace, mcmc_hmc, nss. If the forward
    model and likelihood are correct, posterior means agree within
    a fraction of a σ, and 68 % CI widths agree within a factor of 2.
    A blatant disagreement means at least one backend is wrong.

(c) **NSS sanity**: also verify it produces a sensible evidence
    (log_Z) — finite, with non-trivial uncertainty.

Skips dense_basis (the speed sweep showed nss times out > 600 s there).
Skips ghmc / mclmc / dynamic_hmc — those failed convergence already.
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
MOCK_SEED = 42  # used for mock generation AND inference, so all methods see the same mock + key


def build_model_and_mock():
    ssp = load_ssp_data(str(SSP_FILE))
    _, _, filters = load_filter_set(FILTERS_NAMES)
    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=builders.sfh.dpl(defaults=FREE),
        dust_attenuation={
            "type": "two_component",
            "all_params": FIXED,
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


# ── (a) gradient-flow probe ──────────────────────────────────────────────


def gradient_probe(model, mock):
    """Probe forward-model + likelihood gradient flow.

    Three checks:
    (1) ∇_θ predict_photometry — does the forward model differentiate?
    (2) ∇_θ chi^2 — does the data-likelihood gradient flow?
    Both at prior midpoint and at a 10% perturbation, in physical space.
    """
    spec = model.spec
    free = spec.free_params

    def make_params(scale: float) -> dict:
        p = {}
        for name in free:
            prior = spec.get_distribution(name)
            lo = getattr(prior, "low", getattr(prior, "lo", 0.0))
            hi = getattr(prior, "high", getattr(prior, "hi", 1.0))
            p[name] = float(0.5 * (lo + hi) + scale * (hi - lo))
        return p

    # chi^2(params) = sum( ((flux_obs - predict(params)) / noise)^2 )
    def chi2(params):
        pred = model.predict_photometry(params)
        return jnp.sum(((mock.flux_obs - pred) / mock.noise) ** 2)

    grad_chi2 = jax.grad(chi2)
    grad_pred_sum = jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))

    out: dict[str, dict] = {}
    for scale, label in [(0.0, "midpoint"), (0.1, "+0.1*span")]:
        params = make_params(scale)
        try:
            g_chi2 = grad_chi2(params)
            g_pred = grad_pred_sum(params)
            chi2_arr = np.array([float(g_chi2[k]) for k in free])
            pred_arr = np.array([float(g_pred[k]) for k in free])
            out[label] = {
                "chi2": float(chi2(params)),
                "chi2_grad": {
                    "finite": bool(np.all(np.isfinite(chi2_arr))),
                    "any_zero": bool(np.any(chi2_arr == 0.0)),
                    "abs_min": float(np.min(np.abs(chi2_arr))),
                    "abs_max": float(np.max(np.abs(chi2_arr))),
                    "by_param": {k: float(g_chi2[k]) for k in free},
                },
                "forward_grad": {
                    "finite": bool(np.all(np.isfinite(pred_arr))),
                    "any_zero": bool(np.any(pred_arr == 0.0)),
                    "abs_min": float(np.min(np.abs(pred_arr))),
                    "abs_max": float(np.max(np.abs(pred_arr))),
                },
            }
        except Exception as e:
            out[label] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ── (b) child runs ───────────────────────────────────────────────────────

KW = {
    "map": dict(n_steps=1000, verbose=False),
    "laplace": dict(verbose=False),
    "mcmc_hmc": dict(
        n_warmup=1000,
        n_burnin=200,
        n_samples=2000,
        n_leapfrog_steps=20,
        dense_mass_matrix=True,
        verbose=False,
    ),
    "nss": dict(verbose=False),
}


def child_run(backend: str, out_json: str) -> None:
    model, mock, truth = build_model_and_mock()
    fitter = Fitter(model, mock.flux_obs, mock.noise)
    entry = _BACKENDS[backend]
    kw = KW[backend]

    rec = {"backend": backend, "n_free": model.spec.n_free, "truth": truth}
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


# ── orchestrator ─────────────────────────────────────────────────────────


def main():
    out_dir = Path("/tmp/validate_231_consistency")
    out_dir.mkdir(exist_ok=True)
    out_path = Path(__file__).parent / "_backend_consistency_results.json"
    summary: dict = {}

    # (a) gradient probe — single in-process check
    print("=== gradient flow ===", flush=True)
    model, mock, truth = build_model_and_mock()
    grad_info = gradient_probe(model, mock)
    summary["gradient"] = grad_info
    for label, info in grad_info.items():
        if "error" in info:
            print(f"  {label}: ERROR {info['error']}", flush=True)
        else:
            print(
                f"  {label}: loss={info['loss']:.3e} finite={info['finite']} "
                f"any_zero={info['any_zero']} |grad| in [{info['abs_min']:.2e}, "
                f"{info['abs_max']:.2e}]",
                flush=True,
            )
    summary["truth"] = truth
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    # (b)+(c) per-backend posterior summaries
    print("\n=== posterior consistency ===", flush=True)
    runs = {}
    TIMEOUT = {"map": 120, "laplace": 120, "mcmc_hmc": 900, "nss": 900}
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

    # Cross-method comparison table
    print(
        "\n=== cross-method posterior means (truth → method posterior_mean (± std)) ===",
        flush=True,
    )
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

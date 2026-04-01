#!/usr/bin/env python3
"""Benchmark: quantitative comparison of all inference methods.

Generates a mock galaxy with known truth (tsnorm SFH, D=7 smooth),
fits with MAP, native_geovi, Ray Tracing, NUTS, and geovi_nuts,
then reports wall-clock time, ESS, ESS/s, bias, and CI coverage.

Usage:
    python analysis/bench_inference_quality.py          # full run
    python analysis/bench_inference_quality.py --quick   # reduced samples
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG_DIR, setup_matplotlib

from tengri import (
    Fitter,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SSP_FILE = PROJECT_ROOT / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


# ── Mock setup ────────────────────────────────────────────────────
def build_model_and_truth(key):
    """Build a D=7 smooth model and generate a mock galaxy.

    Returns
    -------
    model : Model
    flux_obs : array
    noise : array
    true_params : dict
    """
    if not SSP_FILE.exists():
        print(f"SSP file not found: {SSP_FILE}")
        print("Skipping benchmark (requires SSP data).")
        sys.exit(0)

    ssp = load_ssp_data(str(SSP_FILE))
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

    spec = ParamSpec(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
    )

    model = Model(spec, ssp, observation=obs)

    # Sample truth from prior
    param_key, noise_key = jax.random.split(key)
    true_params = spec.sample(param_key)

    # Generate mock photometry with SNR ~ 20
    mock = model.mock(true_params, snr=20.0, key=noise_key)
    flux_obs = mock.flux_obs
    noise = mock.noise

    return model, flux_obs, noise, true_params


# ── Method configurations ─────────────────────────────────────────
def get_method_configs(quick: bool):
    """Return {name: (method_str, kwargs)} for each inference method."""
    if quick:
        return {
            "MAP": ("map", dict(
                n_steps=500, learning_rate=0.03,
            )),
            "native_geovi": ("geovi", dict(
                n_iterations=5, n_posterior_samples=50,
            )),
            "Ray Tracing": ("raytrace", dict(
                n_steps=200, n_burnin=50, n_leapfrog_steps=10,
            )),
            "NUTS": ("nuts", dict(
                n_warmup=200, n_samples=200,
                target_accept_rate=0.85,
            )),
            "geovi_nuts": ("geovi_nuts", dict(
                n_iterations=3, n_posterior_samples=100,
            )),
        }
    return {
        "MAP": ("map", dict(
            n_steps=2000, learning_rate=0.03,
        )),
        "native_geovi": ("geovi", dict(
            n_iterations=10, n_posterior_samples=100,
        )),
        "Ray Tracing": ("raytrace", dict(
            n_steps=500, n_burnin=100, n_leapfrog_steps=10,
        )),
        "NUTS": ("nuts", dict(
            n_warmup=500, n_samples=500,
            target_accept_rate=0.85,
        )),
        "geovi_nuts": ("geovi_nuts", dict(
            n_iterations=5, n_posterior_samples=200,
        )),
    }


# ── Single method runner ──────────────────────────────────────────
def run_method(model, flux_obs, noise, method_str, key, **kwargs):
    """Run one inference method, return (Posterior, wall_total, wall_runtime).

    wall_total includes compile; wall_runtime is the second call
    (post-compile) when possible, else equals wall_total.
    """
    fitter = Fitter(model, flux_obs, noise)

    if method_str == "map":
        # MAP: single run, no compile warmup needed
        t0 = time.perf_counter()
        posterior = fitter.run("map", key=key, verbose=False, **kwargs)
        jax.block_until_ready(posterior.params)
        wall_total = time.perf_counter() - t0
        return posterior, wall_total, wall_total

    # For sampling methods: MAP init first
    map_result = fitter.run(
        "map", key=key, n_steps=500, learning_rate=0.03, verbose=False,
    )
    key = jax.random.fold_in(key, 1)

    # First call (includes XLA compile)
    t0 = time.perf_counter()
    posterior = fitter.run(
        method_str, init_from=map_result, key=key, verbose=False, **kwargs,
    )
    if posterior.samples is not None:
        jax.block_until_ready(
            jnp.stack([v for v in posterior.samples.values() if v.ndim == 1])
        )
    wall_total = time.perf_counter() - t0

    # Second call for runtime-only measurement (compiled)
    key2 = jax.random.fold_in(key, 2)
    t1 = time.perf_counter()
    posterior2 = fitter.run(
        method_str, init_from=map_result, key=key2, verbose=False, **kwargs,
    )
    if posterior2.samples is not None:
        jax.block_until_ready(
            jnp.stack([v for v in posterior2.samples.values() if v.ndim == 1])
        )
    wall_runtime = time.perf_counter() - t1

    return posterior2, wall_total, wall_runtime


# ── Metrics computation ───────────────────────────────────────────
def compute_metrics(posterior, true_params, wall_total, wall_runtime):
    """Compute quality metrics for one method.

    Returns
    -------
    dict with keys: wall_total, wall_runtime, n_samples, ess_per_param,
        ess_min, ess_per_sec, bias_per_param, mean_abs_bias, coverage_68.
    """
    metrics = {
        "wall_total": wall_total,
        "wall_runtime": wall_runtime,
    }

    summary = posterior.summary()
    scalar_params = [
        k for k in summary if k != "psd_xi"
    ]

    if posterior.samples is None:
        # MAP: no ESS, compute bias from point estimates
        metrics["n_samples"] = 0
        metrics["ess_per_param"] = {}
        metrics["ess_min"] = 0.0
        metrics["ess_per_sec"] = 0.0

        biases = {}
        for name in scalar_params:
            truth = float(jnp.mean(true_params[name]))
            est = summary[name]["value"]
            biases[name] = abs(est - truth) / max(abs(truth), 1e-6)
        metrics["bias_per_param"] = biases
        metrics["mean_abs_bias"] = float(np.mean(list(biases.values())))
        metrics["coverage_68"] = float("nan")
        return metrics

    # Sampling methods
    n_samples = next(iter(posterior.samples.values())).shape[0]
    metrics["n_samples"] = n_samples

    # ESS
    try:
        ess = posterior.effective_sample_size()
        ess_scalar = {k: v for k, v in ess.items() if k in scalar_params}
    except Exception:
        ess_scalar = {k: float("nan") for k in scalar_params}
    metrics["ess_per_param"] = ess_scalar
    ess_vals = [v for v in ess_scalar.values() if not np.isnan(v)]
    metrics["ess_min"] = float(min(ess_vals)) if ess_vals else 0.0
    metrics["ess_per_sec"] = (
        metrics["ess_min"] / wall_runtime if wall_runtime > 0 else 0.0
    )

    # Bias: |median - truth| / |truth|
    biases = {}
    for name in scalar_params:
        truth = float(jnp.mean(true_params[name]))
        med = summary[name]["median"]
        biases[name] = abs(med - truth) / max(abs(truth), 1e-6)
    metrics["bias_per_param"] = biases
    metrics["mean_abs_bias"] = float(np.mean(list(biases.values())))

    # 68% CI coverage: fraction of params where truth in [lo_68, hi_68]
    n_covered = 0
    n_total = 0
    for name in scalar_params:
        truth = float(jnp.mean(true_params[name]))
        lo = summary[name]["lo_68"]
        hi = summary[name]["hi_68"]
        if lo <= truth <= hi:
            n_covered += 1
        n_total += 1
    metrics["coverage_68"] = n_covered / n_total if n_total > 0 else float("nan")

    return metrics


def compute_kl_proxy(all_metrics, reference="NUTS"):
    """Compute a KL divergence proxy for VI methods vs a reference.

    Approximates KL as sum of (mu_vi - mu_ref)^2 / sigma_ref^2
    over scalar parameters, using posterior means and stds.
    """
    if reference not in all_metrics:
        return {}

    ref = all_metrics[reference]
    if ref.get("_summary") is None:
        return {}

    ref_summary = ref["_summary"]
    kl_proxies = {}

    for method_name, m in all_metrics.items():
        if method_name == reference:
            continue
        if m.get("_summary") is None or m.get("n_samples", 0) == 0:
            continue

        kl = 0.0
        n_params = 0
        for name in ref_summary:
            if name not in m["_summary"]:
                continue
            if "median" not in ref_summary[name] or "median" not in m["_summary"][name]:
                continue
            mu_ref = ref_summary[name]["median"]
            sigma_ref = (ref_summary[name]["hi_68"] - ref_summary[name]["lo_68"]) / 2.0
            mu_vi = m["_summary"][name]["median"]
            if sigma_ref > 1e-10:
                kl += (mu_vi - mu_ref) ** 2 / sigma_ref**2
                n_params += 1

        kl_proxies[method_name] = kl / max(n_params, 1)

    return kl_proxies


# ── Output: tables ────────────────────────────────────────────────
def print_summary_table(all_metrics, kl_proxies):
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Method':<16s} {'Total(s)':>8s} {'Run(s)':>8s} {'Nsamp':>6s} "
        f"{'ESSmin':>7s} {'ESS/s':>7s} {'|Bias|':>7s} {'Cov68':>6s} {'KLprx':>6s}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("INFERENCE QUALITY BENCHMARK")
    print(sep)
    print(header)
    print(sep)

    for name, m in all_metrics.items():
        n_samp = str(m["n_samples"]) if m["n_samples"] > 0 else "---"
        ess_min = f"{m['ess_min']:.0f}" if m["ess_min"] > 0 else "---"
        ess_s = f"{m['ess_per_sec']:.1f}" if m["ess_per_sec"] > 0 else "---"
        cov = f"{m['coverage_68']:.0%}" if not np.isnan(m["coverage_68"]) else "---"
        kl = f"{kl_proxies.get(name, float('nan')):.2f}" if name in kl_proxies else "---"

        print(
            f"{name:<16s} {m['wall_total']:>8.1f} {m['wall_runtime']:>8.1f} "
            f"{n_samp:>6s} {ess_min:>7s} {ess_s:>7s} "
            f"{m['mean_abs_bias']:>7.3f} {cov:>6s} {kl:>6s}"
        )

    print(sep)


def print_latex_table(all_metrics, kl_proxies):
    """Print a LaTeX-ready table."""
    print("\n% LaTeX table: inference quality comparison")
    print("\\begin{tabular}{lrrrrrrr}")
    print("\\toprule")
    print(
        "Method & Total (s) & Run (s) & $N_{\\rm samp}$ & "
        "ESS$_{\\min}$ & ESS/s & $|$Bias$|$ & Cov$_{68}$ \\\\"
    )
    print("\\midrule")

    for name, m in all_metrics.items():
        n_samp = f"{m['n_samples']}" if m['n_samples'] > 0 else "---"
        ess_min = f"{m['ess_min']:.0f}" if m['ess_min'] > 0 else "---"
        ess_s = f"{m['ess_per_sec']:.1f}" if m['ess_per_sec'] > 0 else "---"
        cov = f"{m['coverage_68']:.0%}" if not np.isnan(m['coverage_68']) else "---"

        print(
            f"{name} & {m['wall_total']:.1f} & {m['wall_runtime']:.1f} & "
            f"{n_samp} & {ess_min} & {ess_s} & "
            f"{m['mean_abs_bias']:.3f} & {cov} \\\\"
        )

    print("\\bottomrule")
    print("\\end{tabular}")


def print_ess_detail(all_metrics):
    """Print per-parameter ESS table for each method."""
    # Collect all param names
    param_names = set()
    for m in all_metrics.values():
        param_names.update(m["ess_per_param"].keys())
    param_names = sorted(param_names)

    if not param_names:
        return

    print(f"\n{'Parameter ESS (per method)':^80s}")
    header = f"{'Param':<28s}" + "".join(f" {n:>10s}" for n in all_metrics)
    print(header)
    print("-" * len(header))

    for p in param_names:
        row = f"{p:<28s}"
        for m in all_metrics.values():
            val = m["ess_per_param"].get(p, float("nan"))
            row += f" {val:>10.0f}" if not np.isnan(val) else f" {'---':>10s}"
        print(row)


# ── Output: figures ───────────────────────────────────────────────
def plot_ess_per_sec(all_metrics):
    """Bar chart of ESS/s by method."""
    plt = setup_matplotlib()

    methods = [n for n in all_metrics if all_metrics[n]["ess_per_sec"] > 0]
    vals = [all_metrics[n]["ess_per_sec"] for n in methods]

    colors = {
        "MAP": "#999999",
        "native_geovi": "#d62728",
        "Ray Tracing": "#ff7f0e",
        "NUTS": "#2ca02c",
        "geovi_nuts": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        range(len(methods)), vals,
        color=[colors.get(m, "#1f77b4") for m in methods],
        alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("ESS$_{\\min}$ / second")
    ax.set_title("Inference Efficiency: ESS per Second (D=7 Smooth)")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_wall_time(all_metrics):
    """Bar chart of wall-clock time (total and runtime) by method."""
    plt = setup_matplotlib()

    methods = list(all_metrics.keys())
    totals = [all_metrics[m]["wall_total"] for m in methods]
    runtimes = [all_metrics[m]["wall_runtime"] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods))
    w = 0.35

    bars_total = ax.bar(
        x - w / 2, totals, w, label="Total (incl. compile)",
        color="#1f77b4", alpha=0.85,
    )
    bars_run = ax.bar(
        x + w / 2, runtimes, w, label="Runtime only",
        color="#ff7f0e", alpha=0.85,
    )

    for bar, val in zip(bars_total, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=8,
        )
    for bar, val in zip(bars_run, runtimes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Inference Wall-Clock Time (D=7 Smooth)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)
    ax.legend(loc="upper left")

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference quality across methods.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use fewer samples for a faster run.",
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(42)

    print("Building model and generating mock galaxy (D=7 smooth, 5 SDSS bands)...")
    model, flux_obs, noise, true_params = build_model_and_truth(key)
    print(f"  Free parameters: {model.spec.free_params}")
    print(f"  D = {len(model.spec.free_params)}")

    method_configs = get_method_configs(args.quick)
    all_metrics = {}

    for name, (method_str, kwargs) in method_configs.items():
        print(f"\n--- {name} ({method_str}) ---")
        run_key = jax.random.fold_in(key, abs(hash(name)) % (2**31))

        try:
            posterior, wall_total, wall_runtime = run_method(
                model, flux_obs, noise, method_str, run_key, **kwargs,
            )
            metrics = compute_metrics(posterior, true_params, wall_total, wall_runtime)
            # Stash summary for KL proxy computation
            metrics["_summary"] = posterior.summary()
            all_metrics[name] = metrics

            print(f"  Total: {wall_total:.1f}s  Runtime: {wall_runtime:.1f}s")
            if metrics["n_samples"] > 0:
                print(f"  Samples: {metrics['n_samples']}  ESS_min: {metrics['ess_min']:.0f}")
            print(f"  Mean |bias|: {metrics['mean_abs_bias']:.3f}")
            if not np.isnan(metrics["coverage_68"]):
                print(f"  68% coverage: {metrics['coverage_68']:.0%}")

        except Exception as e:
            print(f"  FAILED: {e}")
            all_metrics[name] = {
                "wall_total": float("nan"),
                "wall_runtime": float("nan"),
                "n_samples": 0,
                "ess_per_param": {},
                "ess_min": 0.0,
                "ess_per_sec": 0.0,
                "bias_per_param": {},
                "mean_abs_bias": float("nan"),
                "coverage_68": float("nan"),
                "_summary": None,
            }

    # KL divergence proxy (VI methods vs NUTS)
    kl_proxies = compute_kl_proxy(all_metrics, reference="NUTS")

    # Tables
    print_summary_table(all_metrics, kl_proxies)
    print_latex_table(all_metrics, kl_proxies)
    print_ess_detail(all_metrics)

    # Figures
    fig_ess = plot_ess_per_sec(all_metrics)
    ess_path = FIG_DIR / "bench_ess_per_sec.pdf"
    fig_ess.savefig(ess_path)
    print(f"\nSaved: {ess_path}")

    fig_time = plot_wall_time(all_metrics)
    time_path = FIG_DIR / "bench_wall_time.pdf"
    fig_time.savefig(time_path)
    print(f"Saved: {time_path}")


if __name__ == "__main__":
    main()

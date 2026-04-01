#!/usr/bin/env python3
"""Figure 7: Speed comparison — MAP vs Ray Tracing vs NUTS vs geoVI.

Measures wall-clock time for each method on both the smooth (5D)
and stochastic (137D) models. Produces a grouped bar chart.

Usage:
    python analysis/fig07_speed_benchmarks.py [--n-repeats 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    FIG_DIR,
    PAPER_FIG_DIR,
    generate_mock_galaxy,
    make_model,
    setup_matplotlib,
)

from tengri import Fitter


def benchmark_method(model, galaxy, method, key, **kwargs):
    """Time a single fit, return (wall_time_s, diagnostics)."""
    fitter = Fitter(model, galaxy.flux_obs, galaxy.noise)

    t0 = time.time()

    if method == "map":
        result = fitter.run("map", key=key, verbose=False, **kwargs)
    else:
        # MAP init first
        result_map = fitter.run("map", key=key, n_steps=500, learning_rate=0.03, verbose=False)
        key = jax.random.fold_in(key, 1)
        result = fitter.run(method, init_from=result_map, key=key, verbose=False, **kwargs)

    wall_time = time.time() - t0
    diag = result.diagnostics.copy()
    diag["wall_time_s"] = wall_time
    diag["n_samples"] = diag.get("n_samples", 0)
    return wall_time, diag


def run_benchmarks(n_repeats: int = 3):
    """Run all benchmarks."""
    key = jax.random.PRNGKey(42)

    # Two model configurations
    configs = {
        "Smooth (5D)": dict(psd_regime="smooth", stochastic=False, n_grid=64),
        "Stochastic (137D)": dict(psd_regime="bursty", stochastic=True, n_grid=128),
    }

    # Methods and their kwargs
    methods = {
        "MAP (Adam)": ("map", dict(n_steps=2000, learning_rate=0.03)),
        "Ray Tracing": ("raytrace", dict(n_steps=400, n_leapfrog_steps=10, n_burnin=100)),
        "NUTS": ("nuts", dict(n_warmup=300, n_samples=200, target_accept_rate=0.85)),
        "geoVI": ("geovi", dict(n_iterations=15, n_posterior_samples=80)),
    }

    results = {}

    for config_name, config_kwargs in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Configuration: {config_name}")

        model = make_model(**config_kwargs, redshift=0.1)
        galaxy = generate_mock_galaxy(
            model, jax.random.fold_in(key, abs(hash(config_name)) % (2**31)), snr=20.0
        )

        results[config_name] = {}

        for method_name, (method, kwargs) in methods.items():
            # Skip NUTS on stochastic (too slow)
            if method == "nuts" and "Stochastic" in config_name:
                print(f"  {method_name}: SKIPPED (too slow for {config_name})")
                results[config_name][method_name] = {
                    "times": [np.nan] * n_repeats,
                    "mean": np.nan,
                    "std": np.nan,
                    "diagnostics": {},
                }
                continue

            times = []
            last_diag = {}

            for rep in range(n_repeats):
                rep_key = jax.random.fold_in(key, rep * 100 + abs(hash(method_name)) % (2**31))
                print(f"  {method_name} rep {rep + 1}/{n_repeats}...", end=" ", flush=True)

                try:
                    wt, diag = benchmark_method(
                        model,
                        galaxy,
                        method,
                        rep_key,
                        **kwargs,
                    )
                    times.append(wt)
                    last_diag = diag
                    print(f"{wt:.1f}s", end="")
                    if "accept_rate" in diag:
                        print(f" (accept={diag['accept_rate']:.0%})", end="")
                    if "n_divergent" in diag:
                        print(f" (div={diag['n_divergent']})", end="")
                    print()
                except Exception as e:
                    print(f"FAILED: {e}")
                    times.append(np.nan)

            results[config_name][method_name] = {
                "times": times,
                "mean": np.nanmean(times),
                "std": np.nanstd(times),
                "diagnostics": last_diag,
            }

    return results


def plot_speed_comparison(results):
    """Plot grouped bar chart."""
    plt = setup_matplotlib()

    configs = list(results.keys())
    methods = list(results[configs[0]].keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(configs))
    width = 0.18
    colors = {
        "MAP (Adam)": "#1f77b4",
        "Ray Tracing": "#ff7f0e",
        "NUTS": "#2ca02c",
        "geoVI": "#d62728",
    }

    for i, method in enumerate(methods):
        means = [results[c][method]["mean"] for c in configs]
        stds = [results[c][method]["std"] for c in configs]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=method,
            color=colors.get(method, f"C{i}"),
            capsize=3,
            alpha=0.85,
        )

        # Annotate with time
        for j, (bar, m) in enumerate(zip(bars, means)):
            if not np.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[j] + 0.5,
                    f"{m:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Inference Speed Comparison")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()
    return fig


def print_table(results):
    """Print LaTeX-ready table."""
    configs = list(results.keys())
    methods = list(results[configs[0]].keys())

    print(f"\n{'Method':<16s}", end="")
    for c in configs:
        print(f" {c:>20s}", end="")
    print()
    print("-" * (16 + 22 * len(configs)))

    for method in methods:
        print(f"{method:<16s}", end="")
        for c in configs:
            r = results[c][method]
            if np.isnan(r["mean"]):
                print(f" {'---':>20s}", end="")
            else:
                n_samples = r["diagnostics"].get("n_samples", "?")
                print(f" {r['mean']:>6.1f}s ({n_samples} samp){'':<5s}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-repeats", type=int, default=3, help="Number of timing repeats (default: 3)"
    )
    args = parser.parse_args()

    results = run_benchmarks(n_repeats=args.n_repeats)

    print_table(results)

    fig = plot_speed_comparison(results)
    out_path = FIG_DIR / "fig07_speed_benchmarks.pdf"
    fig.savefig(out_path)
    print(f"\nSaved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig07_speed_benchmarks.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()

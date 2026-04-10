"""Benchmark precision vs speed tradeoffs in tengri.

Three studies:
1. Mixed precision (float32 vs float64) photometry error
2. Photometry precomputation error by dust law
3. Ray Tracing step_size acceptance curve

Usage::

    python analysis/bench_accuracy_tradeoffs.py          # full run
    python analysis/bench_accuracy_tradeoffs.py --quick   # fast smoke test
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

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
SSP_FILE = PROJECT_ROOT / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

FILTER_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
DUST_LAWS = ["power_law", "calzetti", "kriek_conroy", "smc", "cardelli", "salim"]


def _load_data():
    """Load SSP data and observation, returning None on failure."""
    try:
        from tengri import Observation, Photometry, load_ssp_data

        ssp = load_ssp_data(str(SSP_FILE))
        obs = Observation(photometry=Photometry.from_names(FILTER_NAMES))
        return ssp, obs
    except Exception as exc:
        print(f"[SKIP] Cannot load SSP data: {exc}")
        return None


def _make_smooth_spec(*, redshift=0.1, dust_tau_bc=1.0, dust_law_bc="power_law"):
    """Create a smooth DPL ParamSpec with fixed redshift."""
    from tengri import Fixed, ParamSpec, Uniform

    return ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Fixed(dust_tau_bc),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(redshift),
        dust_law_bc=dust_law_bc,
        dust_law_diff=dust_law_bc,
    )


# ═══════════════════════════════════════════════════════════════════
# Study 1: Mixed precision error (float32 vs float64)
# ═══════════════════════════════════════════════════════════════════


def study_mixed_precision(ssp, obs, *, n_draws=20):
    """Compare float32 vs float64 photometry across redshifts and dust."""
    from tengri import Model

    redshifts = [0.01, 0.1, 0.5, 1.0, 3.0]
    dust_values = [0.0, 1.0, 2.0, 3.0]

    print("\n" + "=" * 74)
    print("STUDY 1: Mixed precision error (float32 vs float64)")
    print("=" * 74)
    print(f"{'z':>5s}  {'tau_bc':>6s}  {'mean_err':>10s}  {'max_err':>10s}  "
          f"{'worst_band':>10s}")
    print("-" * 50)

    rows = []
    for z in redshifts:
        for tau_bc in dust_values:
            spec = _make_smooth_spec(redshift=z, dust_tau_bc=tau_bc)
            model_f64 = Model(spec, ssp, observation=obs, forward_dtype="float64",
                              precompute=False)
            model_f32 = Model(spec, ssp, observation=obs, forward_dtype="float32",
                              precompute=False)

            all_rel = []
            for i in range(n_draws):
                key = jax.random.PRNGKey(i)
                params = spec.sample(key)
                flux_64 = model_f64.predict_photometry(params)
                flux_32 = model_f32.predict_photometry(params)
                rel = jnp.abs(flux_64 - flux_32) / jnp.maximum(jnp.abs(flux_64), 1e-30)
                all_rel.append(np.asarray(rel))

            all_rel = np.stack(all_rel)  # (n_draws, n_bands)
            mean_err = float(np.mean(all_rel))
            max_err = float(np.max(all_rel))
            worst_band = FILTER_NAMES[int(np.argmax(np.max(all_rel, axis=0)))]

            rows.append({
                "z": z, "tau_bc": tau_bc,
                "mean_err": mean_err, "max_err": max_err,
                "worst_band": worst_band,
            })
            print(f"{z:5.2f}  {tau_bc:6.1f}  {mean_err:10.2e}  {max_err:10.2e}  "
                  f"{worst_band:>10s}")

    return rows


# ═══════════════════════════════════════════════════════════════════
# Study 2: Photometry precomputation error by dust law
# ═══════════════════════════════════════════════════════════════════


def study_precompute_accuracy(ssp, obs, *, n_draws=20):
    """Compare precomputed (fast) vs exact photometry per dust law."""
    from tengri import Model

    print("\n" + "=" * 74)
    print("STUDY 2: Photometry precomputation error by dust law")
    print("=" * 74)
    print(f"{'law':>16s}  {'mean_err':>10s}  {'max_err':>10s}  {'worst_band':>10s}")
    print("-" * 54)

    rows = []
    for law in DUST_LAWS:
        spec = _make_smooth_spec(dust_law_bc=law, dust_tau_bc=1.5)
        model_fast = Model(spec, ssp, observation=obs, precompute=True)
        model_exact = Model(spec, ssp, observation=obs, precompute=False)

        all_rel = []
        for i in range(n_draws):
            key = jax.random.PRNGKey(i)
            params = spec.sample(key)
            flux_fast = model_fast.predict_photometry(params)
            flux_exact = model_exact.predict_photometry(params)
            rel = jnp.abs(flux_fast - flux_exact) / jnp.maximum(jnp.abs(flux_exact), 1e-30)
            all_rel.append(np.asarray(rel))

        all_rel = np.stack(all_rel)
        mean_err = float(np.mean(all_rel))
        max_err = float(np.max(all_rel))
        worst_band = FILTER_NAMES[int(np.argmax(np.max(all_rel, axis=0)))]

        rows.append({
            "law": law, "mean_err": mean_err, "max_err": max_err,
            "worst_band": worst_band,
        })
        print(f"{law:>16s}  {mean_err:10.2e}  {max_err:10.2e}  {worst_band:>10s}")

    return rows


# ═══════════════════════════════════════════════════════════════════
# Study 3: Ray Tracing step_size acceptance curve
# ═══════════════════════════════════════════════════════════════════


def study_rt_acceptance(ssp, obs, *, n_steps=50, n_burnin=10):
    """Measure RT acceptance rate vs step_size for a D=7 smooth model."""
    from tengri import Fitter, Model

    print("\n" + "=" * 74)
    print("STUDY 3: Ray Tracing acceptance rate vs step_size (D=7 smooth)")
    print("=" * 74)

    spec = _make_smooth_spec(redshift=0.1, dust_tau_bc=1.0)
    model = Model(spec, ssp, observation=obs, precompute=True)

    # Generate mock data for fitting
    key = jax.random.PRNGKey(99)
    key, mock_key = jax.random.split(key)
    true_params = spec.sample(mock_key)
    mock = model.mock(true_params, snr=20.0, key=mock_key)

    fitter = Fitter(model, mock.flux_obs, mock.noise)
    n_free = len(fitter._free_names)
    print(f"Free parameters: {n_free} ({', '.join(fitter._free_names)})")

    step_sizes = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    print(f"\n{'step_size':>10s}  {'accept_rate':>12s}  {'time_s':>8s}")
    print("-" * 36)

    results = []
    for ss in step_sizes:
        key, run_key = jax.random.split(key)
        t0 = time.perf_counter()
        try:
            post = fitter.run(
                "mcmc_raytrace",
                key=run_key,
                n_burnin=n_burnin,
                n_steps=n_steps,
                step_size=ss,
                n_leapfrog_steps=10,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            acc = post.diagnostics.get("acceptance_rate", float("nan"))
            results.append({"step_size": ss, "acceptance_rate": acc, "time_s": elapsed})
            print(f"{ss:10.3f}  {acc:11.1%}  {elapsed:8.1f}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            results.append({"step_size": ss, "acceptance_rate": 0.0, "time_s": elapsed})
            print(f"{ss:10.3f}  {'FAILED':>12s}  {elapsed:8.1f}  ({exc})")

    # Save figure
    _plot_acceptance_curve(results)
    return results


def _plot_acceptance_curve(results):
    """Save acceptance rate vs step_size figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available, skipping figure.")
        return

    step_sizes = [r["step_size"] for r in results]
    accept_rates = [r["acceptance_rate"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(step_sizes, accept_rates, "o-", color="C0", linewidth=2, markersize=7)
    ax.axhspan(30, 70, alpha=0.15, color="green", label="Ideal range (30-70%)")
    ax.set_xlabel("Step size")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Ray Tracing: Acceptance vs Step Size (D=7 smooth)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / "bench_rt_acceptance_curve.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark precision vs speed tradeoffs in tengri."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer parameter draws and shorter RT chains.",
    )
    args = parser.parse_args()

    n_draws = 5 if args.quick else 20
    rt_steps = 20 if args.quick else 50
    rt_burnin = 5 if args.quick else 10

    loaded = _load_data()
    if loaded is None:
        sys.exit(1)
    ssp, obs = loaded

    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"SSP: {SSP_FILE.name}")
    print(f"Filters: {FILTER_NAMES}")

    study_mixed_precision(ssp, obs, n_draws=n_draws)
    study_precompute_accuracy(ssp, obs, n_draws=n_draws)
    study_rt_acceptance(ssp, obs, n_steps=rt_steps, n_burnin=rt_burnin)

    print("\n" + "=" * 74)
    print("All studies complete.")
    print("=" * 74)


if __name__ == "__main__":
    main()

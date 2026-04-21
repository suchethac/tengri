"""Adam vs L-BFGS MAP comparison with ground-truth recovery."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Parameters,
    SEDModel,
    Uniform,
    load_filter_set,
    load_ssp_data,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SSP_FILE = DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

TRUE_PARAMS = {
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 3.0,
    "sfh_tsnorm_width_gyr": 1.5,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.3,
    "dust_slope": -0.7,
    "redshift": 0.1,
}


def build_fitter():
    ssp = load_ssp_data(str(SSP_FILE))
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    spec = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )
    model = SEDModel(spec, ssp, filters=filters)
    mock = model.mock(TRUE_PARAMS, snr=20.0, key=jr.PRNGKey(42))
    return Fitter(model, mock.flux_obs, mock.noise), spec


def run_map(fitter, optimizer, n_steps=500):
    key = jr.PRNGKey(0)
    kwargs = {"method": "map", "key": key, "optimizer": optimizer,
              "n_steps": n_steps, "verbose": False}

    # Warmup (JIT compilation)
    _ = fitter.run(**kwargs)

    # Timed run
    t0 = time.perf_counter()
    result = fitter.run(**{**kwargs, "key": jr.PRNGKey(1)})
    jax.block_until_ready(result.params)
    wall = time.perf_counter() - t0

    return result, wall


def main():
    if not SSP_FILE.is_file():
        print("SSP file not found — skipping")
        return

    print("Building model + fitter...", end=" ", flush=True)
    fitter, spec = build_fitter()
    free_names = spec.free_params
    print(f"done (D={len(free_names)})")
    print()

    results = {}
    for opt in ("adam", "lbfgs"):
        print(f"Running MAP/{opt}...", end=" ", flush=True)
        result, wall = run_map(fitter, opt)
        results[opt] = (result, wall)
        print(f"done ({wall:.3f}s)")

    # ── Parameter recovery table ──
    print()
    print("=" * 90)
    print("  PARAMETER RECOVERY (warm run, SNR=20)")
    print("=" * 90)
    print()
    print(f"{'Parameter':<30} {'True':>8} {'Adam':>8} {'L-BFGS':>8} "
          f"{'Adam err%':>10} {'LBFGS err%':>10}")
    print("-" * 90)

    adam_result = results["adam"][0]
    lbfgs_result = results["lbfgs"][0]

    adam_errors = []
    lbfgs_errors = []

    for name in free_names:
        true_val = TRUE_PARAMS[name]
        adam_val = float(adam_result.params[name])
        lbfgs_val = float(lbfgs_result.params[name])

        if abs(true_val) > 1e-10:
            adam_err = (adam_val - true_val) / abs(true_val) * 100
            lbfgs_err = (lbfgs_val - true_val) / abs(true_val) * 100
        else:
            adam_err = adam_val - true_val
            lbfgs_err = lbfgs_val - true_val

        adam_errors.append(abs(adam_err))
        lbfgs_errors.append(abs(lbfgs_err))

        print(f"{name:<30} {true_val:>8.3f} {adam_val:>8.3f} {lbfgs_val:>8.3f} "
              f"{adam_err:>+9.2f}% {lbfgs_err:>+9.2f}%")

    # ── Summary ──
    print()
    print("=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print()

    adam_r, adam_w = results["adam"]
    lbfgs_r, lbfgs_w = results["lbfgs"]

    adam_diag = adam_r.diagnostics
    lbfgs_diag = lbfgs_r.diagnostics

    print(f"{'Metric':<30} {'Adam':>15} {'L-BFGS':>15}")
    print("-" * 62)
    print(f"{'Wall time (warm)':.<30} {adam_w:>14.3f}s {lbfgs_w:>14.3f}s")
    print(f"{'Final loss':.<30} {adam_diag['final_loss']:>15.6f} "
          f"{lbfgs_diag['final_loss']:>15.6f}")
    print(f"{'Steps/iterations':.<30} {adam_diag['n_steps']:>15d} "
          f"{lbfgs_diag['n_steps']:>15d}")

    if "converged" in lbfgs_diag:
        print(f"{'Converged':.<30} {'—':>15} "
              f"{'yes' if lbfgs_diag['converged'] else 'no':>15}")
    if "grad_norm" in lbfgs_diag:
        print(f"{'Grad norm':.<30} {'—':>15} "
              f"{lbfgs_diag['grad_norm']:>15.2e}")

    print(f"{'Mean |param error|':.<30} "
          f"{np.mean(adam_errors):>14.2f}% {np.mean(lbfgs_errors):>14.2f}%")
    print(f"{'Max |param error|':.<30} "
          f"{np.max(adam_errors):>14.2f}% {np.max(lbfgs_errors):>14.2f}%")

    # Which is better?
    print()
    adam_loss = adam_diag["final_loss"]
    lbfgs_loss = lbfgs_diag["final_loss"]
    if lbfgs_loss < adam_loss:
        print(f"L-BFGS reaches {(adam_loss - lbfgs_loss)/adam_loss*100:.1f}% lower loss "
              f"and is {adam_w/lbfgs_w:.1f}x {'faster' if lbfgs_w < adam_w else 'slower'} "
              f"(warm).")
    else:
        print(f"Adam reaches {(lbfgs_loss - adam_loss)/lbfgs_loss*100:.1f}% lower loss "
              f"and is {lbfgs_w/adam_w:.1f}x {'faster' if adam_w < lbfgs_w else 'slower'} "
              f"(warm).")


if __name__ == "__main__":
    main()

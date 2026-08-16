"""SFH isolation test for vi NaN debugging.

Tests three SFH models in order of increasing complexity:
  1. exp   — single exponential, fully analytic, no matrix solve
  2. dpl   — double power-law, fully analytic, no matrix solve
  3. dense_basis — GP via linalg.solve (the suspected NaN source)

If exp and dpl succeed but dense_basis fails → NaN is isolated to the GP kernel
matrix path (linalg.solve / its JVP in NIFTy's metric update).

Run from the repo root:
    source .venv/bin/activate
    cd ~/Projects/tengri
    python scripts/isolate_nan_sfh.py
"""

import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Path setup (same as quickstart)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))
_src = os.path.join(_repo_root, "src")
if os.path.isdir(os.path.join(_src, "tengri")):
    sys.path.insert(0, _src)
sys.path.insert(0, _repo_root)
os.chdir(_repo_root)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import warnings

warnings.filterwarnings("ignore")

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Spectroscopy,
    Uniform,
    load_ssp_data,
)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
SSP_FILE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
N_ITERATIONS = 8
N_SAMPLES = 4

print("Loading SSP data...")
ssp_data = load_ssp_data(SSP_FILE)
obs = Observation(spectroscopy=Spectroscopy(wave_obs=WAVE_OBS))
print(f"SSP: {ssp_data.ssp_flux.shape[0]} metallicities × {ssp_data.ssp_flux.shape[1]} ages")
print()


def run_sfh_test(sfh_name: str, spec: Parameters, true_params: dict) -> bool:
    """Run MAP + vi for a given SFH spec. Returns True on success, False on NaN."""
    print(f"{'=' * 60}")
    print(f"Testing SFH: {sfh_name}  (D = {spec.n_free} free params)")
    print(f"  Free params: {spec.free_params}")
    print(f"{'=' * 60}")

    try:
        model = SEDModel(spec, ssp_data, observation=obs)
        model.precompute_spectroscopy(WAVE_OBS)

        # Generate mock
        key = jax.random.PRNGKey(42)
        mock = model.mock_spectrum(true_params, WAVE_OBS, snr=30.0, key=key)
        print(
            f"  Mock spectrum: min={float(mock.flux_obs.min()):.3e}, max={float(mock.flux_obs.max()):.3e}"
        )

        fitter = Fitter(model, mock.flux_obs, mock.noise)

        # MAP
        print("  Running MAP (500 steps)...")
        t0 = time.perf_counter()
        result_map = fitter.run("map", n_steps=500, verbose=False)
        t_map = time.perf_counter() - t0
        print(f"  MAP done: {t_map:.1f}s")

        # vi (geoVI)
        print(f"  Running vi (n_iterations={N_ITERATIONS}, n_samples={N_SAMPLES})...")
        t0 = time.perf_counter()
        result_vi = fitter.run(
            "vi",
            n_iterations=N_ITERATIONS,
            n_samples=N_SAMPLES,
            n_posterior_samples=200,
            verbose=True,
        )
        t_vi = time.perf_counter() - t0
        print(f"  vi done: {t_vi:.1f}s  *** SUCCESS ***")
        return True

    except Exception as e:
        print(f"  *** FAILED: {type(e).__name__}: {e} ***")
        traceback.print_exc()
        return False
    finally:
        print()


# ---------------------------------------------------------------------------
# Test 1: exp (single exponential — fully analytic, no matrix solve)
# ---------------------------------------------------------------------------
spec_exp = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_exp_tau_gyr=Uniform(0.5, 8.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="exp",
)
true_exp = {
    "sfh_exp_log_total_mass": jnp.array(1.0),
    "sfh_exp_tau_gyr": jnp.array(3.0),
    "met_logzsol": jnp.array(-0.3),
    "dust_tau_bc": jnp.array(0.5),
    "dust_tau_diff": jnp.array(0.2),
    "dust_slope": jnp.array(-0.7),
    "redshift": jnp.array(0.1),
}
ok_exp = run_sfh_test("exp (single exponential)", spec_exp, true_exp)


# ---------------------------------------------------------------------------
# Test 2: dpl (double power-law — fully analytic, no matrix solve)
# ---------------------------------------------------------------------------
spec_dpl = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_dpl_alpha=Uniform(0.5, 5.0),
    sfh_dpl_beta=Uniform(0.5, 5.0),
    sfh_dpl_tau_gyr=Uniform(0.1, 5.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)
true_dpl = {
    "sfh_dpl_log_total_mass": jnp.array(1.0),
    "sfh_dpl_alpha": jnp.array(2.0),
    "sfh_dpl_beta": jnp.array(1.5),
    "sfh_dpl_tau_gyr": jnp.array(2.0),
    "met_logzsol": jnp.array(-0.3),
    "dust_tau_bc": jnp.array(0.5),
    "dust_tau_diff": jnp.array(0.2),
    "dust_slope": jnp.array(-0.7),
    "redshift": jnp.array(0.1),
}
ok_dpl = run_sfh_test("dpl (double power-law)", spec_dpl, true_dpl)


# ---------------------------------------------------------------------------
# Test 3: dense_basis (GP SFH via linalg.solve — the suspected NaN source)
# ---------------------------------------------------------------------------
spec_db = Parameters(
    sfh_db_log_total_mass=Uniform(8, 12),
    sfh_db_log_sfr_inst=Uniform(-2, 3),
    sfh_db_tx_frac_0=Uniform(0.05, 0.95),
    sfh_db_tx_frac_1=Uniform(0.05, 0.95),
    sfh_db_tx_frac_2=Uniform(0.05, 0.95),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dense_basis",
)
true_db = {
    "sfh_db_log_total_mass": jnp.array(10.5),
    "sfh_db_log_sfr_inst": jnp.array(0.8),
    "sfh_db_tx_frac_0": jnp.array(0.25),
    "sfh_db_tx_frac_1": jnp.array(0.35),
    "sfh_db_tx_frac_2": jnp.array(0.4),
    "met_logzsol": jnp.array(-0.3),
    "dust_tau_bc": jnp.array(0.5),
    "dust_tau_diff": jnp.array(0.2),
    "dust_slope": jnp.array(-0.7),
    "redshift": jnp.array(0.1),
}
ok_db = run_sfh_test("dense_basis (GP SFH, linalg.solve)", spec_db, true_db)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("ISOLATION SUMMARY")
print("=" * 60)
print(f"  exp         (analytic, no solve) : {'OK' if ok_exp else 'FAILED'}")
print(f"  dpl         (analytic, no solve) : {'OK' if ok_dpl else 'FAILED'}")
print(f"  dense_basis (GP, linalg.solve)   : {'OK' if ok_db else 'FAILED'}")
print()
if ok_exp and ok_dpl and not ok_db:
    print("DIAGNOSIS: NaN is isolated to the dense_basis GP kernel solve path.")
    print("  → Check gp_interpolate nugget size and linalg.solve JVP stability.")
elif not ok_exp:
    print("DIAGNOSIS: NaN occurs even with purely analytic SFH (exp).")
    print("  → NaN is NOT from dense_basis. Suspect dust, metallicity, or NIFTy setup.")
elif ok_exp and not ok_dpl:
    print("DIAGNOSIS: NaN occurs with dpl but not exp.")
    print("  → Suspect something in dpl-specific code path.")
else:
    print("DIAGNOSIS: All models OK — no NaN observed with current settings.")

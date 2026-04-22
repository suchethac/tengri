"""Benchmark NUTS wall time: window adaptation vs. Pathfinder warm-start.

Runs each path twice — the first call pays JIT compile cost; the second is
the representative wall time. Reports wall time, divergences, and step size.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path

from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.inference.fitter import Fitter
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

_DATA = Path(__file__).resolve().parents[1] / "data"
_SSP = _DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


def _load_ssp():
    return load_ssp_data(str(_SSP))


def _make_fitter(ssp, filters):
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 1.0),
        dust_slope=Uniform(-1.5, -0.2),
        redshift=0.1,
    )
    model = SEDModel(spec, ssp, filters=filters)
    true_params = {
        "sfh_dpl_alpha": 1.2,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 4.0,
        "sfh_dpl_log_peak_sfr": 0.9,
        "met_logzsol": -0.3,
        "dust_tau_bc": 1.0,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))
    return Fitter(model, mock.flux_obs, mock.noise)


def _time(fitter, *, pathfinder_warmstart, n_warmup, key_seed):
    # First run: includes JIT compile.
    t0 = time.time()
    fitter.run(
        "mcmc_nuts",
        pathfinder_warmstart=pathfinder_warmstart,
        n_warmup=n_warmup,
        n_burnin=50,
        n_samples=200,
        key=jax.random.PRNGKey(key_seed),
        verbose=False,
    )
    compile_time = time.time() - t0

    # Second run: representative (cached compilation + possibly cached adaptation).
    t0 = time.time()
    r = fitter.run(
        "mcmc_nuts",
        pathfinder_warmstart=pathfinder_warmstart,
        n_warmup=n_warmup,
        n_burnin=50,
        n_samples=200,
        key=jax.random.PRNGKey(key_seed + 1),
        verbose=False,
    )
    warm_time = time.time() - t0
    return compile_time, warm_time, r


def main():
    print("Loading SSP + filters...")
    ssp = _load_ssp()
    from tengri.observation.filters import load_filter_set

    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

    fitter = _make_fitter(ssp, filters)
    n_dim = fitter.spec.n_free
    print(f"Model built, D = {n_dim} free params.\n")

    print("=" * 70)
    print("Window adaptation (default)")
    print("=" * 70)
    c1, w1, r1 = _time(fitter, pathfinder_warmstart=False, n_warmup=300, key_seed=42)
    print(f"  compile+run: {c1:.2f}s  warm run: {w1:.2f}s")
    print(
        f"  divergences: {r1.diagnostics['n_divergent']}/200  "
        f"step_size: {r1.diagnostics['step_size']:.4f}"
    )

    # Clear adaptation cache by building a fresh fitter (avoid warm path reuse).
    fitter = _make_fitter(ssp, filters)
    print()
    print("=" * 70)
    print("Pathfinder warm-start (n_warmup=50)")
    print("=" * 70)
    c2, w2, r2 = _time(fitter, pathfinder_warmstart=True, n_warmup=50, key_seed=42)
    print(f"  compile+run: {c2:.2f}s  warm run: {w2:.2f}s")
    print(
        f"  divergences: {r2.diagnostics['n_divergent']}/200  "
        f"step_size: {r2.diagnostics['step_size']:.4f}"
    )

    # Fair comparison: match n_warmup.
    fitter = _make_fitter(ssp, filters)
    print()
    print("=" * 70)
    print("Pathfinder warm-start (n_warmup=300, matched to window)")
    print("=" * 70)
    c3, w3, r3 = _time(fitter, pathfinder_warmstart=True, n_warmup=300, key_seed=42)
    print(f"  compile+run: {c3:.2f}s  warm run: {w3:.2f}s")
    print(
        f"  divergences: {r3.diagnostics['n_divergent']}/200  "
        f"step_size: {r3.diagnostics['step_size']:.4f}"
    )

    print()
    print("=" * 70)
    print("Summary (warm-run wall time — JIT cache populated)")
    print("=" * 70)
    print(f"  Window adaptation (n_warmup=300):    {w1:.2f}s")
    print(
        f"  Pathfinder warm-start (n_warmup=50): {w2:.2f}s   "
        f"{'faster' if w2 < w1 else 'slower'} by {abs(w1 - w2) / w1 * 100:.1f}%"
    )
    print(
        f"  Pathfinder warm-start (n_warmup=300):{w3:.2f}s   "
        f"{'faster' if w3 < w1 else 'slower'} by {abs(w1 - w3) / w1 * 100:.1f}%"
    )


if __name__ == "__main__":
    main()

"""Compare predict_observables (eager) vs predict_observables_jit (jitted) across cold/warm.

Also log XLA compile events via JAX_LOG_COMPILES.

Set JAX_LOG_COMPILES=1 in the environment when running; this file just measures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Use lower threshold so micro-kernels also persist.
# (Comment this out to see default behaviour.)
# os.environ.setdefault("TENGRI_JAX_MIN_COMPILE_S", "0.0")  # not yet wired
import jax

import tengri

SSP_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "default"
    t0 = time.perf_counter()
    ssp = tengri.load_ssp_data(SSP_PATH)
    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]
        )
    )
    model = tengri.SEDModel.build(
        ssp_data=ssp, observation=obs, **tengri.recipes.mock_recovery_minimal()
    )
    truth = model.spec.sample(jax.random.PRNGKey(0))
    print(f"[{label}] build_model            : {time.perf_counter() - t0:6.2f}s")

    # Eager path (what the quickstart actually calls)
    t = time.perf_counter()
    _ = model.predict_observables(truth)
    print(f"[{label}] predict_observables  #1 : {time.perf_counter() - t:6.2f}s  (eager)")
    t = time.perf_counter()
    _ = model.predict_observables(truth)
    print(f"[{label}] predict_observables  #2 : {time.perf_counter() - t:6.2f}s  (eager)")

    # JIT path
    t = time.perf_counter()
    _ = model.predict_observables_jit(truth)
    jax.block_until_ready(_.phot_fnu)
    print(f"[{label}] predict_observables_jit #1: {time.perf_counter() - t:6.2f}s  (compile)")
    t = time.perf_counter()
    _ = model.predict_observables_jit(truth)
    jax.block_until_ready(_.phot_fnu)
    print(f"[{label}] predict_observables_jit #2: {time.perf_counter() - t:6.2f}s  (warm in-proc)")

    print(f"[{label}] cache_size_total       : {tengri.cache_size_bytes() / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()

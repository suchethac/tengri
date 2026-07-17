#!/usr/bin/env python
"""Test the 00_quickstart notebook's early sections (before NUTS fit)."""
import os
import sys
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*WavePrecomp.*")
warnings.filterwarnings("ignore", message=".*Fitter.*deprecated.*")
warnings.filterwarnings("ignore", message=".*was marked FIXED.*")
warnings.filterwarnings("ignore", message=".*Composable AGN.*")
warnings.filterwarnings("ignore", message=".*before the Big Bang.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import tengri
from tengri import (
    FIXED,
    FREE,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    citations,
    cosmology,
    generate_mock,
    load_ssp_data,
    plot,
)
from tengri.utils.conversions import lnu_to_fnu

print("\n=== SECTION 1: Imports ===")
try:
    plot.setup_style()
    FIG_DIR = Path("_figs")
    FIG_DIR.mkdir(exist_ok=True)
    print("PASS: Imports and setup successful")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== SECTION 2: Load SSP ===")
try:
    SSP_NAME = "fsps_prsc_miles_chabrier"
    ssp_path = Path("../data") / f"{SSP_NAME}.h5"
    if not ssp_path.exists():
        print(f"  Note: {ssp_path} not found, trying to download...")
        ssp_path = Path(tengri.download_ssp(SSP_NAME))
    ssp = load_ssp_data(str(ssp_path))
    print(f"PASS: SSP loaded from {ssp_path}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== SECTION 3: Create observation ===")
try:
    FILTERS = [
        "galex_fuv",
        "galex_nuv",
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "2mass_j",
        "2mass_h",
        "2mass_ks",
        "wise_w1",
        "wise_w2",
    ]
    obs = Observation(photometry=Photometry.from_names(FILTERS))
    print("PASS: Observation created with 12 filters")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== SECTION 4: Build model ===")
try:
    sed_model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="calzetti",
            tau_bc=Uniform(0.0, 1.0),
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    forward = ForwardModel.build(sed=sed_model, observation=obs)
    print("PASS: Model and forward model built")
    print("  Summary:")
    print(sed_model.summary())
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== SECTION 5: Generate mock ===")
try:
    key = jax.random.PRNGKey(9)
    key_truth, key_mock, key_fit = jax.random.split(key, 3)

    truth = sed_model.spec.sample(key_truth)
    mock = generate_mock(sed_model, truth, key=key_mock, snr=30.0)
    flux_obs = np.asarray(mock["flux_obs"])
    noise = np.asarray(mock["noise"])
    print(f"PASS: Mock generated with {len(flux_obs)} filters")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== SECTION 6: JIT compilation test ===")
try:
    import time
    p0 = {**sed_model.spec.get_fixed_values(), **truth}
    predict_phot = jax.jit(sed_model.predict_photometry)
    grad_fn = jax.jit(
        jax.grad(lambda p: 0.5 * jnp.sum(((sed_model.predict_photometry(p) - flux_obs) / noise) ** 2))
    )

    t = time.perf_counter()
    _ = predict_phot(p0).block_until_ready()
    compile_time = time.perf_counter() - t
    print(f"PASS: Forward model JIT compiled in {compile_time:.4f} s")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== SECTION 7: Create fitter ===")
try:
    from tengri.inference.fitter import Fitter
    fitter = Fitter(sed_model, flux_obs, noise, data_type="photometry")
    print("PASS: Fitter created (ready for MAP/NUTS, skipping sampling)")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== All early notebook sections passed ===")

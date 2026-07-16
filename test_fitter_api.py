#!/usr/bin/env python
"""Test the two different Fitter signatures shown in docs."""
import os
import sys
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import jax
import numpy as np
from pathlib import Path
from tengri import (
    FIXED,
    FREE,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    WavePrecomp,
    builders,
    generate_mock,
    load_ssp_data,
)
from tengri.inference.fitter import Fitter

# Load SSP
print("\n=== Setup: Load SSP and create model ===")
try:
    SSP_NAME = "fsps_prsc_miles_chabrier"
    ssp_path = Path("../data") / f"{SSP_NAME}.h5"
    if not ssp_path.exists():
        import tengri
        ssp_path = Path(tengri.download_ssp(SSP_NAME))
    ssp = load_ssp_data(str(ssp_path))

    FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    obs = Observation(photometry=Photometry.from_names(FILTERS))

    sed_model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust=builders.dust.two_component(defaults=FIXED),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    forward = ForwardModel.build(sed=sed_model, observation=obs)

    key = jax.random.PRNGKey(9)
    key_truth, key_mock = jax.random.split(key, 2)
    truth = sed_model.spec.sample(key_truth)
    mock = generate_mock(sed_model, truth, key=key_mock, snr=30.0)
    flux_obs = np.asarray(mock["flux_obs"])
    noise = np.asarray(mock["noise"])

    print("PASS: Setup complete")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test Signature 1: Fitter(forward, flux, noise)
print("\n=== TEST 1: Fitter(forward, flux_obs, noise) ===")
try:
    fitter1 = Fitter(forward, flux_obs, noise)
    print("PASS: Signature works")
except TypeError as e:
    print(f"FAIL: TypeError: {e}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

# Test Signature 2: Fitter(sed_model, flux, noise, data_type="photometry")
print("\n=== TEST 2: Fitter(sed_model, flux_obs, noise, data_type='photometry') ===")
try:
    fitter2 = Fitter(sed_model, flux_obs, noise, data_type="photometry")
    print("PASS: Signature works")
except TypeError as e:
    print(f"FAIL: TypeError: {e}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

# Check docstring
print("\n=== CHECK: Fitter.__init__ signature ===")
try:
    import inspect
    sig = inspect.signature(Fitter.__init__)
    print(f"Fitter.__init__ signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

# Check what the README actually recommends vs what's implemented
print("\n=== README vs Notebook ===")
print("README (line 143): Fitter(forward, mock.flux_obs, mock.noise)")
print("Notebook (line 213): Fitter(sed_model, flux_obs, noise, data_type='photometry')")
print("These are DIFFERENT signatures - which should a beginner use?")

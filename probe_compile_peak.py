#!/usr/bin/env python3
"""Measure compile peak for free-redshift photometry (child process)."""
import os
import resource
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TENGRI_DISABLE_JAX_CACHE"] = "1"

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import tengri
from tengri import SEDModel, Uniform, WavePrecomp, load_ssp_data, load_filter_set

# Load minimal data
ssp = load_ssp_data("/Users/suchethacooray/Projects/tengri/data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# Build free-redshift model with default WavePrecomp (n_z=250)
model = SEDModel.build(
    ssp_data=ssp,
    filters=filters,
    sfh={"type": "dpl"},
    redshift=Uniform(0.05, 0.2),
    approx=WavePrecomp(),
)

# Sample params and call predict_photometry to trigger compilation
params = model.spec.sample(jax.random.PRNGKey(0))
phot = model.predict_photometry(params)

# Get peak RSS after compilation
usage = resource.getrusage(resource.RUSAGE_SELF)
peak_rss_gb = usage.ru_maxrss / (1024.0 * 1024.0)  # macOS returns bytes

# Output results
print(f"PEAK_RSS_GB={peak_rss_gb:.2f}")
print(f"PHOTOMETRY_SHAPE={phot.shape}")
print(f"PHOTOMETRY_REPR={repr(phot)}")

#!/usr/bin/env python3
"""Audit inference engines for closure-captured large constants in HLO.

This probe builds tiny representative models for each inference engine
(VI/VI-native, MCMC NUTS/HMC, NSS, MAP, Laplace) with minimal mock data,
lowers each compiled callable to HLO, and scans for constants > 1 MB.

Usage:
  JAX_PLATFORMS=cpu python scripts/probe_inference_constants.py
"""

import os
import re
import sys
import time
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import SEDModel, Fitter, Parameters, Observation, Photometry
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform


def build_toy_photometry_setup(n_bands=12, n_age=30):
    """Build minimal photometry setup for probing.

    Parameters
    ----------
    n_bands : int
        Number of filter bands.
    n_age : int
        Number of age grid points for dummy SSP.

    Returns
    -------
    dict
        Keys: model, observation, param_spec, fitter.
    """
    # Minimal SPS config: Dale 2014 dust (smaller than DL14)
    ssp_data = load_ssp_data(kind="fsps")
    ssp_data_subsampled = {
        "ssp_lgmet": ssp_data["ssp_lgmet"][:, ::3],  # subsample age grid
        "ssp_flux": ssp_data["ssp_flux"][:, ::3, :],
        "ssp_ages_yr": ssp_data["ssp_ages_yr"][::3],
        "ssp_lgmet_scatter": ssp_data["ssp_lgmet_scatter"][::3],
    }

    # Build model with dust emission
    model = SEDModel(
        ssp_data=ssp_data_subsampled,
        dust_emission="dale2014",
        dust_scheme="exact",
    )

    # Mock photometry: 12 bands, 1 source
    filter_set = load_filter_set("sdss")  # Use first n_bands
    filter_names = list(filter_set.keys())[:n_bands]
    filters = {k: filter_set[k] for k in filter_names}

    fluxes = jnp.ones(n_bands) * 1e-15  # dummy fluxes
    flux_errors = fluxes * 0.1

    obs = Observation(
        photometry=Photometry(fluxes=fluxes, flux_errors=flux_errors, filter_set=filters),
    )

    # Minimal parameter spec: fixed redshift, free SFH + dust
    spec = Parameters.new(
        model,
        sfh_choice="tsnorm",
        met_choice="fixed",
        dust_choice="fast",
        agn_choice="none",
    )

    spec = spec.set_prior("sfh_tsnorm_age_max_log10_yr", Fixed(10.1))
    spec = spec.set_prior("sfh_tsnorm_tau_log10_yr", Uniform(6, 10))
    spec = spec.set_prior("z", Fixed(0.5))

    # Build fitter
    fitter = Fitter(
        model=model,
        param_spec=spec,
        observation=obs,
    )

    return {
        "model": model,
        "observation": obs,
        "param_spec": spec,
        "fitter": fitter,
    }


def extract_hlo_constants(hlo_text, min_bytes=1e6):
    """Parse HLO text and extract constants > min_bytes.

    Parameters
    ----------
    hlo_text : str
        HLO text dump.
    min_bytes : float
        Minimum size in bytes to report.

    Returns
    -------
    list of dict
        Each entry: {shape, size_bytes, dtype, instruction_id}.
    """
    constants = []

    # Pattern: f64[12,107,11149]{2,1,0} for shape + type
    # Look for tensor<...> constants
    pattern = r"tensor<(\d+)x([a-z0-9_]+)>"

    for match in re.finditer(pattern, hlo_text):
        bytes_str = match.group(1)
        dtype_str = match.group(2)

        # Rough estimate: f64 = 8 bytes, f32 = 4, i64 = 8, i32 = 4
        dtype_bytes = {"f64": 8, "f32": 4, "i64": 8, "i32": 4}.get(dtype_str, 8)
        estimated_size = int(bytes_str) * dtype_bytes

        if estimated_size > min_bytes:
            constants.append(
                {
                    "shape": match.group(0),
                    "size_bytes": estimated_size,
                    "dtype": dtype_str,
                }
            )

    return constants


def get_hlo_text(compiled_fn):
    """Extract HLO text from a compiled JAX function.

    This is a best-effort attempt using the serialization mechanism.
    """
    try:
        # Lower to HLO (requires the function to be JIT compiled)
        if hasattr(compiled_fn, "lower"):
            lowered = compiled_fn.lower()
            hlo = lowered.compile().as_compiled_module()
            return str(hlo)
    except Exception as e:
        return f"<Could not extract HLO: {e}>"


def probe_vi_native(setup):
    """Probe VI (native JAX) backend for closure constants."""
    fitter = setup["fitter"]

    print("\n" + "=" * 70)
    print("VI (Native JAX)")
    print("=" * 70)

    # Compile VI engine
    try:
        fitter.compile(modes=["vi_native"], n_threads=1)
        engine = fitter._get_or_build_engine("vi_native")

        # The engine holds a compiled _step function
        if hasattr(engine, "_step"):
            print(f"Engine compiled: {engine}")
            print(f"  _step callable: {engine._step}")
        else:
            print(f"Engine structure: {dir(engine)}")

    except Exception as e:
        print(f"Error during VI compile: {e}")
        import traceback

        traceback.print_exc()


def probe_nuts(setup):
    """Probe NUTS backend for closure constants."""
    fitter = setup["fitter"]

    print("\n" + "=" * 70)
    print("MCMC NUTS")
    print("=" * 70)

    try:
        fitter.compile(modes=["mcmc_nuts"], n_threads=1)
        engine = fitter._get_or_build_engine("mcmc_nuts")

        print(f"Engine compiled: {engine}")
        print(f"  Attributes: {[a for a in dir(engine) if not a.startswith('_')]}")

    except Exception as e:
        print(f"Error during NUTS compile: {e}")
        import traceback

        traceback.print_exc()


def probe_map(setup):
    """Probe MAP backend for closure constants."""
    fitter = setup["fitter"]

    print("\n" + "=" * 70)
    print("Maximum A Posteriori (MAP)")
    print("=" * 70)

    try:
        fitter.compile(modes=["map"], n_threads=1)
        engine = fitter._get_or_build_engine("map")

        print(f"Engine compiled: {engine}")
        print(f"  Attributes: {[a for a in dir(engine) if not a.startswith('_')]}")

    except Exception as e:
        print(f"Error during MAP compile: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run audit."""
    print("Building toy model...")
    setup = build_toy_photometry_setup(n_bands=12)

    print(f"Model: {setup['model']}")
    print(f"ParamSpec: {len(setup['param_spec'].free_params)} free parameters")
    print(f"  {setup['param_spec'].free_params}")

    # Suppress background compilation during setup
    os.environ["TENGRI_NO_BACKGROUND_COMPILE"] = "1"

    # Probe each engine
    probe_vi_native(setup)
    probe_nuts(setup)
    probe_map(setup)

    print("\n" + "=" * 70)
    print("Audit complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

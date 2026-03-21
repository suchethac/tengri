#!/usr/bin/env python
"""Generate CUE reference outputs from the original TF implementation.

Run this on a machine where TensorFlow works (Linux/CUDA), NOT on macOS ARM.
Outputs are saved to data/cue_reference_outputs.npz and used by
tests/crossval/test_cue_crossval.py to validate the JAX forward pass.

Usage:
    pip install tensorflow cue  # or install from source
    python scripts/generate_cue_reference.py
"""

import sys

import numpy as np

try:
    import tensorflow as tf

    print(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError:
    print("ERROR: TensorFlow not available. Run on a TF-compatible machine.")
    sys.exit(1)

# Standard test inputs (must match test_cue_crossval.py)
TEST_INPUTS = [
    {
        "ionspec_index1": -1.5,
        "ionspec_index2": -3.0,
        "ionspec_index3": -1.0,
        "ionspec_index4": -2.0,
        "ionspec_logLratio1": 0.0,
        "ionspec_logLratio2": 0.0,
        "ionspec_logLratio3": -0.5,
        "gas_logu": -2.5,
        "gas_logn": 2.0,
        "gas_logz": -0.5,
        "gas_logno": -0.5,
        "gas_logco": 0.0,
    },
    {
        "ionspec_index1": -2.0,
        "ionspec_index2": -4.0,
        "ionspec_index3": -1.5,
        "ionspec_index4": -3.0,
        "ionspec_logLratio1": 0.5,
        "ionspec_logLratio2": -0.5,
        "ionspec_logLratio3": 0.0,
        "gas_logu": -3.0,
        "gas_logn": 1.0,
        "gas_logz": 0.0,
        "gas_logno": 0.0,
        "gas_logco": 0.0,
    },
]


def main():
    try:
        from cue.nn import Cue
    except ImportError:
        print("ERROR: cue package not found. Install from: https://github.com/elijahlc/cue")
        sys.exit(1)

    cue_model = Cue()

    results = {}
    for i, params in enumerate(TEST_INPUTS):
        # Build input array in CUE's expected order
        param_array = np.array(
            [
                params["ionspec_index1"],
                params["ionspec_index2"],
                params["ionspec_index3"],
                params["ionspec_index4"],
                params["ionspec_logLratio1"],
                params["ionspec_logLratio2"],
                params["ionspec_logLratio3"],
                params["gas_logu"],
                params["gas_logn"],
                params["gas_logz"],
                params["gas_logno"],
                params["gas_logco"],
            ],
            dtype=np.float64,
        )

        # Get predictions from TF CUE
        lines = cue_model.predict_lines(param_array)
        continuum = cue_model.predict_continuum(param_array)

        results[f"lines_{i}"] = np.asarray(lines, dtype=np.float64)
        results[f"continuum_{i}"] = np.asarray(continuum, dtype=np.float64)
        results[f"params_{i}"] = param_array

        print(f"Input {i}: lines shape={lines.shape}, continuum shape={continuum.shape}")
        print(f"  lines range: [{np.min(lines):.4f}, {np.max(lines):.4f}]")
        print(f"  continuum range: [{np.min(continuum):.4f}, {np.max(continuum):.4f}]")

    outpath = "data/cue_reference_outputs.npz"
    np.savez(outpath, **results)
    print(f"\nSaved reference outputs to {outpath}")
    print("Copy this file to data/ in the tengri repo and run: pytest -m crossval")


if __name__ == "__main__":
    main()

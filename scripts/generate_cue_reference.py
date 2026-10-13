#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Generate CUE reference outputs from the upstream TensorFlow Emulator.

Runs the upstream Cue emulator (``cue.Emulator``, Li et al. 2024) on the
standard test inputs and writes the arrays that ``tests/crossval/
test_cue_crossval.py`` compares the JAX forward pass against. The output
layout is ``data/cue_reference_outputs.npz``:

    lines_{i}, continuum_{i}   emulator output per test input i
    continuum_wavelengths      [Angstrom] grid the continuum is evaluated on
    line_ind                   indices of the 128 FSPS-matched lines
    line_wavelengths           [Angstrom] wavelengths of the 128 lines, in output order
    line_wavelengths_all       [Angstrom] wavelengths of all 138 network lines
    tf_time_lines_ms, tf_time_cont_ms   single-call wall times [ms]

The upstream package is TensorFlow-based and must be installed in the
environment that runs this script. It is not vendored here. Upstream
source: https://github.com/yi-jia-li/cue (MIT). Install with::

    pip install tensorflow dill scikit-learn numpy scipy
    git clone https://github.com/yi-jia-li/cue /tmp/cue_upstream
    pip install /tmp/cue_upstream

Usage
-----
    python scripts/generate_cue_reference.py [--output data/cue_reference_outputs.npz]

The script does not overwrite ``data/cue_reference_outputs.npz`` unless
``--output`` names that path. Regenerating the shipped file is a deliberate
step: review the diff before committing it.
"""

import argparse
import sys
import time

import numpy as np

CUE_REPO_URL = "https://github.com/yi-jia-li/cue"

#: Continuum evaluation grid [Angstrom]: 1000 points from 100 to 60000 Angstrom.
CONTINUUM_WAVELENGTHS = np.linspace(100.0, 60000.0, 1000)

#: ``gas_logqion`` passed to every call. Matches the upstream ``Emulator``
#: default and tengri's ``CueBackend`` default (``default_gas_logqion``).
GAS_LOGQION = 49.1

# Standard test inputs. Must match _TEST_PARAMS in tests/crossval/test_cue_crossval.py.
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


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        default="data/cue_reference_outputs.npz",
        help="Path of the .npz file to write (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _import_emulator():
    """Import upstream's ``Emulator``, or exit with an install hint."""
    try:
        from cue import Emulator
    except ImportError as exc:
        print(f"ERROR: upstream cue package not importable ({exc}).")
        print(f"Install from source: {CUE_REPO_URL}")
        sys.exit(1)
    return Emulator


def _timed(fn, *args, **kwargs):
    """Return ``(result, elapsed_ms)`` for one call of ``fn``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, 1e3 * (time.perf_counter() - start)


def generate(emulator_cls):
    """Run the emulator on every test input and collect the reference arrays.

    Parameters
    ----------
    emulator_cls : type
        Upstream ``cue.Emulator`` class.

    Returns
    -------
    dict of str to ndarray
        The keys written to the output npz.
    """
    results = {}
    for i, params in enumerate(TEST_INPUTS):
        emu = emulator_cls(**params, gas_logqion=GAS_LOGQION)
        lines, t_lines_ms = _timed(emu.predict_lines)
        continuum, t_cont_ms = _timed(emu.predict_cont, CONTINUUM_WAVELENGTHS)

        if i == 0:
            # Upstream sorts the network lines by wavelength in predict_lines,
            # then keeps only emu.line_ind (128 of 138). output_line_wavelength
            # is that sorted order, set by predict_lines.
            results["line_ind"] = np.asarray(emu.line_ind, dtype=np.int64)
            results["line_wavelengths"] = np.asarray(
                emu.output_line_wavelength[emu.line_ind], dtype=np.float64
            )
            results["line_wavelengths_all"] = np.asarray(emu.line_wavelength, dtype=np.float64)
            results["continuum_wavelengths"] = CONTINUUM_WAVELENGTHS.copy()
            results["tf_time_lines_ms"] = np.float64(t_lines_ms)
            results["tf_time_cont_ms"] = np.float64(t_cont_ms)

        results[f"lines_{i}"] = np.asarray(lines, dtype=np.float64)
        results[f"continuum_{i}"] = np.asarray(continuum, dtype=np.float64)
        print(f"Input {i}: lines shape={lines.shape}, continuum shape={continuum.shape}")
        print(f"  lines range: [{np.min(lines):.4f}, {np.max(lines):.4f}]")
        print(f"  continuum range: [{np.min(continuum):.4f}, {np.max(continuum):.4f}]")
    return results


def main(argv=None):
    args = _parse_args(argv)
    emulator_cls = _import_emulator()
    results = generate(emulator_cls)
    np.savez(args.output, **results)
    print(f"\nSaved reference outputs to {args.output}")
    print("Copy this file to data/ in the tengri repo and run: pytest -m crossval")
    return 0


if __name__ == "__main__":
    sys.exit(main())

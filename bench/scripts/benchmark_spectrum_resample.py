# SPDX-License-Identifier: BSD-3-Clause
"""#1166: does a flux-conserving resample let the model run on a coarser grid?

For spectroscopy the forward-eval cost scales with the model wavelength grid
``n_wave`` (the CSP einsum is ``O(n_age * n_wave)``); unlike photometry there is
no ``n_wave -> n_filters`` collapse. The only lever is the model grid resolution
itself — evaluate the SED coarser and resample to the observed pixels — which is
safe only if the resample conserves flux.

This sweeps the model ``n_wave`` from an ultra-fine reference down toward the
observed pixel count and compares point interpolation (``compute_spectrum``)
against the bin-integral resample (``compute_spectrum_conserving``, #1166) on:
  * binned-flux error vs. the ultra-fine reference (relative RMS), and
  * per-eval JIT wall-clock.

The resulting curve is the #1166 acceptance test: it decides whether a coarse
model grid is accurate enough to be worth the speed, and therefore whether a
banded ``R_resample`` / SpectrumPrecomp conserving builder is worth building.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python bench/scripts/benchmark_spectrum_resample.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import tengri  # noqa: F401  (enables float64)
from tengri.observation.spectrum import compute_spectrum, compute_spectrum_conserving


def _continuum(wave):
    """Smooth rest-frame continuum only (no lines)."""
    return (wave / 5000.0) ** -1.5


def _model_sed(wave):
    """Continuum + three emission lines (FWHM ~ 1.5 Angstrom)."""
    lines = sum(np.exp(-0.5 * ((wave - c) / 1.5) ** 2) for c in (4861.0, 5007.0, 6564.6))
    return _continuum(wave) + 3.0 * lines


def _sweep(sed_fn, label):
    z, dl = 0.1, 1e27
    # Fixed observed pixel grid (a mid-resolution spectrograph, ~R 1000-2000).
    wave_obs = np.geomspace(4000.0, 7000.0, 1500)
    # Ultra-fine reference model grid — the ground truth both resamplers target.
    fine = np.geomspace(3000.0, 9000.0, 60000)
    ref = np.asarray(
        compute_spectrum_conserving(
            jnp.asarray(sed_fn(fine)), jnp.asarray(fine), jnp.asarray(wave_obs), z, dl
        )
    )
    ref_norm = float(np.sqrt(np.mean(ref**2)))
    print(f"\n[{label}]  observed pixels: {wave_obs.shape[0]}   reference grid: {fine.shape[0]}")
    hdr = f"{'n_wave':>8} {'point_relRMS':>13} {'consv_relRMS':>13}"
    print(f"{hdr} {'point_ms':>10} {'consv_ms':>10}")
    for n_wave in (60000, 20000, 8000, 4000, 2000, 1500):
        wm = np.geomspace(3000.0, 9000.0, n_wave)
        sed = jnp.asarray(sed_fn(wm))
        row = {}
        for fn, tag in ((compute_spectrum, "point"), (compute_spectrum_conserving, "consv")):
            out = np.asarray(fn(sed, jnp.asarray(wm), jnp.asarray(wave_obs), z, dl))
            row[f"{tag}_err"] = float(np.sqrt(np.mean((out - ref) ** 2)) / ref_norm)
            f = jax.jit(fn)
            f(sed, jnp.asarray(wm), jnp.asarray(wave_obs), z, dl).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(50):
                f(sed, jnp.asarray(wm), jnp.asarray(wave_obs), z, dl).block_until_ready()
            row[f"{tag}_ms"] = (time.perf_counter() - t0) / 50 * 1e3
        print(
            f"{n_wave:>8} {row['point_err']:>13.3e} {row['consv_err']:>13.3e} "
            f"{row['point_ms']:>10.3f} {row['consv_ms']:>10.3f}"
        )


def main():
    # Continuum-only isolates #1166's actual claim (flux conservation of the
    # smooth part); the line case shows what happens when the model grid can no
    # longer resolve the emission lines (tengri adds those analytically anyway).
    _sweep(_continuum, "continuum only")
    _sweep(_model_sed, "continuum + emission lines")


if __name__ == "__main__":
    main()

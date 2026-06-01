# SPDX-License-Identifier: BSD-3-Clause
r"""Port Synthesizer's stellar ``incident`` grid to DSPS-shaped HDF5.

Synthesizer stores SSP spectra in HDF5 grids as ``spectra['incident']`` with
shape ``(n_age, n_met, n_wave)`` in :math:`L_\nu` [erg/s/Hz] per :math:`M_\odot`
formed, on a rest-frame Angstrom grid. This script re-shapes and unit-converts
that grid into the layout :func:`tengri.load_ssp_data` reads
(``ssp_flux`` ``(n_met, n_age, n_wave)`` in :math:`L_\odot/\mathrm{Hz}/M_\odot`),
so the §1 SSP head-to-head has both codes reading *identical* templates — the
residual is then interpolation alone, not a different spectral library.

Unit conversion: divide :math:`L_\nu` [erg/s/Hz/M⊙] by
:data:`units.L_SUN_ERG_PER_S` to reach :math:`L_\odot/\mathrm{Hz}/M_\odot`. The
constant must match the one the notebook multiplies back by (it does — both use
``units.L_SUN_ERG_PER_S``), so the round-trip is exact.

The Synthesizer test grid spans an extreme 1e-4 Å – 3e11 Å. We clip to a
science SED range before writing so the DSPS interpolation and tengri's forward
pipeline stay well-conditioned (the clipped band is shared by both sides, so the
comparison is unaffected).

Run once::

    python -m reproduction.synthesizer._drivers.synthesizer_ssp_to_dsps
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import units as U

# Clip the ported grid to a science SED range [Å]. The native grid extends to
# soft-X-ray (1e-4 Å) and radio (3e11 Å); neither is used by the stellar panels
# and both stress the interpolators.
_WAVE_MIN_AA = 10.0
_WAVE_MAX_AA = 1.0e8

_STELLAR_GRID = "test_grid"


def _grid_dir() -> str:
    env = os.environ.get("SYNTHESIZER_GRID_DIR")
    if env:
        return env
    return os.path.expanduser("~/Library/Application Support/Synthesizer/grids")


def port_stellar_grid(out_path: str | Path, *, grid_name: str = _STELLAR_GRID) -> None:
    """Port the Synthesizer stellar ``incident`` grid to a DSPS-shaped HDF5.

    Parameters
    ----------
    out_path : str or Path
        Output HDF5 path.
    grid_name : str
        Synthesizer grid name (default the downloadable ``test_grid``).
    """
    import h5py
    from synthesizer import Grid

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = Grid(grid_name, grid_dir=_grid_dir())
    wave_aa = np.asarray(g.lam.to("angstrom").value, dtype=np.float64)
    ages_yr = np.asarray(g.ages.to("yr").value, dtype=np.float64)
    mets = np.asarray(g.metallicities, dtype=np.float64)
    # incident: (n_age, n_met, n_wave) in erg/s/Hz/Msun
    incident = np.asarray(g.spectra["incident"], dtype=np.float64)

    # Clip wavelength to the science range.
    keep = (wave_aa >= _WAVE_MIN_AA) & (wave_aa <= _WAVE_MAX_AA)
    wave_aa = wave_aa[keep]
    incident = incident[:, :, keep]

    # (n_age, n_met, n_wave) → (n_met, n_age, n_wave) and erg/s/Hz → Lsun/Hz.
    flux_dsps = np.transpose(incident, (1, 0, 2)) / U.L_SUN_ERG_PER_S

    lg_age_gyr = np.log10(ages_yr / 1e9).astype(np.float32)
    lgmet = np.log10(mets).astype(np.float32)  # absolute log10(Z), not Z/Zsun

    n_met, n_age, n_wave = flux_dsps.shape
    # The test grid carries no surviving-mass fraction; the incident spectrum is
    # already per M⊙ formed, so set surviving = formed (no mass-loss tracking).
    # This affects only the surviving-mass annotation, not the spectra.
    mass_remaining = np.ones((n_met, n_age), dtype=np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ssp_flux", data=flux_dsps.astype(np.float32), dtype=np.float32)
        f.create_dataset("ssp_lg_age_gyr", data=lg_age_gyr, dtype=np.float32)
        f.create_dataset("ssp_lgmet", data=lgmet, dtype=np.float32)
        f.create_dataset("ssp_wave", data=wave_aa.astype(np.float32), dtype=np.float32)
        f.create_dataset("ssp_mass_remaining", data=mass_remaining, dtype=np.float32)
        f.attrs["flux_units"] = "Lsun/Hz/Msun"
        f.attrs["wave_units"] = "Angstrom"
        f.attrs["n_met"] = n_met
        f.attrs["n_age"] = n_age
        f.attrs["n_wave"] = n_wave
        f.attrs["source"] = f"Synthesizer {grid_name} incident (ported)"

    print(f"✓ wrote {out_path}")
    print(f"  shape ({n_met}, {n_age}, {n_wave}) [met, age, wave]")
    print(f"  log10 Z: {lgmet}")
    print(f"  age range [Gyr]: {10.0 ** lg_age_gyr.min():.4e} – {10.0 ** lg_age_gyr.max():.4e}")
    print(f"  wave range [Å]: {wave_aa.min():.1f} – {wave_aa.max():.3e}")


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "synthesizer_test_grid.h5"
    port_stellar_grid(out)

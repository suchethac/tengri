"""Nebular emission models for tengri.

Provides modular backends for adding nebular emission (emission lines +
continuum) to the stellar SED. Three backends:

- **BakedIn** (default): Use SSP files with nebular emission pre-included
  (e.g., wNE files with fixed logU=-3, logZ=0). No free parameters.

- **CloudyGrid**: Precomputed CLOUDY photoionization grids with free
  ionization parameter (logU) and gas metallicity. Lines and continuum
  computed on-the-fly via interpolation.

- **Cue**: Neural net emulator (Li et al. 2025) with 12 free parameters
  including abundance ratios. Pure JAX, JIT-compatible, differentiable.

Usage::

    from tengri.models.nebular import CloudyGridBackend, BakedInBackend, CueBackend

    # Load CLOUDY grid
    backend = CloudyGridBackend("data/cloudy_grid_mist.h5", ssp_data)

    # Or use baked-in (default)
    backend = BakedInBackend()

    # Or use Cue neural net emulator
    backend = CueBackend("data/cue_weights.npz")
"""

from pathlib import Path

from tengri.models.nebular.baked_in import BakedInBackend
from tengri.models.nebular.cloudy_grid import CloudyGridBackend
from tengri.models.nebular.cue import CueBackend
from tengri.models.nebular.dig import mix_dig_emission
from tengri.models.nebular.shock import shock_emission_sed, shock_line_ratios

_DEFAULT_CUE_WEIGHTS_PATH = Path(__file__).resolve().parents[4] / "data" / "cue_weights.npz"

__all__ = [
    "_DEFAULT_CUE_WEIGHTS_PATH",
    "BakedInBackend",
    "CloudyGridBackend",
    "CueBackend",
    "mix_dig_emission",
    "shock_emission_sed",
    "shock_line_ratios",
]

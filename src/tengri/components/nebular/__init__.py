"""Nebular emission models for tengri.

Provides modular backends for adding nebular emission (emission lines +
continuum) to the stellar SED. Six backends:

- **BakedIn** (default): Use SSP files with nebular emission pre-included
  (e.g., wNE files with fixed logU=-3, logZ=0). No free parameters.

- **CloudyGrid**: Precomputed CLOUDY photoionization grids with free
  ionization parameter (logU) and gas metallicity. Lines and continuum
  computed on-the-fly via interpolation.

- **Cue**: Neural net emulator (Li et al. 2025) with 12 free parameters
  including abundance ratios. Pure JAX, JIT-compatible, differentiable.

- **CB19Grid**: Charlot & Bruzual 2019 CLOUDY grid from 3MdB_17 (Martinez-
  Paredes et al. 2023). 6D grid over log(O/H), logU, log_nH, log(C/O), ΔN/O,
  HbFrac. **Lines only — no nebular continuum.** Run
  ``scripts/download_cb19_templates.py`` to build the HDF5 file.

- **MappingsPhotoStellar**: MAPPINGS V v5.2.1 photoionization grids (Flury
  et al. 2024, arXiv:2412.06763; Zenodo 14140949). Stellar ionizing spectra
  from Starburst99 or BPASS v2.2, with Nicholls+2017 abundance patterns and
  Jenkins+2009 dust depletion. 4D grid over ζ_O, logU, age, logn. **Lines
  only — no nebular continuum.** Run ``scripts/build_flury2024_grids.py``
  to build the HDF5 file.

- **MappingsPhotoAGN**: MAPPINGS V v5.2.1 AGN grids (Flury et al. 2024).
  Ionizing source is the OPTXAGNF accretion-disc SED. 5D grid over ζ_O,
  logU, log(M_BH), log(λ_Edd), logn. Normalized by ionizing luminosity
  L_ion rather than Q_H. **Lines only — no nebular continuum.**

Usage::

    from tengri.components.nebular import (
        CloudyGridBackend,
        BakedInBackend,
        CueBackend,
        CB19Backend,
        MappingsPhotoStellarBackend,
        MappingsPhotoAGNBackend,
    )

    # Load CLOUDY grid
    backend = CloudyGridBackend("data/cloudy_grid_mist.h5", ssp_data)

    # Or use baked-in (default)
    backend = BakedInBackend()

    # Or use Cue neural net emulator
    backend = CueBackend("data/cue_weights.npz")

    # Or use CB_19 grid (wide abundance coverage, lines only)
    backend = CB19Backend()  # requires data/cb19_templates.h5

    # Or use MAPPINGS V stellar grid (Flury+2024, Nicholls+2017 abundances)
    backend = MappingsPhotoStellarBackend("data/flury2024_grids.h5", model="bpass", density="cpr")

    # Or use MAPPINGS V AGN grid (OPTXAGNF disc, lines only)
    backend = MappingsPhotoAGNBackend("data/flury2024_grids.h5", density="cpr")
"""

from pathlib import Path

from tengri.components.nebular._protocol import NebularBackend, NebularContinuumUnavailableError
from tengri.components.nebular._shared import NebularContinuumFallback
from tengri.components.nebular.agn_nebular import (
    FeltreGridData,
    FeltreNLRBackend,
    SynthesizerGridData,
    SynthesizerNLRBackend,
)
from tengri.components.nebular.baked_in import BakedInBackend, BakedInNebularWarning
from tengri.components.nebular.cloudy23_inputs import (
    Cloudy23Deck,
    build_cloudy23_deck,
)
from tengri.components.nebular.cloudy_cb19 import (
    CB19Backend,
    CB19IonizingSpectrumWarning,
    CB19NoContinuumWarning,
)
from tengri.components.nebular.cloudy_grid import (
    CloudyGridBackend,
    CloudyGridIonizingSpectrumWarning,
    CloudyGridWNESSPWarning,
)
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.cue_model import (
    CueNebularSEDComponent,
    CueNebularSEDComponentConfig,
)
from tengri.components.nebular.dig import mix_dig_emission
from tengri.components.nebular.mappings_photo import (
    IonizingSpectrumInconsistencyError,
    IonizingSpectrumInconsistencyWarning,
    MappingsPhotoAGNBackend,
    MappingsPhotoStellarBackend,
)
from tengri.components.nebular.shock import (
    ShockBackend,
    compute_shock_sed,
    shock_line_ratios,
)

_DEFAULT_CUE_WEIGHTS_PATH = Path(__file__).resolve().parents[4] / "data" / "cue_weights.npz"

__all__ = [
    "_DEFAULT_CUE_WEIGHTS_PATH",
    "BakedInBackend",
    "BakedInNebularWarning",
    "CB19Backend",
    "CB19IonizingSpectrumWarning",
    "CB19NoContinuumWarning",
    "Cloudy23Deck",
    "CloudyGridBackend",
    "CloudyGridIonizingSpectrumWarning",
    "CloudyGridWNESSPWarning",
    "CueBackend",
    "CueNebularSEDComponent",
    "CueNebularSEDComponentConfig",
    "FeltreGridData",
    "FeltreNLRBackend",
    "IonizingSpectrumInconsistencyError",
    "IonizingSpectrumInconsistencyWarning",
    "MappingsPhotoAGNBackend",
    "MappingsPhotoStellarBackend",
    "NebularBackend",
    "NebularContinuumFallback",
    "NebularContinuumUnavailableError",
    "ShockBackend",
    "SynthesizerGridData",
    "SynthesizerNLRBackend",
    "build_cloudy23_deck",
    "compute_shock_sed",
    "mix_dig_emission",
    "shock_line_ratios",
]


def __dir__() -> list[str]:
    """Curated tab-completion list — only the names in ``__all__``."""
    return list(__all__)


# Convenience re-exports for `from tengri.nebular import ...`

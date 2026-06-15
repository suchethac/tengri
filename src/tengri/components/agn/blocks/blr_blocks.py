# SPDX-License-Identifier: BSD-3-Clause
"""Re-export BLR blocks from per-model modules.

This module imports and re-exports individual BLR block implementations
(analytic, synthesizer, synthesizer_spectra) from their respective per-model
modules. Importing this module side-effects the registration of all BLR blocks.
"""

from __future__ import annotations

from tengri.components.agn.blocks._blr_common import (
    _resolve_synthesizer_grid as _resolve_synthesizer_grid,
)
from tengri.components.agn.blocks.blr_analytic import (
    blr_analytic_block as blr_analytic_block,
)
from tengri.components.agn.blocks.blr_synthesizer import (
    blr_synthesizer_block as blr_synthesizer_block,
)
from tengri.components.agn.blocks.blr_synthesizer_spectra import (
    blr_synthesizer_spectra_block as blr_synthesizer_spectra_block,
)

__all__ = [
    "blr_analytic_block",
    "blr_synthesizer_block",
    "blr_synthesizer_spectra_block",
]

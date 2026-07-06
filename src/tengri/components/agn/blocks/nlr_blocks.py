# SPDX-License-Identifier: BSD-3-Clause
"""Re-export NLR blocks from per-model modules.

This module imports and re-exports individual NLR block implementations
(analytic, synthesizer, synthesizer_spectra) from their respective per-model
modules. Importing this module side-effects the registration of all NLR blocks.
"""

from __future__ import annotations

from tengri.components.agn.blocks._nlr_common import (
    _resolve_synthesizer_grid as _resolve_synthesizer_grid,
)
from tengri.components.agn.blocks.nlr_analytic import (
    nlr_analytic_block as nlr_analytic_block,
)
from tengri.components.agn.blocks.nlr_feltre import (
    nlr_feltre_block as nlr_feltre_block,
)
from tengri.components.agn.blocks.nlr_synthesizer import (
    nlr_synthesizer_block as nlr_synthesizer_block,
)
from tengri.components.agn.blocks.nlr_synthesizer_spectra import (
    nlr_synthesizer_spectra_block as nlr_synthesizer_spectra_block,
)

__all__ = [
    "nlr_analytic_block",
    "nlr_feltre_block",
    "nlr_synthesizer_block",
    "nlr_synthesizer_spectra_block",
]

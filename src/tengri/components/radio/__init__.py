# SPDX-License-Identifier: BSD-3-Clause
"""Radio emission: star-formation synchrotron + AGN jets + free-free."""

from tengri.components.radio._models import (
    RADIO_MODELS,
    RadioRegistryEntry,
    register_radio_model,
)
from tengri.components.radio.radio import (
    compute_radio_components,
    radio_agn,
    radio_agn_dpl,
    radio_freefree,
    radio_sfr_bell2003,
    radio_sfr_delvecchio2021,
    radio_sfr_mccheyne2022,
    radio_star_forming,
    radio_total,
    radio_total_dpl,
)
from tengri.components.radio.radio_model import (
    RadioPowerLawSEDComponent,
    RadioPowerLawSEDComponentConfig,
)

# Populate the runtime registry. ``_VALID_RADIO_TYPES`` in
# ``parameters/groups.py`` derives from ``RADIO_MODELS.keys()``.
register_radio_model(
    "none",
    short_doc="Disable radio emission",
)(None)
register_radio_model(
    "condon92",
    citation="Condon 1992 (ARA&A 30, 575) + Yang+2020 (MNRAS 491, 740)",
    short_doc="FIR-radio correlation (q_IR) + AGN radio loudness power-law",
)(radio_total)

__all__ = [
    "RADIO_MODELS",
    "RadioPowerLawSEDComponent",
    "RadioPowerLawSEDComponentConfig",
    "RadioRegistryEntry",
    "compute_radio_components",
    "radio_agn",
    "radio_agn_dpl",
    "radio_freefree",
    "radio_sfr_bell2003",
    "radio_sfr_delvecchio2021",
    "radio_sfr_mccheyne2022",
    "radio_star_forming",
    "radio_total",
    "radio_total_dpl",
    "register_radio_model",
]

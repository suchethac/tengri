"""Free-parameter declarations owned by the radio component.

Single source of truth for radio priors. ``tengri.parameters._param_defs``
derives its legacy ``_RADIO_PARAMS`` bucket dict from this tuple, and
:meth:`RadioSEDComponent.declared_parameters` returns it directly.
Drift between the two paths is structurally impossible because they
share the same in-memory list.

When the JP / KP / Tribble physical synchrotron-aging kernels land
(Harwood+2013), their two free parameters (``radio_alpha_inj``,
``radio_log_nu_break``) get added here alongside the model names in
:data:`tengri.components.radio.component.AGN_RADIO_MODELS`. Until then,
selecting those model names raises :class:`ValueError` at construction.
"""

from __future__ import annotations

from tengri.protocols.component import ParamDeclaration
from tengri.parameters.priors import Fixed

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "radio_q_ir",
        Fixed(2.64),
        "FIR-radio correlation q_IR (Bell 2003: 2.64, evolves with z)",
    ),
    ParamDeclaration(
        "radio_alpha_sf",
        Fixed(0.8),
        "SF synchrotron spectral index (typical 0.7-0.8)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "radio_loudness",
        Fixed(0.0),
        "AGN radio-loudness log10(L_5GHz/L_B) (>1 = radio-loud)",
    ),
    ParamDeclaration(
        "radio_alpha_agn",
        Fixed(0.7),
        "AGN radio spectral index (typical 0.7)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "radio_T_e",
        Fixed(1e4),
        "Electron temperature [K] for thermal free-free emission (typical 1e4)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "radio_alpha_ff",
        Fixed(-0.1),
        "Thermal free-free spectral index (typical -0.1)",
    ),
    # AGNfitter-rx double power-law AGN radio model parameters
    # (Martinez-Ramirez+2024 Eq. 9-10). Activated by
    # ``RadioSEDComponentConfig.agn_radio_model="dpl"``; ignored otherwise.
    ParamDeclaration(
        "radio_alpha_thin",
        Fixed(-0.75),
        "AGN-DPL optically-thin (steep) spectral slope (typical -0.75)",
    ),
    ParamDeclaration(
        "radio_alpha_thick",
        Fixed(-0.1),
        "AGN-DPL optically-thick (flat/inverted) spectral slope (typical -0.1)",
    ),
    ParamDeclaration(
        "radio_log_nu_t",
        Fixed(10.0),
        "AGN-DPL log10(transition frequency / Hz); typical 9-11",
    ),
    ParamDeclaration(
        "radio_log_nu_cut",
        Fixed(13.0),
        "AGN-DPL log10(synchrotron aging exponential cutoff / Hz); typical 12-14",
    ),
)

__all__ = ["PARAMS"]

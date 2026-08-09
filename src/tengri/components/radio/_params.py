# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the radio component.

Single source of truth for radio priors. ``tengri.parameters._builders``
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

from tengri.parameters.priors import Fixed
from tengri.protocols.component import ParamDeclaration


class RadioFIRRCDegeneracyWarning(UserWarning):
    """A FIRRC *slope* coefficient was freed in a single-galaxy fit.

    The FIR-radio correlation slopes vary q_IR(M*, z) *across* a sample;
    at one galaxy's fixed (M*, z) they collapse to a single scalar and are
    degenerate with the normalization (``radio_*_q0`` / ``radio_q_ir``).
    They are identifiable only as ``PopulationFitter`` hyperparameters.
    Filter this category to silence the notice for a deliberate hierarchical
    fit.
    """


#: The FIRRC slope coefficients whose freedom is degenerate per-galaxy
#: (the q0 normalizations are fine — they *are* the radio-excess knob).
FIRRC_SLOPE_PARAMS: frozenset[str] = frozenset(
    {
        "radio_delv_mass_slope",
        "radio_delv_z_slope",
        "radio_mcch_mass_slope",
        "radio_mcch_z_slope",
    }
)

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
        units="K",
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
    # ── FIR-radio correlation (FIRRC) evolution coefficients ──────────────
    # Mass- and redshift-dependent q_IR(M*, z) for the evolving SF-radio
    # models. Surfaced as free parameters so they can be fitted directly or
    # promoted to PopulationFitter hyperparameters (the SF functions in
    # radio.py already accept them as q0/mass_slope/z_slope overrides).
    #
    # The names are MODEL-SPECIFIC because the two calibrations carry
    # genuinely different literature defaults *and* a different mass-slope
    # sign convention (Delvecchio subtracts the mass term, McCheyne adds it),
    # so a single shared name would silently apply the wrong default to the
    # inactive model. Only the active ``sfr_mode``'s triplet is consumed; the
    # other three stay Fixed no-ops (mirrors the DPL-param pattern above).
    #
    # Delvecchio+2021 (1.4 GHz; SEMPER Eq. 4):
    #   q(M*, z) = q0 (1+z)^z_slope - (logM* - 10) * mass_slope
    ParamDeclaration(
        "radio_delv_q0",
        Fixed(2.743),
        "Delvecchio+2021 FIRRC normalization q0 at logM*=10, z=0 (1.4 GHz)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "radio_delv_mass_slope",
        Fixed(0.234),
        "Delvecchio+2021 FIRRC mass slope dq/dlogM* (subtracted; >0 = massive -> more radio)",
    ),
    ParamDeclaration(
        "radio_delv_z_slope",
        Fixed(-0.025),
        "Delvecchio+2021 FIRRC redshift exponent on (1+z) (slight decline with z)",
    ),
    # McCheyne+2022 (150 MHz; SEMPER Eq. 5):
    #   q(M*, z) = q0 (1+z)^z_slope + mass_slope * (logM* - 10)
    ParamDeclaration(
        "radio_mcch_q0",
        Fixed(1.98),
        "McCheyne+2022 FIRRC normalization q0 at logM*=10, z=0 (150 MHz)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "radio_mcch_mass_slope",
        Fixed(-0.22),
        "McCheyne+2022 FIRRC mass slope dq/dlogM* (added; <0 = massive -> more radio)",
    ),
    ParamDeclaration(
        "radio_mcch_z_slope",
        Fixed(0.02),
        "McCheyne+2022 FIRRC redshift exponent on (1+z) (near-zero evolution)",
    ),
)

__all__ = ["FIRRC_SLOPE_PARAMS", "PARAMS", "RadioFIRRCDegeneracyWarning"]

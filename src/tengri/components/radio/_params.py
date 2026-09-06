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

from tengri.parameters.priors import Fixed, Uniform
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
#: (the q0 normalizations are fine: they *are* the radio-excess knob).
#:
#: None of these four carries a ``free_prior``, and that is deliberate (#887).
#: They have perfectly good ranges, but ``all_params: FREE`` is a single-galaxy
#: gesture and at one galaxy's fixed (M*, z) a slope collapses to a scalar that
#: is exactly degenerate with the normalization beside it, which is what
#: :class:`RadioFIRRCDegeneracyWarning` exists to say. Declaring them would make
#: the wildcard emit that warning by construction, i.e. ship a default whose own
#: guard objects to it. They are identifiable as ``PopulationFitter``
#: hyperparameters, where they should be set explicitly.
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
        # The FIRRC is tight -- under 0.3 dex scatter across five decades in
        # luminosity -- so the range is the Bell (2003) value carried to about
        # +/-3 sigma of that scatter. It is the radio-excess knob: a fitted q_IR
        # well below 2.64 is the signature of an AGN contribution.
        free_prior=Uniform(1.8, 3.5, "FIR-radio correlation q_IR", default=2.64),
    ),
    ParamDeclaration(
        "radio_alpha_sf",
        Fixed(0.8),
        "SF synchrotron spectral index (typical 0.7-0.8)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # Star-forming regions are optically thin to synchrotron, giving the
        # characteristic steep index alpha ~ 0.8 (this default). The range
        # spans flatter free-free-contaminated spectra up to the steepest
        # aged-electron populations observed in SF galaxies.
        free_prior=Uniform(0.5, 1.2, "SF synchrotron spectral index", default=0.8),
    ),
    ParamDeclaration(
        "radio_loudness",
        Fixed(0.0),
        "AGN radio-loudness log10(L_5GHz/L_B) (>1 = radio-loud)",
        # log10 of the Kellermann R = L_5GHz/L_B ratio, so the range covers
        # radio-quiet (R ~ 0.01) through the most radio-loud quasars
        # (R ~ 1e4). The R = 10 radio-loud/quiet divide sits at 1.0, inside.
        free_prior=Uniform(-2.0, 4.0, "AGN radio loudness", default=0.0),
    ),
    ParamDeclaration(
        "radio_alpha_agn",
        Fixed(0.7),
        "AGN radio spectral index (typical 0.7)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # Spans flat-spectrum cores (alpha ~ 0, self-absorbed) through steep
        # optically-thin lobes; 0.7 is the canonical mid-range default.
        free_prior=Uniform(0.0, 1.5, "AGN radio spectral index", default=0.7),
    ),
    ParamDeclaration(
        "radio_T_e",
        Fixed(1e4),
        "Electron temperature [K] for thermal free-free emission (typical 1e4)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
        # HII-region electron temperatures run from a few thousand K in
        # metal-rich, efficiently cooled nebulae to ~2e4 K in metal-poor ones;
        # 1e4 K is the canonical value this default takes.
        free_prior=Uniform(5.0e3, 2.0e4, "Free-free electron temperature", units="K", default=1e4),
    ),
    ParamDeclaration(
        "radio_alpha_ff",
        Fixed(-0.1),
        "Thermal free-free spectral index (typical -0.1)",
        # Deliberately NO free_prior. Unlike the synchrotron indices beside it
        # this is not an empirical knob: optically-thin thermal bremsstrahlung
        # has an analytic spectral index of -0.1, essentially independent of
        # density and only weakly dependent on temperature (which is already
        # exposed separately as ``radio_T_e``). Freeing it would let the fit
        # absorb calibration error into a quantity physics fixes. Override
        # explicitly if you specifically want that slack.
    ),
    # AGNfitter-rx double power-law AGN radio model parameters. Activated by
    # ``RadioSEDComponentConfig.agn_radio_model="dpl"``; ignored otherwise.
    # Martinez-Ramirez, L. N. et al. 2024, "AGNfitter-rx: Modelling the
    # radio-to-X-ray SEDs of AGNs," A&A, 688, A46,
    # doi:10.1051/0004-6361/202449329, arXiv:2405.12111 -- Table 1 gives
    # alpha1 (synchrotron aged / optically-thin) in [-1, 1] and alpha2
    # (self-absorbed / optically-thick) in [-1, 0]. The paper's Eq. (2) uses
    # the same L_nu ~ nu^alpha exponent convention as ``radio_agn_dpl`` below,
    # so no sign conversion is needed. The paper's 3-band fallback
    # alpha1 = -0.75 matches this file's existing default, and both existing
    # Fixed defaults (-0.75, -0.1) lie inside these intervals.
    ParamDeclaration(
        "radio_alpha_thin",
        Fixed(-0.75),
        "AGN-DPL optically-thin (steep) spectral slope (typical -0.75)",
        free_prior=Uniform(-1.0, 1.0, "AGN-DPL optically-thin spectral slope", default=-0.75),
    ),
    ParamDeclaration(
        "radio_alpha_thick",
        Fixed(-0.1),
        "AGN-DPL optically-thick (flat/inverted) spectral slope (typical -0.1)",
        free_prior=Uniform(-1.0, 0.0, "AGN-DPL optically-thick spectral slope", default=-0.1),
    ),
    ParamDeclaration(
        "radio_log_nu_t",
        Fixed(10.0),
        "AGN-DPL log10(transition frequency / Hz); typical 9-11",
        # The typical interval this description already states. Safe to declare
        # because the radio wildcard is scoped per model:
        # _RADIO_AGN_PARAMS_BY_MODEL only routes these four under
        # ``agn_radio_model="dpl"``, so a powerlaw fit never sees them.
        free_prior=Uniform(9.0, 11.0, "DPL transition frequency", units="log10(Hz)", default=10.0),
    ),
    ParamDeclaration(
        "radio_log_nu_cut",
        Fixed(13.0),
        "AGN-DPL log10(synchrotron aging exponential cutoff / Hz); typical 12-14",
        free_prior=Uniform(12.0, 14.0, "DPL aging cutoff", units="log10(Hz)", default=13.0),
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
        # Same reasoning as ``radio_q_ir``: a q0 normalization carried to about
        # +/-3 sigma of the FIRRC's sub-0.3-dex scatter. Safe under the wildcard
        # because _RADIO_SF_PARAMS_BY_MODE routes this triplet only under
        # ``radio_sfr_mode="delvecchio2021"`` -- the McCheyne triplet stays
        # pinned, and vice versa, so only one calibration is ever freed.
        free_prior=Uniform(1.8, 3.5, "Delvecchio+2021 FIRRC q0", default=2.743),
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
        # As above but centered on the 150 MHz normalization (1.98 rather than
        # 2.743); the low-frequency q0 sits lower because synchrotron dominates
        # further above the thermal component there.
        free_prior=Uniform(1.0, 3.0, "McCheyne+2022 FIRRC q0", default=1.98),
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

# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the X-ray component.

Single source of truth for the ``xray_*`` priors.
``tengri.parameters._param_defs`` derives its legacy ``_XRAY_PARAMS``
bucket from this tuple, and :meth:`XRaySEDComponent.declared_parameters`
returns it directly. Drift between the two paths is structurally
impossible because they share the same in-memory list.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "xray_gamma_agn",
        Fixed(1.8),
        "AGN X-ray photon index Gamma (typical 1.4-2.4)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # Following ``xray_log_nh`` below: free over the range this declaration
        # itself calls typical, not over the validator's ``> 0``, which admits
        # photon indices no corona produces.
        free_prior=Uniform(1.4, 2.4, "AGN X-ray photon index", default=1.8),
    ),
    ParamDeclaration(
        "xray_delta_alpha_ox",
        Fixed(0.0),
        "Offset [dex] applied to the L_2500-derived alpha_ox (Just+2007,"
        " CIGALE convention). 0 (default) = pure empirical alpha_ox(L_2500);"
        " negative hardens the X-ray corona, positive softens it.",
        # Deliberately NO free_prior. The sensible width of an offset on an
        # empirical relation is that relation's intrinsic scatter, and neither
        # this description nor the declaration records the Just+2007 scatter --
        # so any interval chosen here would be a guess dressed as a default.
        # It is a genuine knob; free it explicitly once you have a scatter to
        # justify, e.g. delta_alpha_ox=Uniform(-0.2, 0.2).
    ),
    # The two X-ray-binary photon indices deliberately get NO free_prior. They
    # are not per-galaxy quantities: Gamma = 2.0 (HMXB) and 1.6 (LMXB) are the
    # fixed spectral assumptions of the Lehmer+2016-style population scalings
    # these components implement, and each description states a single value
    # rather than a range because there is no per-object range to state. The
    # freedom a caller actually wants here -- how far this galaxy's XRB
    # luminosity departs from the scaling relation -- is already exposed, and
    # already free by default, as ``xray_det_hmxb`` / ``xray_det_lmxb`` below.
    # Freeing the index as well would let the fit trade normalization against
    # spectral slope with nothing to break the degeneracy.
    ParamDeclaration(
        "xray_gamma_hmxb",
        Fixed(2.0),
        "HMXB photon index (typical 2.0)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "xray_gamma_lmxb",
        Fixed(1.6),
        "LMXB photon index (typical 1.6)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "xray_E_cut",
        Fixed(300.0),
        "Exponential cutoff energy [keV] for AGN X-ray spectrum (typical 100-500)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="keV",
        # The typical interval this description states.
        free_prior=Uniform(100.0, 500.0, "AGN X-ray cutoff energy", units="keV", default=300.0),
    ),
    ParamDeclaration(
        "xray_log_nh",
        Fixed(20.0),
        "Line-of-sight equivalent hydrogen column density [log10(cm^-2)]. "
        "Typical range 20 (unobscured) to 24 (Compton-thick). Controls "
        "photoelectric absorption below ~2 keV (Morrison & McCammon 1983).",
        lambda lo, hi: 0 <= lo <= 26,
        "must be in [0, 26]",
        # The [0, 26] bound is what the absorption model tolerates, not a
        # sensible prior: it would put most of the mass on columns no galaxy
        # has. Free over the range this declaration itself calls typical —
        # 20 (unobscured) to 24 (Compton-thick).
        free_prior=Uniform(
            20.0, 24.0, "Line-of-sight hydrogen column", units="log10(cm^-2)", default=20.0
        ),
        units="log10(cm^-2)",
    ),
    ParamDeclaration(
        "xray_alpha_irx",
        Fixed(0.3),
        "alpha_IRX = log10(nu*Lnu(12um) / Lx(2-10 keV)) for the lopez24 corona "
        "(Asmus+2015 / Lopez+2024). Higher -> fainter X-ray. Ignored by yang20. "
        "Typical range 0.0-0.6.",
        # The typical interval this description states. Note it is read only by
        # the lopez24 corona -- yang20 ignores it -- so freeing it under a
        # wildcard is only meaningful with that corona selected.
        free_prior=Uniform(0.0, 0.6, "IR-to-X-ray luminosity ratio", units="dex", default=0.3),
    ),
    # X-ray binary offsets (xray_aird component, #1307)
    ParamDeclaration(
        "xray_det_hmxb",
        Uniform(
            -2.0,
            2.0,
            default=0.0,
        ),
        "Deviation from expected HMXB log L_X (Yang+2020 [1]_). "
        "Positive = brighter X-ray. Allows intrinsic scatter or evolution "
        "around the Lehmer+2016 SFR relation.",
        units="dex",
    ),
    ParamDeclaration(
        "xray_det_lmxb",
        Uniform(
            -2.0,
            2.0,
            default=0.0,
        ),
        "Deviation from expected LMXB log L_X (Yang+2020 [1]_). "
        "Positive = brighter X-ray. Allows intrinsic scatter or evolution "
        "around the Lehmer+2016 age/mass relation.",
        units="dex",
    ),
)

__all__ = ["PARAMS"]

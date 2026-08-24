# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP (Buchner+ 2024) AGN model; JAX implementation.

Implements the spectral model from Buchner et al. 2024 ("Genuine Retrieval
of the AGN Host Stellar Population", arXiv:2405.19297). Templates are sourced
verbatim from ``JohannesBuchner/GRAHSP`` (CeCILL-v2) and packaged in
``data/grahsp/grahsp_templates.h5``.

Public API
----------
:class:`GRAHSPParams`
    Parameter container.
:class:`GRAHSPSED`
    Output bundle.
:func:`evaluate_grahsp_agn`
    Compose BBB + lines + FeII + torus + Si + bi-attenuation on a user grid.

Building blocks (also public, for component-level testing)
----------------------------------------------------------
:func:`sbpl_bbb`
    Smooth bending power-law continuum (Ryde 1999).
:func:`gaussian_lines`, :func:`feii_forest`
    Emission line + FeII forest builders.
:func:`torus_dust_continuum`, :func:`si_feature`
    Mid-IR torus components.
:func:`smc_attenuation_curve`, :func:`attenuation_factors`
    SMC-like bi-attenuation (Prevot 1984 / Buchner+ 2024).
:func:`normalized_excess_variance`
    Pan-STARRS1 NEV (Simm+ 2016).
:func:`bolometric_luminosity_bbb`, :func:`bolometric_luminosity_torus`,
:func:`agn_fraction_dale`
    Bolometric quantities.
:func:`load_grahsp_templates`
    HDF5 template bundle loader.

References
----------
.. [1] Buchner, J., Starck, H., Salvato, M., et al. 2024,
       arXiv:2405.19297. The paper implemented here.
"""

from tengri.components.agn.grahsp.attenuation import (
    PREVOT_DEFAULTS,
    attenuation_factors,
    smc_attenuation_curve,
)
from tengri.components.agn.grahsp.bbb import LAMBDA_5100_NM, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import (
    LYMAN_LIMIT_NM,
    agn_fraction_dale,
    bolometric_luminosity_bbb,
    bolometric_luminosity_torus,
)
from tengri.components.agn.grahsp.component import (
    GRAHSPSEDComponent,
    GRAHSPSEDComponentConfig,
    GRAHSPSEDComponentState,
)
from tengri.components.agn.grahsp.lines import (
    AGN_TYPE_BL,
    AGN_TYPE_LINER,
    AGN_TYPE_SY2,
    feii_forest,
    gaussian_lines,
)
from tengri.components.agn.grahsp.model import (
    GRAHSPSED,
    GRAHSPParams,
    compute_grahsp_sed,
    evaluate_grahsp_agn,
)

# Importing ``registry`` side-effects ``AGN_MODELS["grahsp"]`` so the
# tengri unified AGN dispatch picks GRAHSP up alongside qsogen, skirtor,
# kubota_done, silva04, etc.
from tengri.components.agn.grahsp.registry import grahsp
from tengri.components.agn.grahsp.templates import (
    DEFAULT_TEMPLATE_PATH,
    GRAHSPTemplates,
    load_grahsp_templates,
)
from tengri.components.agn.grahsp.torus import (
    SI_DEFAULTS,
    si_feature,
    torus_dust_continuum,
)
from tengri.components.agn.grahsp.variability import normalized_excess_variance

__all__ = [
    "AGN_TYPE_BL",
    "AGN_TYPE_LINER",
    "AGN_TYPE_SY2",
    "DEFAULT_TEMPLATE_PATH",
    "GRAHSPSED",
    "LAMBDA_5100_NM",
    "LYMAN_LIMIT_NM",
    "PREVOT_DEFAULTS",
    "SI_DEFAULTS",
    "GRAHSPParams",
    "GRAHSPSEDComponent",
    "GRAHSPSEDComponentConfig",
    "GRAHSPSEDComponentState",
    "GRAHSPTemplates",
    "agn_fraction_dale",
    "attenuation_factors",
    "bolometric_luminosity_bbb",
    "bolometric_luminosity_torus",
    "compute_grahsp_sed",
    "evaluate_grahsp_agn",
    "feii_forest",
    "gaussian_lines",
    "grahsp",
    "load_grahsp_templates",
    "normalized_excess_variance",
    "sbpl_bbb",
    "si_feature",
    "smc_attenuation_curve",
    "torus_dust_continuum",
]
# GRAHSP parity (Balmer continuum, MN12 torus, Veron-Cetty FeII, Netzer disc): see PR #649

# SPDX-License-Identifier: BSD-3-Clause
"""Static associations between tengri model configuration and citation keys.

Module layout:

* ``CORE_CITATIONS``          always-applicable citations for any tengri run.
* ``DUST_LAW_CITATIONS``      map of dust-law name → citation keys.
* ``NEBULAR_BACKEND_CITATIONS`` map of nebular-backend name → citation keys.
* ``IGM_CITATIONS``           map of IGM-model name → citation keys.
* ``BACKEND_CITATIONS``       map of inference-backend name → citation keys.
* ``FUNCTION_CITATIONS``      optional map of ``"module.function"`` → citation
  keys, populated by the :func:`cites` decorator or by direct registration.

Downstream logic (:func:`tengri.collect_citations`) reads these tables to
assemble a per-Galaxy citation set. If you add a new dust law, nebular
backend, or inference path, add an entry here so the citation machinery
can surface the right paper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


# Citations that apply to every tengri run.
CORE_CITATIONS: list[str] = ["tengri", "jax", "dsps"]


#: Star-formation-history model name → citation keys.
#:
#: Every other subsystem had a table; SFH did not, so the SFH papers lived only
#: in the two hand-written name→key maps and neither ``collect_citations`` nor
#: ``print_components_bibtex`` cited ``delayed`` at all. Each entry below was
#: checked against ``references.bib`` on volume and page, not on author+year:
#: a fuzzy match wanted to send ``conroy2010`` (ApJ 708, 58) to ``Conroy_2010a``
#: (ApJ 712, 833), which is a different paper, so that one is deliberately absent.
SFH_CITATIONS: dict[str, list[str]] = {
    "dpl": ["bagpipes"],
    # Boquien+2019 (A&A 622, A103) for the CIGALE form, Carnall+2018
    # (MNRAS 480, 4379) for the Bagpipes one — the menu row names both.
    "delayed": ["cigale", "bagpipes"],
    "continuity": ["leja2019"],
    "continuity_flex": ["leja2019"],
    "dirichlet": ["leja2019"],
    "dense_basis": ["iyer2020"],
    # The SFH shape is Leja+2019; the Wang+2024 prior paper has no bundled
    # BibTeX entry yet, so only the half that exists is claimed.
    "prospector_beta": ["leja2019"],
    None: [],
}

#: Radio *model* name → citation keys.
#:
#: :data:`RADIO_CITATIONS` is a flat list of keys that apply whenever radio is
#: active; this maps the selectable model names on top of it. ``condon92``
#: names both papers in its own menu row.
RADIO_MODEL_CITATIONS: dict[str, list[str]] = {
    "condon92": ["condon1992", "yang2020"],
    "bell2003": ["bell2003"],
    "none": [],
    None: [],
}


# Dust attenuation laws. Keys match tengri.config.settings.DustConfig.law_*
# and DustConfig.model values.
DUST_LAW_CITATIONS: dict[str, list[str]] = {
    "calzetti": ["calzetti2000"],
    "power_law": [],
    "kriek_conroy": ["kriek_conroy2013"],
    "smc": ["gordon2003_smc"],
    "cardelli": ["cardelli1989"],
    "mw": ["cardelli1989"],
    "salim": ["salim2018"],  # Salim+2018 modified-Calzetti (ADS-verified)
    "reddy15": ["reddy2015"],  # Reddy+2015 MOSDEF high-z curve (R_V=2.505, ADS-verified)
    "li08": ["li2008_ext"],  # Li+2008 four-coefficient analytical curve (ADS-verified)
    "vw07_bc": ["witt_gordon2000"],  # tengri source notes Wild+2007; closest workspace match
    "vw07_diff": ["witt_gordon2000"],  # see above; update once Wild+2007 is in the workspace bib
    "cf00": ["charlot_fall2000"],
}

# Dust emission templates (DustConfig.emission).
DUST_EMISSION_CITATIONS: dict[str, list[str]] = {
    # Canonical modified-blackbody (Hildebrand 1983); da Cunha 2013 supplies
    # the CMB-heating correction applied automatically at redshift > 0.
    "modified_blackbody": ["hildebrand1983", "dacunha2013"],
    "mbb": ["casey2012"],
    "casey2012": ["casey2012"],
    "dale2014": ["dale2014"],
    "draine_li2007": ["draine_li2007"],
    "dl07": ["draine_li2007"],
    "draine_li2014": ["draine2014"],
    "dl14": ["draine2014"],
    "themis": ["jones2013", "jones2017"],
    "astrodust": ["hensley_draine2023"],
    "schreiber2016": ["schreiber2016"],
    "pah_drude": ["smith2007"],
    "bosa": ["cigale"],
    None: [],
}

# Additional citation(s) triggered by the dust *model* wrapper, independent
# of the per-component law. Two-component = Charlot & Fall 2000.
DUST_MODEL_CITATIONS: dict[str, list[str]] = {
    "two_component": ["charlot_fall2000"],
    "single_screen": [],
    # Witt & Gordon (2000) radiative-transfer screen (FSPS dust_type=3).
    "wg00": ["witt_gordon2000"],
}

# Nebular-emission backends. Keys match tengri.config.settings.NebularConfig.backend.
NEBULAR_BACKEND_CITATIONS: dict[str, list[str]] = {
    "cue": ["cue"],
    "cloudy": ["cloudy"],
    # The live backend objects report these ``.name`` values; map them to the
    # same underlying Cloudy photoionization-grid citations.
    "cloudy_grid": ["cloudy"],
    "cb19_grid": ["byler2017", "cloudy"],
    "mappings": ["mappings"],
    # Nebular emission baked into the SSP grid (FSPS ``wNE`` files) uses the
    # Byler+2017 Cloudy photoionization grids.
    "baked_in": ["byler2017", "cloudy"],
    "off": [],
    None: [],
}


# ── SSP provenance ────────────────────────────────────────────────────────
# The SSP grid filename encodes its provenance as
# ``<sps_code>_<isochrone>_<library>_<imf>`` (FSPS / ProGeny convention, e.g.
# ``fsps_prsc_miles_chabrier`` = FSPS + PARSEC isochrones + MILES library +
# Chabrier IMF). ``collect_citations`` splits the SSP ``source`` name on ``_``
# and maps each token through these tables; the IMF is read from ``SSPData.imf``.
# A token that matches none of these tables triggers a provenance warning.

# Initial mass function (SSPData.imf, or a filename token).
IMF_CITATIONS: dict[str, list[str]] = {
    "chabrier": ["chabrier2003"],
    "kroupa": ["kroupa2001"],
    "salpeter": ["salpeter1955"],
}

# SPS code that generated the grid (first filename token).
SSP_CODE_CITATIONS: dict[str, list[str]] = {
    # FSPS (Conroy, Gunn & White 2009 + Conroy & Gunn 2010) generated via the
    # python-fsps interface (Foreman-Mackey et al. 2014). Aringer+2009 (carbon-star
    # library extending TP-AGB stars redward of K) and Villaume+2015 (circumstellar
    # AGB dust, ``add_agb_dust_model`` — on by default) are baked into every
    # FSPS-generated grid, so they fire for any ``fsps_*`` source. See #560.
    "fsps": ["fsps2009", "fsps", "pythonfsps", "aringer2009", "villaume2015"],
    "bc03": ["bc03"],
    "bpss": ["bpass"],
    "bpass": ["bpass"],
    "pgny": ["progeny"],
    "progeny": ["progeny"],
}

# Stellar-evolution isochrone set (second filename token).
SSP_ISOCHRONE_CITATIONS: dict[str, list[str]] = {
    "prsc": ["parsec"],
    "parsec": ["parsec"],
    "mist": ["mist", "mist_dotter2016"],  # Choi+2016 (MIST I) + Dotter+2016 (MIST 0)
    "pdva": ["padova"],
    "padova": ["padova"],
    "bsti": ["basti"],
    "basti": ["basti"],
    "stars": [],  # BPASS handles isochrones internally (see SSP_CODE bpass)
}

# Stellar spectral / atmosphere library (third filename token). ``c3k`` is the
# FSPS theoretical (Kurucz ATLAS12/SYNTHE) library — the FSPS README directs
# users to cite the FSPS papers for it, so it maps to the FSPS code citations.
SSP_LIBRARY_CITATIONS: dict[str, list[str]] = {
    "miles": ["miles"],
    "c3k": ["fsps2009", "fsps"],
    "basel": ["basel"],
    "stelib": ["stelib"],
}


# AGN components — keyed by AGNConfig.disc / torus / blr values.
AGN_DISC_CITATIONS: dict[str, list[str]] = {
    "powerlaw": [],
    "multicolor": ["shakura_sunyaev1973"],
    "shakura_sunyaev": ["shakura_sunyaev1973"],
    "kubota_done": ["kubota_done2018"],
    "adaf": ["mahadevan1997"],
    None: [],
}

AGN_TORUS_CITATIONS: dict[str, list[str]] = {
    "skirtor": ["skirtor", "skirtor_2012"],
    "stalevski": ["skirtor", "skirtor_2012"],
    "clumpy": ["clumpy_nenkova2008"],  # Nenkova+2008 Paper I (ADS-verified)
    "nenkova": ["clumpy_nenkova2008"],
    None: [],
}

# Synthesizer (Lovell et al. 2025 + Roper et al. 2026). BOTH papers MUST be
# cited whenever any Synthesizer-derived grid is used — the upstream authors'
# citation policy is strict on this. Keep these two keys together everywhere.
# https://synthesizer-project.github.io/synthesizer/#citation-acknowledgement
SYNTHESIZER_CITATIONS: list[str] = ["synthesizer", "synthesizer_joss"]

# AGN narrow-line-region blocks (Parameters.agn_nlr_block). The Synthesizer
# Cloudy grid variants cite both Synthesizer papers.
AGN_NLR_CITATIONS: dict[str, list[str]] = {
    "synthesizer": SYNTHESIZER_CITATIONS,
    "synthesizer_spectra": SYNTHESIZER_CITATIONS,
    "grahsp": ["buchner2024"],
    "analytic": [],
    "none": [],
    None: [],
}

# AGN broad-line-region blocks (Parameters.agn_blr_block).
AGN_BLR_CITATIONS: dict[str, list[str]] = {
    "synthesizer": SYNTHESIZER_CITATIONS,
    "synthesizer_spectra": SYNTHESIZER_CITATIONS,
    "qsogen": ["temple2021_qsogen"],
    "grahsp": ["buchner2024"],
    "analytic": [],
    "none": [],
    # Legacy / alternative selector values retained for back-compat.
    "temple": ["temple2021_qsogen"],
    "vanden_berk": ["vandenberk2001"],  # SDSS composite (ADS-verified)
    "sdss_composite": ["vandenberk2001"],
    None: [],
}

# IGM-attenuation models.
IGM_CITATIONS: dict[str, list[str]] = {
    "inoue": ["inoue2014"],
    "inoue14": ["inoue2014"],
    "inoue2014": ["inoue2014"],
    "madau": ["madau1995"],
    "madau1995": ["madau1995"],
    "meiksin06": ["meiksin2006"],
    "meiksin2006": ["meiksin2006"],
    # Asada+2025 builds the CGM damping wing on top of the Inoue+2014 mean IGM,
    # so it cites both.
    "asada25": ["inoue2014", "asada2025"],
    None: [],
}

#: Damped-Lyman-α absorber citation, triggered when ``spec.dla`` is set
#: (independent of the mean-IGM model). The absorber's Voigt profile follows
#: the Tepper-García (2006) analytic Voigt-Hjerting approximation.
DLA_CITATIONS: list[str] = ["teppergarcia2006"]

#: X-ray model name → citation keys. Star-formation X-rays always carry the
#: X-ray-binary scalings (Lehmer+2016); the AGN corona adds the X-CIGALE
#: alpha_ox-L2500 module (Yang+2020).
XRAY_CITATIONS: dict[str, list[str]] = {
    "simple": ["lehmer2016"],
    "yang20": ["yang2020", "lehmer2016"],
    # The Lopez+2024 IRX corona is a variant within the same X-CIGALE X-ray
    # framework (Yang+2020) layered on the XRB scalings.
    "lopez24": ["yang2020", "lehmer2016"],
    None: [],
}

#: Radio: star-forming synchrotron + free-free continuum (Condon 1992) with the
#: IR-radio SFR calibration (Bell 2003). Triggered whenever radio is active.
RADIO_CITATIONS: list[str] = ["condon1992", "bell2003"]

#: Shock model name → citation keys. MAPPINGS V shock/precursor grids.
SHOCK_CITATIONS: dict[str, list[str]] = {
    "mappings": ["mappings"],
    None: [],
}

# Photometric filter-convolution convention (FilterConvention; ADR-0017).
# The AB-system foundations apply to any broadband flux; the per-convention
# entries cite the code each convention reproduces. See docs/units.md.
PHOTOMETRY_CONVENTION_CITATIONS: dict[str, list[str]] = {
    # Always relevant when broadband photometry is computed.
    "core": ["ab_system", "kcorrection", "fukugita1996", "bessell2012"],
    # w = 1/lambda — photon-counting; matches FSPS / DSPS / sedpy.
    "bessell": ["fsps", "dsps", "kcorrection", "fukugita1996"],
    # w = 1/lambda^2 — energy / flat-in-frequency; matches CIGALE / bagpipes.
    "energy": ["cigale", "bagpipes"],
}


# Inference backends (values passed to ``Fitter.run(backend=...)``).
BACKEND_CITATIONS: dict[str, list[str]] = {
    "map": [],
    "laplace": [],
    "pathfinder": ["pathfinder"],
    "mcmc_nuts": ["blackjax"],
    "nuts": ["blackjax"],
    "mcmc_raytrace": ["raytrace_behroozi"],
    "raytrace": ["raytrace_behroozi"],
    "evidence": ["nss"],
    "nss": ["nss"],
    "nested_slice": ["nss"],
    "ess": ["ess_murray2010"],  # Murray, Adams & MacKay (2010, ADS-verified)
    "elliptical_slice": ["ess_murray2010"],
    "mcmc_ess": ["ess_murray2010"],
    "vi": ["nifty", "ift"],
    "vi_native": ["nifty", "ift"],
}


# Optional function-level annotations. Populated by the @cites decorator.
# Key: "module.qualname" string. Value: list of citation registry keys.
FUNCTION_CITATIONS: dict[str, list[str]] = {}


def cites(*keys: str) -> Callable[[F], F]:
    """Decorator that records citations for a function or class.

    Use it to tie a scientific module to its upstream paper(s) at definition
    time — the citation is then visible to :func:`tengri.collect_citations`
    whenever that function is called on the object graph of a Galaxy.

    Parameters
    ----------
    *keys : str
        Citation registry keys (see ``references.bib``).

    Examples
    --------
    >>> from tengri.citations import cites
    >>> @cites("calzetti2000")
    ... def calzetti_law(wave, av):
    ...     '''Calzetti et al. (2000) starburst attenuation law.'''
    ...     ...

    The decorator is transparent — it returns ``func`` unchanged — and
    simply registers the function's fully-qualified name in
    ``FUNCTION_CITATIONS``.
    """

    def _decorate(func: F) -> F:
        qual = f"{func.__module__}.{func.__qualname__}"
        FUNCTION_CITATIONS.setdefault(qual, []).extend(keys)
        # Also expose on the function object for introspection.
        existing = list(getattr(func, "_tengri_cites", ()))
        existing.extend(keys)
        import contextlib

        with contextlib.suppress(AttributeError, TypeError):
            # Some callables (builtins, slot wrappers) reject attributes.
            func._tengri_cites = tuple(existing)  # type: ignore[attr-defined]
        return func

    return _decorate


def register_function_citations(qualname: str, keys: list[str]) -> None:
    """Register citations for a function by fully-qualified name.

    Equivalent to ``@cites`` but usable without touching the function
    definition (e.g. annotating third-party code or JIT-wrapped callables).
    """
    FUNCTION_CITATIONS.setdefault(qualname, []).extend(keys)


__all__ = [
    "AGN_BLR_CITATIONS",
    "AGN_DISC_CITATIONS",
    "AGN_NLR_CITATIONS",
    "AGN_TORUS_CITATIONS",
    "BACKEND_CITATIONS",
    "CORE_CITATIONS",
    "DLA_CITATIONS",
    "DUST_LAW_CITATIONS",
    "DUST_MODEL_CITATIONS",
    "FUNCTION_CITATIONS",
    "IGM_CITATIONS",
    "IMF_CITATIONS",
    "NEBULAR_BACKEND_CITATIONS",
    "PHOTOMETRY_CONVENTION_CITATIONS",
    "RADIO_CITATIONS",
    "SHOCK_CITATIONS",
    "SSP_CODE_CITATIONS",
    "SSP_ISOCHRONE_CITATIONS",
    "SSP_LIBRARY_CITATIONS",
    "SYNTHESIZER_CITATIONS",
    "XRAY_CITATIONS",
    "cites",
    "register_function_citations",
]

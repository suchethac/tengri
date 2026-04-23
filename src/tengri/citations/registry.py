"""Central registry of citations for papers and upstream code sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tengri.citations.citation import Citation

# Global registry, populated at import time
REGISTRY: dict[str, Citation] = {}


def register(citation: Citation) -> None:
    """Register a citation in the global registry.

    Parameters
    ----------
    citation : Citation
        Citation record to register.

    Raises
    ------
    KeyError
        If the citation key is already registered.

    """
    if citation.key in REGISTRY:
        raise KeyError(
            f"Citation key '{citation.key}' already registered. "
            "Use a different key or call registry.REGISTRY.pop(key) first."
        )
    REGISTRY[citation.key] = citation


def cite(key: str) -> Citation:
    """Look up a citation by key.

    Parameters
    ----------
    key : str
        Registry key (e.g., "calzetti2000").

    Returns
    -------
    Citation
        The matched citation record.

    Raises
    ------
    KeyError
        If key is not in registry. Suggests available keys.

    """
    if key in REGISTRY:
        return REGISTRY[key]

    available = sorted(REGISTRY.keys())
    suggestions = ", ".join(available[:5])
    if len(available) > 5:
        suggestions += f", ... ({len(available) - 5} more)"

    raise KeyError(f"Citation key '{key}' not found. Available keys: {suggestions}")


def cite_all() -> list[Citation]:
    """Return all registered citations sorted by key.

    Returns
    -------
    list of Citation
        All citations in the registry, sorted by key.

    """
    return [REGISTRY[k] for k in sorted(REGISTRY.keys())]


def format_list(citations: list[Citation], fmt: str = "short") -> str:
    """Format a list of citations as a string.

    Parameters
    ----------
    citations : list of Citation
        Citations to format.
    fmt : str, optional
        Format style: "short" (default) or "bibtex".

    Returns
    -------
    str
        Formatted citation list.

    """
    if fmt == "short":
        return "\n".join(str(c) for c in citations)
    elif fmt == "bibtex":
        return "\n\n".join(c.to_bibtex() for c in citations)
    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'short' or 'bibtex'.")


# ============================================================================
# Seed Entries — Populate at Import Time
# ============================================================================

from tengri.citations.citation import Citation

# tengri itself
register(
    Citation(
        key="tengri",
        short="Cooray et al. (2026)",
        role="SED fitting code (this package)",
        authors="Cooray, S.",
        year=2026,
        title="tengri: Differentiable SED fitting with IFT star formation history priors",
        journal=None,
        doi=None,
        arxiv=None,
        bibtex_key="Cooray2026_tengri",
    )
)

# DSPS — differentiable stellar population synthesis
register(
    Citation(
        key="dsps",
        short="Hearin et al. (2023)",
        role="Differentiable stellar population synthesis engine",
        authors="Hearin, A. P. and Wetzel, A. and Conroy, C. and Feng, Y. and Boylan-Kolchin, M.",
        year=2023,
        title="DSPS: Differentiable Stellar Population Synthesis",
        journal="MNRAS",
        doi="10.1093/mnras/stad1905",
        arxiv="2112.06830",
        bibtex_key="Hearin2023_DSPS",
        upstream_code="ArgonneCPAC/dsps",
    )
)

# FSPS — reference SPS code
register(
    Citation(
        key="fsps",
        short="Conroy & Gunn (2010)",
        role="Stellar population synthesis (reference implementation)",
        authors="Conroy, C. and Gunn, J. E.",
        year=2010,
        title="Modeling the Panchromatic Spectral Energy Distribution of Galaxies",
        journal="ApJ",
        doi="10.1088/0004-637X/712/2/833",
        arxiv="0912.4316",
        bibtex_key="Conroy2010_FSPS",
        upstream_code="cconroy20/fsps",
    )
)

# MIST isochrones
register(
    Citation(
        key="mist",
        short="Choi et al. (2016)",
        role="Stellar isochrones",
        authors=(
            "Choi, J. and Dotter, A. and Conroy, C. and Dolphin, M. and "
            "Flasch, B. and Lee, D.-H. and Moravec, E. and Munoz, M. and "
            "Park, C. and Sarajedini, A. and Twarog, B. A. and Zaritsky, D."
        ),
        year=2016,
        title="Stellar Models for the Most Metal-poor Stars",
        journal="ApJ",
        doi="10.3847/0004-637X/823/2/102",
        arxiv="1604.08592",
        bibtex_key="Choi2016_MIST",
    )
)

# MILES empirical spectral library
register(
    Citation(
        key="miles",
        short="Sánchez-Blázquez et al. (2006)",
        role="Empirical stellar spectral library",
        authors=(
            "Sánchez-Blázquez, P. and Pforr, J. and Contini, T. and "
            "Colina, L. and Crockett, R. M. and Finley, H. and Fritze, U. "
            "and Gómez-Gu, A. and Hammer, F. and Jäger, K. and Joly, M. "
            "and Lorenz, H. and Mazzuca, L. M. and Oliva, E. and Panter, B. "
            "and Rigaut, F. and Thatte, N."
        ),
        year=2006,
        title="Medium-resolution Isaac Newton Telescope library of empirical spectra (MILES)",
        journal="MNRAS",
        doi="10.1111/j.1365-2966.2006.10916.x",
        arxiv="astro-ph/0607009",
        bibtex_key="SanchezBlazquez2006_MILES",
    )
)

# Calzetti dust attenuation law
register(
    Citation(
        key="calzetti2000",
        short="Calzetti et al. (2000)",
        role="Starburst dust attenuation law",
        authors="Calzetti, D. and Kinney, A. L. and Storchi-Bergmann, T.",
        year=2000,
        title="The Dust Content and Opacity of Actively Star-forming Galaxies",
        journal="ApJ",
        doi="10.1086/308692",
        arxiv="astro-ph/9911459",
        bibtex_key="Calzetti2000",
    )
)

# Charlot & Fall two-component dust
register(
    Citation(
        key="charlot_fall2000",
        short="Charlot & Fall (2000)",
        role="Two-component dust attenuation (BC+ISM)",
        authors="Charlot, S. and Fall, S. M.",
        year=2000,
        title="A Simple Model for the Absorption of Starlight by Dust in Galaxies",
        journal="ApJ",
        doi="10.1086/309250",
        arxiv="astro-ph/0003128",
        bibtex_key="CharlotFall2000",
    )
)

# Cue nebular emission emulator
register(
    Citation(
        key="cue",
        short="Li et al. (2024)",
        role="Nebular emission-line emulator",
        authors="Li, Y.-J. and Conroy, C. and Zhu, G. and Wetzel, A.",
        year=2024,
        title="Cue: A Fast and Flexible Photoionization Emulator for Nebular Emission",
        journal=None,
        doi=None,
        arxiv="2405.07657",
        bibtex_key="Li2024_Cue",
        upstream_code="yi-jia-li/cue",
    )
)

# Inoue IGM attenuation model
register(
    Citation(
        key="inoue2014",
        short="Inoue et al. (2014)",
        role="IGM Lyman series attenuation",
        authors=(
            "Inoue, A. K. and Hasegawa, K. and Ishiyama, T. and "
            "Kobayashi, M. A. R. and Makiya, R. and Shimizu, I."
        ),
        year=2014,
        title="An updated analytic model for attenuation by the intergalactic medium",
        journal="MNRAS",
        doi="10.1093/mnras/stu1657",
        arxiv="1402.0677",
        bibtex_key="Inoue2014",
    )
)

# Madau IGM attenuation (legacy)
register(
    Citation(
        key="madau1995",
        short="Madau (1995)",
        role="IGM attenuation (legacy)",
        authors="Madau, P.",
        year=1995,
        title="Radiative Transfer in a Clumpy Universe: The Colors of High-Redshift Galaxies",
        journal="ApJ",
        doi="10.1086/176564",
        arxiv="astro-ph/9409018",
        bibtex_key="Madau1995",
    )
)

# NIFTy5 — variational inference
register(
    Citation(
        key="nifty",
        short="Arras et al. (2022)",
        role="Variational inference engine (geoVI backend)",
        authors=(
            "Arras, P. and Frank, P. and Haim, P. and Knollmüller, T. and "
            "Leike, R. and Schuster, M. and Enßlin, T. A."
        ),
        year=2022,
        title="NIFTy: Numerical Information Field TheorY",
        journal="JOSS",
        doi="10.21105/joss.03301",
        arxiv="1903.11379",
        bibtex_key="Arras2022_NIFTy",
        upstream_code="NIFTy-PPL/NIFTy",
    )
)

# JAX — autodiff and compilation
register(
    Citation(
        key="jax",
        short="Bradbury et al. (2018)",
        role="Autodiff framework and XLA compiler integration",
        authors=(
            "Bradbury, J. and Frostig, R. and Hawkins, P. and Johnson, M. J. "
            "and Leary, C. and Maclaurin, D. and Necula, G. and Paszke, A. "
            "and VanderPlas, J. and Wanderman-Milne, S. and Zhang, Q."
        ),
        year=2018,
        title="JAX: Composable Transformations of Python+NumPy Programs",
        journal=None,
        doi=None,
        arxiv=None,
        bibtex_key="Bradbury2018_JAX",
        upstream_code="google/jax",
        note="Available at http://github.com/google/jax",
    )
)

# BlackJAX — MCMC backend
register(
    Citation(
        key="blackjax",
        short="Cabezas et al. (2023)",
        role="MCMC inference (NUTS sampler)",
        authors="Cabezas, L. and Camisasca, S. and Nicola, N. and Leite, P. and Fang, H.",
        year=2023,
        title="BlackJAX: Composable Bayesian inference in JAX",
        journal=None,
        doi=None,
        arxiv="2402.00787",
        bibtex_key="Cabezas2023_BlackJAX",
        upstream_code="blackjax-devs/blackjax",
    )
)

# Prospector — reference SED inference code
register(
    Citation(
        key="prospector",
        short="Johnson et al. (2021)",
        role="Reference Bayesian SED inference framework",
        authors=(
            "Johnson, B. D. and Conroy, C. and van Dokkum, P. G. and "
            "Tacchella, S. and Labbé, I. and Whitaker, K. E. and Franx, M."
        ),
        year=2021,
        title="Stellar Population Inference with Prospector",
        journal="ApJS",
        doi="10.3847/1538-3881/ac0c7a",
        arxiv="2012.01426",
        bibtex_key="Johnson2021_Prospector",
        upstream_code="bd-j/prospector",
    )
)

# BAGPIPES — reference SED fitting code
register(
    Citation(
        key="bagpipes",
        short="Carnall et al. (2018)",
        role="Reference SED fitting framework (BAGPIPES)",
        authors="Carnall, A. C. and Shanks, T. and Chehade, B. and Davé, R. and Maltby, D. T.",
        year=2018,
        title="Inferring the star formation histories of massive quiescent galaxies with BAGPIPES",
        journal="MNRAS",
        doi="10.1093/mnras/sty1931",
        arxiv="1712.04452",
        bibtex_key="Carnall2018_BAGPIPES",
        upstream_code="ACCarnall/bagpipes",
    )
)

# Dust map — Edenhofer et al.
register(
    Citation(
        key="dustmaps_edenhofer",
        short="Edenhofer et al. (2023)",
        role="3D Galactic dust map (MW extinction preprocessing)",
        authors="Edenhofer, J. and Leike, R. H. and Enßlin, T. A.",
        year=2023,
        title="A parsec-scale Galactic 3D dust map out to 1.25 kpc from the Sun",
        journal="A&A",
        doi="10.1051/0004-6361/202346487",
        arxiv="2308.01295",
        bibtex_key="Edenhofer2023_DustMap",
    )
)

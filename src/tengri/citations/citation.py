# SPDX-License-Identifier: BSD-3-Clause
"""Citation record for papers and upstream code sources."""

from __future__ import annotations

from dataclasses import dataclass

# Canonical component-type categories. Any string is allowed in a Citation's
# ``category`` field, but these are the ones tengri uses for grouping output.
CATEGORIES: tuple[str, ...] = (
    "framework",  # this code + theoretical foundations (tengri, JAX, IFT)
    "ssp",  # stellar population synthesis (DSPS, FSPS, MIST, MILES)
    "sfh",  # star formation history priors / models
    "dust_attenuation",  # attenuation laws
    "dust_emission",  # dust emission templates
    "nebular",  # nebular emission
    "igm",  # intergalactic medium
    "agn",  # AGN components
    "inference",  # inference backends
    "preprocessing",  # data hygiene (MW extinction, zero points, ...)
    "reference_code",  # peer SED-fitting codes (implemented or compared against)
    "other",  # anything that doesn't fit a canonical bucket
)


@dataclass(frozen=True)
class Citation:
    """Citation record for a paper or upstream code source.

    Attributes
    ----------
    key: str
        Registry key, lowercase (e.g., "calzetti2000", "dsps").
    short: str
        Human-readable short form (e.g., "Calzetti et al. (2000)").
    role: str
        Purpose in tengri (e.g., "Starburst dust attenuation law").
    category: str
        Component-type category (one of :data:`CATEGORIES`: "framework",
        "ssp", "dust_attenuation", "nebular", "igm", "inference", ...).
        Used by ``Bibliography.by_category()`` to group the report.
    authors: str
        Full author list in BibTeX style.
    year: int
        Publication year.
    title: str
        Publication title.
    journal: str | None
        Journal name (None for preprints or code packages).
    doi: str | None
        DOI URL or identifier (e.g., "10.1086/512090").
    arxiv: str | None
        ArXiv identifier (e.g., "astro-ph/0309170").
    bibtex_key: str
        BibTeX citation key (e.g., "Calzetti2000").
    upstream_code: str | None
        Repository path of the upstream reference
        (e.g., "bd-j/prospector").
    license: str | None
        License of upstream code if applicable.
    note: str | None
        Additional notes or caveats.

    """

    key: str
    short: str
    role: str
    authors: str
    year: int
    title: str
    journal: str | None
    doi: str | None
    arxiv: str | None
    bibtex_key: str
    category: str = "other"
    upstream_code: str | None = None
    license: str | None = None
    note: str | None = None

    def __str__(self) -> str:
        """Return one-line human-readable representation.

        Format: "[role]; short. DOI:... (upstream: ...)"

        Returns
        -------
        str
            Single-line citation string.

        """
        parts = [f"[{self.role}]; {self.short}"]

        if self.doi:
            parts.append(f"DOI: {self.doi}")
        elif self.arxiv:
            parts.append(f"arXiv: {self.arxiv}")

        if self.upstream_code:
            parts.append(f"(upstream: {self.upstream_code})")

        return ". ".join(parts)

    def to_bibtex(self) -> str:
        """Generate a minimal BibTeX entry.

        Returns
        -------
        str
            BibTeX entry in @article or @misc format.

        """
        # Build author list (keep as-is, don't try to parse)
        lines = [f"@article{{{self.bibtex_key},"]
        lines.append(f"  author = {{{self.authors}}},")
        lines.append(f"  title = {{{self.title}}},")
        lines.append(f"  year = {{{self.year}}},")

        if self.journal:
            lines.append(f"  journal = {{{self.journal}}},")

        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")

        if self.arxiv:
            lines.append("  archivePrefix = {arXiv},")
            lines.append(f"  eprint = {{{self.arxiv}}},")

        lines.append("}")

        return "\n".join(lines)

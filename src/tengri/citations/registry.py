"""Central registry of citations — populated at import time from ``references.bib``.

The BibTeX file at :data:`BIB_PATH` is the single source of truth. Do not
hard-code citation data in this module — add or edit entries in the .bib.
"""

from __future__ import annotations

import importlib.resources as _resources
from pathlib import Path

from tengri.citations._bibparser import parse_bibtex
from tengri.citations.citation import Citation

# ---------------------------------------------------------------------------
# Locate the canonical .bib file (shipped with the package).
# ---------------------------------------------------------------------------

BIB_FILENAME = "references.bib"


def _locate_bib() -> Path:
    """Find ``references.bib`` via importlib.resources (works inside wheels)."""
    res = _resources.files("tengri.citations") / BIB_FILENAME
    return Path(str(res))


BIB_PATH: Path = _locate_bib()


# ---------------------------------------------------------------------------
# Registry — populated at import time by parsing BIB_PATH.
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Citation] = {}


def _entry_to_citation(entry: dict) -> Citation:
    """Convert a parsed BibTeX entry dict into a :class:`Citation`."""
    registry_key = entry.get("registry_key") or entry["bibtex_key"].lower()
    bibtex_key = entry["bibtex_key"]

    year_raw = entry.get("year")
    try:
        year = int(year_raw) if year_raw is not None else 0
    except (TypeError, ValueError):
        year = 0

    return Citation(
        key=registry_key,
        short=entry.get("short") or f"{bibtex_key} ({year})",
        role=entry.get("role") or "",
        category=entry.get("category") or "other",
        authors=entry.get("author") or "",
        year=year,
        title=entry.get("title") or "",
        journal=entry.get("journal"),
        doi=entry.get("doi"),
        arxiv=entry.get("arxiv") or entry.get("eprint"),
        bibtex_key=bibtex_key,
        upstream_code=entry.get("upstream_code"),
        license=entry.get("upstream_license"),
        note=entry.get("note"),
    )


def register(citation: Citation) -> None:
    """Register a :class:`Citation` in the global registry.

    Parameters
    ----------
    citation : Citation

    Raises
    ------
    KeyError
        If the citation's key is already present.
    """
    if citation.key in REGISTRY:
        raise KeyError(
            f"Citation key '{citation.key}' already registered. "
            "Update the source entry in references.bib instead."
        )
    REGISTRY[citation.key] = citation


def _load_from_bib() -> None:
    """Parse BIB_PATH and fill REGISTRY. Safe to call multiple times."""
    REGISTRY.clear()
    text = BIB_PATH.read_text(encoding="utf-8")
    for entry in parse_bibtex(text):
        cit = _entry_to_citation(entry)
        REGISTRY[cit.key] = cit


_load_from_bib()


# ---------------------------------------------------------------------------
# Public lookup helpers.
# ---------------------------------------------------------------------------


def cite(key: str) -> Citation:
    """Look up a citation by registry key (e.g. ``"calzetti2000"``).

    Raises
    ------
    KeyError
        If ``key`` is not registered. The exception message lists available
        keys to aid recovery from typos.
    """
    if key in REGISTRY:
        return REGISTRY[key]

    available = sorted(REGISTRY.keys())
    suggestions = ", ".join(available[:5])
    if len(available) > 5:
        suggestions += f", ... ({len(available) - 5} more)"
    raise KeyError(f"Citation key '{key}' not found. Available keys: {suggestions}")


def cite_all() -> list[Citation]:
    """Return every registered citation, sorted by key."""
    return [REGISTRY[k] for k in sorted(REGISTRY.keys())]


def format_list(citations: list[Citation], fmt: str = "short") -> str:
    """Format a list of citations as text.

    Parameters
    ----------
    fmt : {"short", "bibtex"}
        "short" prints one line per citation. "bibtex" emits full BibTeX blocks
        separated by blank lines.
    """
    if fmt == "short":
        return "\n".join(str(c) for c in citations)
    if fmt == "bibtex":
        return "\n\n".join(c.to_bibtex() for c in citations)
    raise ValueError(f"Unknown format: {fmt}. Use 'short' or 'bibtex'.")


def reload() -> None:
    """Reparse the .bib file. Useful after editing references.bib in-place."""
    _load_from_bib()

# SPDX-License-Identifier: BSD-3-Clause
"""Convenience helpers for the tengri paper citations.

The tengri paper series lives in :file:`CITATION.bib` at the repository
root (ADS-formatted). These helpers load that file so users can do:

    >>> import tengri as tg
    >>> print(tg.paper_citation())  # full BibTeX block, all papers
    >>> print(tg.paper_citation(paper=1))  # just Paper I
    >>> print(tg.paper_citation(paper="II"))

The same content is mirrored into the main registry (see
:mod:`tengri.citations.registry`) under the keys ``tengri``,
``tengri_paper2``, ``tengri_paper3``.
"""

from __future__ import annotations

from pathlib import Path

# CITATION.bib lives at the repo root. We locate it by walking up from this
# file. In a wheel install the file is not necessarily shipped: callers can
# still use ``tengri.cite("tengri")`` from the registry as a fallback.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent.parent  # .../tengri
CITATION_BIB_PATH = _REPO_ROOT / "CITATION.bib"


_PAPER_KEYS = {
    1: "Cooray_2026",
    "I": "Cooray_2026",
    "i": "Cooray_2026",
    2: "Cooray_2026a",
    "II": "Cooray_2026a",
    "ii": "Cooray_2026a",
    3: "Cooray_2026b",
    "III": "Cooray_2026b",
    "iii": "Cooray_2026b",
}


def _read_citation_bib() -> str:
    """Return the contents of CITATION.bib, or a fallback built from the registry."""
    if CITATION_BIB_PATH.exists():
        return CITATION_BIB_PATH.read_text(encoding="utf-8")

    # Fallback: synthesize BibTeX blocks from registry entries.
    from tengri.citations.registry import REGISTRY

    keys = ("tengri", "tengri_paper2", "tengri_paper3")
    blocks = []
    for k in keys:
        if k in REGISTRY:
            blocks.append(REGISTRY[k].to_bibtex())
    return "\n\n".join(blocks)


def paper_citation(paper: int | str | None = None) -> str:
    """Return the BibTeX citation for the tengri paper(s).

    Parameters
    ----------
    paper : int, str, or None
        Which paper. Accepts ``1``/``2``/``3`` or ``"I"``/``"II"``/``"III"``.
        If ``None`` (default), returns the full contents of ``CITATION.bib``
        every paper plus the ``tengri`` alias.

    Returns
    -------
    str
        BibTeX block(s). Copy-paste into a ``.bib`` file or paper draft.

    Raises
    ------
    ValueError
        If ``paper`` is given but not one of the known identifiers.
    """
    text = _read_citation_bib()

    if paper is None:
        return text

    if paper not in _PAPER_KEYS:
        raise ValueError(
            f"Unknown paper identifier {paper!r}. "
            "Use 1/2/3 or 'I'/'II'/'III', or pass None for all papers."
        )

    wanted = _PAPER_KEYS[paper]
    return _extract_entry(text, wanted)


def _extract_entry(bibtex_text: str, key: str) -> str:
    """Return the single @ENTRY block with the matching citation key."""
    # Find the @ENTRYTYPE{key, pattern and then the matching closing brace.
    needle = "{" + key + ","
    idx = bibtex_text.find(needle)
    if idx == -1:
        raise KeyError(
            f"BibTeX key {key!r} not found in CITATION.bib. "
            "This is a bug in tengri: please report it."
        )

    # Walk backward to find the '@' that starts this entry.
    at_idx = bibtex_text.rfind("@", 0, idx)
    if at_idx == -1:
        raise ValueError("Malformed BibTeX: '@' not found before citation key")

    # Walk forward counting braces. We are already INSIDE the entry body
    # (``needle`` consumed the opening '{'), so depth starts at 1.
    i = idx + len(needle)
    depth = 1
    while i < len(bibtex_text):
        ch = bibtex_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return bibtex_text[at_idx : i + 1]
        i += 1
    raise ValueError(f"Malformed BibTeX: unterminated entry {key}")


def print_paper_citation(paper: int | str | None = None) -> None:
    """Convenience: print :func:`paper_citation` to stdout."""
    print(paper_citation(paper))

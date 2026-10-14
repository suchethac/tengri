# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: every AGN menu citation resolves to a bibliography entry."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def _parse_bib_entries() -> dict[tuple[str, str], str]:
    """Parse references.bib and extract (first_author_surname, year) -> entry_key."""
    bib_path = (
        Path(__file__).parent.parent.parent / "src" / "tengri" / "citations" / "references.bib"
    )
    content = bib_path.read_text()

    entries: dict[tuple[str, str], str] = {}

    # Match @TYPE{key, ... author = {Author Name and ...}, ... year = {YYYY}, ...}
    # We need to extract author and year for each entry
    entry_pattern = r"@\w+\{([^,]+),"
    current_entry_key = None
    current_author = None
    current_year = None

    for line in content.split("\n"):
        # Check if this is the start of an entry
        entry_match = re.match(r"@\w+\{([^,]+),", line)
        if entry_match:
            # Save previous entry if complete
            if current_entry_key is not None and current_author and current_year:
                entries[(current_author, current_year)] = current_entry_key
            current_entry_key = entry_match.group(1).strip()
            current_author = None
            current_year = None

        # Extract author field (first author surname)
        if "author" in line.lower():
            # Match author = {{AuthorSurname}, ... (including Unicode letters)
            author_match = re.search(r"author\s*=\s*\{\{(\w[\w'\-]*)", line, re.IGNORECASE)
            if author_match:
                current_author = author_match.group(1)

        # Extract year field
        if "year" in line.lower():
            year_match = re.search(r"year\s*=\s*\{?((?:19|20)\d\d)", line, re.IGNORECASE)
            if year_match:
                current_year = year_match.group(1)

    # Don't forget the last entry
    if current_entry_key is not None and current_author and current_year:
        entries[(current_author, current_year)] = current_entry_key

    return entries


def _extract_citation_components(citation_str: str) -> tuple[str, str] | None:
    r"""Extract (author_surname, year) from a citation string using regex.

    Matches first author surname (word characters including Unicode) and 4-digit year.
    Returns None if no year is found.
    """
    # Match first author surname (using \w which includes Unicode letters)
    # followed by optional apostrophes and hyphens, then a 4-digit year
    match = re.search(r"^(\w[\w'\-]*).*?((?:19|20)\d\d)", citation_str)
    if match:
        return (match.group(1), match.group(2))
    return None


def test_agn_menu_citations_resolve_to_bib():
    """Contract: every AGN menu citation resolves against references.bib."""
    import tengri
    from tengri.components.agn import unified

    # Collect citations from _SELF_CONTAINED_AGN_MODELS
    menu_citations: dict[str, str] = {}
    for model_name, model_dict in unified._SELF_CONTAINED_AGN_MODELS.items():
        if "citation" in model_dict:
            menu_citations[model_name] = model_dict["citation"]

    # Collect citations from list_agn_models() — each entry is a dict with 'name' field
    agn_list = tengri.list_agn_models()
    for model_entry in agn_list:
        if isinstance(model_entry, dict):
            name = model_entry.get("name")
            citation = model_entry.get("citation")
        else:
            # Fallback if it's an object
            name = getattr(model_entry, "name", None)
            citation = getattr(model_entry, "citation", None)

        if name and citation:
            menu_citations[name] = citation

    # Parse the bibliography
    bib_entries = _parse_bib_entries()

    # Strings with no year — allow-list for known special cases
    # (composable is not a real paper)
    allowed_no_year = {
        "composable",  # recipe, not a paper
    }
    no_year_citations: list[tuple[str, str]] = []  # (name, citation)

    # Check each menu citation
    failures: list[str] = []
    for name, citation in menu_citations.items():
        components = _extract_citation_components(citation)

        if components is None:
            if name not in allowed_no_year:
                no_year_citations.append((name, citation))
            continue

        author, year = components

        if (author, year) not in bib_entries:
            # Find similar entries for the author
            similar = [
                f"  - ({a}, {y}): {key}" for (a, y), key in bib_entries.items() if a == author
            ]
            similar_str = "\n".join(similar) if similar else "  (no entries for this author)"

            failures.append(
                f"Model '{name}' citation '{citation}' → ({author}, {year}) not in bib\n"
                f"  Similar entries for author '{author}':\n{similar_str}"
            )

    # Verify no_year_citations is empty
    assert not no_year_citations, (
        f"Unexpected citations with no year (not in allow-list): {no_year_citations}\n"
        f"Update allowed_no_year if these are valid."
    )

    # Assert all citations resolve
    assert not failures, (
        f"The following AGN menu citations do not resolve against references.bib:\n\n"
        f"{chr(10).join(failures)}"
    )

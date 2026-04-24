"""Citation records and attribution helpers for tengri components.

Provides a central registry of papers and upstream code that tengri depends on
or ports from. Call ``tengri.cite("calzetti")`` for a single entry or
``tengri.cite_all()`` for every registered source.

See Also
--------
tengri.citations.citation.Citation : the record type.
tengri.citations.registry : seed entries and registry functions.

"""

from __future__ import annotations

from tengri.citations.citation import Citation
from tengri.citations.papers import (
    CITATION_BIB_PATH,
    paper_citation,
    print_paper_citation,
)
from tengri.citations.registry import REGISTRY, cite, cite_all, format_list, register

__all__ = [
    "CITATION_BIB_PATH",
    "REGISTRY",
    "Citation",
    "cite",
    "cite_all",
    "format_list",
    "paper_citation",
    "print_paper_citation",
    "register",
]

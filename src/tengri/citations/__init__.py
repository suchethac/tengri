# SPDX-License-Identifier: BSD-3-Clause
"""Citation records and attribution helpers for tengri components.

Provides a central registry of papers and reference codes that tengri depends
on, implements, or compares against. Call ``tengri.cite("calzetti")`` for a single entry or
``tengri.cite_all()`` for every registered source.

See Also
--------
tengri.citations.citation.Citation : the record type.
tengri.citations.registry : seed entries and registry functions.

"""

from __future__ import annotations

from tengri.citations.associations import (
    BACKEND_CITATIONS,
    CORE_CITATIONS,
    DUST_LAW_CITATIONS,
    DUST_MODEL_CITATIONS,
    IGM_CITATIONS,
    NEBULAR_BACKEND_CITATIONS,
    cites,
    register_function_citations,
)
from tengri.citations.bibliography import Bibliography
from tengri.citations.citation import Citation
from tengri.citations.collect import (
    citations_bibtex,
    citations_report,
    collect_citations,
    print_bibtex,
    print_citations,
)
from tengri.citations.papers import (
    CITATION_BIB_PATH,
    paper_citation,
    print_paper_citation,
)
from tengri.citations.registry import REGISTRY, cite, cite_all, format_list, register

__all__ = [
    "BACKEND_CITATIONS",
    "CITATION_BIB_PATH",
    "CORE_CITATIONS",
    "DUST_LAW_CITATIONS",
    "DUST_MODEL_CITATIONS",
    "IGM_CITATIONS",
    "NEBULAR_BACKEND_CITATIONS",
    "REGISTRY",
    "Bibliography",
    "Citation",
    "citations_bibtex",
    "citations_report",
    "cite",
    "cite_all",
    "cites",
    "collect_citations",
    "format_list",
    "paper_citation",
    "print_bibtex",
    "print_citations",
    "print_paper_citation",
    "register",
    "register_function_citations",
]

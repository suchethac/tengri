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
from tengri.citations.registry import REGISTRY, cite, cite_all, format_list, register

__all__ = ["REGISTRY", "Citation", "cite", "cite_all", "format_list", "register"]

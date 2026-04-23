"""User-facing filter discovery helpers.

Thin wrappers around tengri.observation.filters.load_filter_set plus an
in-memory directory of the curated filter library. No network calls.

Functions
---------
list_filters(instrument=None) -> list[str]
    Names of filters shipped with tengri. Filter by instrument prefix.
load(names) -> filter objects
    Alias for ``tengri.observation.filters.load_filter_set`` for discoverability.
describe(name) -> str
    One-line description of a filter (wavelength range, instrument).
suggest(redshift, coverage='visible_to_nir') -> list[str]
    Filter names that cover the requested rest-frame span at this redshift.
"""

from tengri.filters.core import describe, list_filters, load, suggest

__all__ = ["describe", "list_filters", "load", "suggest"]

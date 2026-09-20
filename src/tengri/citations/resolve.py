# SPDX-License-Identifier: BSD-3-Clause
"""One place that turns a component name into BibTeX keys.

There were three, and they disagreed by omission:

===================================  =====  ==============================
map                                  names  read by
===================================  =====  ==============================
``registry.py::_NAME_TO_BIBKEY``        48  ``print_components_bibtex``
``collect.py::_LIVE_NAME_TO_BIBKEY``    41  ``collect_citations``
``associations.py::*_CITATIONS``       101  ``collect_citations`` (explicit)
===================================  =====  ==============================

14 names were in the first and not the second, 7 the other way, and 23 lived in
a hand-written map with no association table at all: so the two public
citation surfaces handed a reader different bibliographies for the same model.
No name mapped to *conflicting* keys; every difference was a gap, which is why
nothing ever looked wrong.

This module holds the merged hand-written map once, and
:func:`citation_keys_for` unions it with the association tables. Both surfaces
call it, so a mapping added anywhere reaches both.

Notes
-----
The association tables stay the preferred home for new entries: they are what
:func:`tengri.collect_citations` reads per subsystem, and they carry one name to
*several* keys. :data:`NAME_TO_BIBKEY` is for the leftovers; names no
subsystem table covers, such as the SFH types and the always-on frameworks.
"""

from __future__ import annotations

__all__ = [
    "NAME_TO_BIBKEY",
    "association_keys_for",
    "citation_key_hint",
    "citation_keys_for",
    "registry_names_for_citation_key",
]

#: Component name → BibTeX key, for names no association table covers.
#:
#: Merged from the two hand-written maps that used to live in ``registry.py``
#: and ``collect.py``. Where both had a name they agreed, so the merge is a
#: union with no arbitration.
NAME_TO_BIBKEY: dict[str, str] = {
    # SFH lives in ``associations.SFH_CITATIONS``: a name there can carry
    # several keys, which ``delayed`` (CIGALE *and* Bagpipes) needs and this
    # one-key-per-name map cannot express.
    # ─ AGN ─
    "skirtor": "skirtor",
    "stalevski": "skirtor",
    "skirtor_agnfitter": "skirtor",
    "schartmann2005_skirtor_atten": "skirtor",
    "kubota_done": "kubota_done2018",
    "kubota_done_full": "kubota_done2018",
    "multicolor_agn": "kubota_done2018",
    "adaf": "mahadevan1997",
    "qsogen": "temple2021_qsogen",
    "qsogen_smc": "temple2021_qsogen",
    "qsogen_balmer": "temple2021_qsogen",
    # AGN composable blocks: bibkeys verified against each block's registered
    # ``citation=`` string (never guessed). Blocks whose paper has no bundled
    # BibTeX (fritz, cat3d_wind, feltre, richards2006, boroson_green, …) fall
    # through to the free-form citation note.
    "grahsp": "buchner2024",
    "agn_grahsp": "buchner2024",
    "grahsp_sbpl": "buchner2024",
    "grahsp_biatten": "buchner2024",
    "nenkova": "clumpy_nenkova2008",
    "nenkova_agnfitter": "clumpy_nenkova2008",
    "multicolor": "shakura_sunyaev1973",
    "synthesizer": "synthesizer",
    "synthesizer_spectra": "synthesizer",
    # ─ Dust attenuation ─
    "calzetti": "calzetti2000",
    "cardelli": "cardelli1989",
    "kriek_conroy": "kriek_conroy2013",
    "noll09": "noll2009",
    "salim": "salim2018",
    "salim_sbl18": "salim2018",
    "li08": "li2008_ext",
    "smc": "gordon2003_smc",
    "lmc": "gordon2003_smc",
    "power_law": "charlot_fall2000",
    "hd23_mwrv31": "hensley_draine2023",
    # ─ Dust attenuation *model* selector (dust_model, not a per-component law) ─
    "wg00": "witt_gordon2000",  # Witt & Gordon (2000) RT screen (FSPS dust_type=3)
    "single_component": "calzetti2000",
    # ─ Dust emission ─
    "dl07": "draine_li2007",
    "draine_li2007": "draine_li2007",
    "dl14": "draine2014",
    "dale2014": "dale2014",
    "dale2014_cigale": "dale2014",
    "casey2012": "casey2012",
    "mbb": "casey2012",
    # ─ Nebular ─
    "cue": "cue",
    "cloudy_grid": "cloudy",
    # ─ Inference ─
    "mcmc_nuts": "blackjax",
    "mcmc": "blackjax",
    "mcmc_ess": "ess_murray2010",
    "vi": "nifty",
    "vi_nonlinear_fast": "nifty",
    "mcmc_raytrace": "raytrace_behroozi",
    "pathfinder": "pathfinder",
    "nss": "nss",
    # ─ Frameworks (always-on) ─
    "tengri": "tengri",
    "dsps": "dsps",
    "jax": "jax",
}

#: Suffixes a registered variant may carry that the base name does not.
#: ``dl07_tabulated`` is the same physics, and the same paper, as ``dl07``.
_STRIPPABLE_SUFFIXES = ("_tabulated",)


def association_keys_for(name: str) -> list[str]:
    """BibTeX keys the per-subsystem association tables record for ``name``.

    Parameters
    ----------
    name : str
        Component name as a registry or menu row spells it.

    Returns
    -------
    list[str]
        Keys in table order, de-duplicated; empty when no table names it.

    Notes
    -----
    Scans every ``*_CITATIONS`` table in
    :mod:`tengri.citations.associations` rather than a hard-coded list, so a
    new subsystem table is picked up without editing this function. Matching by
    name across all tables is safe only while no name means two different
    things; ``test_bibtex_uses_the_association_tables`` pins that.
    """
    from tengri.citations import associations as _assoc

    keys: list[str] = []
    for attr in sorted(dir(_assoc)):
        if not attr.endswith("_CITATIONS"):
            continue
        table = getattr(_assoc, attr)
        if isinstance(table, dict):
            keys.extend(table.get(name) or [])
        elif isinstance(table, list) and name in table:
            # Flat tables (RADIO_CITATIONS) list registry keys directly, so a
            # component whose own name is one of them resolves to itself.
            keys.append(name)
    return list(dict.fromkeys(keys))


def citation_keys_for(name: str | None) -> list[str]:
    """Every BibTeX key for a component name.

    Parameters
    ----------
    name : str or None
        Component name; ``None`` and empty strings return ``[]`` so callers can
        pass an unset config field straight through.

    Returns
    -------
    list[str]
        Keys from :data:`NAME_TO_BIBKEY` first, then the association tables,
        de-duplicated in that order.

    Notes
    -----
    The two sources are a **union, not a fallback chain with a winner**.
    ``skirtor`` is only in :data:`NAME_TO_BIBKEY`; ``two_component`` cites
    Charlot & Fall and is only in ``DUST_MODEL_CITATIONS``. Consulting one of
    them loses the other's names.

    Lookup is case-insensitive on the explicit map because config fields arrive
    lower-cased while some registry rows are capitalized (``DSPS``, ``JAX``).
    """
    if not name:
        return []
    text = str(name)
    keys = [k for k in (NAME_TO_BIBKEY.get(text), NAME_TO_BIBKEY.get(text.lower())) if k]
    keys.extend(association_keys_for(text))
    if not keys:
        for suffix in _STRIPPABLE_SUFFIXES:
            if text.lower().endswith(suffix):
                return citation_keys_for(text[: -len(suffix)])
    return list(dict.fromkeys(keys))


def registry_names_for_citation_key(key: str) -> tuple[str, ...]:
    """Registry names that cite a given BibTeX key.

    Reverse of :func:`citation_keys_for`: if a registry name ``foo`` maps to
    citation key ``bar``, then ``registry_names_for_citation_key("bar")``
    includes ``"foo"``.

    Parameters
    ----------
    key : str
        A BibTeX key (e.g., ``"charlot_fall2000"``).

    Returns
    -------
    tuple[str, ...]
        Registry names citing this key, in the order they appear in the forward
        maps. Empty tuple if the key is unknown.

    Notes
    -----
    Lookup is case-insensitive to handle variation in key spelling. The reverse
    mapping is built once per module import and cached.

    Examples
    --------
    >>> registry_names_for_citation_key("charlot_fall2000")
    ('power_law',)
    >>> registry_names_for_citation_key("skirtor")
    ('skirtor', 'stalevski', 'skirtor_agnfitter', 'schartmann2005_skirtor_atten')
    """
    key_lower = key.lower()
    names: list[str] = []

    # Build reverse of NAME_TO_BIBKEY
    for name, bibkey in NAME_TO_BIBKEY.items():
        if bibkey.lower() == key_lower:
            names.append(name)

    # Build reverse of association tables
    from tengri.citations import associations as _assoc

    for attr in sorted(dir(_assoc)):
        if not attr.endswith("_CITATIONS"):
            continue
        table = getattr(_assoc, attr)
        if isinstance(table, dict):
            for name, keys in table.items():
                if isinstance(keys, list):
                    for k in keys:
                        if k.lower() == key_lower:
                            names.append(name)
                elif isinstance(keys, str) and keys.lower() == key_lower:
                    names.append(name)
        elif isinstance(table, list):
            # Flat tables: check if key is in the list
            for item in table:
                if isinstance(item, str) and item.lower() == key_lower:
                    names.append(item)

    return tuple(dict.fromkeys(names))


def citation_key_hint(unknown: str, valid_names: list[str], *, kind: str, keyword: str) -> str:
    """Error message fragment if unknown name is a citation key.

    When a user provides a citation key (like ``"charlot_fall2000"``) instead
    of a registry name (like ``"power_law"``), this function returns a helpful
    message fragment explaining the confusion. If ``unknown`` is not a known
    citation key, returns an empty string so difflib suggestions can follow.

    Parameters
    ----------
    unknown : str
        The unknown name the user provided.
    valid_names : list[str]
        List of valid registry names for this parameter.
    kind : str
        The type of thing being validated (e.g., ``"dust law"``).
    keyword : str
        The parameter keyword used (e.g., ``"law"`` or ``"law_bc"``).

    Returns
    -------
    str
        A sentence(s) or empty string. If non-empty, describes the citation key
        and which registry name(s) it corresponds to.

    Examples
    --------
    >>> citation_key_hint("charlot_fall2000", valid_names, kind="dust law", keyword="law")
    "That is a citation key for power_law (dust law). Use law='power_law'."
    """
    names_for_key = registry_names_for_citation_key(unknown)
    if not names_for_key:
        return ""

    # Filter to names that are actually valid at this site
    valid_at_site = [n for n in names_for_key if n in valid_names]

    if valid_at_site:
        # This citation key is valid here
        names_str = ", ".join(valid_at_site)
        if len(valid_at_site) == 1:
            return (
                f"That is a citation key for {valid_at_site[0]} ({kind}). "
                f"Use {keyword}='{valid_at_site[0]}'."
            )
        else:
            return (
                f"That is a citation key for {names_str} ({kind}). "
                f"Use {keyword}='{valid_at_site[0]}' or {keyword}='"
                f"{valid_at_site[1]}' (or another from: {', '.join(valid_at_site)})."
            )
    else:
        # This citation key exists but not for this group
        # Report the first name it maps to
        first_name = names_for_key[0]
        return (
            f"That is a citation key for {first_name} (a different {kind}), "
            f"not a valid {kind}. Use {keyword}='{valid_names[0]}' or another "
            f"from the valid list."
        )

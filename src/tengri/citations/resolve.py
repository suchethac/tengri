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

__all__ = ["NAME_TO_BIBKEY", "association_keys_for", "citation_keys_for"]

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
    name: str
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
    name: str or None
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

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


#: Association tables whose keys are true registry namespaces: values a
#: :func:`tengri.SEDModel.build` group's ``type=``/``law=`` selector actually
#: accepts, and so the ones :mod:`tengri.parameters.groups` validates unknown
#: names against. Paired with the human-readable label used when an orphaned
#: citation key's name is reported (:func:`citation_key_hint`'s "not valid
#: here" branch).
#:
#: Deliberately excludes:
#:
#: * SSP provenance/alias tables (``SSP_CODE_CITATIONS``,
#:   ``SSP_ISOCHRONE_CITATIONS``, ``SSP_LIBRARY_CITATIONS``, ``IMF_CITATIONS``)
#:   -- filename tokens and their misspelling aliases (``"bsti"`` for
#:   ``"basti"``), never a value any group's grammar accepts.
#: * ``PHOTOMETRY_CONVENTION_CITATIONS`` and ``BACKEND_CITATIONS`` -- real
#:   selectors, but for ``FilterConvention`` / ``Fitter.run(backend=...)``,
#:   neither of which ``parameters/groups.py`` validates through this path.
#: * The always-on flat lists (``CORE_CITATIONS``, ``RADIO_CITATIONS``,
#:   ``DLA_CITATIONS``, ``SYNTHESIZER_CITATIONS``) -- unconditional citations
#:   triggered by a subsystem being active at all, not name -> key maps.
#: * ``FUNCTION_CITATIONS`` -- keyed by ``"module.qualname"``, not a
#:   component name.
#:
#: Including any of these reversed a citation key to a name from the wrong
#: universe: ``sfh={'type': 'basti'}`` (an SSP isochrone token, not an SFH
#: type) was reported as "citation key for bsti", and ``bessell2012`` /
#: ``chabrier2003`` similarly pointed at a photometry convention / an IMF
#: token (#2429 opus review M2).
_REGISTRY_NAMESPACE_TABLES: tuple[tuple[str, str], ...] = (
    ("SFH_CITATIONS", "SFH type"),
    ("RADIO_MODEL_CITATIONS", "radio model"),
    ("DUST_LAW_CITATIONS", "dust law"),
    ("DUST_EMISSION_CITATIONS", "dust_emission type"),
    ("DUST_MODEL_CITATIONS", "dust_attenuation type"),
    ("NEBULAR_BACKEND_CITATIONS", "nebular type"),
    ("AGN_DISC_CITATIONS", "agn.disc type"),
    ("AGN_TORUS_CITATIONS", "agn.torus type"),
    ("AGN_NLR_CITATIONS", "agn.nlr type"),
    ("AGN_BLR_CITATIONS", "agn.blr type"),
    ("IGM_CITATIONS", "IGM type"),
    ("XRAY_CITATIONS", "X-ray type"),
    ("SHOCK_CITATIONS", "shock type"),
)


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
    Swept sources are :data:`NAME_TO_BIBKEY` and the tables listed in
    :data:`_REGISTRY_NAMESPACE_TABLES` -- deliberately not every
    ``*_CITATIONS`` table in :mod:`tengri.citations.associations`; see that
    constant's docstring for which ones are excluded and why. Lookup is
    case-insensitive to handle variation in key spelling.

    Examples
    --------
    >>> registry_names_for_citation_key("charlot_fall2000")
    ('power_law', 'cf00', 'two_component')
    >>> registry_names_for_citation_key("skirtor")
    ('skirtor', 'stalevski', 'skirtor_agnfitter', 'schartmann2005_skirtor_atten')
    """
    key_lower = key.lower()
    names: list[str] = []

    # Build reverse of NAME_TO_BIBKEY
    for name, bibkey in NAME_TO_BIBKEY.items():
        if bibkey.lower() == key_lower:
            names.append(name)

    # Build reverse of the registry-namespace association tables only.
    from tengri.citations import associations as _assoc

    for attr, _kind in _REGISTRY_NAMESPACE_TABLES:
        table = getattr(_assoc, attr, None)
        if not isinstance(table, dict):
            continue
        for name, keys in table.items():
            if isinstance(keys, list):
                if any(isinstance(k, str) and k.lower() == key_lower for k in keys):
                    names.append(name)
            elif isinstance(keys, str) and keys.lower() == key_lower:
                names.append(name)

    return tuple(dict.fromkeys(names))


def _registry_namespace_kind(name: str) -> str | None:
    """The human-readable registry namespace ``name`` belongs to, if any.

    Consulted only once a citation key has resolved to a name that is *not*
    valid at the current call site, so the error can say what kind of entry
    it actually is (e.g. ``"power_law (dust law)"``) instead of a bare,
    context-free name.

    Parameters
    ----------
    name : str
        A registry name, e.g. one returned by
        :func:`registry_names_for_citation_key`.

    Returns
    -------
    str or None
        The label from :data:`_REGISTRY_NAMESPACE_TABLES` for the first table
        whose keys contain ``name``, or ``None`` if no table does (e.g. a name
        known only through :data:`NAME_TO_BIBKEY`, which mixes several kinds
        under one map and so cannot be attributed to a single one).
    """
    from tengri.citations import associations as _assoc

    for attr, label in _REGISTRY_NAMESPACE_TABLES:
        table = getattr(_assoc, attr, None)
        if isinstance(table, dict) and name in table:
            return label
    return None


def citation_key_hint(
    unknown: str,
    valid_names: list[str],
    *,
    kind: str,
    keyword: str | None,
    fallback: str = "",
) -> str:
    """Error message fragment recognizing ``unknown`` as a citation key.

    When a user provides a citation key (like ``"charlot_fall2000"``) where a
    registry name (like ``"power_law"``) belongs, this explains the mix-up
    instead of leaving the caller with a difflib "did you mean" guess that has
    nothing close to suggest.

    Parameters
    ----------
    unknown : str
        The unknown name the user provided.
    valid_names : list of str
        Registry names valid at this call site.
    kind : str
        The kind of thing being validated (e.g., ``"dust law"``), used
        verbatim in the message.
    keyword : str or None
        The parameter keyword to show in the "Use ``keyword='...'``" remedy
        (e.g. ``"law"``, ``"law_bc"``). ``None`` suppresses that remedy clause
        entirely -- for call sites where ``unknown`` is a *dict key* name
        rather than a value assigned via ``keyword=``, for which
        "Use key='...'" does not parse as an instruction a caller could type.
    fallback : str
        The difflib "Did you mean: ...?" sentence (or an alias hint), already
        computed by the caller. Returned unchanged when ``unknown`` is not a
        citation key at all; appended after the citation note (never
        replaced) when it is a citation key but not valid here, so a real
        typo suggestion is never lost to a true-but-unhelpful citation
        observation (#2429 opus review M2).

    Returns
    -------
    str
        The hint sentence(s), or ``fallback`` unchanged when ``unknown`` is
        not a known citation key of any name other than itself.

    Examples
    --------
    >>> citation_key_hint(
    ...     "charlot_fall2000", ["calzetti", "power_law"], kind="dust law", keyword="law"
    ... )
    "That is a citation key for power_law (dust law). Use law='power_law'."
    """
    names_for_key = registry_names_for_citation_key(unknown)
    # A name that cites itself (``"cue"`` -> ``"cue"``, ``"tengri"`` ->
    # ``"tengri"``) has no *other* spelling to redirect to; reporting it as
    # "a citation key for cue" is a tautology that also throws away the
    # difflib fallback for no benefit (#2429 opus review M1).
    unknown_lower = str(unknown).lower()
    names_for_key = tuple(n for n in names_for_key if n.lower() != unknown_lower)
    if not names_for_key:
        return fallback

    valid_at_site = [n for n in names_for_key if n in valid_names]

    if valid_at_site:
        names_str = ", ".join(valid_at_site)
        if keyword is None:
            if len(valid_at_site) == 1:
                return f"That is a citation key for {valid_at_site[0]} ({kind})."
            return f"That is a citation key for {names_str} ({kind})."
        if len(valid_at_site) == 1:
            return (
                f"That is a citation key for {valid_at_site[0]} ({kind}). "
                f"Use {keyword}='{valid_at_site[0]}'."
            )
        remedy = f"Use {keyword}='{valid_at_site[0]}' or {keyword}='{valid_at_site[1]}'"
        if len(valid_at_site) > 2:
            # Redundant for exactly two names -- both are already named in
            # the remedy sentence (#2429 opus review L5).
            remedy += f" (or another from: {names_str})"
        return f"That is a citation key for {names_str} ({kind}). {remedy}."

    # This citation key exists, but not for a name valid at this site.
    first_name = names_for_key[0]
    entry_kind = _registry_namespace_kind(first_name) or "a model"
    note = f"That is a citation key for {first_name} ({entry_kind}), not a valid {kind}."
    return f"{fallback} {note}" if fallback else note

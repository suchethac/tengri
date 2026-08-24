# SPDX-License-Identifier: BSD-3-Clause
"""Per-object citation collection: the public ``collect_citations()`` API.

Inspects a Galaxy / SEDModel / Fitter / SEDModelConfig / Parameters instance and
returns the subset of the citation registry that applies to *that specific
configuration*. Users only cite what they actually use.
"""

from __future__ import annotations

import contextlib
from typing import Any

from tengri.citations.associations import (
    AGN_BLR_CITATIONS,
    AGN_NLR_CITATIONS,
    BACKEND_CITATIONS,
    CORE_CITATIONS,
    DLA_CITATIONS,
    DUST_LAW_CITATIONS,
    DUST_MODEL_CITATIONS,
    FUNCTION_CITATIONS,
    IGM_CITATIONS,
    IMF_CITATIONS,
    NEBULAR_BACKEND_CITATIONS,
    PHOTOMETRY_CONVENTION_CITATIONS,
    RADIO_CITATIONS,
    SHOCK_CITATIONS,
    SSP_CODE_CITATIONS,
    SSP_ISOCHRONE_CITATIONS,
    SSP_LIBRARY_CITATIONS,
    XRAY_CITATIONS,
)
from tengri.citations.citation import Citation
from tengri.citations.registry import cite
from tengri.citations.resolve import citation_keys_for


def _dedup(keys: list[str]) -> list[str]:
    """Return ``keys`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _citations_from_components(obj: Any) -> list[str]:
    """Walk obj's components and union their declared citations.

    Inspects whether ``obj`` exposes ``.components`` (a sequence of
    SEDComponent instances) and calls ``citations()`` on each, collecting
    bib keys. Returns a flat list; deduplication is handled upstream in
    ``_collect_keys``.

    Parameters
    ----------
    obj: Any
        Object that may expose a ``.components`` attribute (e.g.
        SEDModel, SEDModelConfig, or any container of SEDComponent
        instances). Components without a ``citations()`` method are
        skipped.

    Returns
    -------
    list of str
        Zero or more citation registry keys, in the order encountered.
        Not deduplicated; that happens in ``_collect_keys``.

    Notes
    -----
    This function is additive: it only reads from the component graph
    and does not modify any state. Designed for use inside
    :func:`_collect_keys` alongside the static association tables.
    """
    out: list[str] = []
    components = getattr(obj, "components", None)
    if components is None:
        return out
    # Iterate over the sequence (may be a list, tuple, or other iterable).
    try:
        for component in components:
            # Call citations() on each component if it exists.
            citations_fn = getattr(component, "citations", None)
            if citations_fn is not None and callable(citations_fn):
                try:
                    keys = citations_fn()
                    if keys:
                        out.extend(keys)
                except Exception:
                    # Silently skip components with broken citations() methods.
                    continue
    except TypeError:
        # components is not iterable; silently skip.
        return out
    return out


def _find_model_config(obj: Any) -> Any:
    """Return a ModelConfig-like object from ``obj``, or ``None``."""
    mc = getattr(obj, "model_config", None)
    if mc is not None:
        return mc
    # obj itself might already be a ModelConfig (has dust/nebular/igm fields).
    if any(hasattr(obj, f) for f in ("dust", "nebular", "igm", "sfh")):
        return obj
    return None


def _backend_from(obj: Any) -> str | None:
    """Extract the inference backend name from ``obj``, if known."""
    # ``Fitter.run`` stamps the canonical registry key on the result. Prefer it:
    # ``Posterior.method`` is a human-readable display string ("NUTS (BlackJAX)",
    # "MAP (ADAM, 5 restarts)") built with an f-string, so it never matches a
    # ``BACKEND_CITATIONS`` key and must not be parsed back into one.
    bk = getattr(obj, "_backend_key", None)
    if bk:
        return str(bk)
    # Galaxy sets _last_backend after fit().
    bk = getattr(obj, "_last_backend", None)
    if bk:
        return str(bk)
    # A Fitter may expose .backend / .backend_name.
    for attr in ("backend", "backend_name"):
        bk = getattr(obj, attr, None)
        if bk:
            return str(bk)
    # A Posterior / FitResult might too.
    inner = getattr(obj, "inner", None)
    if inner is not None:
        for attr in ("backend", "backend_name"):
            bk = getattr(inner, attr, None)
            if bk:
                return str(bk)
    return None


def _ssp_provenance_keys(ssp: Any) -> list[str]:
    """Citation keys inferred from an ``SSPData`` provenance (name tokens + IMF).

    The grid filename follows ``<sps_code>_<isochrone>_<library>_<imf>`` (e.g.
    ``fsps_prsc_miles_chabrier`` → FSPS + PARSEC + MILES + Chabrier). We scan
    the underscore/dash/dot-delimited tokens of ``SSPData.source`` against the
    SPS-code, isochrone, and spectral-library tables, and read the IMF from
    ``SSPData.imf`` (falling back to a token).

    If neither an isochrone (stellar-evolution library) nor a spectral library
    (stellar atmospheres) can be inferred: i.e. the filename does not match the
    convention; a provenance warning is emitted so the user supplies those
    citations manually.
    """
    import re
    import warnings

    source = str(getattr(ssp, "source", "") or "")
    tokens = [t.lower() for t in re.split(r"[_\-.]+", source) if t]

    def _match(table: dict[str, list[str]]) -> tuple[list[str], bool]:
        hits: list[str] = []
        recognized = False
        for t in tokens:
            if t in table:
                recognized = True
                hits.extend(table[t])
        return hits, recognized

    out: list[str] = []
    code_keys, _ = _match(SSP_CODE_CITATIONS)
    iso_keys, iso_ok = _match(SSP_ISOCHRONE_CITATIONS)
    lib_keys, lib_ok = _match(SSP_LIBRARY_CITATIONS)
    out.extend(code_keys)
    out.extend(iso_keys)
    out.extend(lib_keys)

    imf = str(getattr(ssp, "imf", "") or "").lower()
    if imf in IMF_CITATIONS:
        out.extend(IMF_CITATIONS[imf])
    else:
        imf_keys, _ = _match(IMF_CITATIONS)
        out.extend(imf_keys)

    # ``wNE`` grids carry baked-in nebular emission (Byler+2017 Cloudy grids).
    if "wne" in tokens:
        out.extend(["byler2017", "cloudy"])

    # Warn when the stellar-evolution isochrone set and/or the spectral
    # (atmosphere) library cannot be inferred from the filename.
    if not iso_ok or not lib_ok:
        missing = []
        if not iso_ok:
            missing.append("stellar-evolution isochrone set")
        if not lib_ok:
            missing.append("spectral/atmosphere library")
        warnings.warn(
            f"Could not infer the {' and '.join(missing)} from SSP source "
            f"{source!r}: expected the <code>_<isochrone>_<library>_<imf> "
            f"convention (e.g. 'fsps_prsc_miles_chabrier'). Provenance citations "
            f"for these ingredients may be missing; add them manually.",
            stacklevel=3,
        )
    return out


def _citable_chain(obj: Any) -> list[Any]:
    """Return ``obj`` followed by every object it delegates its configuration to.

    The documented per-fit surface is ``print_components_bibtex(result)``, but a
    :class:`~tengri.inference.posterior.Posterior` stores its model under
    ``_model`` (not ``model``), and a
    :class:`~tengri.forward.forward_model.ForwardModel` hides the SED behind
    ``_inner_sed_for_delegation()``. Neither hop is reachable through the plain
    ``getattr(obj, "model", None)`` probe the collectors use, so citing a *result*
    silently degraded to the three core keys while citing the *model* returned the
    full set.

    The configuration is spread across the chain rather than concentrated at
    either end; the sampler is known only to the result, the photometry only to
    the forward model, the physics components only to the SED: so callers union
    over the whole chain instead of resolving to a single root.

    Parameters
    ----------
    obj: Any
        Any citable object (``Posterior``, ``ForwardModel``, ``SEDModel``, ...).

    Returns
    -------
    list
        ``obj`` first, then each delegation target reachable from it, visited
        once. Cycles and repeated references are collapsed by identity.

    Notes
    -----
    **JIT-compatible**: no; pure Python attribute introspection.
    """
    seen: set[int] = set()
    chain: list[Any] = []
    queue: list[Any] = [obj]
    while queue:
        cur = queue.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        chain.append(cur)
        # Posterior -> _model; Fitter/others -> model; wrappers -> sed.
        for attr in ("_model", "model", "sed", "_sed"):
            queue.append(getattr(cur, attr, None))
        # ForwardModel exposes its wrapped SED only through this accessor.
        inner = getattr(cur, "_inner_sed_for_delegation", None)
        if callable(inner):
            # The accessor raises deliberately on a multi-population forward,
            # where ``populations[0]`` would be an arbitrary pick: that refusal
            # is what this guard is for, and it arrives as ValueError/TypeError.
            # Catching everything meant a citation silently going missing looked
            # identical to a model that legitimately has none, and a bibliography
            # that quietly drops a reference is worse than one that fails to
            # build.
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                queue.append(inner())
    return chain


def _collect_keys(obj: Any, *, include_backend: bool = True) -> list[str]:
    """Return the flat, ordered, deduplicated registry keys for ``obj``.

    Unions :func:`_collect_keys_for_one` over :func:`_citable_chain` so that
    citing a fit result reports everything that actually ran, not just the
    core keys reachable from the result object itself.
    """
    keys: list[str] = []
    for depth, node in enumerate(_citable_chain(obj)):
        # The ``@cites`` sweep below reads *every* public attribute of its
        # target. That is safe on the object the caller handed us, but forcing
        # it on delegated nodes would touch every property of the wrapped
        # SEDModel: including ones that compile or allocate; purely to look
        # for citation annotations. Restrict it to the root; the delegated
        # nodes contribute through the targeted extractors, which is where the
        # component, SSP and backend keys come from anyway.
        keys.extend(
            _collect_keys_for_one(
                node, include_backend=include_backend, scan_attributes=depth == 0
            )
        )
    return _dedup(keys)


def _collect_keys_for_one(
    obj: Any, *, include_backend: bool = True, scan_attributes: bool = True
) -> list[str]:
    """Return the registry keys contributed by ``obj`` alone (no delegation).

    Parameters
    ----------
    scan_attributes: bool, optional
        Sweep every public attribute for ``@cites`` annotations. Default
        ``True``. Callers set it ``False`` for delegated nodes so the sweep
        never forces attribute evaluation on an object the user did not pass.
    """
    keys: list[str] = list(CORE_CITATIONS)

    mc = _find_model_config(obj)
    if mc is not None:
        # Dust model wrapper (e.g. two_component → Charlot & Fall).
        dust = getattr(mc, "dust", None)
        if dust is not None:
            model = getattr(dust, "model", None)
            if model in DUST_MODEL_CITATIONS:
                keys.extend(DUST_MODEL_CITATIONS[model])
            # Per-component laws.
            for law_attr in ("law", "law_bc", "law_diff"):
                law = getattr(dust, law_attr, None)
                if law in DUST_LAW_CITATIONS:
                    keys.extend(DUST_LAW_CITATIONS[law])

        # Nebular backend.
        neb = getattr(mc, "nebular", None)
        if neb is not None:
            backend = getattr(neb, "backend", None)
            if backend in NEBULAR_BACKEND_CITATIONS:
                keys.extend(NEBULAR_BACKEND_CITATIONS[backend])

        # IGM model.
        igm = getattr(mc, "igm", None)
        if igm is not None:
            model = getattr(igm, "model", None)
            if model in IGM_CITATIONS:
                keys.extend(IGM_CITATIONS[model])

        # DLA foreground absorber (Tepper-García Voigt profile), independent of
        # the mean-IGM model. Flagged via ``dla`` on the igm config or the
        # top-level model config.
        uses_dla = bool(getattr(igm, "dla", False)) or bool(getattr(mc, "dla", False))
        if uses_dla:
            keys.extend(DLA_CITATIONS)

    # Photometry: AB-system + filter-convolution convention (ADR-0017). Any
    # run that produces broadband fluxes cites the AB foundations; the
    # per-convention entry cites the code each convention reproduces.
    obs = getattr(obj, "observation", None) or getattr(
        getattr(obj, "model", None), "observation", None
    )
    phot = getattr(obs, "photometry", None)
    if phot is not None:
        keys.extend(PHOTOMETRY_CONVENTION_CITATIONS["core"])
        conv = str(getattr(phot, "convention", "bessell"))
        if conv in PHOTOMETRY_CONVENTION_CITATIONS:
            keys.extend(PHOTOMETRY_CONVENTION_CITATIONS[conv])

    # SSP provenance: SPS code + isochrone + spectral library + IMF, inferred
    # from the grid filename tokens (warns if the library/atmosphere is
    # unrecognizable).
    ssp = getattr(obj, "ssp_data", None) or getattr(getattr(obj, "model", None), "ssp_data", None)
    if ssp is not None:
        keys.extend(_ssp_provenance_keys(ssp))

    # Precomputation method: WavePrecomp filter pre-integration cites
    # Zacharegkas+2025 (DSPS is already a core citation).
    approx = getattr(obj, "_approx", None) or getattr(getattr(obj, "model", None), "_approx", None)
    if isinstance(approx, dict) and approx.get("wave_precomp"):
        keys.append("zacharegkas2025")

    # Inference backend.
    if include_backend:
        bk = _backend_from(obj)
        if bk is not None and bk in BACKEND_CITATIONS:
            keys.extend(BACKEND_CITATIONS[bk])

    # Function-level annotations attached via @cites decorator, if any
    # function objects are exposed on obj (e.g. obj.run or obj.forward).
    for attr_name in dir(obj) if scan_attributes else ():
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            continue
        ckeys = getattr(attr, "_tengri_cites", None)
        if ckeys:
            keys.extend(ckeys)
        # Also check the static FUNCTION_CITATIONS dict by qualname.
        mod = getattr(attr, "__module__", None)
        qn = getattr(attr, "__qualname__", None)
        if mod and qn:
            fq = f"{mod}.{qn}"
            if fq in FUNCTION_CITATIONS:
                keys.extend(FUNCTION_CITATIONS[fq])

    # Live registry walk (Parameters / SEDModel / Posterior): picks up
    # citations registered via ``@register_agn_model("…", citation=…)``,
    # ``@register_dust_law``, SFH ``_register``, etc.  This bridges the
    # static association tables above (which only know the canonical
    # alternatives) with whatever is in the registry RIGHT NOW so that
    # contributor models also appear in ``print_citations(model)``.
    keys.extend(_keys_from_live_registry(obj))

    # Component-graph walk: collect citations declared by each SEDComponent
    # in the object's component chain. Additive: the static tables and
    # function annotations are still the primary source.
    keys.extend(_citations_from_components(obj))

    return _dedup(keys)


def _keys_from_live_registry(obj: Any) -> list[str]:
    """Walk obj's structural choices and map to bibtex keys.

    Reads ``mean_sfh_type``, ``agn_model``, ``dust_emission``, ``dust_law``,
    ``dust_law_bc``, ``dust_law_diff`` off the spec; falls through silently
    for objects that don't expose them.
    """
    out: list[str] = []
    spec = getattr(obj, "spec", obj)

    def _push(name):
        # citation_keys_for owns the name → key mapping for BOTH public
        # surfaces, including the "_tabulated" suffix strip this used to do
        # locally. Two maps is how collect_citations and
        # print_components_bibtex came to disagree on 21 component names.
        out.extend(citation_keys_for(str(name).lower() if name else name))

    sfh_types = getattr(spec, "mean_sfh_type", None)
    if isinstance(sfh_types, str):
        sfh_types = [sfh_types]
    if sfh_types:
        for s in sfh_types:
            _push(s)

    _push(getattr(spec, "agn_model", None))

    # Composable AGN NLR/BLR blocks map to a LIST of keys (the Synthesizer
    # variants cite BOTH Synthesizer papers: Lovell 2025 + Roper 2026).
    for attr, table in (
        ("agn_nlr_block", AGN_NLR_CITATIONS),
        ("agn_blr_block", AGN_BLR_CITATIONS),
    ):
        block = getattr(spec, attr, None)
        if block:
            out.extend(table.get(str(block).lower(), []))

    _push(getattr(spec, "dust_model", None))  # e.g. wg00 → witt_gordon2000
    _push(getattr(spec, "dust_emission", None))
    _push(getattr(spec, "dust_law", None))
    _push(getattr(spec, "dust_law_bc", None))
    _push(getattr(spec, "dust_law_diff", None))

    # IGM mean-model + DLA (observed-frame absorption). The config-based
    # ``_collect_keys`` path only fires when ``_find_model_config`` resolves a
    # ModelConfig; a live SEDModel exposes these on the spec, so map them here
    # too (mean-IGM inoue/madau/meiksin06/asada25 + DLA cite their papers).
    if getattr(spec, "apply_igm", False):
        igm_model = getattr(spec, "igm_model", None)
        if igm_model in IGM_CITATIONS:
            out.extend(IGM_CITATIONS[igm_model])
    if getattr(spec, "dla", False):
        out.extend(DLA_CITATIONS)

    # Nebular backend, X-ray, radio, shock: read off the live model (``obj``);
    # these are not exposed as flat spec attributes the way the mean-IGM model
    # is. The config-based ``_collect_keys`` path handles nebular only when
    # ``_find_model_config`` resolves a ModelConfig (it returns None for a live
    # SEDModel), so map them here too. See #938.
    backend = getattr(obj, "_nebular_backend", None)
    backend_name = getattr(backend, "name", None)
    if backend_name in NEBULAR_BACKEND_CITATIONS:
        out.extend(NEBULAR_BACKEND_CITATIONS[backend_name])

    if getattr(obj, "_uses_xray", False):
        xray_model = getattr(spec, "xray_model", None)
        out.extend(XRAY_CITATIONS.get(xray_model, []))

    if getattr(obj, "_uses_radio", False):
        out.extend(RADIO_CITATIONS)

    if getattr(obj, "_uses_shock", False):
        # The canonical shock component is the MAPPINGS V grid.
        out.extend(SHOCK_CITATIONS["mappings"])

    # Inference method (Posterior exposes .method)
    method = getattr(obj, "method", None)
    if method:
        _push(method)

    return out


def collect_citations(
    obj: Any,
    *,
    include_backend: bool = True,
) -> list[Citation]:
    """Return citations for everything ``obj`` is configured to use.

    Inspects the object's model configuration (dust law, nebular backend,
    IGM model, AGN sub-components), the last inference backend if a fit has
    been run, and any ``@cites``-annotated callables exposed as attributes.
    Returns *only* the citations that apply to this specific run: no
    dust-emission citation if dust emission is off, no Cue citation if
    nebular emission is off, no NUTS citation if MAP was the backend.

    Parameters
    ----------
    obj: Galaxy | SEDModel | Fitter | SEDModelConfig | FitResult | Any
        Object whose configuration is inspected. If ``obj`` exposes a
        ``bibliography`` attribute, its contents are returned verbatim; for
        plain objects the citations are reconstructed from ``model_config``.
        Unknown objects yield only the core citations (``tengri``, ``jax``,
        ``dsps``).
    include_backend: bool, optional
        Include citations for the last inference backend used. Default
        ``True``. Pass ``False`` to get citations for the model
        configuration alone, independent of how it will be fit.

    Returns
    -------
    list of Citation
        Deduplicated, registry-resolved records, ordered by first use.
        Keys that are not in :data:`tengri.citations.registry.REGISTRY` are
        silently skipped rather than raising.

    Raises
    ------
    No exceptions are raised for missing citation keys. A missing ``obj``
    attribute is also not an error; the function degrades gracefully.

    Notes
    -----
    **JIT-compatible**: no, pure Python, does attribute introspection.

    The static association tables live in
    :mod:`tengri.citations.associations`:

    - ``DUST_LAW_CITATIONS`` / ``DUST_MODEL_CITATIONS`` / ``DUST_EMISSION_CITATIONS``
    - ``NEBULAR_BACKEND_CITATIONS``
    - ``IGM_CITATIONS``
    - ``AGN_DISC_CITATIONS`` / ``AGN_TORUS_CITATIONS`` / ``AGN_BLR_CITATIONS``
    - ``BACKEND_CITATIONS``
    - ``FUNCTION_CITATIONS`` (populated by the :func:`cites` decorator)

    Adding a new component value? Extend the corresponding table so this
    function picks it up automatically.

    See Also
    --------
    tengri.citations.Bibliography.from_object: the class method this wraps.
    citations_report: formatted grouped text version.
    citations_bibtex: BibTeX version of the same list.

    Examples
    --------
    Galaxy-level; only cites what this galaxy uses:

    >>> import numpy as np, tengri as tg
    >>> g = tg.Galaxy.from_arrays(
    ...     filters=["sdss_u", "sdss_g", "sdss_r"],
    ...     flux=np.array([1e-28, 2e-28, 3e-28]),
    ...     flux_err=np.array([1e-29] * 3),
    ...     redshift=0.1,
    ...     ssp_path="data/ssp.h5",
    ...     preset="starforming",
    ... )  # doctest: +SKIP
    >>> [c.key for c in tg.collect_citations(g)]  # doctest: +SKIP
    ['tengri', 'jax', 'dsps', 'charlot_fall2000', 'calzetti2000']

    Config-only, no backend:

    >>> from tengri.config.settings import SEDModelConfig, DustConfig
    >>> mc = SEDModelConfig(dust=DustConfig(law_bc="kriek_conroy"))
    >>> [c.key for c in tg.collect_citations(mc, include_backend=False)]
    ['tengri', 'jax', 'dsps', 'charlot_fall2000', 'kriek_conroy2013']
    """
    keys = _collect_keys(obj, include_backend=include_backend)
    out: list[Citation] = []
    for k in keys:
        try:
            out.append(cite(k))
        except KeyError:
            continue
    return out


def citations_report(obj: Any, *, include_backend: bool = True) -> str:
    """Build a multi-line human-readable citation list for ``obj``.

    Parameters
    ----------
    obj: Any
        Any object understood by :func:`collect_citations` (Galaxy,
        SEDModelConfig, Fitter, FitResult, ...).
    include_backend: bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    str
        Multi-line report. One paragraph per citation: short name and role
        on the first line, title on the second, DOI / arXiv / upstream code
        links on the third. Intended for terminal or notebook display.

    Notes
    -----
    **JIT-compatible**: no; pure Python string formatting.

    The default report is flat and numbered. For a category-grouped layout
    (Stellar populations / Dust / Nebular / Inference …) use
    :meth:`tengri.citations.Bibliography.report` directly.

    See Also
    --------
    print_citations: same output, printed to stdout.
    citations_bibtex: BibTeX-formatted companion.

    Examples
    --------
    >>> from tengri.config.settings import SEDModelConfig, DustConfig
    >>> mc = SEDModelConfig(dust=DustConfig(law_bc="calzetti"))
    >>> report = tg.citations_report(mc, include_backend=False)  # doctest: +SKIP
    >>> "Calzetti" in report  # doctest: +SKIP
    True
    """
    cites = collect_citations(obj, include_backend=include_backend)
    if not cites:
        return "No citations inferred for this object.\n"

    lines: list[str] = [
        "Please cite the following when publishing results that use tengri:",
        "",
    ]
    for i, c in enumerate(cites, 1):
        lines.append(f"  [{i}] {c.short} ; {c.role}")
        if c.title:
            lines.append(f"       {c.title}")
        link_bits: list[str] = []
        if c.doi:
            link_bits.append(f"DOI: {c.doi}")
        if c.arxiv:
            link_bits.append(f"arXiv: {c.arxiv}")
        if c.upstream_code:
            link_bits.append(f"code: {c.upstream_code}")
        if link_bits:
            lines.append("       " + "   ".join(link_bits))
        lines.append("")
    return "\n".join(lines)


def citations_bibtex(obj: Any, *, include_backend: bool = True) -> str:
    """Build a BibTeX block containing every citation applicable to ``obj``.

    Parameters
    ----------
    obj: Any
        Any object understood by :func:`collect_citations`.
    include_backend: bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    str
        Concatenated BibTeX entries separated by blank lines. Ready to
        paste into a paper's ``.bib`` file.

    Notes
    -----
    **JIT-compatible**: no; pure Python string formatting.

    See Also
    --------
    tengri.citations.Citation.to_bibtex: single-entry BibTeX formatter.
    print_bibtex: same output printed to stdout.

    Examples
    --------
    >>> from tengri.config.settings import SEDModelConfig, DustConfig
    >>> mc = SEDModelConfig(dust=DustConfig(law_bc="calzetti"))
    >>> bibtex = tg.citations_bibtex(mc, include_backend=False)  # doctest: +SKIP
    >>> "@article{Calzetti_2000" in bibtex  # doctest: +SKIP
    True
    """
    cites = collect_citations(obj, include_backend=include_backend)
    return "\n\n".join(c.to_bibtex() for c in cites)


def print_citations(obj: Any, *, include_backend: bool = True) -> None:
    """Print the human-readable citation report for ``obj`` to stdout.

    Convenience wrapper around :func:`citations_report`.

    Parameters
    ----------
    obj: Any
        Any object understood by :func:`collect_citations`.
    include_backend: bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no; performs I/O.

    Examples
    --------
    >>> import numpy as np, tengri as tg
    >>> g = tg.Galaxy.from_arrays(..., preset="starforming")  # doctest: +SKIP
    >>> tg.print_citations(g)  # doctest: +SKIP
    Please cite the following when publishing results that use tengri:
      [1] Cooray et al. (2026, Paper I); SED fitting framework ...
      ...
    """
    print(citations_report(obj, include_backend=include_backend))


def print_bibtex(obj: Any, *, include_backend: bool = True) -> None:
    """Print the BibTeX block for every citation applicable to ``obj``.

    Convenience wrapper around :func:`citations_bibtex`.

    Parameters
    ----------
    obj: Any
        Any object understood by :func:`collect_citations`.
    include_backend: bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no; performs I/O.

    Examples
    --------
    >>> import tengri as tg
    >>> tg.print_bibtex(my_galaxy)  # doctest: +SKIP
    @article{Cooray_2026, ... }
    @article{Bradbury2018_JAX, ... }
    ...
    """
    print(citations_bibtex(obj, include_backend=include_backend))


__all__ = [
    "citations_bibtex",
    "citations_report",
    "collect_citations",
    "print_bibtex",
    "print_citations",
]

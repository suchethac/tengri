"""Per-object citation collection — the public ``collect_citations()`` API.

Inspects a Galaxy / SEDModel / Fitter / ModelConfig / Parameters instance and
returns the subset of the citation registry that applies to *that specific
configuration*. Users only cite what they actually use.
"""

from __future__ import annotations

from typing import Any

from tengri.citations.associations import (
    BACKEND_CITATIONS,
    CORE_CITATIONS,
    DUST_LAW_CITATIONS,
    DUST_MODEL_CITATIONS,
    FUNCTION_CITATIONS,
    IGM_CITATIONS,
    NEBULAR_BACKEND_CITATIONS,
)
from tengri.citations.citation import Citation
from tengri.citations.registry import cite


def _dedup(keys: list[str]) -> list[str]:
    """Return ``keys`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
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


def _collect_keys(obj: Any, *, include_backend: bool = True) -> list[str]:
    """Return the flat, ordered, deduplicated list of registry keys for ``obj``."""
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

    # Inference backend.
    if include_backend:
        bk = _backend_from(obj)
        if bk is not None and bk in BACKEND_CITATIONS:
            keys.extend(BACKEND_CITATIONS[bk])

    # Function-level annotations attached via @cites decorator, if any
    # function objects are exposed on obj (e.g. obj.run or obj.forward).
    for attr_name in dir(obj):
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

    return _dedup(keys)


def collect_citations(
    obj: Any,
    *,
    include_backend: bool = True,
) -> list[Citation]:
    """Return citations for everything ``obj`` is configured to use.

    Inspects the object's model configuration (dust law, nebular backend,
    IGM model, AGN sub-components), the last inference backend if a fit has
    been run, and any ``@cites``-annotated callables exposed as attributes.
    Returns *only* the citations that apply to this specific run — no
    dust-emission citation if dust emission is off, no Cue citation if
    nebular emission is off, no NUTS citation if MAP was the backend.

    Parameters
    ----------
    obj : Galaxy | SEDModel | Fitter | ModelConfig | FitResult | Any
        Object whose configuration is inspected. If ``obj`` exposes a
        ``bibliography`` attribute, its contents are returned verbatim; for
        plain objects the citations are reconstructed from ``model_config``.
        Unknown objects yield only the core citations (``tengri``, ``jax``,
        ``dsps``).
    include_backend : bool, optional
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
    attribute is also not an error — the function degrades gracefully.

    Notes
    -----
    **JIT-compatible**: no — pure Python, does attribute introspection.

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
    tengri.citations.Bibliography.from_object : the class method this wraps.
    citations_report : formatted grouped text version.
    citations_bibtex : BibTeX version of the same list.

    Examples
    --------
    Galaxy-level — only cites what this galaxy uses:

    >>> import numpy as np, tengri as tg
    >>> g = tg.Galaxy.from_arrays(
    ...     filters=["sdss_u", "sdss_g", "sdss_r"],
    ...     flux=np.array([1e-28, 2e-28, 3e-28]),
    ...     flux_err=np.array([1e-29]*3),
    ...     redshift=0.1, ssp_path="data/ssp.h5",
    ...     preset="starforming",
    ... )  # doctest: +SKIP
    >>> [c.key for c in tg.collect_citations(g)]  # doctest: +SKIP
    ['tengri', 'jax', 'dsps', 'charlot_fall2000', 'calzetti2000']

    Config-only, no backend:

    >>> from tengri.config.settings import ModelConfig, DustConfig
    >>> mc = ModelConfig(dust=DustConfig(law_bc="kriek_conroy"))
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
    obj : Any
        Any object understood by :func:`collect_citations` (Galaxy,
        ModelConfig, Fitter, FitResult, ...).
    include_backend : bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    str
        Multi-line report. One paragraph per citation: short name and role
        on the first line, title on the second, DOI / arXiv / upstream code
        links on the third. Intended for terminal or notebook display.

    Notes
    -----
    **JIT-compatible**: no — pure Python string formatting.

    The default report is flat and numbered. For a category-grouped layout
    (Stellar populations / Dust / Nebular / Inference …) use
    :meth:`tengri.citations.Bibliography.report` directly.

    See Also
    --------
    print_citations : same output, printed to stdout.
    citations_bibtex : BibTeX-formatted companion.

    Examples
    --------
    >>> from tengri.config.settings import ModelConfig, DustConfig
    >>> mc = ModelConfig(dust=DustConfig(law_bc="calzetti"))
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
        lines.append(f"  [{i}] {c.short}  —  {c.role}")
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
    obj : Any
        Any object understood by :func:`collect_citations`.
    include_backend : bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    str
        Concatenated BibTeX entries separated by blank lines. Ready to
        paste into a paper's ``.bib`` file.

    Notes
    -----
    **JIT-compatible**: no — pure Python string formatting.

    See Also
    --------
    tengri.citations.Citation.to_bibtex : single-entry BibTeX formatter.
    print_bibtex : same output printed to stdout.

    Examples
    --------
    >>> from tengri.config.settings import ModelConfig, DustConfig
    >>> mc = ModelConfig(dust=DustConfig(law_bc="calzetti"))
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
    obj : Any
        Any object understood by :func:`collect_citations`.
    include_backend : bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no — performs I/O.

    Examples
    --------
    >>> import numpy as np, tengri as tg
    >>> g = tg.Galaxy.from_arrays(..., preset="starforming")  # doctest: +SKIP
    >>> tg.print_citations(g)  # doctest: +SKIP
    Please cite the following when publishing results that use tengri:
      [1] Cooray et al. (2026, Paper I) — SED fitting framework ...
      ...
    """
    print(citations_report(obj, include_backend=include_backend))


def print_bibtex(obj: Any, *, include_backend: bool = True) -> None:
    """Print the BibTeX block for every citation applicable to ``obj``.

    Convenience wrapper around :func:`citations_bibtex`.

    Parameters
    ----------
    obj : Any
        Any object understood by :func:`collect_citations`.
    include_backend : bool, optional
        Include inference-backend citations. Default ``True``.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no — performs I/O.

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

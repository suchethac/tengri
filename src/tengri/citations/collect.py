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

    Parameters
    ----------
    obj : Galaxy | SEDModel | Fitter | ModelConfig | FitResult | Any
        Any object whose configuration can be inspected for dust, nebular,
        IGM, and inference-backend choices. Unknown objects return just
        the core citations (``tengri``, ``jax``, ``dsps``).
    include_backend : bool
        If ``True`` (default), also include citations for the last inference
        backend used (``_last_backend`` on Galaxy, or ``backend_name`` on
        Fitter/FitResult). Set to ``False`` to get citations for the model
        configuration alone, independent of how it will be fit.

    Returns
    -------
    list[Citation]
        Deduplicated, registry-resolved citations. Any key that is not
        registered in ``references.bib`` is silently skipped.

    Examples
    --------
    >>> import tengri as tg
    >>> g = tg.Galaxy.from_arrays(..., preset="starforming")
    >>> cites = tg.collect_citations(g)
    >>> [c.short for c in cites]
    ['Cooray et al. (2026, Paper I)', 'Bradbury et al. (2018)', ...]
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
    """Return a multi-line human-readable citation list for ``obj``.

    The format is designed for a terminal / notebook — one paragraph per
    citation with short name, full title, and a DOI/arXiv link if available.
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
    """Return a BibTeX block containing every citation applicable to ``obj``."""
    cites = collect_citations(obj, include_backend=include_backend)
    return "\n\n".join(c.to_bibtex() for c in cites)


def print_citations(obj: Any, *, include_backend: bool = True) -> None:
    """Print the human-readable citation report for ``obj``."""
    print(citations_report(obj, include_backend=include_backend))


def print_bibtex(obj: Any, *, include_backend: bool = True) -> None:
    """Print the BibTeX block for every citation applicable to ``obj``."""
    print(citations_bibtex(obj, include_backend=include_backend))


__all__ = [
    "citations_bibtex",
    "citations_report",
    "collect_citations",
    "print_bibtex",
    "print_citations",
]

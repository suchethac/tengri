"""``Bibliography`` — a live citation container carried by tengri objects.

Each object that performs scientific work (``Galaxy``, ``SEDModel``, a future
``Fitter`` wrapper, ``FitResult``) owns a :class:`Bibliography` instance.
Components register themselves into that bibliography at construction time
or when they are invoked. Reading the bibliography gives the user exactly
the citations that *their* configured run requires — no more, no less.

Basic usage
-----------

>>> import tengri as tg
>>> g = tg.Galaxy.from_arrays(..., preset="starforming")
>>> print(g.bibliography.report())       # human-readable list
>>> print(g.bibliography.to_bibtex())    # copy-paste BibTeX block
>>> g.bibliography.keys                  # ['tengri', 'jax', 'dsps', ...]

The container is intentionally small: it is an ordered, deduplicated list
of registry keys plus helpers for human / BibTeX output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tengri.citations.citation import Citation


@dataclass
class Bibliography:
    """Ordered, deduplicated set of citation keys for a specific run.

    Attributes
    ----------
    keys : list[str]
        Registry keys (e.g. ``"calzetti2000"``), in insertion order, without
        duplicates.
    source : str
        Human-readable label describing what the bibliography is tied to
        (e.g. ``"Galaxy(preset=starforming)"``) — used in ``report()``.
    """

    keys: list[str] = field(default_factory=list)
    source: str = ""
    _seen: set[str] = field(default_factory=set, repr=False)

    # -- mutation -----------------------------------------------------------

    def add(self, *keys: str) -> Bibliography:
        """Add one or more citation keys. Duplicates are silently ignored.

        Returns ``self`` for chaining.
        """
        for k in keys:
            if not k:
                continue
            if k not in self._seen:
                self._seen.add(k)
                self.keys.append(k)
        return self

    def extend(self, other: Bibliography | Iterable[str]) -> Bibliography:
        """Absorb keys from another Bibliography or iterable of keys."""
        if isinstance(other, Bibliography):
            self.add(*other.keys)
        else:
            self.add(*list(other))
        return self

    def remove(self, key: str) -> None:
        """Remove a key (silently no-op if not present)."""
        if key in self._seen:
            self._seen.discard(key)
            self.keys = [k for k in self.keys if k != key]

    # -- queries ------------------------------------------------------------

    def __contains__(self, key: object) -> bool:
        return key in self._seen

    def __iter__(self):
        """Iterate over :class:`Citation` records (missing keys skipped)."""
        from tengri.citations.registry import cite

        for k in self.keys:
            try:
                yield cite(k)
            except KeyError:
                continue

    def __len__(self) -> int:
        return len(self.keys)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def to_list(self) -> list[Citation]:
        """Return a list of :class:`Citation` records (missing keys skipped)."""
        return list(self)

    def by_category(self) -> dict[str, list[Citation]]:
        """Group citations by :attr:`Citation.category`, preserving insertion order."""
        groups: dict[str, list[Citation]] = {}
        for c in self:
            groups.setdefault(c.category or "other", []).append(c)
        return groups

    # -- serialization ------------------------------------------------------

    # Stable print order for categories (others appended alphabetically).
    _CATEGORY_ORDER: ClassVar[tuple[str, ...]] = (
        "framework",
        "ssp",
        "sfh",
        "dust_attenuation",
        "dust_emission",
        "nebular",
        "agn",
        "igm",
        "preprocessing",
        "inference",
        "reference_code",
        "other",
    )

    _CATEGORY_HEADINGS: ClassVar[dict[str, str]] = {
        "framework": "Framework & theory",
        "ssp": "Stellar populations",
        "sfh": "Star formation history",
        "dust_attenuation": "Dust attenuation",
        "dust_emission": "Dust emission",
        "nebular": "Nebular emission",
        "agn": "AGN",
        "igm": "Intergalactic medium",
        "preprocessing": "Preprocessing",
        "inference": "Inference",
        "reference_code": "Peer codes (for comparison / porting credit)",
        "other": "Other",
    }

    def report(self, *, group_by_category: bool = True) -> str:
        """Human-readable multi-line report.

        Parameters
        ----------
        group_by_category : bool
            If True (default), citations are grouped under category headings
            (Stellar populations / Dust / Nebular / Inference …). If False,
            print one flat numbered list.
        """
        cites = self.to_list()
        if not cites:
            return "No citations registered.\n"
        header = "Please cite the following when publishing results:"
        if self.source:
            header += f"  ({self.source})"
        lines = [header, ""]

        if group_by_category:
            groups = self.by_category()
            ordered: list[str] = [c for c in self._CATEGORY_ORDER if c in groups]
            ordered += sorted(c for c in groups if c not in self._CATEGORY_ORDER)
            for cat in ordered:
                heading = self._CATEGORY_HEADINGS.get(cat, cat.replace("_", " ").title())
                lines.append(f"  ── {heading} " + "─" * max(2, 50 - len(heading)))
                for c in groups[cat]:
                    lines.extend(self._render_citation(c))
                lines.append("")
        else:
            for i, c in enumerate(cites, 1):
                lines.extend(self._render_citation(c, index=i))
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_citation(c: Citation, *, index: int | None = None) -> list[str]:
        """Render a single citation into 2-3 output lines."""
        prefix = f"  [{index}] " if index is not None else "    • "
        out = [f"{prefix}{c.short}  —  {c.role}"]
        if c.title:
            out.append(f"        {c.title}")
        bits: list[str] = []
        if c.doi:
            bits.append(f"DOI: {c.doi}")
        if c.arxiv:
            bits.append(f"arXiv: {c.arxiv}")
        if c.upstream_code:
            bits.append(f"code: {c.upstream_code}")
        if bits:
            out.append("        " + "   ".join(bits))
        return out

    def to_bibtex(self) -> str:
        """BibTeX block of every citation, ready to paste into a .bib file."""
        return "\n\n".join(c.to_bibtex() for c in self)

    def __str__(self) -> str:  # short summary, e.g. in repr-heavy REPLs
        if not self.keys:
            return "Bibliography(empty)"
        preview = ", ".join(self.keys[:4])
        more = "" if len(self.keys) <= 4 else f" (+{len(self.keys) - 4} more)"
        return f"Bibliography({len(self.keys)}: {preview}{more})"

    # -- construction helpers ----------------------------------------------

    @classmethod
    def from_object(cls, obj: Any, *, include_backend: bool = True) -> Bibliography:
        """Build a Bibliography by inspecting any tengri object.

        Looks for the ``bibliography`` attribute first (in which case it's
        returned verbatim), otherwise falls back to inspecting
        ``model_config``, the last inference backend used, and any
        ``@cites``-annotated callables found on the object.
        """
        existing = getattr(obj, "bibliography", None)
        if isinstance(existing, Bibliography):
            return existing

        # Determine source label from class name if none provided.
        source = type(obj).__name__
        preset = getattr(obj, "preset_name", None)
        if preset:
            source = f"{source}(preset={preset})"

        bib = cls.from_config(getattr(obj, "model_config", None), source=source)

        if include_backend:
            backend = (
                getattr(obj, "_last_backend", None)
                or getattr(obj, "backend", None)
                or getattr(obj, "backend_name", None)
            )
            bib.add_backend(backend)

        # Function-level annotations via @cites decorator.
        from tengri.citations.associations import FUNCTION_CITATIONS

        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(obj, attr_name)
            except Exception:
                continue
            for key in getattr(attr, "_tengri_cites", ()):
                bib.add(key)
            mod = getattr(attr, "__module__", None)
            qn = getattr(attr, "__qualname__", None)
            if mod and qn:
                fq = f"{mod}.{qn}"
                if fq in FUNCTION_CITATIONS:
                    bib.add(*FUNCTION_CITATIONS[fq])

        return bib

    @classmethod
    def from_config(cls, model_config: Any, *, source: str = "") -> Bibliography:
        """Build a Bibliography from a :class:`ModelConfig`-like object.

        Inspects ``dust``, ``nebular``, ``igm`` sub-configs and adds the
        citations implied by their values. Core citations (``tengri``,
        ``jax``, ``dsps``) are always included.
        """
        from tengri.citations.associations import (
            CORE_CITATIONS,
            DUST_LAW_CITATIONS,
            DUST_MODEL_CITATIONS,
            IGM_CITATIONS,
            NEBULAR_BACKEND_CITATIONS,
        )

        bib = cls(source=source)
        bib.add(*CORE_CITATIONS)

        if model_config is None:
            return bib

        dust = getattr(model_config, "dust", None)
        if dust is not None:
            model = getattr(dust, "model", None)
            bib.add(*DUST_MODEL_CITATIONS.get(model, []))
            for law_attr in ("law", "law_bc", "law_diff"):
                law = getattr(dust, law_attr, None)
                bib.add(*DUST_LAW_CITATIONS.get(law, []))

        neb = getattr(model_config, "nebular", None)
        if neb is not None:
            backend = getattr(neb, "backend", None)
            bib.add(*NEBULAR_BACKEND_CITATIONS.get(backend, []))

        igm = getattr(model_config, "igm", None)
        if igm is not None:
            igm_model = getattr(igm, "model", None)
            bib.add(*IGM_CITATIONS.get(igm_model, []))

        return bib

    def add_backend(self, backend: str | None) -> Bibliography:
        """Append citations for an inference backend (``"vi"``, ``"mcmc_nuts"``, …)."""
        if not backend:
            return self
        from tengri.citations.associations import BACKEND_CITATIONS

        self.add(*BACKEND_CITATIONS.get(str(backend), []))
        return self

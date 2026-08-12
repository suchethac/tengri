# SPDX-License-Identifier: BSD-3-Clause
"""Public introspection façade — list and describe everything available in tengri.

This is the first thing a new user should reach for. Open a notebook, run::

    import tengri

    tengri.help()  # the cheatsheet
    tengri.summary()  # one-line counts of every menu

then narrow down with::

    tengri.list_agn_models()  # all 12 AGN models, as a pretty table
    tengri.list_dust_laws(status="production")
    tengri.list_inference_methods(tier="primary")
    tengri.describe("skirtor")  # full metadata for any name

Returned values are normal Python lists, but they print as readable tables
in a notebook or REPL — no need to wrap them in `pprint`.
"""

from __future__ import annotations

import functools
import re
from typing import Any

from tengri._display import _display

# ──────────────────────────────────────────────────────────────────
# Pretty-printed return types (still real lists/dicts — just nicer repr)
# ──────────────────────────────────────────────────────────────────


class _RegistryTable(list):
    """A `list[dict]` that prints as a column-aligned table.

    Behaves identically to a list otherwise — indexing, iteration,
    JSON serialization, etc. all work as usual.
    """

    _PREFERRED_COLS = (
        "component",
        "name",
        "kind",
        "tier",
        "status",
        "citation",
        "short_doc",
        "use",
    )
    # ``fn`` holds a live callable (list_laws) — its str() is an address,
    # so it is carried in the row but never rendered.
    _ALWAYS_HIDDEN = ("module", "requires", "params", "fn")  # surfaced via describe()

    def _columns(self) -> list[str]:
        """Decide which columns to render.

        - ``kind`` is shown only when results span more than one kind
          (e.g. cross-menu search) so single-menu tables stay narrow.
        - ``use`` (the call-site hint) is shown only on cross-menu /
          search tables — single-menu tables expose it via describe()
          to keep the row width readable.

        """
        kinds = {d.get("kind") for d in self}
        hidden = set(self._ALWAYS_HIDDEN)
        if len(kinds) <= 1:
            hidden.add("kind")
            hidden.add("use")
        all_keys = list(self[0].keys())
        cols = [k for k in self._PREFERRED_COLS if k in all_keys and k not in hidden]
        cols += [k for k in all_keys if k not in cols and k not in hidden]
        return cols

    def __repr__(self) -> str:
        if not self:
            return "(empty)"
        cols = self._columns()

        # Truncate very long fields for readability.
        def _cell(v: Any) -> str:
            s = "" if v is None else str(v)
            return s if len(s) <= 80 else s[:77] + "..."

        widths = {k: max(len(k), *(len(_cell(d.get(k, ""))) for d in self)) for k in cols}
        header = "  ".join(k.ljust(widths[k]) for k in cols)
        sep = "  ".join("─" * widths[k] for k in cols)
        rows = "\n".join(
            "  ".join(_cell(d.get(k, "")).ljust(widths[k]) for k in cols) for d in self
        )
        # Footer "kind" reads "mixed" for cross-menu (search) results.
        kinds = {d.get("kind", "entry") for d in self}
        kind_label = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        footer = f"\n[{len(self)} result{'s' if len(self) != 1 else ''} — {kind_label}]"
        # Hint at how to actually use any row — only shown on single-kind
        # tables (where the `use` column is hidden) so search tables don't
        # repeat themselves.
        if len(kinds) == 1 and self and "use" in self[0]:
            example = next((d.get("use", "") for d in self if d.get("use")), "")
            if example:
                footer += f"\n  Use:  tengri.describe({self[0]['name']!r})  →  {example}"
        return f"{header}\n{sep}\n{rows}{footer}"

    def filter(self, **criteria: Any) -> _RegistryTable:
        """Narrow the table by per-field criteria.

        Each keyword either does an exact match (``status="production"``)
        or uses a ``field__op`` operator suffix:

        - ``field=value``           — case-insensitive equality
        - ``field__contains=value`` — case-insensitive substring match
        - ``field__in=(a, b, c)``   — membership in a sequence
        - ``field__startswith=v``   — prefix match (case-insensitive)

        All criteria must match (logical AND).

        On filter tables, ``survey=`` is **smart**: a query like
        ``survey="SDSS"`` matches both the SVO ``survey`` field
        (``SLOAN``) AND the ``instrument`` field (``SDSS``) so the
        astronomer-conventional name finds the right rows.  Same for
        DES/DECam, VISTA, HSC, UKIDSS, PS1.

        Examples
        --------
        >>> tengri.list_filters().filter(survey="SDSS")
        >>> tengri.list_filters().filter(survey="HST", band__contains="F814")
        >>> tengri.list_agn_models().filter(status="production")
        >>> tengri.list_dust_laws().filter(citation__contains="Calzetti")
        """
        out: list[dict] = []
        for entry in self:
            ok = True
            for key, val in criteria.items():
                if "__" in key:
                    field, op = key.rsplit("__", 1)
                else:
                    field, op = key, "eq"
                cell = entry.get(field)

                # Smart-survey: filter tables follow the SVO convention
                # where SDSS lives in `instrument`, not `survey`. Match
                # both fields so astronomer-speak just works.
                is_filter_row = entry.get("kind") == "filter"
                if op == "eq" and field == "survey" and is_filter_row:
                    q = str(val).lower().strip()
                    target = _SURVEY_ALIASES.get(q, (q, q))
                    sv_lc = str(entry.get("survey", "")).lower()
                    in_lc = str(entry.get("instrument", "")).lower()
                    ok = q in (sv_lc, in_lc) or target[0] == sv_lc or target[1] == in_lc
                elif op == "eq":
                    # Default equality is case-insensitive for strings,
                    # exact for everything else (numbers, sequences).
                    if isinstance(cell, str) and isinstance(val, str):
                        ok = cell.lower() == val.lower()
                    else:
                        ok = cell == val
                elif op == "contains":
                    ok = str(val).lower() in str(cell or "").lower()
                elif op == "startswith":
                    ok = str(cell or "").lower().startswith(str(val).lower())
                elif op == "in":
                    ok = cell in val
                else:
                    raise ValueError(
                        f"Unknown filter operator '{op}'.  Valid: eq (default), "
                        "contains, startswith, in."
                    )
                if not ok:
                    break
            if ok:
                out.append(entry)
        return _RegistryTable(out)

    def names(self) -> list[str]:
        """Just the list of names — convenient for ``Photometry.from_names``."""
        return [d["name"] for d in self]

    def to_dict(self, value: str = "short_doc") -> dict[str, Any]:
        """Collapse to ``{name: value}`` — the shape some ``list_*`` once returned.

        ``list_known_ssps`` and ``list_filter_conventions`` returned plain
        ``dict[str, str]`` while every other ``list_*`` returned a table
        (#1285). They now return tables too; this is the mechanical migration
        for callers that wanted the mapping.

        Parameters
        ----------
        value : str, optional
            Which column becomes the dict value. Defaults to ``"short_doc"``.

        Returns
        -------
        dict
            ``{row["name"]: row[value]}`` in table order.

        Raises
        ------
        KeyError
            If ``value`` is not a column. Silently returning ``None`` values
            would look like an empty catalog rather than a wrong column name.
        """
        if self and value not in self[0]:
            raise KeyError(
                f"{value!r} is not a column of this table. Available: {sorted(self[0])}."
            )
        return {d["name"]: d[value] for d in self}

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — renders as a real HTML table in notebooks."""
        if not self:
            return "<i>(empty)</i>"
        cols = self._columns()
        head = "".join(f"<th style='text-align:left'>{k}</th>" for k in cols)
        body = "".join(
            "<tr>"
            + "".join(f"<td style='text-align:left'>{d.get(k, '')}</td>" for k in cols)
            + "</tr>"
            for d in self
        )
        kinds = {d.get("kind", "entry") for d in self}
        kind_label = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        return (
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            f"<i>{len(self)} result{'s' if len(self) != 1 else ''} — {kind_label}</i>"
        )


class _DescribeRecord(dict):
    """A `dict` that prints as a labeled block. Plain dict otherwise."""

    def __repr__(self) -> str:
        if not self:
            return "(empty)"
        # Render every field except param_details first; render that last
        # as an indented sub-table since it's a list-of-dicts.
        non_details = [(k, v) for k, v in self.items() if k != "param_details"]
        width = max((len(k) for k, _ in non_details), default=0) + 2
        lines: list[str] = []
        for k, v in non_details:
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                if len(v) <= 4:
                    lines.append(f"  {k.ljust(width)}{', '.join(v)}")
                else:
                    lines.append(f"  {k.ljust(width)}{v[0]}")
                    for item in v[1:]:
                        lines.append(f"  {' ' * width}{item}")
            else:
                lines.append(f"  {k.ljust(width)}{v}")

        # Sub-table for parameter defaults and descriptions.
        details = self.get("param_details")
        if details:
            name_w = max(len(d["name"]) for d in details)
            def_w = max(len(d["default"]) for d in details)
            lines.append("")
            lines.append("  param_details (free-parameter priors):")
            for d in details:
                desc = d.get("description", "")
                lines.append(f"    {d['name'].ljust(name_w)}  {d['default'].ljust(def_w)}  {desc}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        def _fmt(v):
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return "<br>".join(v)
            return str(v)

        _row_tpl = (
            "<tr><th style='text-align:left'>{k}</th><td style='text-align:left'>{v}</td></tr>"
        )
        rows = "".join(_row_tpl.format(k=k, v=_fmt(v)) for k, v in self.items())
        return f"<table>{rows}</table>"


# ──────────────────────────────────────────────────────────────────
# Internal entry-normalization
# ──────────────────────────────────────────────────────────────────


def _extract_params(entry: Any, kind: str) -> list[str]:
    """Best-effort free-parameter list for a registry entry.

    AGN entries → introspect callable signature, keep names starting with ``agn_``.
    SFH entries → ``callable.params`` (an ``SFHModelSpec`` field) → key list.
    Dust laws / others → empty (parameters come from the caller, not the
    registered function).
    """
    import inspect

    if kind == "sfh_model":
        spec = getattr(entry, "callable", None)
        params = getattr(spec, "params", None)
        if isinstance(params, dict):
            return list(params)

    if kind == "agn_model":
        fn = getattr(entry, "callable", None)
        if fn is None:
            return []
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return []
        return [p.name for p in sig.parameters.values() if p.name.startswith("agn_")]

    return []


def _usage_hint(name: str, kind: str) -> str:
    """Return a copy-pasteable one-liner showing how to actually use this
    entry — the missing piece between "I found a thing called skirtor"
    and "now what?"

    Patterns are based on the canonical spec/filter/fitter call sites
    used in ``docs/spine/00_quickstart.py``.
    """
    if kind == "filter":
        return f'Photometry.from_names(["{name}"])'
    if kind == "agn_model":
        return f"SEDModel.build(..., agn={{'type': '{name}'}})"
    if kind == "dust_model":
        return f"SEDModel.build(..., dust={{'type': '{name}'}})"
    if kind == "dust_attenuation":
        return f"SEDModel.build(..., dust={{'type': 'single_component', 'law_bc': '{name}'}})"
    if kind == "dust_emission":
        return f"SEDModel.build(..., dust={{'emission': {{'type': '{name}'}}}})"
    if kind == "sfh_model":
        return f"SEDModel.build(..., sfh={{'type': '{name}'}})"
    if kind == "nebular_backend":
        return f"SEDModel.build(..., neb={{'type': '{name}'}})"
    if kind == "inference_method":
        return f'fitter.run("{name}")'
    if kind == "xray_model":
        return f"SEDModel.build(..., xray={{'type': '{name}'}})"
    if kind == "radio_model":
        return f"SEDModel.build(..., radio={{'type': '{name}'}})"
    if kind == "shock_model":
        return f"SEDModel.build(..., shock={{'type': '{name}'}})"
    if kind == "metallicity_mode":
        # ``met={'type': ...}``, the parallel of ``sfh={'type': ...}`` (#1720),
        # which replaced ``stellar={'met_mode': ...}``. One spelling, and it is
        # the one the rest of the grammar uses.
        return f"SEDModel.build(..., met={{'type': '{name}'}})"
    if kind == "igm_model":
        return f"SEDModel.build(..., igm={{'type': '{name}'}})"
    if kind == "component":
        menus = _COMPONENT_MENUS.get(name)
        return f"tengri.{name}  (see {menus} for alternatives)" if menus else f"tengri.{name}"
    return ""


# Each SEDComponent maps to its real discovery menu(s). The menu names are
# irregular (``list_sfh_models`` vs ``list_dust_laws`` vs
# ``list_nebular_backends``), so this must be a lookup, not an
# ``f"list_{name}_models"`` formula — the formula advertised
# ``list_stellar_models`` / ``list_agn_laws`` and other functions that do not
# exist, sending a fresh user into an ``AttributeError``. (``list_dust_models``
# was another of that formula's phantoms; it is real now, but it names the
# structural axis only — the formula still guesses wrong for every other group.)
_COMPONENT_MENUS: dict[str, str] = {
    "met": "list_metallicity_modes()",
    "dust": "list_dust_models() / list_dust_laws() / list_dust_emission_models()",
    "agn": "list_agn_models() / list_agn_blocks()",
    "nebular": "list_nebular_backends()",
    "radio": "list_radio_models() / list_radio_blocks()",
    "igm": "list_igm_models()",
    "xray": "list_xray_models()",
    "shock": "list_shock_models()",
}


def _extract_param_details(entry: Any, kind: str) -> list[dict]:
    """Per-parameter details (default prior, description) when discoverable.

    Used by ``describe()`` to show *what range to put in your Uniform()*
    for a model the user hasn't seen before — the most common new-user
    question after "what models are available?"
    """
    out: list[dict] = []

    if kind == "sfh_model":
        spec = getattr(entry, "callable", None)
        params = getattr(spec, "params", None)
        if isinstance(params, dict):
            for name, pdef in params.items():
                out.append(
                    {
                        "name": name,
                        "default": str(getattr(pdef, "default", "")),
                        "description": getattr(pdef, "description", ""),
                    }
                )
        return out

    if kind == "agn_model":
        # AGN param defaults live in _AGN_PARAMS keyed by param name; we
        # match against the names that appear in the callable's signature.
        import inspect

        fn = getattr(entry, "callable", None)
        if fn is None:
            return out
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return out
        agn_names = [p.name for p in sig.parameters.values() if p.name.startswith("agn_")]
        try:
            from tengri.parameters._builders import _resolve_lazy_bucket

            agn_params = _resolve_lazy_bucket("_AGN_PARAMS")
            for n in agn_names:
                meta = agn_params.get(n)
                if meta is None:
                    continue
                description, _check, _err, default = meta
                out.append(
                    {
                        "name": n,
                        "default": str(default),
                        "description": description,
                    }
                )
        except (ImportError, AttributeError):
            pass
        return out

    return out


def _entry_to_dict(name: str, entry: Any, *, kind: str) -> dict:
    """Normalize a registry entry (varied dataclass shapes) into a uniform dict."""
    out = {
        "name": name,
        "kind": kind,
        "status": getattr(entry, "status", "production"),
        "citation": getattr(entry, "citation", ""),
        "short_doc": getattr(entry, "short_doc", ""),
        "use": _usage_hint(name, kind),
    }
    params = _extract_params(entry, kind)
    if params:
        out["params"] = params
    details = _extract_param_details(entry, kind)
    if details:
        out["param_details"] = details
    return out


def _component_entry(name: str, *, kind: str) -> dict:
    """Build a menu row for a name that lives in the ``SEDModelComponent`` registry.

    Several grammar axes accept the union of a legacy ``*_MODELS`` dict and the
    ``_REGISTRY`` components carrying the matching prefix. The legacy entries
    are dataclasses with ``status`` / ``citation`` / ``short_doc`` attributes;
    the component classes carry none of those, so their summary is taken from
    the first line of the class docstring. Deriving it here — rather than
    listing these names in a metadata table — keeps the menu honest when a new
    component is registered: the row appears with whatever docstring it has,
    instead of the name silently going missing.
    """
    from tengri.forward.component_factory import _REGISTRY

    cls = _REGISTRY.get(name)
    doc_lines = (getattr(cls, "__doc__", "") or "").strip().splitlines()
    return {
        "name": name,
        "kind": kind,
        "status": getattr(cls, "status", "production"),
        "citation": getattr(cls, "citation", ""),
        "short_doc": doc_lines[0].strip() if doc_lines else "",
        "use": _usage_hint(name, kind),
    }


# ──────────────────────────────────────────────────────────────────
# Menu filtering
# ──────────────────────────────────────────────────────────────────

#: The word for "do not filter". Every menu accepts it, because it is what a
#: reader types first and because ``status='all'`` used to return nothing at all
#: — the worst possible answer to "show me everything".
ALL = "all"


@functools.cache
def _menu_vocabulary(column: str) -> tuple[str, ...]:
    """Every value any discovery menu publishes for ``column``.

    Derived from the live menus rather than pinned, so it cannot rot the way a
    hard-coded list would. Safe from recursion: the listers are called with no
    filter, and :func:`_filter_menu` returns before reaching here when the
    requested value is ``None``.
    """
    values: set[str] = set()
    for lister in _menu_listers():
        for row in lister():
            if column in row:
                values.add(row[column])
    return tuple(sorted(values))


def _filter_menu(rows: list[dict], column: str, value: str | None, *, listing: str) -> list[dict]:
    """Narrow ``rows`` to one ``column`` value, refusing a value that is not one.

    Every menu used to filter with a bare ``[r for r in rows if r[column] ==
    value]``, which answers a typo and a genuine "nothing matches" with the same
    empty list. ``list_sfh_models(status='producton')`` returned zero of
    thirty-four rows and said nothing, and so did the natural
    ``status='all'`` (#1679).

    The distinction kept here is between a value that is not a ``column`` value
    at all — a typo, which raises — and one that is real but absent from *this*
    menu, which is a legitimate empty answer: there simply are no unvalidated
    dust laws.
    """
    if value is None or str(value).lower() == ALL:
        return rows
    here = sorted({r[column] for r in rows if column in r})
    # Union, not just the global set: a menu may surface rows that the default
    # listing hides, and those values are legitimate. `list_inference_methods`
    # passes `include_broken=(tier == "broken")`, so `tier='broken'` is a
    # documented query whose rows exist only once it has been asked for — a
    # vocabulary derived from default listings alone would reject it. That is
    # the same too-narrow-census mistake this helper exists to fix.
    vocabulary = sorted(set(_menu_vocabulary(column)) | set(here))
    if value not in vocabulary:
        raise ValueError(
            f"{listing}({column}={value!r}) — {value!r} is not a {column} any menu "
            f"uses. Valid values: {vocabulary}. This menu currently has: "
            f"{here}. Pass {column}={ALL!r} (or omit it) to list everything."
        )
    return [r for r in rows if r.get(column) == value]


def _resolve_category(value: str | None, accepted, *, listing: str) -> str | None:
    """The category to filter on, or ``None`` for "do not filter".

    The category axis is validated where the rows are *built*, not after, so it
    cannot go through :func:`_filter_menu`. It gets the same contract anyway:
    ``list_agn_blocks`` already refused an unknown category while
    ``list_radio_blocks`` silently returned zero of seven rows — one sibling
    right, one wrong, which is the giveaway that nothing enforced the rule.
    """
    if value is None or str(value).lower() == ALL:
        return None
    if value not in accepted:
        raise ValueError(
            f"{listing}(category={value!r}) — {value!r} is not a category this "
            f"menu has. Accepted: {sorted(accepted)}. Pass category={ALL!r} "
            f"(or omit it) to list everything."
        )
    return value


# ──────────────────────────────────────────────────────────────────
# Listing functions
# ──────────────────────────────────────────────────────────────────


def list_agn_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered AGN SED models.

    Parameters
    ----------
    status : str, optional
        Filter by ``"production"``, ``"experimental"``, ``"demo"``, or
        ``"deprecated"``.

    Returns
    -------
    _RegistryTable
        List of metadata dicts. Prints as a table in a notebook.
    """
    from tengri.components.agn.unified import AGN_MODELS

    out = [_entry_to_dict(n, e, kind="agn_model") for n, e in AGN_MODELS.items()]
    out = _filter_menu(out, "status", status, listing="list_agn_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_agn_blocks(*, category: str | None = None, status: str | None = None) -> _RegistryTable:
    """List all registered composable AGN block implementations.

    Blocks are the fine-grained components of AGN SEDs: disc, nlr, blr,
    feii, torus, and attenuation. Users compose an AGN by selecting one
    block per category inside the ``agn`` group dict, e.g.
    ``SEDModel.build(..., agn={"disc": {"type": "multicolor"},
    "nlr": {"type": "analytic"}})``.

    This coexists with monolithic AGN models (:func:`list_agn_models`) —
    blocks offer mix-and-match flexibility while monolithic models bundle
    a complete recipe.

    Parameters
    ----------
    category : str, optional
        Filter to a specific category: ``"disc"``, ``"nlr"``, ``"blr"``,
        ``"feii"``, ``"torus"``, or ``"attenuation"``. If ``None``, list
        all categories.
    status : str, optional
        Filter by ``"production"``, ``"experimental"``, ``"demo"``, or
        ``"deprecated"``.

    Returns
    -------
    _RegistryTable
        List of metadata dicts. Prints as a table in a notebook.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_agn_blocks(category="disc")
    >>> tengri.list_agn_blocks(status="production")
    """
    from tengri.components.agn.blocks._protocol import AGN_BLOCK_META, AGN_BLOCKS

    # AGN_BLOCKS categories are the human-readable labels; the ``agn`` group
    # grammar keys match them except for 'attenuation', whose structural key is
    # the terser 'atten' (see parameters.groups._AGN_SUBBLOCK_KEYS). Map so the
    # advertised ``use:`` string names the exact key SEDModel.build accepts.
    category_to_group_key = {"attenuation": "atten"}

    # Accept the grammar key as well as the registry label. Five of the six
    # slots have category == grammar key, so the mismatch showed up only for
    # attenuation — and it was this function's own ``use:`` string that told
    # users to type ``agn={'atten': ...}``, so ``list_agn_blocks('atten')``
    # returned an empty table for the exact name it had just advertised
    # (#1451). Normalize before filtering rather than at each comparison.
    group_key_to_category = {v: k for k, v in category_to_group_key.items()}
    # `'all'` means "do not filter" on every other menu axis (#1679); refusing
    # it only here would be a second, smaller version of the same trap.
    if category is not None and str(category).lower() == ALL:
        category = None
    if category is not None:
        category = group_key_to_category.get(category, category)
        # ...and fail loudly on anything else. The filter used to fall through
        # to "no category matched", so a typo, an empty string and a valid-but-
        # wrong key were indistinguishable from a category that genuinely has
        # no blocks — a guard that fails open is the bug.
        if category not in AGN_BLOCKS:
            accepted = sorted(set(AGN_BLOCKS) | set(group_key_to_category))
            raise ValueError(
                f"Unknown AGN block category {category!r}. "
                f"Accepted: {', '.join(repr(c) for c in accepted)}. "
                "('atten' and 'attenuation' both work — the first is the "
                "build-grammar key, the second the registry label.)"
            )

    out: list[dict] = []
    for cat in AGN_BLOCKS:
        if category is not None and cat != category:
            continue
        group_key = category_to_group_key.get(cat, cat)
        for name in AGN_BLOCKS[cat]:
            meta = AGN_BLOCK_META.get((cat, name), {})
            entry_dict = {
                "name": name,
                "category": cat,
                "kind": "agn_block",
                "status": meta.get("status", "production"),
                "citation": meta.get("citation", ""),
                "short_doc": meta.get("short_doc", ""),
                "use": f"SEDModel.build(..., agn={{'{group_key}': {{'type': '{name}'}}}})",
            }
            out.append(entry_dict)

    out = _filter_menu(out, "status", status, listing="list_agn_blocks")
    return _RegistryTable(sorted(out, key=lambda m: (m["category"], m["name"])))


# Dust *structural* models — the ``dust={'type': ...}`` axis. Keyed by the
# names in ``_VALID_DUST_TYPES``, which is what the build validator accepts;
# the listing derives its names from that same set (see
# :func:`list_dust_models`) so the menu and the validator cannot drift.
_DUST_MODEL_METADATA: dict[str, dict[str, str]] = {
    "single_component": {
        "status": "production",
        "citation": "Calzetti et al. 2000 (ApJ 533, 682)",
        "short_doc": "One screen over all stars; `law_bc` sets the curve",
    },
    "two_component": {
        "status": "production",
        "citation": "Charlot & Fall 2000 (ApJ 539, 718)",
        "short_doc": "Birth-cloud + diffuse screens; young stars see both",
    },
    "wg00": {
        "status": "production",
        "citation": "Witt & Gordon 2000 (ApJ 528, 799)",
        "short_doc": "Radiative-transfer grid (geometry/structure/curve selectors)",
    },
}


def list_dust_models(*, status: str | None = None) -> _RegistryTable:
    """List the dust **structural** models — the ``dust={'type': ...}`` choice.

    Dust is selected along three independent axes, and this is the first one:
    how the dust is *arranged* relative to the stars. The attenuation
    **curve** is a separate choice (:func:`list_dust_laws`, via ``law_bc`` /
    ``law_diff``), and the IR **emission** template a third
    (:func:`list_dust_emission_models`, via ``dust={'emission': ...}``).

    The other two axes had menus; this one did not, so the structural
    names — including ``two_component``, the type the recipes themselves
    build with — were absent from every discovery surface: no ``list_*``
    named them, ``describe('two_component')`` raised ``KeyError`` and
    ``search('two_component')`` returned nothing.

    Names are derived from :data:`tengri.parameters.groups._VALID_DUST_TYPES`,
    the same set ``SEDModel.build`` validates against, so this menu cannot
    advertise a type the builder rejects (nor omit one it accepts).
    """
    from tengri.parameters.groups import _VALID_DUST_TYPES

    out = []
    for name in _VALID_DUST_TYPES:
        meta = _DUST_MODEL_METADATA.get(name, {})
        out.append(
            {
                "name": name,
                "kind": "dust_model",
                "status": meta.get("status", "production"),
                "citation": meta.get("citation", ""),
                "short_doc": meta.get("short_doc", ""),
                "use": _usage_hint(name, "dust_model"),
            }
        )
    out = _filter_menu(out, "status", status, listing="list_dust_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_dust_laws(*, status: str | None = None) -> _RegistryTable:
    """List all registered dust **attenuation** laws.

    Attenuation describes how UV/optical photons are absorbed/scattered
    by dust along the line of sight (Calzetti, Cardelli, Charlot-Fall, …).
    For dust **emission** templates (DL07, Dale, MBB, …), see
    :func:`list_dust_emission_models`. For how the dust is *arranged*
    (one screen vs birth-cloud + diffuse), see :func:`list_dust_models`.
    """
    from tengri.components.dust.attenuation import DUST_LAWS

    out = [_entry_to_dict(n, e, kind="dust_attenuation") for n, e in DUST_LAWS.items()]
    out = _filter_menu(out, "status", status, listing="list_dust_laws")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


# Dust emission metadata keyed by the canonical registry name in
# ``DUST_EMISSION_MODELS``. Each entry supplies citation/short_doc; the
# accepted set of build-time names is derived from the live registry (see
# :func:`list_dust_emission_models` below), so the validator and the
# introspection helper can never drift apart (closes #495 — same pattern
# as PR #489 for AGN blocks).
_DUST_EMISSION_METADATA: dict[str, dict[str, str]] = {
    "dl07": {
        "status": "production",
        "citation": "Draine & Li 2007 (ApJ 657, 810)",
        "short_doc": "Diffuse + PAH grain mixture, Umin/Umax/qpah (alias of draine_li2007)",
    },
    "draine_li2007": {
        "status": "production",
        "citation": "Draine & Li 2007 (ApJ 657, 810)",
        "short_doc": "Diffuse + PAH grain mixture, Umin/Umax/qpah",
    },
    "dl14": {
        "status": "production",
        "citation": "Draine et al. 2014 (ApJ 780, 172)",
        "short_doc": "Updated DL with extended PAH/silicate features (alias of draine_li2014)",
    },
    "draine_li2014": {
        "status": "production",
        "citation": "Draine et al. 2014 (ApJ 780, 172)",
        "short_doc": "Updated DL with extended PAH and silicate features",
    },
    "dale2014": {
        "status": "production",
        "citation": "Dale et al. 2014 (ApJ 784, 83)",
        "short_doc": "SFR-driven empirical IR template family (alpha_sf)",
    },
    "astrodust": {
        "status": "experimental",
        "citation": "Hensley & Draine 2023 (ApJ 948, 55)",
        "short_doc": "Astrodust + PAH unified grain model",
    },
    "themis": {
        "status": "experimental",
        "citation": "Jones et al. 2017 (A&A 602, A46)",
        "short_doc": "THEMIS amorphous-carbon grain model",
    },
    "bosa": {
        "status": "experimental",
        "citation": "Boquien et al. 2019 (CIGALE BOSA grids)",
        "short_doc": "BOSA dust SED templates",
    },
    "mbb": {
        "status": "production",
        "citation": "Casey 2012 (MNRAS 425, 3094)",
        "short_doc": "Single-temperature modified blackbody (alias of modified_blackbody)",
    },
    "modified_blackbody": {
        "status": "production",
        "citation": "Casey 2012 (MNRAS 425, 3094)",
        "short_doc": "Single-temperature modified blackbody (analytic)",
    },
    "casey2012": {
        "status": "production",
        "citation": "Casey 2012 (MNRAS 425, 3094)",
        "short_doc": "Modified blackbody + mid-IR power law (analytic)",
    },
    "dale2014_cigale": {
        "status": "production",
        "citation": "Dale et al. 2014 (ApJ 784, 83)",
        "short_doc": "Dale+2014 IR templates, CIGALE alpha_sf + fracAGN parameterization",
    },
    "schreiber2018": {
        "status": "experimental",
        "citation": "Schreiber et al. 2018 (A&A 609, A30)",
        "short_doc": "Tabulated IR template library (T_dust, f_PAH)",
    },
    "schreiber2016": {
        "status": "production",
        "citation": "Schreiber et al. 2016 (A&A 589, A35)",
        "short_doc": "Modified-blackbody (beta=1.5) + PAH mix; (T_dust, f_PAH)",
    },
    "pah_drude": {
        "status": "production",
        "citation": "Smith et al. 2007 (ApJ 656, 770) Drude profiles",
        "short_doc": "Drude-profile PAH emission features",
    },
    "energy_balance_split": {
        "status": "experimental",
        "citation": "Kokorev et al. 2021 (ApJ 921, 40)",
        "short_doc": "Two-temperature (warm+cold) + AGN-IR emission; f_cold, L_agn_ir",
    },
}


def list_dust_emission_models(*, status: str | None = None) -> _RegistryTable:
    """List all available dust **emission** template families.

    Emission templates describe the IR re-radiation of energy absorbed by
    dust (DL07, DL14, Dale+2014, THEMIS, MBB, …). For UV/optical
    **attenuation** laws, see :func:`list_dust_laws`.

    Calls the ``SEDModel.build`` grammar validator itself
    (:func:`tengri.parameters.groups._valid_dust_emission_types`) rather than
    re-deriving what it derives. The ``DUST_EMISSION_MODELS`` loader cache is
    **not** consulted — it is load-only (closes #495).

    The call matters. This menu used to re-implement the validator's
    derivation (``_REGISTRY`` components publishing ``sed_dust_ir``, union the
    alias map) under a docstring promising the two "can never drift". They
    drifted anyway: the validator also unions ``_LAZY_DUST_EMISSION_TYPES``,
    which is declared inside ``groups.py`` and is therefore invisible from
    here, so ``dh02_ce01`` was builder-accepted and named by no menu. A copy of
    a derivation is a second source of truth no matter how faithful it is on
    the day it is written.
    """
    # Import triggers component registration into _REGISTRY + the alias map.
    import tengri.components.dust.emission  # noqa: F401
    from tengri.parameters.groups import _valid_dust_emission_types

    names = set(_valid_dust_emission_types())

    out = []
    for name in names:
        meta = _DUST_EMISSION_METADATA.get(
            name,
            {"status": "production", "citation": "", "short_doc": ""},
        )
        entry = {
            "name": name,
            **meta,
            "use": _usage_hint(name, "dust_emission"),
            "kind": "dust_emission",
        }
        out.append(entry)
    out = _filter_menu(out, "status", status, listing="list_dust_emission_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_sfh_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered star formation history models.

    SFH types that are registered but not yet wired into the DSPS forward
    path (:data:`~tengri.components.stellar.sfh.registry.UNVALIDATED_SFH_TYPES`)
    are reported with ``status='unvalidated'`` rather than the registry's
    default ``'production'``: ``SEDModel.build(sfh={'type': ...})`` rejects
    them, so advertising them as production would send a fresh user into a
    build-time ``ValueError``. Filter to the buildable set with
    ``list_sfh_models(status='production')``.
    """
    from tengri.components.stellar.sfh.registry import (
        SFH_REGISTRY,
        UNVALIDATED_SFH_TYPES,
    )

    out = [_entry_to_dict(n, e, kind="sfh_model") for n, e in SFH_REGISTRY.items()]
    for m in out:
        if m["name"] in UNVALIDATED_SFH_TYPES:
            m["status"] = "unvalidated"
            if "not builder-available" not in m["short_doc"]:
                suffix = " [not builder-available — registered, not yet DSPS-validated]"
                m["short_doc"] = f"{m['short_doc']}{suffix}"
            # The status said "unvalidated" while ``use:`` still advertised
            # ``SEDModel.build(..., sfh={'type': X})`` — a copy-pasteable call
            # that raises ValueError. That is the "advice that raises" class
            # (#1275) reappearing in the hint field: the sweep guarding it
            # checked that every ``use:`` *starts with* ``SEDModel.build(``,
            # a shape test that these eight passed while failing to run.
            # Carry the reason and the next step instead of a call, so nothing
            # in the menu is copy-pasteable-and-broken.
            m["use"] = (
                "not builder-available — SEDModel.build rejects it; "
                "browse the buildable set with list_sfh_models(status='production')"
            )
        # ``mixture`` (burst) and ``modulator`` (field) SFH components cannot
        # stand alone — ``sfh={'type': 'burst'}`` raises "At least one additive
        # (smooth) SFH component required". Advertise the composed list form so
        # the ``use:`` hint is something the builder actually accepts.
        ctype = getattr(SFH_REGISTRY[m["name"]], "composition_type", "additive")
        if ctype in ("mixture", "modulator"):
            m["use"] = f"SEDModel.build(..., sfh={{'type': ['const', '{m['name']}']}})"
    out = _filter_menu(out, "status", status, listing="list_sfh_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_nebular_backends(*, status: str | None = None) -> _RegistryTable:
    """List all registered nebular emission backends.

    Reads :data:`tengri.components.nebular.NEBULAR_MODELS` so the
    listing stays in lock-step with what the grammar-layer validator
    will actually accept (#331). The names match the keys consumed by
    ``SEDModel.build(..., neb={'type': ...})`` — ``'none'`` /
    ``'ssp'`` / ``'cue'`` / ``'cloudy'`` / ``'cb19'``.
    """
    from tengri.components.nebular import NEBULAR_MODELS

    out = [_entry_to_dict(n, e, kind="nebular_backend") for n, e in NEBULAR_MODELS.items()]
    for m in out:
        # The generic CLOUDY backend needs a user-supplied grid file:
        # ``neb={'type': 'cloudy'}`` raises "The CLOUDY nebular backend needs a
        # grid file." The key is ``grid``; this hint advertised ``gridfile``,
        # so the line printed to fix one failure raised a different one —
        # "Unknown key 'gridfile' in group 'neb'. Did you mean: grid?" — and
        # nothing checked either. (``cb19`` ships its own grid and stands alone.)
        if m["name"] == "cloudy":
            m["use"] = "SEDModel.build(..., neb={'type': 'cloudy', 'grid': 'grid.h5'})"
    out = _filter_menu(out, "status", status, listing="list_nebular_backends")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_xray_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered X-ray emission models.

    The X-ray group composes the AGN corona (Yang+2020 ``alpha_ox(L_2500)``
    relation), the Lehmer+2016 high- and low-mass X-ray binary fits,
    and optional thermal hot-gas emission. The ``'none'`` entry disables
    the whole block.

    Names come from the grammar validator itself
    (:func:`tengri.parameters.groups._valid_xray_types`), which accepts the
    union of :data:`XRAY_MODELS` and the ``SEDModelComponent`` X-ray variants
    in ``_REGISTRY``. Reading only ``XRAY_MODELS`` — as this menu did — hid
    ``xray_aird`` and ``agn_xray_corona`` from every discovery surface from the
    moment #1323 made them builder-reachable, while the validator's own
    docstring asserted the menu and builder "cannot drift".

    See also: :func:`list_radio_models`, :func:`list_igm_models`,
    :mod:`tengri.builders.xray`.
    """
    from tengri.components.xray._models import XRAY_MODELS
    from tengri.parameters.groups import _valid_xray_types

    out = [
        _entry_to_dict(n, XRAY_MODELS[n], kind="xray_model")
        if n in XRAY_MODELS
        else _component_entry(n, kind="xray_model")
        for n in _valid_xray_types()
    ]
    out = _filter_menu(out, "status", status, listing="list_xray_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_radio_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered radio emission models.

    The radio group adds the Condon+1992 FIR-radio correlation for the
    star-forming-galaxy contribution plus an optional AGN radio
    power-law via the radio-loudness parameter. ``'none'`` disables
    the block.

    Names come from the grammar validator itself
    (:func:`tengri.parameters.groups._valid_radio_types`), which accepts the
    union of :data:`RADIO_MODELS` and the ``SEDModelComponent`` radio variants
    in ``_REGISTRY`` — so ``radio_powerlaw`` and ``radio_dpl`` are listed here
    rather than being builder-only. This is the ``radio={'type': ...}`` axis;
    the ``radio={'sf'/'agn': ...}`` sub-blocks live in
    :func:`list_radio_blocks`.

    See also: :func:`list_xray_models`, :func:`list_igm_models`,
    :mod:`tengri.builders.radio`.
    """
    from tengri.components.radio._models import RADIO_MODELS
    from tengri.parameters.groups import _valid_radio_types

    out = [
        _entry_to_dict(n, RADIO_MODELS[n], kind="radio_model")
        if n in RADIO_MODELS
        else _component_entry(n, kind="radio_model")
        for n in _valid_radio_types()
    ]
    out = _filter_menu(out, "status", status, listing="list_radio_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


# Radio *sub-block* variants — the ``radio={'sf': ...}`` / ``radio={'agn': ...}``
# axes, which are separate from the legacy ``radio={'type': ...}`` menu in
# :func:`list_radio_models`. Keyed by ``(category, name)`` like AGN_BLOCK_META.
# The names themselves are NOT listed here: they are derived from the tuples the
# validator checks against (see :func:`list_radio_blocks`), so a variant added to
# the physics can never be missing from the menu.
_RADIO_BLOCK_METADATA: dict[tuple[str, str], dict[str, str]] = {
    ("sf", "none"): {
        "short_doc": "No star-forming synchrotron (AGN radio only)",
    },
    ("sf", "bell2003"): {
        "citation": "Bell 2003 (ApJ 586, 794)",
        "short_doc": "Fixed-q FIR-radio correlation",
    },
    ("sf", "delvecchio2021"): {
        "citation": "Delvecchio+2021 FIRRC (SEMPER Eq. 4, arXiv:2503.20525)",
        "short_doc": "Mass- and z-dependent q_IR at 1.4 GHz",
    },
    ("sf", "mccheyne2022"): {
        "citation": "McCheyne+2022 FIRRC (SEMPER Eq. 5, arXiv:2503.20525)",
        "short_doc": "Mass- and z-dependent q_IR at 150 MHz (LOFAR)",
    },
    ("agn", "none"): {
        "short_doc": "No AGN radio (star-forming synchrotron only)",
    },
    ("agn", "powerlaw"): {
        "short_doc": "Single power-law AGN radio scaled by radio-loudness",
    },
    ("agn", "dpl"): {
        "citation": "Martinez-Ramirez+2024 (A&A 692, A85)",
        "short_doc": "Broken double power-law with exp aging cutoff",
    },
}


def list_radio_blocks(*, category: str | None = None, status: str | None = None) -> _RegistryTable:
    """List the composable radio sub-block variants — ``sf`` and ``agn``.

    Radio is selected along two independent axes: the star-forming
    synchrotron model (``radio={'sf': {'type': ...}}``) and the AGN radio
    model (``radio={'agn': {'type': ...}}``). Either may be ``'none'`` to
    run the other alone.

    This is a different axis from :func:`list_radio_models`, which lists the
    **legacy** ``radio={'type': ...}`` key. That key predates the SF/AGN split
    and cannot be combined with these sub-blocks; mixing the two raises.

    Parameters
    ----------
    category : str, optional
        Filter to ``"sf"`` or ``"agn"``. If ``None``, list both.
    status : str, optional
        Filter by ``"production"``, ``"experimental"``, ``"demo"``, or
        ``"deprecated"``.

    Returns
    -------
    _RegistryTable
        List of metadata dicts. Prints as a table in a notebook.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_radio_blocks(category="sf")
    >>> tengri.list_radio_blocks()
    """
    from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS

    # Derived from the validator's own tuples, never from a hand-written list.
    names_by_category = {"sf": SF_RADIO_MODELS, "agn": AGN_RADIO_MODELS}
    category = _resolve_category(category, names_by_category, listing="list_radio_blocks")

    out: list[dict] = []
    for cat, names in names_by_category.items():
        if category is not None and cat != category:
            continue
        for name in names:
            meta = _RADIO_BLOCK_METADATA.get((cat, name), {})
            out.append(
                {
                    "name": name,
                    "category": cat,
                    "kind": "radio_block",
                    "status": meta.get("status", "production"),
                    "citation": meta.get("citation", ""),
                    "short_doc": meta.get("short_doc", ""),
                    "use": f"SEDModel.build(..., radio={{'{cat}': {{'type': '{name}'}}}})",
                }
            )

    out = _filter_menu(out, "status", status, listing="list_radio_blocks")
    return _RegistryTable(sorted(out, key=lambda m: (m["category"], m["name"])))


# Shock models — the ``shock={'type': ...}`` axis. Names derive from
# ``_VALID_SHOCK_TYPES`` (see :func:`list_shock_models`).
_SHOCK_MODEL_METADATA: dict[str, dict[str, str]] = {
    "none": {
        "short_doc": "No shock component",
    },
    "mappings": {
        "citation": "Allen+2008 (ApJS 178, 20); Sutherland & Dopita 2017 (ApJS 229, 34)",
        "short_doc": "MAPPINGS shock + precursor emission, additive to any neb backend",
    },
}


def list_shock_models(*, status: str | None = None) -> _RegistryTable:
    """List the shock emission models — the ``shock={'type': ...}`` choice.

    The shock component is **additive**: it composes with whichever
    photoionized nebular backend is active rather than replacing it, so both
    can be on at once.

    Parameters
    ----------
    status : str, optional
        Filter by ``"production"``, ``"experimental"``, ``"demo"``, or
        ``"deprecated"``.

    Returns
    -------
    _RegistryTable
        List of metadata dicts. Prints as a table in a notebook.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_shock_models()
    """
    from tengri.parameters.groups import _VALID_SHOCK_TYPES

    out: list[dict] = []
    for name in _VALID_SHOCK_TYPES:
        meta = _SHOCK_MODEL_METADATA.get(name, {})
        out.append(
            {
                "name": name,
                "kind": "shock_model",
                "status": meta.get("status", "production"),
                "citation": meta.get("citation", ""),
                "short_doc": meta.get("short_doc", ""),
                "use": _usage_hint(name, "shock_model"),
            }
        )
    out = _filter_menu(out, "status", status, listing="list_shock_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


# Metallicity modes — the ``met={'type': ...}`` axis (#1720, replacing the
# ``stellar={'met_mode': ...}`` spelling of #311). Names are NOT
# listed here: :func:`list_metallicity_modes` derives them from ``MET_REGISTRY``
# itself, the same dict ``_translate_met`` validates against, so a mode
# added to the physics cannot be missing from the menu. This table carries only
# the prose, lifted from each mode's section header in ``met_registry.py``.
_METALLICITY_MODE_METADATA: dict[str, dict[str, str]] = {
    "delta": {"short_doc": "Single metallicity (the default)"},
    "ramp": {"short_doc": "Linear evolving metallicity"},
    "two_step": {"short_doc": "Step function at a lookback time"},
    "psb_two_step": {"short_doc": "Step at post-starburst burst age"},
    "bins": {"short_doc": "Per-bin metallicities (pairs with continuity SFH)"},
    "bins_continuity": {"short_doc": "Cumulative delta-log-Z steps"},
    "chem_evol": {"short_doc": "Gas-regulator model (Z derived from SFH)"},
    "table": {"short_doc": "User-provided Z(t)"},
    "massmap_lin": {"short_doc": "Linear metallicity tied to cumulative mass formed"},
    "massmap_box": {"short_doc": "Closed-box metallicity tied to cumulative mass formed"},
}


def list_metallicity_modes(*, status: str | None = None) -> _RegistryTable:
    """List the metallicity modes — the ``met={'type': ...}`` choice.

    The metallicity axis is structural in the same sense as the SFH or dust
    axis: it decides whether a fit carries one metallicity, an evolving one, or
    a per-bin vector, and which ``met_*`` parameters exist as a result.

    Names derive from :data:`MET_REGISTRY`, the dict
    ``parameters.groups._translate_stellar`` validates against, so this menu
    cannot advertise a mode the builder rejects or omit one it accepts. Until
    this menu existed the axis had **no** discovery surface at all: nine of its
    ten values resolved to nothing, and ``describe('table')`` answered with the
    unrelated *SFH* table model.

    Parameters
    ----------
    status : str, optional
        Filter by ``"production"``, ``"experimental"``, ``"demo"``, or
        ``"deprecated"``.

    Returns
    -------
    _RegistryTable
        List of metadata dicts. Prints as a table in a notebook.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_metallicity_modes()
    """
    from tengri.components.stellar.sfh.met_registry import MET_REGISTRY

    out: list[dict] = []
    for name in MET_REGISTRY:
        meta = _METALLICITY_MODE_METADATA.get(name, {})
        entry = {
            "name": name,
            "kind": "metallicity_mode",
            "status": meta.get("status", "production"),
            "citation": meta.get("citation", ""),
            "short_doc": meta.get("short_doc", ""),
            "use": _usage_hint(name, "metallicity_mode"),
        }
        params = tuple(getattr(MET_REGISTRY[name], "params", {}) or ())
        if params:
            entry["params"] = list(params)
        out.append(entry)
    out = _filter_menu(out, "status", status, listing="list_metallicity_modes")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_igm_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered IGM transmission models.

    Optional sub-flags on the IGM group (``patchy=True``, ``dla=True``)
    layer extra free parameters on top of the chosen mean transmission
    curve. Those are not separate models here — they apply to either
    ``'inoue14'`` or ``'madau'``.

    See also: :func:`list_xray_models`, :func:`list_radio_models`,
    :mod:`tengri.builders.igm`.
    """
    from tengri.components.igm._models import IGM_MODELS

    out = [_entry_to_dict(n, e, kind="igm_model") for n, e in IGM_MODELS.items()]
    out = _filter_menu(out, "status", status, listing="list_igm_models")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


#: The SFH→SSP age-weight kernels, as a menu. Not a registry-backed dispatch
#: (there are exactly two, hand-written in the stellar component), but a
#: structural axis of ``SEDModel.build`` all the same — and a builder-accepted
#: value named by no menu is undiscoverable by construction (#1446).
_AGE_KERNELS: tuple[tuple[str, str, str], ...] = (
    (
        "cic",
        "production",
        "Cloud-in-cell on a 16x dense integrand — the accuracy default (#964)",
    ),
    (
        "dsps",
        "comparison",
        "DSPS histogram kernel — cross-code parity only; biases optical CSP +1.2 %",
    ),
)


def list_age_kernels(*, status: str | None = None) -> _RegistryTable:
    """List the SFH→SSP age-weight kernels selectable via ``sfh={'age_kernel': ...}``.

    The kernel decides how the star-formation history is integrated onto the SSP
    age grid. ``'cic'`` splits each ``SFR(t)*dt`` parcel between its bracketing
    SSP nodes with log-age cloud-in-cell weights on a dense integrand;
    ``'dsps'`` hands the coarse per-SSP-age table to DSPS's histogram kernel,
    which interpolates ``log10(M(<t))`` in ``log10(t)``.

    They are not interchangeable. The DSPS kernel annihilates the mass of any
    table segment straddling the SFH's maximum age — the first SSP node older
    than the SFH start keeps ~1e-5 of its share — which biases the optical CSP
    +1.2 % versus FSPS / bagpipes / a dense reference (#964). It is offered for
    comparison against DSPS-native pipelines, not for science.

    Leaving ``age_kernel`` unset auto-selects: ``'cic'`` on the parametric path,
    ``'dsps'`` on the GP-field path (whose draw lives on its own coarse lookback
    grid, so there is no dense integrand to cloud-in-cell).

    Parameters
    ----------
    status : str, optional
        Filter to one status — ``"production"`` or ``"comparison"``.

    Returns
    -------
    _RegistryTable
        One row per kernel: ``name``, ``status``, ``short_doc``.

    See also: :func:`list_sfh_models`, :mod:`tengri.builders.sfh`.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_age_kernels()  # doctest: +SKIP
    """
    out = [
        {
            "name": name,
            "kind": "age_kernel",
            "status": st,
            "citation": "hearin2021" if name == "dsps" else "",
            "short_doc": doc,
            "use": f"SEDModel.build(sfh={{'age_kernel': {name!r}}})",
        }
        for name, st, doc in _AGE_KERNELS
    ]
    out = _filter_menu(out, "status", status, listing="list_age_kernels")
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


_COMPONENT_DOCS: tuple[tuple[str, str, str], ...] = (
    (
        "stellar",
        "tengri.components.stellar.component",
        "SSP integration over SFH (DSPS), metallicity history, mass remaining",
    ),
    (
        "dust",
        "tengri.components.dust.component",
        "Two-component attenuation (BC + diffuse); see list_dust_laws()",
    ),
    (
        "agn",
        "tengri.components.agn.component",
        "Disc + torus + polar dust + BLR/NLR; see list_agn_models / list_agn_blocks",
    ),
    (
        "nebular",
        "tengri.components.nebular.component",
        "Emission lines + continuum — BakedIn / CUE / CloudyGrid / CB19",
    ),
    (
        "radio",
        "tengri.components.radio.component",
        "SF synchrotron + AGN jets + free-free continuum",
    ),
    ("igm", "tengri.components.igm.component", "IGM transmission (Inoue+2014) + DLA absorption"),
    ("xray", "tengri.components.xray.component", "X-ray binaries + AGN corona"),
)


_PLOT_HELPERS: tuple[tuple[str, str], ...] = (
    ("plot_sed_fit", "Plot observed photometry/spectrum + posterior SED + uncertainty band"),
    ("plot_spectrum_fit", "Plot spectrum + posterior model with calibration polynomial"),
    ("plot_sfh", "Plot single SFH(t) curve from a Parameters / Posterior"),
    ("plot_sfh_comparison", "Overlay multiple SFH(t) curves (e.g. truth vs posterior)"),
    ("plot_corner_comparison", "Two-posterior corner plot (e.g. with truth)"),
    ("safe_corner", "Corner plot wrapper that handles fixed parameters gracefully"),
    ("plot_1d_posterior", "Marginal histogram of one parameter + median/16/84"),
    ("plot_calibration", "Chebyshev calibration polynomial with 16/84 band"),
    ("setup_style", "Apply tengri matplotlib style (serif, tight, 150 dpi)"),
    ("diagnostics_table", "ESS / R-hat / divergences table for sampling diagnostics"),
)


def list_plots() -> _RegistryTable:
    """List the plotting helpers in ``tengri.plot``.

    These render directly with matplotlib — no separate plotting framework.
    Each row carries the canonical call site so a notebook user can
    discover and copy without leaving the cell.
    """
    return _RegistryTable(
        [
            {
                "name": name,
                "kind": "plot",
                "status": "production",
                "short_doc": doc,
                "use": f"tengri.plot.{name}(...)",
            }
            for name, doc in _PLOT_HELPERS
        ]
    )


# SVO filter filenames are ``Telescope_Instrument_Band.dat``; for several
# common cases the astronomer-conventional name (what people *say* in
# papers and seminars) is the *instrument*, not the SVO "telescope":
#
#     SLOAN_SDSS_g   → people say "SDSS g"
#     CTIO_DECam_*   → "DES" / "DECam"
#     Subaru_HSC_*   → "HSC"
#     Paranal_VISTA_*→ "VISTA"
#     UKIRT_UKIDSS_* → "UKIDSS"
#     PAN-STARRS_PS1_* → "PS1"
#
# So ``list_filters(survey="SDSS")`` should return the SLOAN/SDSS rows.
# This map translates an astronomer-spoken survey name into the
# ``(survey, instrument)`` pair we should match against, lowercase.
_SURVEY_ALIASES: dict[str, tuple[str, str]] = {
    "sdss": ("sloan", "sdss"),
    "des": ("ctio", "decam"),
    "decam": ("ctio", "decam"),
    "vista": ("paranal", "vista"),
    "hsc": ("subaru", "hsc"),
    "suprime": ("subaru", "suprime"),
    "ukidss": ("ukirt", "ukidss"),
    "pan-starrs": ("pan-starrs", "ps1"),
    "panstarrs": ("pan-starrs", "ps1"),
    "ps1": ("pan-starrs", "ps1"),
}


def cite_components(obj=None) -> _RegistryTable:
    """Citations for every physics component a Parameters / SEDModel / Posterior uses.

    This is the "live" citation walk: read straight from the registry
    metadata you (or contributors) populated via the
    ``citation=`` kwarg on ``@register_agn_model`` /
    ``@register_dust_law`` / SFH ``_register`` / etc.

    Goes beyond :func:`tengri.collect_citations` (which uses a static
    BibTeX-key association table) — every per-alternative citation is
    pulled live, so a contributor adding a new model with a fresh
    ``citation="Author+Year"`` immediately appears here.

    Parameters
    ----------
    obj : Parameters or SEDModel or Posterior, optional
        Object whose structural choices to inspect.  If ``None``, returns
        the citations attached to the four core dependencies (tengri,
        JAX, DSPS) plus an empty per-component slate.

    Returns
    -------
    _RegistryTable
        One row per component used.  Columns: ``component`` (where in
        the SED model), ``name`` (the registered alternative chosen),
        ``citation`` (the live string from the registry), ``kind``.

    Examples
    --------
    >>> spec = tengri.Parameters(
    ...     mean_sfh_type="dpl", agn_model="skirtor", dust_emission="dl07_tabulated"
    ... )
    >>> tengri.cite_components(spec)
    >>>
    >>> # Same call works on the SEDModel and Posterior — they expose .spec
    >>> tengri.cite_components(model)
    >>> tengri.cite_components(posterior)
    """
    rows: list[dict] = []

    def _add(component: str, name: str | None, table_fn) -> None:
        if not name:
            return
        # Strip the suffix Parameters internally adds (e.g. "dl07_tabulated"
        # vs the registered "dl07") — fall back to the raw name if exact.
        candidates = [name]
        if name.endswith("_tabulated"):
            candidates.append(name[: -len("_tabulated")])
        for entry in table_fn():
            if entry["name"] in candidates:
                rows.append(
                    {
                        "component": component,
                        "name": entry["name"],
                        "citation": entry.get("citation", ""),
                        "kind": entry.get("kind", "?"),
                    }
                )
                return

    # Resolve obj → underlying Parameters spec
    spec = obj
    if spec is None:
        rows.append(
            {
                "component": "framework",
                "name": "tengri",
                "citation": "Cooray et al. (2026, Paper I)",
                "kind": "framework",
            }
        )
        return _RegistryTable(rows)

    # Resolve the spec through the delegation chain. A ``SEDModel`` exposes
    # ``.spec`` directly, but a ``Posterior`` does not — it holds the fitted
    # model under ``_model``, which may itself be a ``ForwardModel`` wrapping
    # the SED. A bare ``getattr(obj, "spec", obj)`` fell back to the Posterior
    # itself, which has no structural fields, so citing a *fit result* emitted
    # only the core dependencies.
    from tengri.citations.collect import _citable_chain

    spec = next(
        (
            s
            for s in (getattr(node, "spec", None) for node in _citable_chain(obj))
            if s is not None
        ),
        obj,
    )

    # Always-present dependencies
    rows.append(
        {
            "component": "framework",
            "name": "tengri",
            "citation": "Cooray et al. (2026, Paper I)",
            "kind": "framework",
        }
    )
    rows.append(
        {
            "component": "ssp",
            "name": "DSPS",
            "citation": "Hearin et al. 2023 (MNRAS 521, 1741)",
            "kind": "framework",
        }
    )
    rows.append(
        {
            "component": "framework",
            "name": "JAX",
            "citation": "Bradbury et al. 2018",
            "kind": "framework",
        }
    )

    # SFH (mean_sfh_type can be str or list[str])
    sfh_types = getattr(spec, "mean_sfh_type", None)
    if isinstance(sfh_types, str):
        sfh_types = [sfh_types]
    if sfh_types:
        for sfh in sfh_types:
            _add("sfh", sfh, list_sfh_models)

    # AGN — composable models fan out into their six block slots (disc, torus,
    # nlr, blr, feii, attenuation), each carrying its own citation. Citing the
    # bare ``agn_model`` would report only the "composable" wrapper, whose entry
    # has no paper — silently dropping every real AGN citation (Stalevski for
    # SKIRTOR, Fritz, Nenkova, Kubota & Done, …) that the model actually uses.
    # Block names are not unique across categories (``skirtor`` is both a disc
    # and a torus), so each slot is resolved within its own category.
    _AGN_BLOCK_SLOTS = (
        ("agn_disc_block", "disc"),
        ("agn_torus_block", "torus"),
        ("agn_nlr_block", "nlr"),
        ("agn_blr_block", "blr"),
        ("agn_feii_block", "feii"),
        ("agn_attenuation_block", "attenuation"),
    )
    active_agn_blocks = [
        (getattr(spec, attr, None), category)
        for attr, category in _AGN_BLOCK_SLOTS
        if getattr(spec, attr, None) not in (None, "none")
    ]
    agn_model = getattr(spec, "agn_model", None)
    if active_agn_blocks:
        for block_name, category in active_agn_blocks:
            _add(
                f"agn_{category}",
                block_name,
                lambda c=category: list_agn_blocks(category=c),
            )
    elif agn_model and agn_model != "composable":
        _add("agn", agn_model, list_agn_models)

    # Dust model — the birth-cloud + diffuse *geometry*, a separate paper from
    # the attenuation *curve* below (Charlot & Fall 2000 vs Calzetti et al.
    # 2000). ``two_component`` is the recommended default path, so leaving this
    # menu out of the walk dropped that citation from nearly every fit.
    _add("dust", getattr(spec, "dust_model", None), list_dust_models)

    # Dust attenuation — bc + diff (skip plain "power_law" default if both equal it)
    for attr in ("dust_law_bc", "dust_law_diff", "dust_law"):
        _add("dust_attenuation", getattr(spec, attr, None), list_dust_laws)

    # Dust emission
    _add("dust_emission", getattr(spec, "dust_emission", None), list_dust_emission_models)

    # Nebular
    nebular_mode = getattr(spec, "nebular_mode", None)
    if nebular_mode and nebular_mode != "off":
        _add("nebular", nebular_mode, list_nebular_backends)

    # Radio, shock, IGM and X-ray — four menus the walk never consulted, so a
    # spec that explicitly requested them produced no row and no warning
    # (#1447). Each is gated on its own boolean, because the slot attributes
    # keep real defaults ("bell2003", "yang20") while the component is switched
    # off: an ungated walk would credit Bell 2003 and Yang+2020 to a plain
    # stellar+dust fit. Over-citing is as wrong as under-citing here.

    # Radio — two independent slots. Block names are not unique across the
    # categories ("none" is registered in both), so each is resolved inside its
    # own category, exactly as the composable AGN blocks are above.
    if getattr(spec, "radio", False):
        for attr, category in (("radio_sfr_mode", "sf"), ("radio_agn_model", "agn")):
            block = getattr(spec, attr, None)
            if block and block != "none":
                _add(
                    f"radio_{category}",
                    block,
                    lambda c=category: list_radio_blocks(category=c),
                )

    # Shock — the spec records only the on/off gate plus physics parameters,
    # with no ``shock_model`` attribute, so an enabled gate implies the single
    # selectable entry. A contract test fails loudly if a second shock model is
    # ever registered and that inference stops holding.
    if getattr(spec, "shock", False):
        _add("shock", "mappings", list_shock_models)

    # IGM — applied by default, so most fits genuinely ran Inoue+2014 (or the
    # chosen alternative) and have never cited it.
    if getattr(spec, "apply_igm", False):
        igm_model = getattr(spec, "igm_model", None)
        if igm_model and igm_model != "none":
            _add("igm", igm_model, list_igm_models)

    # X-ray. ``agn_xray_corona`` and ``xray_aird`` are accepted by the builder
    # but absent from ``list_xray_models()``, so they resolve to no row until
    # that menu derives from the same source as the builder (#1446); the walk
    # then picks them up with no change here.
    if getattr(spec, "xray", False):
        xray_model = getattr(spec, "xray_model", None)
        if xray_model and xray_model != "none":
            _add("xray", xray_model, list_xray_models)

    # Inference method (from a Posterior)
    method = getattr(obj, "method", None)
    if method:
        _add("inference", method, list_inference_methods)

    # Deduplicate by (component, name) preserving order
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        key = (r["component"], r["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return _RegistryTable(out)


def print_components_bibtex(obj=None) -> None:
    """Print BibTeX entries for every component used by ``obj``.

    Walks :func:`cite_components` to discover the components, then for
    each component looks up the formal BibTeX entry in the bundled
    citation registry (``tengri.citations.references.bib``).  Components
    with a registered free-form ``citation=`` string but no BibTeX entry
    in the bundled registry are emitted as a ``%`` comment line so the
    user knows to track down the BibTeX themselves.

    Output is paste-ready into a paper's ``.bib`` file.

    Examples
    --------
    >>> spec = tengri.Parameters(
    ...     mean_sfh_type="dpl", agn_model="skirtor", dust_emission="dl07_tabulated"
    ... )
    >>> tengri.print_components_bibtex(spec)
    @article{Stalevski_2016, ...}
    @article{Carnall_2018, ...}
    @article{Draine_2007, ...}
    ...
    """

    rows = cite_components(obj)
    _display("% ────────────────────────────────────────────────────────────────")
    _display(
        f"%  Citations for {len(rows)} component{'s' if len(rows) != 1 else ''} "
        "used by the model.  Paste into your .bib file."
    )
    _display("% ────────────────────────────────────────────────────────────────")
    _display("")

    try:
        from tengri.citations import cite as _cite_lookup
        from tengri.citations.resolve import citation_keys_for
    except ImportError:
        _cite_lookup = None

        def citation_keys_for(_name):
            return []

    seen_keys: set[str] = set()
    for row in rows:
        name = row["name"]
        comp = row["component"]
        emitted = False
        for bibkey in citation_keys_for(name):
            if bibkey in seen_keys or _cite_lookup is None:
                emitted = emitted or bibkey in seen_keys
                continue
            try:
                citation = _cite_lookup(bibkey)
                bib_method = getattr(citation, "to_bibtex", None)
                if callable(bib_method):
                    _display(f"% [{comp}] {name}")
                    _display(bib_method())
                    _display("")
                    seen_keys.add(bibkey)
                    emitted = True
            except Exception:
                continue
        if emitted:
            continue
        # Fallback: free-form citation note. Say what is actually missing —
        # a *mapping* from this component name to a key, not necessarily the
        # entry. Four references (Charlot & Fall 2000, Bell 2003, Inoue+2014,
        # Yang+2020) were in references.bib the whole time and still printed
        # "no bib entry", so readers pasted the output and silently lost them.
        cit = row.get("citation", "")
        if cit:
            _display(f"% [{comp}] {name}: {cit}")
            _display(
                f"%   (no BibTeX key is mapped to {name!r} — add one to "
                "tengri.citations.associations)"
            )
            _display("")


def list_filters(survey: str | None = None) -> _RegistryTable:
    """List every filter curve bundled with tengri.

    Filter files live in ``data/filters/`` (relative to the install root)
    and follow the SVO naming convention ``Telescope_Instrument_Band.dat``.

    Parameters
    ----------
    survey : str, optional
        Narrow to one survey/instrument family (case-insensitive).  Smart
        about the SVO-vs-astronomer-speak mismatch: ``survey="SDSS"``
        finds the ``SLOAN_SDSS_*`` rows even though SDSS is technically
        the *instrument* in SVO's filename schema.  Other recognized
        astronomer aliases: ``DES``/``DECam`` → CTIO/DECam,
        ``VISTA`` → Paranal/VISTA, ``HSC`` → Subaru/HSC,
        ``UKIDSS`` → UKIRT/UKIDSS, ``PS1`` → PAN-STARRS/PS1.

    Returns
    -------
    _RegistryTable
        One row per filter, with columns ``name`` (file stem),
        ``survey``, ``instrument``, ``band``, and ``alias`` — the short
        spelling the loaders also accept (``"sdss_r"`` for
        ``"SLOAN_SDSS_r"``), empty when the curve has no alias. Prints as
        a table, also renders as HTML in Jupyter.

    See Also
    --------
    tengri.observation.filters.list_filter_aliases : the same curves keyed
        by short alias. Both names were once ``list_filters`` (#1574).

    Notes
    -----
    Counts and groupings are computed from the live filesystem on each
    call — no hardcoding. Add a ``Survey_Instrument_Band.dat`` file to
    ``data/filters/`` and it appears here automatically.

    Examples
    --------
    >>> tengri.list_filters(survey="SDSS")  # 5 rows
    >>> tengri.list_filters(survey="JWST")  # all JWST instruments
    >>> tengri.list_filters().filter(  # finer-grained
    ...     instrument="NIRCam", band__contains="F150"
    ... )
    """
    import os

    # Resolve filter directory: prefer repo-local ``data/filters/``,
    # fall back to package-local if installed wheel-style.
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "filters"),
        os.path.join(os.path.dirname(__file__), "data", "filters"),
        os.path.join(os.getcwd(), "data", "filters"),
    ]
    filter_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if filter_dir is None:
        return _RegistryTable([])

    # Short aliases the loaders also accept ("sdss_r" for "SLOAN_SDSS_r").
    # Imported here, not at module scope: observation.filters imports this
    # module, so a module-level import would cycle.
    from tengri.observation.filters import FILTER_REGISTRY

    # A curve may carry more than one alias (2MASS/2MASS.J is both
    # "2mass_j" and "johnson_j"), so collect them all rather than letting
    # the last one silently win.
    _aliases: dict[str, list[str]] = {}
    for short, svo_id in FILTER_REGISTRY.items():
        _aliases.setdefault(svo_id.replace("/", "_").replace(".", "_"), []).append(short)
    alias_by_stem = {stem: ", ".join(sorted(v)) for stem, v in _aliases.items()}

    out = []
    for fname in sorted(os.listdir(filter_dir)):
        if not fname.endswith(".dat"):
            continue
        stem = fname[:-4]
        parts = stem.split("_", 2)
        sv = parts[0] if len(parts) >= 1 else ""
        instr = parts[1] if len(parts) >= 2 else ""
        band = parts[2] if len(parts) >= 3 else ""
        out.append(
            {
                "name": stem,
                "kind": "filter",
                "survey": sv,
                "instrument": instr,
                "band": band,
                "alias": alias_by_stem.get(stem, ""),
                "use": _usage_hint(stem, "filter"),
            }
        )

    # Apply optional survey filter, with astronomer-friendly aliases.
    if survey is not None:
        q = str(survey).lower().strip()
        target = _SURVEY_ALIASES.get(q, (q, q))  # (survey_lc, instrument_lc)

        def _match(entry: dict) -> bool:
            sv_lc = str(entry.get("survey", "")).lower()
            in_lc = str(entry.get("instrument", "")).lower()
            # Match either field against either component of the alias —
            # so "SDSS" hits SLOAN/SDSS rows and "SLOAN" still works too.
            return q in (sv_lc, in_lc) or target[0] == sv_lc or target[1] == in_lc

        out = [e for e in out if _match(e)]

    return _RegistryTable(out)


def list_components() -> _RegistryTable:
    """List the SEDComponent adapters currently wired into the forward model."""
    out = []
    for name, module_path, short_doc in _COMPONENT_DOCS:
        try:
            __import__(module_path)
            out.append(
                {
                    "name": name,
                    "kind": "component",
                    "status": "production",
                    "module": module_path,
                    "short_doc": short_doc,
                    "use": _usage_hint(name, "component"),
                }
            )
        except Exception as e:
            out.append(
                {
                    "name": name,
                    "kind": "component",
                    "status": "broken",
                    "module": module_path,
                    "short_doc": f"import failed: {e!r}",
                    "use": "",
                }
            )
    return _RegistryTable(out)


def _inference_method_row(entry, target: object | None = None) -> dict:
    """Build the table row for one backend entry.

    Shared by :func:`list_inference_methods` and
    :func:`describe_inference_method` so the menu and the per-name lookup
    can never disagree about what a backend is.
    """
    from tengri.inference._strategy import resolve_status

    use = _usage_hint(entry.name, "inference_method")
    if entry.tier == "broken":
        # Advice that raises is the bug, not the help (#1364). The plain
        # ``fitter.run("pathfinder")`` hint is refused by the tier gate, and
        # a broken backend only became visible here at all once describe
        # stopped hiding it (#1560) — so ship the invocation that works.
        use = f'fitter.run("{entry.name}", allow_unvalidated=True)  # tier=broken'
    return {
        "name": entry.name,
        "kind": "inference_method",
        "tier": entry.tier,
        "short_doc": entry.short_doc,
        "requires": list(entry.requires),
        "status": resolve_status(entry, target).value,
        "use": use,
    }


def list_inference_methods(
    *, tier: str | None = None, target: object | None = None
) -> _RegistryTable:
    """List all registered inference methods.

    Parameters
    ----------
    tier : str, optional
        Filter by ``"primary"`` (recommended for new users),
        ``"experimental"``, or ``"broken"``. Backends registered as
        ``"broken"`` — those whose own ``short_doc`` reports wrong answers or
        crashes — are **excluded from the default listing** (#1287); pass
        ``tier="broken"`` to see them.
    target : Fitter | InferenceContext, optional
        If supplied, each entry's ``status`` column reflects whether the
        backend's ``is_compatible`` predicate (if any) accepts the
        target. If ``None``, ``status`` reflects only whether the
        backend's optional dependencies are importable.

    Returns
    -------
    _RegistryTable
        Rows: ``{name, kind, tier, short_doc, requires, status, use}``.
        ``status`` is one of ``"ok"`` / ``"missing_dep"`` / ``"incompatible"``
        (see :class:`tengri.inference._strategy.BackendStatus`).
    """
    from tengri.inference._backend_registry import all_backends

    # Broken backends are listed only on explicit request. Offering a sampler
    # that returns R-hat ~ 3 in the same table as one that works is what let
    # users pick it on the strength of its speed (#1287).
    out = [
        _inference_method_row(entry, target)
        for entry in all_backends(include_broken=tier == "broken")
    ]
    out = _filter_menu(out, "tier", tier, listing="list_inference_methods")
    return _RegistryTable(out)


# ──────────────────────────────────────────────────────────────────
# Describe + overview
# ──────────────────────────────────────────────────────────────────


def _menu_listers() -> tuple:
    """The canonical set of per-menu ``list_*`` functions.

    :func:`describe`, :func:`search`, and :func:`list_all` all walk this one
    tuple, so adding a new physics group (a new ``list_*`` menu) can't
    silently leave one of them behind. That drift is exactly what once made
    ``describe()`` and ``search()`` blind to the xray / radio / igm menus and
    to the composable AGN blocks — the models auto-register into their own
    registries, but these aggregators re-listed which registries to consult by
    hand and fell out of sync.
    """
    return (
        list_inference_methods,
        list_agn_models,
        list_agn_blocks,
        list_dust_models,
        list_dust_laws,
        list_dust_emission_models,
        list_sfh_models,
        list_age_kernels,
        list_metallicity_modes,
        list_nebular_backends,
        list_xray_models,
        list_radio_models,
        list_radio_blocks,
        list_shock_models,
        list_igm_models,
    )


#: Menus that are not a physics group but are still menus a user can look a
#: name up in. ``describe`` and ``search`` used to append these by hand at each
#: call site — a second and third hand-written enumeration of "every menu".
_EXTRA_MENU_LISTER_NAMES = ("list_components", "list_filters", "list_plots", "list_recipes")


@functools.cache
def _every_menu_lister() -> tuple:
    """Every ``list_*`` menu a name can be looked up in, derived not listed.

    Returns
    -------
    tuple of callable
        Zero-argument listers, de-duplicated, physics groups first so that a
        name living in several menus reports them in a stable order.

    Notes
    -----
    :func:`_menu_listers` exists so ``describe``/``search``/``list_all`` cannot
    fall out of sync when a physics group is added — its docstring says so, and
    names that drift (#1120, #1446). It drifted again anyway, because the guard
    against a hand-written list *is itself a hand-written list*:
    ``list_instruments`` and ``list_known_ssps`` were never added, so
    ``describe('GALEX')`` answered ``Unknown name 'GALEX'`` for a name
    ``list_instruments()`` advertises — 30 of 490 advertised names.

    So the set is discovered: every public ``tengri.list_*`` returning rows
    with a ``name`` column is a menu. Adding a menu now costs nothing, and
    forgetting to register it here is not possible. Measured when this replaced
    the hand-written unions: **+2 menus, 0 lost, 460 -> 490 rows walked, and 0
    new multi-menu names**, so no existing lookup changes its answer.

    The population scanned is the **union of both export lists**, and that
    matters as much as deriving the set at all. ``dir(tengri)`` is not the
    public surface: it is curated down to ~30 obvious entry points on purpose
    ("not the 175-item kitchen sink of every public symbol",
    :mod:`tengri.__init__`). Scanning it alone made the derivation inherit the
    curation — ``list_parameters``, ``list_properties``,
    ``list_filter_conventions`` and ``list_available_ssps`` are exported and
    not curated, so they stayed invisible and **410 further advertised names
    stayed refused**, 358 of them parameters. Unioning ``__all__`` is the rule
    #1608 established when the same blind spot made ``check_api_coverage.py``
    report 0 missing while 6 were. Measured at that widening: **+4 menus, 0
    lost, 490 -> 935 rows walked, 463 -> 887 names (+424), and 0 answers
    changed** — the last of which is true only because of the scan order
    below, not for free.

    Cached because discovery *calls* each ``list_*`` to check its shape:
    uncached that was 95 ms, **49% of a 193 ms** ``describe()``, paid again on
    every lookup. The set of menus cannot change within a process, so it is
    computed once. A test that monkeypatches a ``list_*`` into ``tengri`` must
    call ``_every_menu_lister.cache_clear()``.
    """
    import tengri

    seen: dict[str, callable] = {}
    for fn in (*_menu_listers(), *(getattr(tengri, n, None) for n in _EXTRA_MENU_LISTER_NAMES)):
        if fn is not None:
            seen[fn.__name__] = fn
    # Curated names first, export-only names after. Order is not cosmetic:
    # ``describe`` reports ``matches[0]``, so a menu discovered earlier wins a
    # name that several menus print. Scanning the union in one sorted pass put
    # ``list_available_ssps`` ahead of ``list_known_ssps`` and moved all 21 SSP
    # names onto the newly-found menu — 21 answers changed, for a change whose
    # whole claim is that it adds names without altering any.
    curated = sorted(dir(tengri))
    export_only = sorted(set(tengri.__all__) - set(dir(tengri)))
    for attr in (*curated, *export_only):
        if not attr.startswith("list_") or attr in seen:
            continue
        fn = getattr(tengri, attr, None)
        if not callable(fn):
            continue
        try:
            rows = fn()
        except Exception:
            continue
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "name" in rows[0]:
            seen[attr] = fn
    return tuple(seen.values())


def _menu_name_aliases() -> dict[str, tuple[str, callable]]:
    """Map each menu's own name, in prose, to that menu.

    Derived from :func:`_menu_listers` rather than hand-written, because a
    hand-written copy is a second source of truth that drifts the day a menu
    is added — the failure this module has already had twice (#1120, #1446).
    ``list_age_kernels`` becomes ``"age kernels"`` and ``"age kernel"``, so a
    new menu is searchable by its own name the moment it is registered.

    This is deliberately separate from the hand-written concept synonyms in
    :func:`search`: those map words a beginner invents ("extinction") onto a
    menu, which cannot be derived from anything.
    """
    out: dict[str, tuple[str, callable]] = {}
    for fn in _menu_listers():
        stem = fn.__name__.removeprefix("list_")
        plural = stem.replace("_", " ")
        forms = {plural}
        if plural.endswith("s"):
            forms.add(plural[:-1])
        for form in forms:
            out[form] = (f"{fn.__name__}()", fn)
    return out


def describe(name: str) -> _DescribeRecord:
    """Universal lookup across every menu.

    Parameters
    ----------
    name : str
        Name of a model, method, or component.

    Returns
    -------
    _DescribeRecord
        Metadata dict; prints as a labeled block.

    Raises
    ------
    KeyError
        If the name is not registered anywhere.
    """
    # Core classes — ForwardModel, SEDModel, Parameters, Fitter, Posterior, Observation.
    if name in _CORE_CLASSES:
        return _DescribeRecord(_CORE_CLASSES[name])

    matches = [entry for fn in _every_menu_lister() for entry in fn() if entry["name"] == name]
    if matches:
        record = dict(matches[0])
        # Some names are registered in more than one menu or AGN category —
        # e.g. 'skirtor' is both a disc and a torus, 'simple' is both a torus
        # block and an X-ray model, 'cue' is both an NLR block and a nebular
        # backend. Returning the first match silently would describe the wrong
        # component; disclose every place the name lives so the user can pick.
        if len(matches) > 1:

            def _where(entry: dict) -> str:
                category = entry.get("category")
                kind = entry.get("kind", "?")
                return f"{kind} ({category})" if category else kind

            # Quote each alternative's own ``use:`` line rather than naming a
            # lookup helper. The message used to recommend
            # ``describe_agn_block(name, category=...)`` for every ambiguity,
            # which was true only while AGN blocks were the sole categorized
            # menu: once radio gained sf/agn categories the advice raised
            # ("Unknown AGN block 'dpl' ... Known names: []") for exactly the
            # user it was written to help. Each row already carries an exact,
            # copy-pasteable build call, so quote that and it cannot go stale.
            def _how(entry: dict) -> str:
                use = entry.get("use", "")
                return use or f"see {_where(entry)}"

            others = "; ".join(_how(m) for m in matches[1:])
            record["also_registered_as"] = (
                f"'{name}' is registered in {len(matches)} places "
                f"[{'; '.join(_where(m) for m in matches)}] — showing the first "
                f"({_how(matches[0])}). The others: {others}."
            )
        return _DescribeRecord(record)

    # The sweep above walks the *curated* menus, so it cannot see a name the
    # menu hides by design: a ``tier="broken"`` backend, or an alias that
    # rows under its canonical name. Both are still dispatchable, and this
    # generic entry point must answer for them exactly as
    # :func:`describe_inference_method` does (#1560).
    from tengri.inference._backend_registry import lookup_backend

    if lookup_backend(name) is not None:
        return describe_inference_method(name)

    raise KeyError(
        f"Unknown name '{name}'.  Try tengri.summary() for a menu of every "
        "core class, AGN model, dust law, SFH variant, nebular backend, "
        "component, filter, or inference method that exists."
    )


# ── Recipe discovery (#310 proposal 2) ─────────────────────────────────────


def list_recipes() -> _RegistryTable:
    """List every callable in :mod:`tengri.recipes`.

    Each entry carries the recipe name, the first sentence of its docstring,
    and the SSP-requirement tag (parsed from the ``**SSP requirement:**``
    line in the docstring, when present). Use :func:`describe_recipe` for the
    full per-recipe block (component selectors, parameter freedoms,
    suggested redshift, etc.).

    Returns
    -------
    _RegistryTable
        Rows: ``{name, short_doc, ssp_requirement, kind="recipe"}``.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_recipes()
    >>> tengri.describe_recipe("star_forming_photometry")
    """
    import inspect

    from tengri import recipes as _recipes

    out: list[dict] = []
    for name in sorted(_recipes.__all__):
        fn = getattr(_recipes, name, None)
        if fn is None or not callable(fn):
            continue
        doc = inspect.getdoc(fn) or ""
        short_doc = doc.split("\n\n", 1)[0].strip().replace("\n", " ")
        ssp_req = _parse_ssp_requirement(doc)
        # Calling the recipe is cheap — it returns a kwargs dict and builds
        # nothing — so the table can report what each one actually needs
        # rather than what its prose claims.
        try:
            data_status = _recipe_data_status(fn())
        except Exception:
            data_status = "unknown"
        out.append(
            {
                "name": name,
                "kind": "recipe",
                "short_doc": short_doc,
                "ssp_requirement": ssp_req,
                "data": data_status,
                "use": f"recipes.{name}() → SEDModel.build(ssp_data=ssp, **recipe)",
            }
        )
    return _RegistryTable(out)


#: Component types whose data ships separately from tengri, mapped to the
#: ``kind`` argument their loader resolves. Keyed by the value a recipe puts in
#: a block's ``type``.
_EXTERNAL_GRID_BLOCKS: dict[str, str] = {
    "synthesizer": "nlr",
    "synthesizer_spectra": "nlr",
}


def _recipe_data_status(kwargs: dict) -> str:
    """Report whether a recipe's non-SSP data is present on this machine.

    ``list_recipes`` presented all ten recipes as equals while one of them —
    ``unified_agn`` — cannot produce a number without a Synthesizer AGN grid
    that does not ship with tengri. A recipe is by definition the thing a new
    user is told to start from, so "this one needs a download" belongs in the
    table rather than in a traceback (#1462 §3).

    The check calls the **loader's own resolver** rather than re-deriving the
    search path. A second copy of "where does this file live" would drift, and
    a column that says ``ready`` while the loader disagrees is worse than no
    column at all.

    Returns
    -------
    str
        ``"ready"`` when nothing extra is needed, or a short note naming what
        is missing. Never raises: an unresolvable requirement is the answer,
        not an error.
    """
    needed: set[str] = set()

    def _walk(node):
        if isinstance(node, dict):
            block_type = node.get("type")
            if isinstance(block_type, str) and block_type in _EXTERNAL_GRID_BLOCKS:
                needed.add(_EXTERNAL_GRID_BLOCKS[block_type])
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)

    _walk(kwargs)
    if not needed:
        return "ready"

    from tengri.components.agn.blocks.nlr import _resolve_synthesizer_grid

    missing = []
    for kind in sorted(needed):
        try:
            _resolve_synthesizer_grid(kind)
        except Exception:
            missing.append(kind)
    if not missing:
        return "ready"
    kinds = "/".join(k.upper() for k in missing)
    return f"needs Synthesizer AGN {kinds} grid (synthesizer-download --agn-test-grids)"


def _parse_ssp_requirement(doc: str) -> str:
    """Pull the ``**SSP requirement:**`` value from a recipe docstring.

    The requirement is a *paragraph*, not a line. Numpydoc wraps it at the
    line limit, and every docstring puts the label first and the consequence
    second — so a first-line-only read keeps ``bare-stellar (Cue nebular
    backend; see`` and drops ``doing so raises CueWNESSPError``. Read to the
    paragraph break instead, matching how ``short_doc`` is derived just above.
    """
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if "SSP requirement" in line:
            after = line.split(":", 1)[1] if ":" in line else line
            parts = [after]
            for cont in lines[i + 1 :]:
                # A blank line ends the paragraph; a new ``**Field:**`` marker
                # ends it too, for docstrings that run fields together.
                if not cont.strip() or cont.lstrip().startswith("**"):
                    break
                parts.append(cont)
            return _plain_text(" ".join(p.strip() for p in parts))
    return "any"


def _plain_text(text: str) -> str:
    """Strip the reST inline markup a docstring carries but a table should not.

    ``list_recipes()`` renders in a terminal, not in Sphinx, so ``:func:`x```
    and ````x```` reach the user as literal punctuation. Dropping the roles and
    the backticks leaves the prose the docstring author actually wrote.
    """
    out = re.sub(r":(?:func|meth|class|mod|data|attr|ref):`~?([^`]*)`", r"\1", text)
    out = out.replace("``", "").replace("*", "")
    return " ".join(out.split()).strip()


def describe_recipe(name: str) -> _DescribeRecord:
    """Return the full descriptor block for a single recipe.

    Parameters
    ----------
    name : str
        Recipe name (see :func:`list_recipes`).

    Returns
    -------
    _DescribeRecord
        Dict-like with ``name``, ``docstring``, ``ssp_requirement``,
        ``returns`` (the actual nested-dict the recipe builds), and a
        ready-to-paste ``use`` example.

    Raises
    ------
    KeyError
        If ``name`` is not in :mod:`tengri.recipes`.
    """
    import inspect

    from tengri import recipes as _recipes

    fn = getattr(_recipes, name, None)
    if fn is None or not callable(fn):
        known = sorted(_recipes.__all__)
        raise KeyError(f"Unknown recipe '{name}'. Known recipes: {known}")
    doc = inspect.getdoc(fn) or ""
    # Materialize the recipe dict so users can see the actual selectors.
    try:
        recipe_dict = fn()
        component_keys = sorted(k for k in recipe_dict if not k.startswith("_"))
    except Exception:
        recipe_dict = {}
        component_keys = []
    return _DescribeRecord(
        {
            "name": name,
            "kind": "recipe",
            "short_doc": doc.split("\n\n", 1)[0].strip().replace("\n", " "),
            "docstring": doc,
            "ssp_requirement": _parse_ssp_requirement(doc),
            "components": component_keys,
            "use": f"model = tengri.SEDModel.build(ssp_data=ssp, **tengri.recipes.{name}())",
        }
    )


# ── Symmetric describe_* per kind (#310 proposal 1) ────────────────────────


def _describe_from_list(name: str, list_fn, kind_label: str, fn_label: str) -> _DescribeRecord:
    """Common path for ``describe_*(name)`` — look up ``name`` in ``list_fn()``."""
    for entry in list_fn():
        if entry["name"] == name:
            return _DescribeRecord(entry)
    known = sorted(e["name"] for e in list_fn())
    raise KeyError(
        f"Unknown {kind_label} '{name}'. Known names: {known}. See {fn_label}() for the full menu."
    )


def describe_agn_model(name: str) -> _DescribeRecord:
    """Return the descriptor row for one AGN model (citation, status, short doc).

    Symmetric with :func:`list_agn_models`. Use the generic :func:`describe`
    for a cross-kind lookup.
    """
    return _describe_from_list(name, list_agn_models, "AGN model", "list_agn_models")


def describe_agn_block(
    name: str, *, category: str | None = None
) -> _DescribeRecord | list[_DescribeRecord]:
    """Return the descriptor record(s) for one composable AGN block.

    Since block names may not be globally unique (e.g., "grahsp" appears in
    disc, nlr, blr, torus, attenuation categories), this function returns:

    - A single `_DescribeRecord` if ``category`` is specified or the name
      is unambiguous.
    - A list of `_DescribeRecord` (one per matching category) if the name is
      ambiguous and ``category=None``.

    Parameters
    ----------
    name : str
        Block name (e.g., ``"grahsp"``, ``"multicolor"``, ``"analytic"``).
    category : str, optional
        Category to narrow the search: ``"disc"``, ``"nlr"``, ``"blr"``,
        ``"feii"``, ``"torus"``, or ``"attenuation"``. If ``None``, search
        all categories.

    Returns
    -------
    _DescribeRecord or list[_DescribeRecord]
        Single record if unambiguous, list of records if ambiguous.

    Raises
    ------
    KeyError
        If the name is not found in the specified (or any) category.
    """
    blocks = list_agn_blocks(category=category)
    matches = [b for b in blocks if b["name"] == name]

    if not matches:
        # Try a cross-category search
        all_blocks = list_agn_blocks()
        matches = [b for b in all_blocks if b["name"] == name]
        if not matches:
            known_in_category = (
                sorted(b["name"] for b in list_agn_blocks(category=category))
                if category
                else sorted(set(b["name"] for b in list_agn_blocks()))
            )
            raise KeyError(
                f"Unknown AGN block '{name}' {f'in category {category!r}' if category else ''}. "
                f"Known names: {known_in_category}. See list_agn_blocks() for the full menu."
            )

    # If category was specified, we expect exactly one match.
    if category:
        if len(matches) != 1:
            raise KeyError(
                f"Expected 1 match for block '{name}' in category {category!r}, "
                f"got {len(matches)}."
            )
        return _DescribeRecord(matches[0])

    # If category was not specified and there are multiple matches,
    # return them all. Otherwise return the single match.
    if len(matches) == 1:
        return _DescribeRecord(matches[0])
    return [_DescribeRecord(m) for m in matches]


def describe_dust_law(name: str) -> _DescribeRecord:
    """Return the descriptor row for one dust **attenuation** law."""
    return _describe_from_list(name, list_dust_laws, "dust law", "list_dust_laws")


def describe_dust_emission_model(name: str) -> _DescribeRecord:
    """Return the descriptor row for one dust **emission** template family."""
    return _describe_from_list(
        name, list_dust_emission_models, "dust emission model", "list_dust_emission_models"
    )


def describe_sfh_model(name: str) -> _DescribeRecord:
    """Return the descriptor row for one star-formation history model."""
    return _describe_from_list(name, list_sfh_models, "SFH model", "list_sfh_models")


def describe_nebular_backend(name: str) -> _DescribeRecord:
    """Return the descriptor row for one nebular emission backend."""
    return _describe_from_list(
        name, list_nebular_backends, "nebular backend", "list_nebular_backends"
    )


def describe_inference_method(name: str) -> _DescribeRecord:
    """Return the descriptor row for one inference backend (MAP / VI / NUTS / NSS / …).

    Resolves any name the fitter dispatches: every tier — including
    ``"broken"``, which :func:`list_inference_methods` hides by default
    (#1287) — and every registered alias.

    ``list`` and ``describe`` answer different questions. ``list`` asks
    "what should I pick?", so it curates. ``describe`` asks "what is this
    name?", so it must not: a name the fitter accepts is never "unknown".
    Deriving this lookup from the curated listing reported six dispatchable
    names as unknown (#1560) — the five broken backends, plus
    ``"vi_nonlinear"``, a ``tier="primary"`` alias named in ``fit()``'s own
    docstring. A confidently wrong answer is worse than a warning (#1446).

    Parameters
    ----------
    name : str
        A method name or alias, as passed to ``fit(method=...)``.

    Returns
    -------
    _DescribeRecord
        Fields: ``{name, kind, tier, short_doc, requires, status, use}``,
        plus ``alias_of`` when ``name`` is an alias. Check ``tier`` before
        acting on the result — ``"broken"`` backends resolve here but are
        refused by ``Fitter.run`` without ``allow_unvalidated=True``.

    Raises
    ------
    KeyError
        If no backend is registered under ``name``.

    Examples
    --------
    >>> import tengri
    >>> tengri.describe_inference_method("vi_nonlinear")["alias_of"]
    'vi'
    """
    from tengri.inference._backend_registry import lookup_backend

    entry = lookup_backend(name)
    if entry is None:
        from tengri.inference._backend_registry import _BACKENDS

        known = sorted(_BACKENDS)
        raise KeyError(
            f"Unknown inference method '{name}'. Known names: {known}. "
            "See list_inference_methods() for the full menu."
        )
    row = _inference_method_row(entry)
    if name != entry.name:
        row = {**row, "name": name, "alias_of": entry.name}
    return _DescribeRecord(row)


def suggest_parameters(
    *,
    mean_sfh_type: str | list[str] = "dpl",
    agn_model: str | None = None,
    dust_law: str | None = None,
    dust_law_bc: str = "power_law",
    dust_law_diff: str | None = None,
    dust_emission: str | None = None,
    dust_model: str = "two_component",
    nebular_backend: str | None = None,
    eline_mode: str = "off",
    radio: bool = False,
    xray: bool = False,
    shock: bool = False,
    chem_evol: bool = False,
    evolving_metallicity: bool = False,
) -> _RegistryTable:
    """Print the full kwargs cheatsheet for a chosen Parameters() config.

    ``Parameters`` accepts ``**kwargs`` — the legal parameter names are
    determined dynamically by the structural choices (mean_sfh_type,
    agn_model, dust_law, dust_emission, nebular_backend, …).  This
    function answers the working question:

        "I want a DPL SFH with SKIRTOR AGN and DL07 dust emission —
        what kwargs can I pass to Parameters()?"

    by building the full param registry for that configuration and
    returning a printable table with every parameter, its default
    Distribution, and a one-line description.

    Parameters
    ----------
    mean_sfh_type : str or list[str], default "dpl"
        SFH model name (or list including "field" for stochastic).  See
        ``tengri.list_sfh_models()`` for the menu.
    agn_model : str, optional
        Name from ``tengri.list_agn_models()``.  ``None`` → AGN off.
    dust_law : str, optional
        Single-component attenuation curve.  Use either ``dust_law=`` or
        ``dust_law_bc=`` / ``dust_law_diff=`` for two-component.
    dust_emission : str, optional
        IR emission template family from
        ``tengri.list_dust_emission_models()``.
    nebular_backend : str, optional
        Nebular emission backend from ``tengri.list_nebular_backends()``:
        ``"ssp"`` (emission baked into the SSP grid), ``"cue"``,
        ``"cloudy"``, or ``"cb19"``.
    radio, xray, shock, chem_evol, evolving_metallicity : bool
        Toggle the corresponding physics module.
    eline_mode : str, default "off"
        ``"off"`` | ``"marginalize"`` | ``"sample"`` for emission lines.

    Returns
    -------
    _RegistryTable
        One row per parameter with columns ``name``, ``default``,
        ``description``.  Prints as a column-aligned table in the REPL,
        as an HTML table in Jupyter.

    Examples
    --------
    >>> tengri.suggest_parameters(
    ...     mean_sfh_type="dpl", agn_model="skirtor", dust_emission="dl07_tabulated"
    ... )

    >>> # Stochastic SFH with the IFT field
    >>> tengri.suggest_parameters(mean_sfh_type=["dpl", "field"])
    """
    from tengri.parameters._builders import _build_param_registry

    nebular_flag = nebular_backend is not None
    if dust_law and not dust_law_diff:
        # Treat single dust_law= as the diffuse component for two-component
        dust_law_diff = dust_law
    registry, defaults = _build_param_registry(
        mean_sfh_type=mean_sfh_type,
        nebular=nebular_flag,
        dust_model=dust_model,
        dust_law_bc=dust_law_bc,
        dust_law_diff=dust_law_diff,
        dust_emission=dust_emission,
        agn_model=agn_model,
        radio=radio,
        xray=xray,
        shock=shock,
        evolving_metallicity=evolving_metallicity,
        chem_evol=chem_evol,
        eline_mode=eline_mode,
    )

    rows: list[dict] = []
    for name, info in registry.items():
        # registry entry shape: (description, bound_check, bound_error, [default])
        description = info[0] if len(info) >= 1 else ""
        default = defaults.get(name, "—")
        rows.append(
            {
                "name": name,
                "kind": "parameter",
                "default": str(default),
                "description": description,
            }
        )
    rows.sort(key=lambda r: r["name"])

    # Print the configuration banner so the user can see what they
    # asked for echoed back.
    parts = []
    parts.append(f"mean_sfh_type={mean_sfh_type!r}")
    if agn_model:
        parts.append(f"agn_model={agn_model!r}")
    if dust_emission:
        parts.append(f"dust_emission={dust_emission!r}")
    if dust_law or dust_law_diff:
        parts.append(f"dust_law_diff={(dust_law_diff or dust_law)!r}")
    if nebular_backend:
        parts.append(f"nebular_backend={nebular_backend!r}")
    _display(f"\nParameters configuration: {', '.join(parts)}")
    _display(f"  → {len(rows)} parameters.  Pass any subset as kwargs to tengri.Parameters().\n")

    return _RegistryTable(rows)


def search(query: str) -> _RegistryTable:
    """Cross-menu fuzzy search by name, short_doc, citation, or status.

    Walks every menu — components, inference methods, AGN models, dust
    attenuation laws, dust emission templates, SFH models, nebular
    backends — and returns every entry whose name, short_doc, citation,
    or status (case-insensitively) contains ``query``.

    Parameters
    ----------
    query : str
        Substring to match (case-insensitive).

    Returns
    -------
    _RegistryTable
        Matching entries from every menu, with a ``kind`` column so
        you can tell them apart. Prints as a table in the REPL.

    Examples
    --------
    >>> tengri.search("torus")  # find every torus model anywhere
    >>> tengri.search("pah")  # find every PAH-related thing
    >>> tengri.search("Leja")  # find everything Leja-cited
    """
    q = query.lower().strip()

    # Shortcut: when the query *is* a kind name, the user almost certainly
    # wants the menu, not 242 rows that happen to all have kind=filter.
    # Redirect to the appropriate list_*() with a hint.
    _KIND_SHORTCUT: dict[str, tuple[str, callable]] = {
        "filter": ("list_filters()", list_filters),
        "filters": ("list_filters()", list_filters),
        "agn": ("list_agn_models()", list_agn_models),
        "agn_model": ("list_agn_models()", list_agn_models),
        "agn_models": ("list_agn_models()", list_agn_models),
        "sfh": ("list_sfh_models()", list_sfh_models),
        "sfh_model": ("list_sfh_models()", list_sfh_models),
        "sfh_models": ("list_sfh_models()", list_sfh_models),
        "dust_attenuation": ("list_dust_laws()", list_dust_laws),
        "dust_emission": ("list_dust_emission_models()", list_dust_emission_models),
        "nebular": ("list_nebular_backends()", list_nebular_backends),
        "nebular_backend": ("list_nebular_backends()", list_nebular_backends),
        "component": ("list_components()", list_components),
        "components": ("list_components()", list_components),
        "inference_method": ("list_inference_methods()", list_inference_methods),
        "inference": ("list_inference_methods()", list_inference_methods),
    }
    if q in _KIND_SHORTCUT:
        call, fn = _KIND_SHORTCUT[q]
        _display(f"  '{query}' is a menu name — redirecting to tengri.{call}\n")
        return fn()

    # Concept synonyms: natural-language terms a beginner types that do not
    # substring-match the terse model short_docs. "star formation" would
    # otherwise return only an AGN model whose citation title happens to
    # contain the phrase (and none of the 26 SFH models), and "dust emission"
    # nothing at all — so point the user at the menu that actually holds those
    # models. Same replace-with-menu behavior as the kind-name shortcut above.
    _CONCEPT_ALIAS: dict[str, tuple[str, callable]] = {
        "star formation": ("list_sfh_models()", list_sfh_models),
        "star formation history": ("list_sfh_models()", list_sfh_models),
        "star-forming": ("list_sfh_models()", list_sfh_models),
        "star forming": ("list_sfh_models()", list_sfh_models),
        "dust emission": ("list_dust_emission_models()", list_dust_emission_models),
        "infrared emission": ("list_dust_emission_models()", list_dust_emission_models),
        "extinction": ("list_dust_laws()", list_dust_laws),
        "reddening": ("list_dust_laws()", list_dust_laws),
        "emission line": ("list_nebular_backends()", list_nebular_backends),
        "emission lines": ("list_nebular_backends()", list_nebular_backends),
        # Without these, "metallicity" substring-matches only the modes whose
        # terse short_doc happens to use the word — 5 of 10, silently omitting
        # the gas-regulator and per-bin models.
        "metallicity": ("list_metallicity_modes()", list_metallicity_modes),
        "chemical evolution": ("list_metallicity_modes()", list_metallicity_modes),
        "chemical enrichment": ("list_metallicity_modes()", list_metallicity_modes),
    }
    # Concept synonyms are hand-curated; the menu's own name in prose is
    # derived from _menu_listers(). Consult both, and try the query with
    # hyphens normalized so "x-ray models" reaches list_xray_models() and
    # "star-forming" keeps working. Concept synonyms win on a tie: they are
    # the more specific statement of intent.
    _spellings = (q, q.replace("-", " "), q.replace("-", ""))
    _menu_names = _menu_name_aliases()
    for table in (_CONCEPT_ALIAS, _menu_names):
        for spelling in _spellings:
            if spelling in table:
                call, fn = table[spelling]
                _display(f"  '{query}' → tengri.{call} (the menu these models live in)\n")
                return fn()

    # ``kind`` and ``use`` are structural/internal — searching them gives
    # spurious 100%-of-table hits (e.g. "filter" matching every filter
    # row's kind, or "fitter" matching every inference method's "use"
    # which contains the literal word "fitter"). Match user-content
    # fields only: name, short_doc, citation, status, survey, instrument,
    # band — i.e. everything except kind/use.
    _SKIP_FIELDS = {"kind", "use"}
    hits: list[dict] = []
    # The same derived set describe() walks. This call site listed four extra
    # menus by hand and omitted list_recipes, so all ten recipes returned zero
    # hits from search() while describe() resolved every one.
    for fn in _every_menu_lister():
        for entry in fn():
            haystack = " ".join(
                str(v) for k, v in entry.items() if k not in _SKIP_FIELDS and isinstance(v, str)
            ).lower()
            if q in haystack:
                hits.append(entry)
    return _RegistryTable(hits)


def list_all() -> dict[str, _RegistryTable]:
    """Return everything available — useful for a single notebook cell overview.

    Returns
    -------
    dict
        One key per menu, named after its lister without the ``list_`` prefix
        (``list_dust_laws`` -> ``dust_laws``). Each value is a
        `_RegistryTable` (list[dict]) that prints as a table.

    Notes
    -----
    The keys are derived from the same census :func:`describe` and
    :func:`search` sweep, not listed here. :func:`_menu_listers` names this
    function as one of the three that "all walk this one tuple" — it was the
    one that never walked it, returning a hand-written dict of nine literals
    while 25 menus existed. So it showed 9 of 25, and its own ``Returns``
    section named only six of the nine it did return, while
    :mod:`tengri.__init__` tells readers it "enumerates every registry live".

    Deriving the keys preserves all nine that were there — every one is its
    lister's name minus the prefix — and adds the sixteen that were missing,
    so this widens the result without renaming anything: 9 -> 25 categories,
    921 rows.
    """
    return {fn.__name__.removeprefix("list_"): fn() for fn in _every_menu_lister()}


def list_properties(*, group: str | None = None) -> _RegistryTable:
    """List all globally-registered derived properties.

    Properties are computed on-demand from the orchestrator :class:`ForwardState`
    and are available on any :class:`Prediction` object. They are grouped
    (e.g., ``"sfh"``, ``"sed"``) to organize related quantities.

    Parameters
    ----------
    group : str, optional
        Filter by property group (e.g., ``"sfh"`` for star-formation-history
        properties). If None, lists all properties across all groups.

    Returns
    -------
    _RegistryTable
        Table with columns: name, group, units, component, description.
        Each row is one registered property.

    Notes
    -----
    Returned table prints as column-aligned text in a REPL or notebook.
    Use ``describe_property(name)`` to get full metadata for one property.

    Examples
    --------
    **List all properties:**

    >>> tengri.list_properties()
    stellar_mass      Msun   sfh    stellar  Total formed stellar mass
    ...

    **List SFH-group properties only:**

    >>> tengri.list_properties(group="sfh")

    **Convert to dict for programmatic use:**

    >>> props = tengri.list_properties()
    >>> # props is a list[dict]; each dict has keys: name, group, units, ...
    >>> sfh_mass = [p for p in props if p["name"] == "stellar_mass"]
    """
    from tengri.forward.properties import PROPERTY_REGISTRY

    out = []
    for _name, entries in PROPERTY_REGISTRY.items():
        for entry in entries:
            row = {
                "name": entry.name,
                "units": entry.units,
                "group": entry.group,
                "component": entry.component_name,
                "description": entry.doc,
            }
            if group is None or entry.group == group:
                out.append(row)
    return _RegistryTable(sorted(out, key=lambda r: r["name"]))


def describe_property(name: str) -> _DescribeRecord:
    """Return full metadata for one derived property.

    Parameters
    ----------
    name : str
        Property name (e.g., ``"stellar_mass"``).

    Returns
    -------
    _DescribeRecord
        Labeled record with name, units, group, component, and description.

    Raises
    ------
    KeyError
        If the property name is not registered. Call :func:`list_properties`
        to see all available names.

    Examples
    --------
    >>> tengri.describe_property("stellar_mass")
    """
    return _describe_from_list(name, list_properties, "property", "list_properties")


# ──────────────────────────────────────────────────────────────────
# The two functions a new user calls first
# ──────────────────────────────────────────────────────────────────


_CORE_CLASSES: dict[str, dict[str, str]] = {
    "SEDModel": {
        "kind": "core_class",
        "name": "SEDModel",
        "module": "tengri.forward.sed_model",
        "purpose": (
            "Differentiable SED forward chain "
            "(stellar → dust → nebular → AGN → IGM → radio → X-ray)."
        ),
        "see_also": "tengri.SEDModel.build(...)",
    },
    "ForwardModel": {
        "kind": "core_class",
        "name": "ForwardModel",
        "module": "tengri.forward.forward_model",
        "purpose": (
            "Thin outer shell that inference talks to. Owns an SED chain and "
            "an observation; exposes .predict / .predict_observables(params) → channel dict."
        ),
        "see_also": "tengri.ForwardModel.build(sed=..., observation=...)",
    },
    "Parameters": {
        "kind": "core_class",
        "name": "Parameters",
        "module": "tengri.parameters.parameters",
        "purpose": "Free / fixed parameter spec with priors.",
        "see_also": "tengri.Parameters(...)",
    },
    "Fitter": {
        "kind": "core_class",
        "name": "Fitter",
        "module": "tengri.inference.fitter",
        "purpose": (
            "Inference driver. Runs MAP / NUTS / VI / Pathfinder / Ray Tracing "
            "/ geoVI / evidence against a forward model."
        ),
        "see_also": 'fitter.run("map") or "nuts" / "vi" / "pathfinder" / ...',
    },
    "Posterior": {
        "kind": "core_class",
        "name": "Posterior",
        "module": "tengri.inference.posterior",
        "purpose": ("Return value of fitter.run(). Posterior samples / mean / median / 68% CI."),
        "see_also": "posterior.summary()",
    },
    "Observation": {
        "kind": "core_class",
        "name": "Observation",
        "module": "tengri.observation.observation",
        "purpose": "Frozen configuration container for photometry + spectroscopy.",
        "see_also": "tengri.Observation(photometry=..., spectroscopy=...)",
    },
}


def summary() -> None:
    """Print a one-line count of every menu in tengri.

    Notes
    -----
    This is what to run right after ``import tengri`` to see what's available.
    """
    counts = [
        (len(list_components()), "physics components", "list_components()"),
        (len(list_agn_models()), "AGN models", "list_agn_models()"),
        (len(list_dust_laws()), "dust attenuation laws", "list_dust_laws()"),
        (
            len(list_dust_emission_models()),
            "dust emission templates",
            "list_dust_emission_models()",
        ),
        # Two rows, mirroring the inference pair below: the raw registry holds
        # SFH types that are registered but not yet wired into the DSPS forward
        # path, and ``SEDModel.build`` rejects those. A single total would send
        # a fresh user shopping among models that cannot build.
        (
            len(list_sfh_models(status="production")),
            "buildable SFH models",
            "list_sfh_models(status='production')",
        ),
        (len(list_sfh_models()), "total SFH models", "list_sfh_models()"),
        (len(list_nebular_backends()), "nebular backends", "list_nebular_backends()"),
        (len(list_filters()), "photometric filters", "list_filters()"),
        (len(list_plots()), "plotting helpers", "list_plots()"),
        (
            len(list_inference_methods(tier="primary")),
            "primary inference methods",
            "list_inference_methods(tier='primary')",
        ),
        (len(list_inference_methods()), "total inference methods", "list_inference_methods()"),
    ]
    _display("\ntengri — what's available:\n")
    _display("  Core classes (the four nouns inference talks to):")
    _display("    tengri.SEDModel         differentiable SED forward chain")
    _display(
        "    tengri.ForwardModel     outer shell — owns SED + observation, "
        "exposes .predict / .predict_observables"
    )
    _display("    tengri.Parameters       priors + fixed values")
    _display("    tengri.Fitter           inference driver")
    _display("")
    width = max(len(label) for _, label, _ in counts)
    for n, label, call in counts:
        _display(f"  {n:>4}  {label.ljust(width)}    tengri.{call}")
    _display("\n  Look up any name:                          tengri.describe('skirtor')")
    _display("  Curated cheatsheet for new users:          tengri.help()\n")


_TOPIC_HELP: dict[str, tuple[str, callable]] = {
    "agn": ("AGN models", lambda: list_agn_models()),
    "dust": (
        "dust attenuation + emission",
        lambda: _RegistryTable(list(list_dust_laws()) + list(list_dust_emission_models())),
    ),
    "sfh": ("SFH models", lambda: list_sfh_models()),
    "nebular": ("nebular backends", lambda: list_nebular_backends()),
    "components": ("physics components", lambda: list_components()),
    "inference": ("inference methods", lambda: list_inference_methods()),
    "filters": ("photometric filters", lambda: list_filters()),
    "properties": ("derived properties", lambda: list_properties()),
    "plot": ("plotting helpers", lambda: list_plots()),
    "plots": ("plotting helpers", lambda: list_plots()),
    # "citations" is handled specially in _help_topic to print the
    # tengri.cite_components / print_components_bibtex flow
    "citations": ("citation API", None),
    "cite": ("citation API", None),
}


def help(topic: str | None = None) -> None:
    """Print a curated cheatsheet covering the entry points new users need.

    Parameters
    ----------
    topic : str, optional
        If given, narrow the cheatsheet to one menu. Recognized topics:
        ``"agn"``, ``"dust"``, ``"sfh"``, ``"nebular"``, ``"components"``,
        ``"inference"``, ``"filters"``, ``"properties"``. Without a topic
        the full cheatsheet is printed.

    Notes
    -----
    Counts are read live from the registries — adding a new model or
    inference backend updates the cheatsheet immediately, no edit needed.

    This shadows :func:`builtins.help` only when accessed as ``tengri.help``;
    the global ``help()`` builtin is unaffected.
    """
    if topic is not None:
        return _help_topic(topic)

    # ── Live counts so that the cheatsheet is never stale. ──
    n_agn = len(list_agn_models())
    n_atte = len(list_dust_laws())
    n_emis = len(list_dust_emission_models())
    n_sfh = len(list_sfh_models())
    # Registered != buildable: the raw registry carries SFH types that are not
    # yet wired into the DSPS forward path, and ``SEDModel.build`` rejects them.
    # ``summary()`` already reports the pair; this cheatsheet is aimed squarely
    # at first-time users, so it must not headline a count they cannot build.
    n_sfh_ok = len(list_sfh_models(status="production"))
    n_neb = len(list_nebular_backends())
    n_inf = len(list_inference_methods(tier="primary"))
    n_recipes = len(list_recipes())
    # The one default, read from the registry rather than written down here —
    # the whole point of #1289 was that hard-coded defaults drifted apart.
    from tengri.inference._backend_registry import DEFAULT_METHOD as default_method

    try:
        from tengri import list_filters  # avoid circular import on cold load

        n_filt = len(list_filters())
    except Exception:
        n_filt = "?"

    text = f"""
tengri — differentiable galaxy SED fitting in JAX

────────────────────────────────────────────────────────────────────
1.  See what's available
────────────────────────────────────────────────────────────────────
    tengri.summary()                      one-line counts of every menu
    tengri.list_agn_models()              {n_agn} AGN models
    tengri.list_dust_laws()               {n_atte} attenuation curves (UV/optical)
    tengri.list_dust_emission_models()    {n_emis} IR emission templates
    tengri.list_sfh_models()              {n_sfh_ok} buildable SFH variants ({n_sfh} registered)
    tengri.list_nebular_backends()        {n_neb} nebular backends
    tengri.list_inference_methods(tier="primary")  {n_inf} primary methods
    tengri.list_filters()                 {n_filt} filter curves
    tengri.list_properties()              derived quantities (M*, SFR, age, lines, …)
    tengri.describe("skirtor")            full metadata for any name
    tengri.search("torus")                cross-menu fuzzy search
    tengri.doctor()                       env / install / SSP health check
    tengri.help("dust")                   topical cheatsheet for one menu

────────────────────────────────────────────────────────────────────
2.  Learn the design — interactive tutorials
────────────────────────────────────────────────────────────────────
    tengri.tutorial()                     list every available recipe
    tengri.tutorial("philosophy")         layered architecture + IFT framework
    tengri.tutorial("key_classes")        Parameters / SEDModel / Fitter / Posterior
    tengri.tutorial("use_cases")          common patterns (catalog / hierarchical / mock)
    tengri.tutorial("first_fit")          end-to-end mock-recovery recipe
    tengri.tutorial("register_a_model", run=True)   register a new alternative
    tengri.tutorial("custom_likelihood")  Student-t / calibration / Protocol
    tengri.tutorial("swap_inference")     same model, NUTS → geoVI → MCMC
    tengri.tutorial("diagnostics")        ESS / R-hat / convergence checking
    tengri.tutorial("properties")         derived quantities catalog (M*, SFR, …)
    tengri.tutorial("mock_catalog")       batch mock catalogs via vmap (no fit)
    tengri.tutorial("fast_vs_exact")      exact vs fast photometry paths

    tengri.explain(tengri.SEDModel)       architectural role of any class
    tengri.examples()                     list every runnable example script

────────────────────────────────────────────────────────────────────
3.  Build a fit
────────────────────────────────────────────────────────────────────
    from tengri.observation import Observation, Photometry   # canonical path

    obs       = Observation(photometry=Photometry.from_names([...]))
    sed       = tengri.SEDModel.build(ssp_data=ssp, observation=obs,
                                      **tengri.recipes.star_forming_photometry())
    forward   = tengri.ForwardModel.build(sed=sed, observation=obs)
    posterior = forward.fit(flux, flux_err, method="{default_method}")
    posterior.summary()                        # median ± 68% CI per param

    `forward.fit(...)` is the canonical entry point. It wraps
    `Fitter(forward, data, noise).run(method)`; reach for the explicit
    Fitter only when you need to hold the engine itself.

    `tengri.list_recipes()` lists the {n_recipes} starting points. To hand-roll a
    model instead, pass group dicts to SEDModel.build:
      sfh={{'type': 'dpl', 'all_params': tengri.FREE}},
      dust={{'type': 'two_component', 'law_bc': 'calzetti'}}, …
      tengri.describe_recipe("star_forming_photometry")   # see what one sets

    Pick a method:
      tengri.list_inference_methods(tier="primary")   # {n_inf} that are validated
      "map" is a point estimate — fast, but no uncertainties.
      "{default_method}" and "mcmc_nuts" give you a posterior.

    Extract derived quantities:
      posterior.properties["stellar_mass"]     # array (n_samples,)
      posterior.properties.ci("stellar_mass")  # credible interval
      tengri.list_properties()                 # see all available names

    Predict without fitting:
      pred = sed.predict(params)               # rich + cached, one forward pass
      sed.predict_photometry(params)           # lean, JIT/vmap-safe

────────────────────────────────────────────────────────────────────
4.  Contribute a new physics alternative
────────────────────────────────────────────────────────────────────
    Copy examples/contrib/example_new_agn_torus.py.  It registers a new
    AGN torus model with metadata (citation, status), then exercises the
    introspection API end-to-end.  See CONTRIBUTING.md for the 5-step recipe.

────────────────────────────────────────────────────────────────────
5.  Cite the components used
────────────────────────────────────────────────────────────────────
    tengri.cite_components(model_or_posterior)  live walk of every component used
    tengri.print_components_bibtex(spec)        BibTeX-only output
    tengri.print_citations(model)               formal Bibliography report
    tengri.help('citations')                    full citation cheatsheet
"""
    _display(text)


def _help_topic(topic: str) -> None:
    """Topical help for one menu — used by ``help(topic=…)``."""
    topic_l = topic.lower()
    if topic_l not in _TOPIC_HELP:
        valid = sorted(_TOPIC_HELP)
        raise ValueError(
            f"Unknown help topic '{topic}'.  Valid topics: {valid}.  "
            "Or call tengri.help() with no argument for the full cheatsheet."
        )

    # "citations" / "cite" → narrative cheatsheet, not a list_*() table.
    if topic_l in ("citations", "cite"):
        text = """
tengri.help('citations') — citation API

Acknowledging the right papers when you publish a fit:

  Quick (live, walks the model's structural choices):
    tengri.cite_components(spec_or_model_or_posterior)
        → table of every component used + its citation string,
          read live from the registry (so contributor models appear too).
        → on a Posterior, also picks up the inference method.

    tengri.print_components_bibtex(spec)   # BibTeX-only output

  Canonical (formal Bibliography registry):
    tengri.print_citations(model)          # human-readable report
    tengri.print_bibtex(model)             # all BibTeX in one block
    tengri.collect_citations(model)        # list[Citation]
    tengri.cite('calzetti2000')            # look up a single key
    tengri.cite_all()                      # every key in the registry
    tengri.print_paper_citation()          # how to cite tengri itself

The two paths coexist:

  • cite_components()    — live, picks up every contributor model with a
                           citation= string, returns a registry table.
  • print_citations()    — formal, uses the static association table of
                           well-known canonical alternatives → BibTeX
                           entries in references.bib.

Use cite_components for "did I cite everything I'm using?" — it
reflects whatever is in the registry RIGHT NOW.  Use print_citations
for paste-ready BibTeX of the canonical pieces.
"""
        _display(text)
        return

    label, fetch = _TOPIC_HELP[topic_l]
    entries = fetch()
    _display(f"\ntengri.help('{topic_l}') — {label}: {len(entries)} available\n")
    _display(str(entries))
    _display("")

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
    _ALWAYS_HIDDEN = ("module", "requires", "params")  # surfaced via describe()

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
    if kind == "igm_model":
        return f"SEDModel.build(..., igm={{'type': '{name}'}})"
    if kind == "component":
        return f"tengri.{name}  (see list_{name}_models / list_{name}_laws for alternatives)"
    return ""


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
    if status:
        out = [m for m in out if m["status"] == status]
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

    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: (m["category"], m["name"])))


def list_dust_laws(*, status: str | None = None) -> _RegistryTable:
    """List all registered dust **attenuation** laws.

    Attenuation describes how UV/optical photons are absorbed/scattered
    by dust along the line of sight (Calzetti, Cardelli, Charlot-Fall, …).
    For dust **emission** templates (DL07, Dale, MBB, …), see
    :func:`list_dust_emission_models`.
    """
    from tengri.components.dust.attenuation import DUST_LAWS

    out = [_entry_to_dict(n, e, kind="dust_attenuation") for n, e in DUST_LAWS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
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

    Derived from the SAME source as the ``SEDModel.build`` grammar validator
    (:func:`tengri.parameters.groups._valid_dust_emission_types`): the
    ``_REGISTRY`` emission components (those publishing ``sed_dust_ir``) plus the
    canonical grammar alias map. The ``DUST_EMISSION_MODELS`` loader cache is
    **not** consulted — it is load-only — so the menu and the validator can
    never drift (closes #495).
    """
    # Import triggers component registration into _REGISTRY + the alias map.
    import tengri.components.dust.emission  # noqa: F401
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES

    names = {
        name
        for name, cls in _REGISTRY.items()
        if "sed_dust_ir" in {o.name for o in getattr(cls, "_outputs_tuple", ())}
    } | set(_EMISSION_TYPE_ALIASES)

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
    if status:
        out = [m for m in out if m["status"] == status]
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
    if status:
        out = [m for m in out if m["status"] == status]
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
    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_xray_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered X-ray emission models.

    The X-ray group composes the AGN corona (Yang+2020 ``alpha_ox(L_2500)``
    relation), the Lehmer+2016 high- and low-mass X-ray binary fits,
    and optional thermal hot-gas emission. The ``'none'`` entry disables
    the whole block.

    See also: :func:`list_radio_models`, :func:`list_igm_models`,
    :mod:`tengri.builders.xray`.
    """
    from tengri.components.xray._models import XRAY_MODELS

    out = [_entry_to_dict(n, e, kind="xray_model") for n, e in XRAY_MODELS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
    return _RegistryTable(sorted(out, key=lambda m: m["name"]))


def list_radio_models(*, status: str | None = None) -> _RegistryTable:
    """List all registered radio emission models.

    The radio group adds the Condon+1992 FIR-radio correlation for the
    star-forming-galaxy contribution plus an optional AGN radio
    power-law via the radio-loudness parameter. ``'none'`` disables
    the block.

    See also: :func:`list_xray_models`, :func:`list_igm_models`,
    :mod:`tengri.builders.radio`.
    """
    from tengri.components.radio._models import RADIO_MODELS

    out = [_entry_to_dict(n, e, kind="radio_model") for n, e in RADIO_MODELS.items()]
    if status:
        out = [m for m in out if m["status"] == status]
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
    if status:
        out = [m for m in out if m["status"] == status]
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

    # SEDModel / Posterior expose .spec
    spec = getattr(obj, "spec", obj)

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

    # Dust attenuation — bc + diff (skip plain "power_law" default if both equal it)
    for attr in ("dust_law_bc", "dust_law_diff", "dust_law"):
        _add("dust_attenuation", getattr(spec, attr, None), list_dust_laws)

    # Dust emission
    _add("dust_emission", getattr(spec, "dust_emission", None), list_dust_emission_models)

    # Nebular
    nebular_mode = getattr(spec, "nebular_mode", None)
    if nebular_mode and nebular_mode != "off":
        _add("nebular", nebular_mode, list_nebular_backends)

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
    # Map registry-entry names → bib-key in tengri.citations.registry.REGISTRY.
    # Best-effort — the registry currently knows ~50 keys; new contributor
    # models that lack a bibtex entry print a TODO comment instead.
    _NAME_TO_BIBKEY: dict[str, str] = {
        # SFH
        "dpl": "bagpipes",
        "continuity": "leja2019",
        "dirichlet": "leja2019",
        "dense_basis": "iyer2020",
        # AGN
        "skirtor": "skirtor",
        "stalevski": "skirtor",
        "kubota_done": "kubota_done2018",
        "kubota_done_full": "kubota_done2018",
        "multicolor_agn": "kubota_done2018",
        "adaf": "mahadevan1997",
        "qsogen": "temple2021_qsogen",
        # AGN composable blocks — bibkeys verified against each block's
        # registered ``citation=`` string (never guessed). Blocks whose paper
        # has no bundled BibTeX (fritz, cat3d_wind, feltre, richards2006,
        # boroson_green, …) fall through to the free-form citation note.
        "grahsp": "buchner2024",
        "grahsp_sbpl": "buchner2024",
        "grahsp_biatten": "buchner2024",
        "nenkova": "clumpy_nenkova2008",
        "nenkova_agnfitter": "clumpy_nenkova2008",
        "multicolor": "shakura_sunyaev1973",
        "synthesizer": "synthesizer",
        "synthesizer_spectra": "synthesizer",
        "qsogen_smc": "temple2021_qsogen",
        "qsogen_balmer": "temple2021_qsogen",
        # Dust attenuation
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
        # Dust emission
        "dl07": "draine_li2007",
        "draine_li2007": "draine_li2007",
        "dl14": "draine2014",
        "dale2014": "dale2014",
        "casey2012": "casey2012",
        "mbb": "casey2012",
        # Nebular
        "cue": "cue",
        "cloudy_grid": "cloudy",
        # Inference
        "mcmc_nuts": "blackjax",
        "vi": "nifty",
        "vi_nonlinear_fast": "nifty",
        "mcmc_raytrace": "raytrace_behroozi",
        "pathfinder": "pathfinder",
        "nss": "nss",
        # Frameworks (always-on)
        "tengri": "tengri",
        "DSPS": "dsps",
        "JAX": "jax",
    }

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
    except ImportError:
        _cite_lookup = None

    seen_keys: set[str] = set()
    for row in rows:
        name = row["name"]
        comp = row["component"]
        bibkey = _NAME_TO_BIBKEY.get(name) or _NAME_TO_BIBKEY.get(name.lower())
        if bibkey and bibkey not in seen_keys and _cite_lookup is not None:
            try:
                citation = _cite_lookup(bibkey)
                bib_method = getattr(citation, "to_bibtex", None)
                if callable(bib_method):
                    _display(f"% [{comp}] {name}")
                    _display(bib_method())
                    _display("")
                    seen_keys.add(bibkey)
                    continue
            except Exception:
                pass
        # Fallback: free-form citation note
        cit = row.get("citation", "")
        if cit:
            _display(f"% [{comp}] {name}: {cit}")
            _display("%   (no bib entry in tengri.citations — please add manually)")
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
        ``survey``, ``instrument``, ``band``. Prints as a table, also
        renders as HTML in Jupyter.

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


def list_inference_methods(
    *, tier: str | None = None, target: object | None = None
) -> _RegistryTable:
    """List all registered inference methods.

    Parameters
    ----------
    tier : str, optional
        Filter by ``"primary"`` (recommended for new users) or
        ``"experimental"``.
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
    from tengri.inference._strategy import resolve_status

    out = []
    for entry in all_backends():
        out.append(
            {
                "name": entry.name,
                "kind": "inference_method",
                "tier": entry.tier,
                "short_doc": entry.short_doc,
                "requires": list(entry.requires),
                "status": resolve_status(entry, target).value,
                "use": _usage_hint(entry.name, "inference_method"),
            }
        )
    if tier:
        out = [m for m in out if m["tier"] == tier]
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
        list_dust_laws,
        list_dust_emission_models,
        list_sfh_models,
        list_nebular_backends,
        list_xray_models,
        list_radio_models,
        list_igm_models,
    )


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

    matches = [
        entry
        for fn in (*_menu_listers(), list_components, list_filters, list_plots, list_recipes)
        for entry in fn()
        if entry["name"] == name
    ]
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

            record["also_registered_as"] = (
                f"'{name}' is registered in {len(matches)} places "
                f"[{'; '.join(_where(m) for m in matches)}] — showing the first. "
                "Use the category-specific list (e.g. describe_agn_block"
                "(name, category=...)) to select another."
            )
        return _DescribeRecord(record)
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
        out.append(
            {
                "name": name,
                "kind": "recipe",
                "short_doc": short_doc,
                "ssp_requirement": ssp_req,
                "use": f"recipes.{name}() → SEDModel.build(ssp_data=ssp, **recipe)",
            }
        )
    return _RegistryTable(out)


def _parse_ssp_requirement(doc: str) -> str:
    """Pull the ``**SSP requirement:**`` value from a recipe docstring."""
    for line in doc.splitlines():
        if "SSP requirement" in line:
            after = line.split(":", 1)[1] if ":" in line else line
            return after.replace("*", "").strip()
    return "any"


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
    """Return the descriptor row for one inference backend (MAP / VI / NUTS / NSS / …)."""
    return _describe_from_list(
        name, list_inference_methods, "inference method", "list_inference_methods"
    )


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
        ``"baked_in"``, ``"cue"``, ``"cloudy_grid"``, or ``"cb19"``.
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
    }
    if q in _CONCEPT_ALIAS:
        call, fn = _CONCEPT_ALIAS[q]
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
    for fn in (*_menu_listers(), list_components, list_filters, list_plots):
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
        Keys: components, inference_methods, agn_models, dust_laws, sfh_models,
        nebular_backends. Each value is a `_RegistryTable` (list[dict]) that
        prints as a table.
    """
    return {
        "components": list_components(),
        "inference_methods": list_inference_methods(),
        "agn_models": list_agn_models(),
        "dust_laws": list_dust_laws(),
        "dust_emission_models": list_dust_emission_models(),
        "sfh_models": list_sfh_models(),
        "nebular_backends": list_nebular_backends(),
        "filters": list_filters(),
        "plots": list_plots(),
    }


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
        (len(list_sfh_models()), "SFH models", "list_sfh_models()"),
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
    n_neb = len(list_nebular_backends())
    n_inf = len(list_inference_methods(tier="primary"))
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
    tengri.list_sfh_models()              {n_sfh} SFH variants
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
    obs        = tengri.Observation(photometry=tengri.Photometry.from_names([...]))
    parameters = tengri.Parameters(...)        # priors + fixed values
    sed        = tengri.SEDModel.build(ssp_data=..., observation=obs, ...)
    forward    = tengri.ForwardModel.build(sed=sed, observation=obs)
    fitter     = tengri.Fitter(forward, data, noise)
    posterior  = fitter.run("map")             # or "nuts", "vi", …
    posterior.summary()                        # median ± 68% CI per param

    Extract derived quantities:
      posterior.properties["stellar_mass"]     # array (n_samples,)
      posterior.properties.ci("stellar_mass") # credible interval
      tengri.list_properties()                # see all available names

    Pick the right kwargs:
      tengri.suggest_parameters(mean_sfh_type="dpl", agn_model="skirtor")

    The outer shell:
      tengri.ForwardModel — thin shell inference talks to. Owns the SED
        chain and the observation; exposes a single .predict(params)
        method that returns a {{channel: array}} dict (phot_fnu, spec_fnu).
        Inference doesn't need to know which prediction method to call.

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

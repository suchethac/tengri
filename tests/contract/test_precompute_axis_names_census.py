# SPDX-License-Identifier: BSD-3-Clause
"""Every precompute module's ``AXIS_PARAMS`` must name parameters that can exist (#1738).

``AXIS_PARAMS`` is a tuple of plain strings matched against
``Parameters.get_fixed_values()`` keys. Nothing forces the two namespaces to
agree, and the failure mode is inertness rather than an exception: when a name
drifts on either side, ``pname in fixed_values`` is simply always ``False``, the
axis is never collapsed, and the grid reduction the module's docstring
advertises silently never happens. Nothing raises, and the SED is unchanged —
only larger and slower to interpolate than intended.

``collapse_fixed_axes`` warns when it meets this at run time, but only for a
model someone actually builds, and the affected modules are never driven with a
real ``Parameters`` anywhere in the suite — which is how the drift survived.
This census is the static counterpart: it needs no template grids, so it runs
in CI, and it enumerates modules from the live precompute registry so a newly
added one cannot escape it.

The ledger records the *exact* set of unresolvable names per module and asserts
it in both directions. Repairing a name makes the ledger wrong and the test
fail, which is the prompt to edit it; a newly broken name fails the same test.
An exemption therefore cannot outlive the defect it records, and cannot quietly
grow.
"""

from __future__ import annotations

import functools
import itertools

import pytest

from tengri.forward.precompute.registry import registered_components, resolve

pytestmark = pytest.mark.contract

#: Declared axis names that match no parameter any ``Parameters`` can hold, so
#: the axis they govern can never be collapsed. Measured on 969e43066 (#1738),
#: when ten of the seventeen registered adapters were affected.
#:
#: Three distinct causes. Two are now repaired and so are absent below:
#:
#: * a near-miss spelling — ``xray_gamma`` -> ``xray_gamma_agn`` and
#:   ``agn_alpha_pl`` -> ``agn_alpha``;
#: * a component-local prefix that is not the component's ``parameter_prefix``
#:   — the four AGN torus adapters, each realigned onto the name its *own*
#:   physics module already documented: ``cat3d_{cos_inc,a,fwd}`` ->
#:   ``agn_{cos_inc,a_cat3d,fwd_cat3d}``, ``silva04_log_NH`` ->
#:   ``agn_log_nh_silva``, ``nenkova_agnfitter_cos_inc`` -> ``agn_cos_inc``,
#:   ``skirtor_agnfitter_{oa,incl,tv}`` -> ``agn_{oa_skirtor,incl_skirtor,tv_skirtor}``.
#:
#: What remains is the third cause, which a rename cannot fix: an internal
#: grid-axis label that was never a user parameter at all. ``cb19``'s seven,
#: ``log_age`` in the two CLOUDY-family adapters, and qsogen's pair have no
#: live counterpart to point at — the parameter genuinely does not exist, so
#: closing these means either declaring the parameter or declaring the axis
#: internal and not user-collapsible. Tracked as #1827.
#:
#: ``cb19`` needs more than that again: it also carries seven axes over a
#: photometry array with six grid dimensions, so repairing the names alone would
#: make the collapse contract the filter axis. Fix both together.
DEAD_AXIS_NAMES: dict[str, frozenset[str]] = {
    "tengri.components.agn.qsogen_precompute": frozenset({"agn_plslp1", "agn_ebv"}),
    "tengri.components.nebular.cb19_precompute": frozenset(
        {
            "log_OH_total",
            "log_age_yr_ssp",
            "log_U",
            "log_nH",
            "log_CO",
            "dNO",
            "HbFrac",
        }
    ),
    "tengri.components.nebular.cloudy_precompute": frozenset({"log_age"}),
    "tengri.components.nebular.mappings_photo_precompute": frozenset({"log_age", "neb_logn"}),
}

#: Modules whose dead names still collapse, because they pass ``defaults=`` to
#: :func:`collapse_fixed_axes` for parameters the model never declares. The
#: names are still wrong — a user who declares one as free cannot free the axis
#: — but the grid does reduce, so these are a milder case than the rest.
#:
#: GRAHSP and the composable AGN runner do the same and are absent only because
#: neither is in the precompute ``_REGISTRY``; they are reached through the AGN
#: block machinery, so this census never sees them.
COLLAPSE_VIA_DEFAULTS: frozenset[str] = frozenset({"tengri.components.agn.qsogen_precompute"})


@functools.lru_cache(maxsize=1)
def _declarable_names() -> frozenset[str]:
    """Every parameter name any ``Parameters`` could hold.

    Union of the classic registry over a spread of configurations and every
    ``SEDModelComponent`` subclass's declared parameters. Deliberately a
    superset: the only claim this census makes is that a name absent from here
    is one no configuration can produce, so over-collecting weakens nothing.
    """
    import tengri  # noqa: F401  -- populates the component registry
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.parameters._builders import _build_param_registry

    names: set[str] = set()
    for sfh, neb, agn in itertools.product(
        [["dpl", "field"], ["delayed"], ["tsnorm"], ["dense_basis"], ["continuity"]],
        [False, True, "cloudy", "cue"],
        ["skirtor", "composable", "unified", None],
    ):
        for extra in (
            {},
            {
                "radio": True,
                "xray": True,
                "shock": True,
                "evolving_metallicity": True,
                "eline_mode": "on",
                "eline_broad": True,
                "igm_patchy": True,
                "dla": True,
                "dust_emission": "dale2014",
            },
        ):
            try:
                registry, _ = _build_param_registry(sfh, nebular=neb, agn_model=agn, **extra)
            except (KeyError, TypeError, ValueError):
                continue
            names |= set(registry)

    for cls in _REGISTRY.values():
        prefix = getattr(cls, "parameter_prefix", "") or ""
        for attr, value in vars(cls).items():
            if attr.startswith("_"):
                continue
            if hasattr(value, "sample") or hasattr(value, "unstandardize"):
                names |= {attr, f"{prefix}{attr}"}
    return frozenset(names)


def _axis_names(module) -> tuple[str, ...]:
    """Flatten ``AXIS_PARAMS`` whether it is a tuple or a per-model dict."""
    declared = getattr(module, "AXIS_PARAMS", ())
    if isinstance(declared, dict):
        return tuple(sorted({name for names in declared.values() for name in names}))
    return tuple(declared or ())


@functools.lru_cache(maxsize=1)
def _modules_with_axes() -> dict[str, tuple[str, ...]]:
    """Registered precompute modules declaring at least one axis parameter.

    Keyed by dotted module path, so the several component names sharing one
    adapter (seven dust-IR names, three radio, three X-ray) are censused once.
    """
    found: dict[str, tuple[str, ...]] = {}
    for component in registered_components():
        module = resolve(component)
        if module is None:
            continue
        names = _axis_names(module)
        if names:
            found[module.__name__] = names
    return found


def _dead_names(module_path: str) -> frozenset[str]:
    """Declared axis names of ``module_path`` that no ``Parameters`` can hold."""
    declarable = _declarable_names()
    return frozenset(n for n in _modules_with_axes()[module_path] if n not in declarable)


def test_the_census_is_not_empty():
    """Guard the guard: a broken enumeration must not read as universal success."""
    modules = _modules_with_axes()
    assert len(modules) >= 15, (
        f"only {len(modules)} precompute modules declare axis parameters; the "
        "registry enumeration is probably broken, which would make every other "
        "assertion in this file vacuous"
    )
    assert len(_declarable_names()) >= 200, (
        f"only {len(_declarable_names())} declarable parameter names found; the "
        "namespace build is probably broken, which would mark every name dead"
    )


@pytest.mark.parametrize("module_path", sorted(_modules_with_axes()))
def test_declared_axis_names_are_parameters_that_can_exist(module_path):
    """Every declared axis name resolves, or is recorded verbatim in the ledger.

    An axis whose name cannot resolve is an axis that can never be collapsed,
    however the parameter is declared — the grid stays at full size and no
    error is raised.
    """
    dead = _dead_names(module_path)
    ledgered = DEAD_AXIS_NAMES.get(module_path, frozenset())

    assert dead == ledgered, (
        f"{module_path}: unresolvable axis names are {sorted(dead)}, but the "
        f"ledger records {sorted(ledgered)}. If you repaired a name, delete it "
        f"from DEAD_AXIS_NAMES; if a new one broke, align AXIS_PARAMS with the "
        f"name the component actually declares (#1738)."
    )


def test_the_ledger_records_a_known_defect_not_an_empty_habit():
    """The ledger must describe reality: every entry non-empty and still failing.

    Without this, emptying a module's entry to ``frozenset()`` would satisfy the
    per-module test above while pretending the module was never broken.
    """
    for module_path, names in DEAD_AXIS_NAMES.items():
        assert names, f"{module_path} has an empty ledger entry; delete it instead (#1738)"
        assert module_path in _modules_with_axes(), (
            f"{module_path} is on the ledger but is not a registered precompute "
            "module declaring axis parameters. Delete the entry (#1738)."
        )


def test_defaults_tier_entries_are_also_ledgered_dead():
    """A module collapsing via defaults still has wrong names, and must say so."""
    for module_path in COLLAPSE_VIA_DEFAULTS:
        assert module_path in DEAD_AXIS_NAMES, (
            f"{module_path} is recorded as collapsing via defaults, which is only "
            "meaningful because its declared names do not resolve — it must also "
            "carry a DEAD_AXIS_NAMES entry (#1738)."
        )


def test_most_modules_resolve_their_axis_names():
    """A regression canary: the healthy count must not quietly erode.

    Four of seventeen registered adapters carry at least one dead name, down
    from ten: two near-miss spellings and the four AGN torus prefix drifts have
    been repaired. The bound is a ratchet — it is the count of healthy modules
    at the last repair, so fixing more names fails this test and is the prompt
    to raise it, while a newly broken name fails it too.
    """
    modules = _modules_with_axes()
    healthy = [m for m in modules if not _dead_names(m)]
    assert len(healthy) >= 13, (
        f"only {len(healthy)}/{len(modules)} precompute modules resolve every "
        f"declared axis name: {sorted(set(modules) - set(healthy))} (#1738)"
    )

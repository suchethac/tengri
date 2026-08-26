# SPDX-License-Identifier: BSD-3-Clause
"""Auto-collapse correctness tests for precompute adapters.

With one grid axis pinned to a Fixed value, the collapsed lookup must return
exactly what the un-collapsed lookup returns at that value. This guards the
``slice_fixed_axes`` collapse machinery.

Three tests here never ran, for three stacked reasons
-----------------------------------------------------

``_DATA`` was ``Path(__file__).parent.parent.parent.parent / "data"``. Four
``.parent`` steps from ``tests/contract/`` land one level *above* the
repository, so the guards resolved ``<repo>/../data`` and the skip messages
printed that path without anyone reading them. Paths now come from
:mod:`tests._data_skip`, which computes the root once.

The filenames were wrong too, independently, so fixing the directory alone
would have left them skipped:

======= ================================= ==========================
class   looked for                        actually shipped
======= ================================= ==========================
Silva04 ``silva04_wind_torus_grid.h5``    ``silva04_torus_grid.h5``
CB19    ``cb19_grid.h5``                  ``cb19_templates.h5``
======= ================================= ==========================

``cb19_templates.h5`` is confirmed by ``cb19_precompute.precompute``'s own
docstring, which documents ``filepath`` as defaulting to it.

The Silva04 test then failed, with ``silva04_phot() missing 1 required
positional argument: 'agn_torus_frac'``. That is not a defect in the adapter:
``build_lookup`` returns a callable taking ``(agn_log_lbol, agn_log_nh_silva,
agn_torus_frac)`` -- one grid axis plus one *runtime* parameter that is
deliberately not a grid axis (CLAUDE.md: ``agn_torus_frac`` must not be
auto-derived in the forward pass). The old caller passed positionally and
assumed an arity of ``1 + len(axes)``. Supplying it makes the collapse
bit-exact: ``max|diff| = 0.0`` at ``agn_torus_frac`` of 0.25, 0.5 and 0.9.

The two lookups disagree on how it is passed -- the full one takes it
positionally, the collapsed one keyword-only after ``*free_axis_values`` -- so
``_call_lookup`` passes every non-axis argument by keyword.

An all-zero comparison is not a match
--------------------------------------

``agn_torus_frac=0.0`` makes every Silva04 filter return exactly ``0.0``, and
``assert_allclose`` of zeros against zeros passes against any collapse
implementation. Every case now asserts the full lookup is non-zero somewhere
before comparing.

Three classes that looked like coverage and held no tests
----------------------------------------------------------

``TestDiscPrecomputeAxisCollapse``, ``TestQsogenPrecomputeAxisCollapse`` and
``TestCat3DPrecomputeAxisCollapse`` had a docstring, in one case a skip
fixture, and no test methods -- ``--collect-only`` showed nothing while a
reader scanning the file saw three covered adapters. They are gone; the
adapters they named are in ``_UNCOVERED``, which is asserted rather than
implied.

The reason those three recorded ("build_lookup requires runtime parameters not
present in precompute grid axes") is the same one Silva04's failure looked
like, and for Silva04 it was wrong. A signature-driven harness was tried on
all of them: it revives Silva04 exactly, and the others still mismatch.
Whether that is a real collapse defect or a harness calling them wrongly is
not decidable from the test side, so they stay listed and unclaimed.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._data_skip import DATA_DIR, requires_cb19

pytestmark = pytest.mark.contract

_SILVA04_GRID = DATA_DIR / "silva04_torus_grid.h5"
_CB19_GRID = DATA_DIR / "cb19_templates.h5"

requires_silva04_torus = pytest.mark.skipif(
    not _SILVA04_GRID.is_file(), reason=f"Silva04 torus grid not found at {_SILVA04_GRID}"
)

# Standard synthetic filter set (used across test adapters)
_CENTERS = np.array([3e5, 1e7, 1e8, 1e10])  # FIR-radio Angstrom
_WIDTHS = np.array([1e5, 3e6, 3e7, 3e9])

#: Values for lookup arguments that are runtime parameters, not grid axes.
#: ``agn_torus_frac`` must be non-zero: at 0.0 the Silva04 lookup returns all
#: zeros and the comparison below becomes vacuous.
_RUNTIME_ARGS = {"agn_torus_frac": 0.5}


@pytest.fixture(scope="module")
def filter_set_radio():
    """Synthetic 4-filter set for radio precompute."""
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(_CENTERS, _WIDTHS):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        waves.append(wv)
        trans.append(np.exp(-0.5 * ((wv - c) / w) ** 2))
    return waves, trans


@pytest.fixture(scope="module")
def filter_set_xray():
    """Synthetic 4-filter set for X-ray precompute (0.1-100 keV)."""
    centers = np.array([1.0, 5.0, 50.0, 500.0])  # 0.1-100 keV in Angstrom
    widths = np.array([0.3, 1.5, 15.0, 150.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 1e-2), c + 3 * w, 64)
        waves.append(wv)
        trans.append(np.exp(-0.5 * ((wv - c) / w) ** 2))
    return waves, trans


def _mock_params(fixed: dict[str, float]) -> MagicMock:
    """A stand-in Parameters whose ``get_fixed_values()`` returns ``fixed``."""
    mock = MagicMock()
    mock.get_fixed_values.return_value = fixed
    mock.free_params = []
    return mock


def _call_lookup(lookup, scale, axis_values):
    """Call a lookup with axis values positionally and everything else by name.

    The full and collapsed lookups do not agree on how a runtime parameter is
    passed -- Silva04's full lookup takes ``agn_torus_frac`` positionally, its
    collapsed one keyword-only after ``*free_axis_values``. Keyword works for
    both.
    """
    accepted = inspect.signature(lookup).parameters
    kwargs = {k: v for k, v in _RUNTIME_ARGS.items() if k in accepted}
    return np.asarray(jax.jit(lookup)(scale, *axis_values, **kwargs))


def _axis_params(adapter, model: str | None) -> tuple[str, ...]:
    """``AXIS_PARAMS``, resolved through the model key when it is a mapping."""
    declared = adapter.AXIS_PARAMS
    return tuple(declared[model] if isinstance(declared, dict) else declared)


def _assert_collapse_matches(adapter, filter_set, *, model, fix_idx, n_axes, kw=None):
    """Pin axis ``fix_idx`` and require the collapsed lookup to reproduce the full one.

    ``AXIS_PARAMS`` is read off the adapter rather than restated here: a copy in
    the test cannot disagree with the declaration it copies, and the mock
    answers to whatever name it is handed, so a drifted name still collapses and
    still passes. Both X-ray coronae declared ``xray_gamma`` against a parameter
    named ``xray_gamma_agn`` and this suite stayed green throughout (#1738).
    """
    waves, trans = filter_set
    kw = dict(kw or {})
    if model is not None:
        kw["model"] = model
    redshift = 0.5

    full = adapter.precompute(waves, trans, redshift, parameters=None, **kw)
    lookup_kw = {"model": model} if model is not None else {}
    full_lookup = adapter.build_lookup(full, **lookup_kw)
    assert len(full["axes"]) == n_axes, (
        f"expected {n_axes} axes before collapse, got {len(full['axes'])}"
    )

    axes = [np.asarray(a) for a in full["axes"]]
    midpoints = [float(a[len(a) // 2]) for a in axes]
    param_name = _axis_params(adapter, model)[fix_idx]

    coll = adapter.precompute(
        waves, trans, redshift, parameters=_mock_params({param_name: midpoints[fix_idx]}), **kw
    )
    coll_lookup = adapter.build_lookup(coll, **lookup_kw)
    assert len(coll["axes"]) == n_axes - 1, (
        f"pinning {param_name} should leave {n_axes - 1} axes, got {len(coll['axes'])}"
    )

    scale = jnp.float64(1.0)
    full_result = _call_lookup(full_lookup, scale, midpoints)
    coll_result = _call_lookup(
        coll_lookup, scale, [m for i, m in enumerate(midpoints) if i != fix_idx]
    )

    assert np.any(full_result != 0.0), (
        f"every filter returned 0.0, so comparing the collapsed lookup against it "
        f"would pass for any implementation. Pinned {param_name}={midpoints[fix_idx]}; "
        f"check that the filter set overlaps this component's emission and that the "
        f"runtime arguments in _RUNTIME_ARGS are not switching it off."
    )
    np.testing.assert_allclose(
        coll_result,
        full_result,
        rtol=1e-10,
        atol=0.0,
        err_msg=f"pinning {param_name} changed the result",
    )


# ── The covered adapters ──────────────────────────────────────────

#: (id, package, module, model or None, axis index to pin, axis count, filter
#: fixture suffix, extra precompute kwargs, marks).
_CASES = [
    ("radio_synchrotron", "radio", "radio_precompute", "radio_synchrotron", 0, 1, "radio", {}, ()),
    ("radio_freefree", "radio", "radio_precompute", "radio_freefree", 0, 1, "radio", {}, ()),
    ("radio_agn_jet", "radio", "radio_precompute", "radio_agn_jet", 0, 1, "radio", {}, ()),
    ("xray_xrb_ax0", "xray", "xray_precompute", "xray_xrb", 0, 2, "xray", {}, ()),
    ("xray_xrb_ax1", "xray", "xray_precompute", "xray_xrb", 1, 2, "xray", {}, ()),
    ("xray_corona_ax0", "xray", "xray_precompute", "xray_corona", 0, 2, "xray", {}, ()),
    ("xray_corona_ax1", "xray", "xray_precompute", "xray_corona", 1, 2, "xray", {}, ()),
    ("xray_lopez24_ax0", "xray", "xray_precompute", "xray_corona_lopez24", 0, 2, "xray", {}, ()),
    ("xray_lopez24_ax1", "xray", "xray_precompute", "xray_corona_lopez24", 1, 2, "xray", {}, ()),
    ("dust_mbb", "dust", "dust_analytic_precompute", "modified_blackbody", 0, 2, "radio", {}, ()),
    ("dust_casey", "dust", "dust_analytic_precompute", "casey2012", 0, 3, "radio", {}, ()),
    (
        "silva04",
        "agn",
        "silva04_precompute",
        None,
        0,
        1,
        "radio",
        {"grid_path": str(_SILVA04_GRID)},
        (requires_silva04_torus,),
    ),
]


@pytest.mark.parametrize(
    ("package", "module", "model", "fix_idx", "n_axes", "filters", "kw"),
    [pytest.param(*c[1:8], id=c[0], marks=c[8]) for c in _CASES],
)
def test_axis_collapse_matches_full_lookup(
    request, package, module, model, fix_idx, n_axes, filters, kw
):
    """Pinning one axis reproduces the un-collapsed lookup exactly."""
    adapter = importlib.import_module(f"tengri.components.{package}.{module}")
    filter_set = request.getfixturevalue(f"filter_set_{filters}")
    _assert_collapse_matches(
        adapter, filter_set, model=model, fix_idx=fix_idx, n_axes=n_axes, kw=kw
    )


@requires_cb19
def test_cb19_collapse_axis0(filter_set_radio):
    """CB19's 7-axis grid collapses on log_OH_total.

    Kept out of the table because it takes ``filepath``, not ``grid_path``, and
    declares no ``model``. The grid is not tracked in git and CI does not fetch
    it, so this skips there.
    """
    from tengri.components.nebular import cb19_precompute as adapter

    waves, trans = filter_set_radio
    full = adapter.precompute(waves, trans, 0.5, parameters=None, filepath=str(_CB19_GRID))
    assert len(full["axes"]) == 7, f"CB19 should have 7 axes, got {len(full['axes'])}"

    full_lookup = adapter.build_lookup(full)
    axes = [np.asarray(a) for a in full["axes"]]
    midpoints = [float(a[len(a) // 2]) for a in axes]

    coll = adapter.precompute(
        waves,
        trans,
        0.5,
        parameters=_mock_params({adapter.AXIS_PARAMS[0]: midpoints[0]}),
        filepath=str(_CB19_GRID),
    )
    coll_lookup = adapter.build_lookup(coll)
    assert len(coll["axes"]) == 6, f"expected 6 axes after collapse, got {len(coll['axes'])}"

    scale = jnp.float64(1.0)
    full_result = _call_lookup(full_lookup, scale, midpoints)
    coll_result = _call_lookup(coll_lookup, scale, midpoints[1:])

    assert np.any(full_result != 0.0), "CB19 full lookup is all zeros; comparison would be vacuous"
    np.testing.assert_allclose(coll_result, full_result, rtol=1e-10, atol=0.0)


# ── Which adapters this file actually covers ──────────────────────

#: Adapters that declare a collapsible axis and have no test here. An entry is
#: a decision, not a dismissal -- ``test_every_adapter_with_axes_is_listed``
#: fails when a new adapter appears in neither this map nor ``_CASES``.
_UNCOVERED: dict[str, str] = {
    "cat3d_precompute": "signature-driven collapse mismatches; cause not decidable from here",
    "disc_precompute": "signature-driven collapse mismatches; cause not decidable from here",
    "qsogen_precompute": "signature-driven collapse mismatches; cause not decidable from here",
    "nenkova_agnfitter_precompute": "signature-driven collapse mismatches; not diagnosed",
    "skirtor_precompute": "5 axes; no collapse test written",
    "skirtor_agnfitter_precompute": "3 axes; no collapse test written",
    "cloudy_precompute": "3 axes; needs the untracked CLOUDY MIST grid",
    "feltre_precompute": "4 axes; no collapse test written",
    "mappings_photo_precompute": "4 axes; no collapse test written",
    "mappings_shock_precompute": "3 axes; no collapse test written",
    "dust_emission_precompute": "8 models with axes; no collapse test written",
}


def _adapters_declaring_axes() -> set[str]:
    """Every ``*_precompute`` module declaring at least one collapsible axis."""
    import tengri.components as components

    found: set[str] = set()
    for pkg in (
        components.agn,
        components.nebular,
        components.radio,
        components.xray,
        components.dust,
    ):
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.name.endswith("_precompute"):
                continue
            mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
            declared = getattr(mod, "AXIS_PARAMS", None)
            if isinstance(declared, dict):
                declared = tuple(p for v in declared.values() for p in v)
            if declared:
                found.add(info.name)
    return found


def test_every_adapter_with_axes_is_listed():
    """Every collapsible adapter is either tested here or listed as uncovered.

    The file's docstring claims to guard ``slice_fixed_axes``. It guards five
    of the sixteen adapters that have anything to collapse, and the gap was
    invisible: three of the missing ones had an empty class standing in for a
    test. A new adapter must now land in ``_CASES`` or ``_UNCOVERED``.
    """
    live = _adapters_declaring_axes()
    covered = {module for _id, _pkg, module, *_rest in _CASES} | {"cb19_precompute"}

    missing = live - covered - set(_UNCOVERED)
    assert not missing, (
        f"precompute adapters declaring collapsible axes with neither a case in "
        f"_CASES nor an entry in _UNCOVERED: {sorted(missing)}"
    )

    stale = (covered | set(_UNCOVERED)) - live
    assert not stale, (
        f"listed here but no longer declaring collapsible axes (rename or removal?): "
        f"{sorted(stale)}"
    )

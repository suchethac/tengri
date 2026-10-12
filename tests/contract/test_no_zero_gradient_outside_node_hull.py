# SPDX-License-Identifier: BSD-3-Clause
"""Contract: a fitted parameter never has an exactly-zero gradient inside its prior.

A lookup that clamps its query to a table's node hull before interpolating
(``jnp.clip``, ``compute_grid_weights`` clamp mode, a ``z_bracket`` weight clip)
returns a value that does not move with the query beyond the hull. If a fitted
parameter's declared prior reaches past that hull, the gradient is exactly zero
there and the sampler gets no signal (#1586, #1911, #2808).

The rule this test enforces, per lookup in :data:`FITTED_LOOKUPS`:

* **Numerical gap** (the prior exceeds the hull by less than half the adjacent
  node spacing): the lookup extrapolates, so the gradient is finite and nonzero
  over the FULL prior.
* **Physical gap** (the prior reaches half a spacing or more past the grid): the
  lookup does not extrapolate, because that would fabricate physics. The gap is
  registered in :data:`tengri.components.grid_support.GRID_SUPPORT`, and the
  gradient is checked over ``prior intersect hull`` only.
* **Boundary** (a derivative at an outermost node): the derivative equals the
  one-sided derivative from inside. ``jnp.clip`` halves it there.

:data:`CENSUS` records every clamping lookup in ``src/`` with its verdict. The
last test greps ``src/`` for the lookup symbols and fails on any call site not
in the census, so a new clamp is caught when it is added.

References
----------
Gap classification follows the orchestrator's rule for #2808 (SKIRTOR
``cos_inc``, stored from ``6.1e-17`` against a prior from ``0``): numerical gaps
are extrapolated, physical gaps are registered.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components._z_response import interp_z_table, ln1pz_nodes
from tengri.components.agn.kd18_agnfitter import KD18AGNfitterGrid, kd18_agnfitter_sed_from_grid
from tengri.components.agn.skirtor import _cos_inc_extrapolate
from tengri.utils.grid_interp import interp_nd_pchip, pchip_interp_1d, pchip_interp_local

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tengri"
_N_SWEEP = 201


@dataclass(frozen=True)
class FittedLookup:
    """One fitted lookup: its prior, its node hull, and how the sweep is taken.

    ``build()`` returns ``f(theta)``, a scalar function; ``theta`` is the fitted parameter value.
    ``gap`` is ``"numerical"`` (sweep the full prior) or ``"physical"`` (sweep
    ``prior intersect hull``, and the gap must be in GRID_SUPPORT).
    """

    lookup_id: str
    prior: tuple[float, float]
    hull: tuple[float, float]
    gap: str
    build: Callable[[], Callable[[jnp.ndarray], jnp.ndarray]]


def _sweep(lo: float, hi: float) -> np.ndarray:
    """Evenly spaced points over ``[lo, hi]`` that include both ends."""
    return np.linspace(lo, hi, _N_SWEEP)


def _z_table_case(extrapolate: bool) -> tuple[Callable, tuple[float, float]]:
    """Free-redshift table read: the stellar pchip z-table and the term-response table."""
    z_nodes = np.expm1(np.asarray(ln1pz_nodes(0.001, 20.2, 60)))
    vals = np.exp(-z_nodes / 3.0) * (1.0 + z_nodes)
    table = {"ln1pz": jnp.log1p(jnp.asarray(z_nodes)), "values": jnp.asarray(vals)}

    def f(z):
        return interp_z_table(table, z, extrapolate=extrapolate)

    return f, (float(z_nodes[0]), float(z_nodes[-1]))


def _pchip_local_case(z_lo: float, z_hi: float) -> Callable:
    """The stellar free-z read: ``pchip_interp_local`` on a z-space node grid."""
    z_nodes = jnp.asarray(np.expm1(np.asarray(ln1pz_nodes(0.001, 20.2, 60))))
    table = jnp.exp(-z_nodes / 3.0) * (1.0 + z_nodes)

    def f(z):
        return pchip_interp_local(z_nodes, table, z, extrapolate=True)

    return f


def _kd18_grid() -> KD18AGNfitterGrid:
    """A small synthetic KD18 grid whose SHAPE varies with both axes.

    The SED is renormalized to a fixed luminosity, so an axis that only scales
    the template has no effect on its shape. The spectral slope here depends on
    both ``log_mbh`` and ``log_edd``, so each axis moves the normalized SED.
    """
    log_mbh = jnp.linspace(6.0, 10.0, 9)
    log_edd = jnp.linspace(-1.5, 0.0, 7)
    wave = jnp.logspace(2.0, 5.0, 40)
    slope = -0.8 - 0.1 * log_edd[None, :, None] - 0.15 * (log_mbh[:, None, None] - 8.0)
    amp = jnp.exp(0.3 * log_mbh[:, None] - 0.5 * log_edd[None, :])
    template = amp[:, :, None] * (wave[None, None, :] / 1e3) ** slope
    return KD18AGNfitterGrid(template=template, wave_grid=wave, log_mbh=log_mbh, log_edd=log_edd)


def _kd18_case(axis: str) -> Callable:
    """KD18 disc SED shape at one wavelength, as a function of one fitted parameter."""
    grid = _kd18_grid()
    wave = jnp.logspace(2.5, 4.5, 30)

    def f(theta):
        kwargs = {"agn_log_mbh": 8.0, "agn_log_ledd": -0.7}
        kwargs[axis] = theta
        sed = kd18_agnfitter_sed_from_grid(grid, wave, **kwargs)
        # Shape functional: the SED is renormalized to a fixed luminosity, so the
        # total is constant and only the spectral shape carries the parameter.
        return sed[5] / jnp.sum(sed)

    return f


def _skirtor_cos_inc_case() -> Callable:
    """SKIRTOR's cos_inc axis: stored from cos(90 deg)=6.1e-17, prior reaches 0."""
    ax_oa = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    ax_cos = jnp.asarray([6.1e-17, 0.25, 0.5, 0.75, 1.0])
    rng = np.random.default_rng(3)
    grid = jnp.asarray(1.0 + 0.2 * rng.standard_normal((4, 5)))
    extrap = _cos_inc_extrapolate((ax_oa, ax_cos))

    def f(cos_inc):
        return interp_nd_pchip(grid, (ax_oa, ax_cos), (25.0, cos_inc), extrapolate=extrap)

    return f


def _fritz_psy_case() -> Callable:
    """Fritz library read at psy = fritz_psy_from_cos_inc(cos i), over the real psy axis.

    Reads the library's psy axis from the shipped grid file. The lookup is the
    end-cubic extension the Fritz psy lookup uses, so the check covers the psy
    derivation and the extension together.
    """
    import h5py

    from tengri._data_setup import find_data
    from tengri.components.agn.fritz import fritz_psy_from_cos_inc

    path = find_data("fritz2006_torus_grid.h5")
    if path is None:
        pytest.skip("fritz2006_torus_grid.h5 not installed")
    with h5py.File(path, "r") as f:
        psy_nodes = jnp.asarray(f["fritz2006/psy_axis"][:], dtype=jnp.float64)
    values = jnp.sin(jnp.radians(psy_nodes)) + 0.1 * psy_nodes / 90.0

    def f(cos_inc):
        psy = fritz_psy_from_cos_inc(cos_inc)
        return pchip_interp_1d(psy_nodes, values, psy, extrapolate=True)

    return f


def _bosa_case() -> Callable:
    """BOSA template read at a fitted dust_log_ssfr, integrated over the band.

    The prior [-11, -8.4] lies inside the stored axis [-11, -8.3999996]; this row
    is a regression guard that the in-hull read keeps a nonzero gradient.
    """
    from tengri._data_setup import find_data
    from tengri.components.dust.emission_templates import create_bosa_from_grid

    path = find_data("bosa_templates.h5")
    if path is None:
        pytest.skip("bosa_templates.h5 not installed")
    emit = create_bosa_from_grid(str(path))
    wave = jnp.logspace(3.5, 4.5, 24)

    def f(log_ssfr):
        return jnp.sum(emit(wave, 1.0e44, dust_log_ssfr=log_ssfr, log_L_ir=10.0))

    return f


FITTED_LOOKUPS: tuple[FittedLookup, ...] = (
    FittedLookup(
        "z_table.pchip_interp_local",
        prior=(0.0, 20.0),
        hull=(0.001, 20.2),
        gap="numerical",
        build=lambda: _pchip_local_case(0.001, 20.2),
    ),
    FittedLookup(
        "z_table.interp_z_table",
        prior=(0.0, 20.0),
        hull=(0.001, 20.2),
        gap="numerical",
        build=lambda: _z_table_case(extrapolate=True)[0],
    ),
    FittedLookup(
        "kd18.agn_log_mbh",
        prior=(6.0, 10.0),
        hull=(6.0, 10.0),
        gap="none",
        build=lambda: _kd18_case("agn_log_mbh"),
    ),
    FittedLookup(
        "kd18.agn_log_ledd",
        prior=(-2.0, 0.5),
        hull=(-1.5, 0.0),
        gap="physical",
        build=lambda: _kd18_case("agn_log_ledd"),
    ),
    FittedLookup(
        "skirtor.cos_inc",
        prior=(0.0, 1.0),
        hull=(6.1e-17, 1.0),
        gap="numerical",
        build=lambda: _skirtor_cos_inc_case(),
    ),
    FittedLookup(
        "fritz.psy_from_cos_inc",
        prior=(0.0, 1.0),
        hull=(0.001, 89.99),
        gap="numerical",
        build=lambda: _fritz_psy_case(),
    ),
    FittedLookup(
        "bosa.dust_log_ssfr",
        prior=(-11.0, -8.4),
        hull=(-11.0, -8.3999996),
        gap="none",
        build=lambda: _bosa_case(),
    ),
)

#: Physical gaps: a fitted prior reaching past a stored axis by at least half a node
#: spacing. The clamp stays (extrapolating would invent physics); the gap is
#: registered in GRID_SUPPORT so a build warns. ``gap_over_spacing`` was measured
#: from the shipped grid file (gap divided by the adjacent node spacing).
PHYSICAL_ROWS: tuple[tuple[str, str, str, tuple[float, float], float, str], ...] = (
    (
        "agn.torus",
        "nenkova_agnfitter",
        "agn_cos_inc",
        (0.0, 1.0),
        1.0,
        "nenkova_agnfitter_torus_grid.h5",
    ),
    ("agn.disc", "relagn", "agn_log_mbh", (6.0, 10.0), 6.3, "relagn_disc_grid.h5"),
    ("dust.attenuation", "wg00", "dust_tau_v", (0.0, 4.0), 1.0, "wg00_attenuation_grid.h5"),
)

#: Every src file that calls a lookup helper (``pchip_interp_local``,
#: ``interp_z_table``, ``z_bracket``, ``interp_nd_pchip``, ``interp_nd_triweight``):
#: relative path -> verdict. ``fitted-fixed`` means the fitted prior reached a
#: numerical gap and the lookup now extrapolates or the call site was changed.
#: ``fitted-ok`` means the prior is inside the hull or the grid is built over it.
#: ``open:<gap>`` marks a BUG row from the census that is NOT yet fixed; it is
#: reported, not hidden. ``fixed-config`` and ``unknown`` are not fitted inputs
#: or could not be traced.
CENSUS: dict[str, str] = {
    "components/stellar/component.py": "fitted-fixed (z-table, numerical gap 0.001 in z)",
    "components/_z_response.py": "fitted-fixed (z-table read, extrapolate flag)",
    "components/_term_response.py": "fitted-fixed (redshift table read)",
    "components/radio/component.py": (
        "fitted-ok (IR-window fraction z-table: one node at Fixed z; for free z the nodes span "
        "the redshift prior padded 1%, so the read stays inside the grid, default hold-edge)"
    ),
    "components/dust/emission/_component_base.py": "fitted-fixed (redshift table read)",
    "components/dust/energy_balance_precompute.py": "fitted-fixed (redshift table read)",
    "components/nebular/nebular_band_z.py": "fitted-fixed (redshift table read)",
    "components/nebular/nebular_grid_precompute.py": (
        "fitted-fixed (redshift table read; kinds=linear axes ok by construction)",
    ),
    "components/agn/kd18_agnfitter.py": (
        "fitted-ok; agn_log_ledd physical gap registered (GRID_SUPPORT kd18)",
    ),
    "components/agn/skirtor.py": "fitted-fixed (cos_inc extrapolate, #2808)",
    "components/agn/fritz.py": (
        "fitted-fixed (psy numerical gap: psy extended, cap continued linearly)"
    ),
    "components/agn/_template_grid.py": (
        "fitted-ok (cat3d grid cos 6.1e-17 vs prior 0; numerical, see skirtor pattern)",
    ),
    "components/agn/cat3d_precompute.py": "fitted-ok (cos 6.1e-17 vs prior 0)",
    "components/agn/cat3d_wind_lowfwd_precompute.py": "fitted-ok (cos 6.1e-17 vs prior 0)",
    "components/agn/silva04.py": "fitted-ok",
    "components/agn/silva04_precompute.py": "fitted-ok",
    "components/agn/nenkova_agnfitter_precompute.py": (
        "fitted-physical-registered (GRID_SUPPORT nenkova_agnfitter, gap 1.0 spacing)"
    ),
    "components/agn/nenkova_agnfitter_2p_precompute.py": "fitted-ok",
    "components/agn/nenkova_agnfitter_3p_precompute.py": "fitted-ok",
    "components/agn/skirtor_agnfitter_precompute.py": "fitted-ok",
    "components/agn/skirtor_agnfitter_1p_precompute.py": "fitted-ok",
    "components/agn/skirtor_agnfitter_2p_precompute.py": "fitted-ok",
    "components/agn/skirtor_precompute.py": "fitted-ok (triweight, grid over prior)",
    "components/agn/disc_precompute.py": "fitted-ok by construction (axes over prior)",
    "components/agn/disc.py": "fitted-physical-registered (GRID_SUPPORT relagn, gap 6.3 spacings)",
    "components/agn/torus.py": "fitted-ok (agn_tau hull = prior)",
    "components/agn/torus_library_transmission.py": (
        "open:_normalized_ratio cos_inc, tracked #2860 (file on #2837, absent here)"
    ),
    "components/agn/blocks/composable_precompute.py": "fitted-ok by construction",
    "components/agn/agn_nebular.py": "unknown (6-D grid file not identified)",
    "components/nebular/agn_nebular.py": "unknown (6-D grid file not identified)",
    "components/nebular/feltre_precompute.py": "fitted-ok",
    "components/nebular/mappings_shock_precompute.py": (
        "fitted-ok (jnp.clip upstream; velocity/density inside hull)",
    ),
    "components/nebular/shock.py": "fitted-ok (jnp.clip upstream; velocity/density inside hull)",
    "components/nebular/cb19_precompute.py": "unknown (cb19 grid absent in worktree)",
    "components/nebular/cloudy_cb19.py": "fitted-ok",
    "components/nebular/mappings_photo.py": "fitted-ok",
    "components/_collapsed_lookup.py": "fitted-ok by construction (axes over prior)",
    "components/dust/wg00.py": "fitted-physical-registered (GRID_SUPPORT wg00, gap 1.0 spacing)",
    "components/dust/dust_emission_precompute.py": (
        "fitted-ok (GRID_SUPPORT registered); log_ltir derived"
    ),
    "components/dust/emission_templates.py": (
        "fitted-ok (BOSA prior [-11,-8.4] inside stored [-11,-8.3999996])"
    ),
    "components/dust/dust_analytic_precompute.py": (
        "fitted-ok by construction; pinned axes fixed-config",
    ),
    "components/stellar/sps/precompute.py": "fixed-config (redshift at build)",
    "forward/precompute/templates.py": "unknown (callers not traced)",
    "utils/grid_interp.py": "mechanism (defines the lookups)",
    "utils/interpolation.py": "mechanism (defines compute_grid_weights clamp)",
}


def _grad_at(f: Callable, theta: float) -> float:
    return float(jax.grad(lambda t: f(t))(jnp.asarray(theta, dtype=jnp.float64)))


@pytest.mark.contract
@pytest.mark.parametrize("lookup", FITTED_LOOKUPS, ids=lambda lk: lk.lookup_id)
def test_gradient_is_finite_and_nonzero_across_prior(lookup: FittedLookup) -> None:
    """Gradient is finite and never exactly zero over the swept prior."""
    lo, hi = lookup.prior
    if lookup.gap == "physical":
        lo, hi = max(lo, lookup.hull[0]), min(hi, lookup.hull[1])
    f = lookup.build()
    grads = np.array([_grad_at(f, float(t)) for t in _sweep(lo, hi)])
    assert np.all(np.isfinite(grads)), f"{lookup.lookup_id}: non-finite gradient"
    assert np.any(grads != 0.0), f"{lookup.lookup_id}: gradient identically zero"
    zeros = np.flatnonzero(grads == 0.0)
    assert zeros.size == 0, (
        f"{lookup.lookup_id}: exactly-zero gradient at "
        f"{_sweep(lo, hi)[zeros[:5]]} (prior {lookup.prior}, hull {lookup.hull})"
    )


_PCHIP_TIE_PR = (
    "depends on #2857: its where-based clamp in _pchip_eval_axis0 removes the tie halving"
)


@pytest.mark.contract
@pytest.mark.parametrize(
    "which",
    [
        pytest.param(
            "interp_nd_pchip",
            marks=pytest.mark.xfail(strict=True, reason=_PCHIP_TIE_PR),
        ),
        "pchip_interp_local",
    ],
)
def test_clamped_pchip_boundary_derivative_is_not_halved(which: str) -> None:
    """The default (clamped) PCHIP read keeps its derivative on the outermost node.

    Exercises the clamp path (no ``extrapolate``) with the query exactly on the
    hull ends. ``jnp.clip`` would return half the one-sided derivative there.
    """
    x = jnp.asarray(np.linspace(0.0, 2.0, 9) ** 1.3)
    y = jnp.sin(3.0 * x) + 0.2 * x

    def f(q):
        if which == "interp_nd_pchip":
            return interp_nd_pchip(y, (x,), (q,))
        return pchip_interp_local(x, y, q)

    for edge, inside in ((float(x[0]), float(x[0]) + 1e-7), (float(x[-1]), float(x[-1]) - 1e-7)):
        assert _grad_at(f, edge) == pytest.approx(_grad_at(f, inside), rel=1e-5), (which, edge)


@pytest.mark.contract
@pytest.mark.parametrize(
    "selector, name, pname, prior, gap_over_spacing, filename",
    PHYSICAL_ROWS,
    ids=[row[1] for row in PHYSICAL_ROWS],
)
def test_physical_gap_is_registered_and_warns(
    selector, name, pname, prior, gap_over_spacing, filename
) -> None:
    """A physical gap has a GRID_SUPPORT entry, read from the grid file, and warns."""
    from tengri.components import grid_support as gs

    assert gap_over_spacing >= 0.5, "a physical row must be at least half a spacing"
    assert (selector, name) in gs.GRID_SUPPORT, f"{selector}/{name} not registered"
    extent = gs.grid_support(selector, name)[pname]
    assert extent == pytest.approx(_axis_extent_from_file(filename), rel=1e-9)
    findings = gs.check_grid_support([(selector, name)], {pname: prior})
    assert findings, f"{selector}/{name}: prior {prior} overhang not reported"


def _axis_extent_from_file(filename: str) -> tuple[float, float]:
    """Independent read of a stored axis, used to check each registry extent."""
    import h5py

    from tengri._data_setup import find_data

    path = find_data(filename)
    with h5py.File(path, "r") as f:
        if filename.startswith("nenkova"):
            incl = np.asarray(f["nenkova_agnfitter/incl_axis"][:], dtype=np.float64)
            cos = np.cos(np.radians(incl))
            return float(cos.min()), float(cos.max())
        if filename.startswith("relagn"):
            ax = np.asarray(f["log_mbh"][:], dtype=np.float64)
        else:
            ax = np.asarray(f["wg00/tau_v_axis"][:], dtype=np.float64)
    return float(ax.min()), float(ax.max())


@pytest.mark.contract
def test_narrow_z_table_warns_at_build_and_holds_at_the_band_edge() -> None:
    """A redshift table narrower than its prior is a support gap: warned, held at the edge.

    The table covers z in [0.5, 1.0]; the prior is [0, 3]. The overhang past z = 1
    is far more than half a node spacing, so it must not be extrapolated: the value
    at z = 3 equals the value at the top band edge, not a continuation of the
    edge cubic.
    """
    from tengri.components._z_response import (
        interp_z_table,
        ln1pz_nodes,
        warn_if_z_table_narrower_than_prior,
    )
    from tengri.config.exceptions import ZTableCoverageWarning

    nodes = ln1pz_nodes(0.5, 1.0, 40)
    z_nodes = np.expm1(np.asarray(nodes))
    vals = jnp.asarray(np.exp(-z_nodes) * (1.0 + z_nodes))
    table = {"ln1pz": nodes, "values": vals}

    with pytest.warns(ZTableCoverageWarning, match=r"\[0, 3\]"):
        warn_if_z_table_narrower_than_prior(nodes, (0.0, 3.0))

    h_hi = float(nodes[-1] - nodes[-2])
    z_band_top = float(np.expm1(float(nodes[-1]) + 0.5 * h_hi))
    v_band = float(interp_z_table(table, z_band_top, extrapolate=True))
    v_at_3 = float(interp_z_table(table, 3.0, extrapolate=True))
    assert v_at_3 == pytest.approx(v_band, rel=1e-12)
    g_at_3 = float(jax.grad(lambda z: interp_z_table(table, z, extrapolate=True))(3.0))
    assert g_at_3 == 0.0


@pytest.mark.contract
def test_narrow_z_table_numerical_overhang_still_extrapolates() -> None:
    """Inside the half-spacing band the extrapolation is live: the gradient is nonzero."""
    from tengri.components._z_response import interp_z_table, ln1pz_nodes

    nodes = ln1pz_nodes(0.5, 1.0, 40)
    z_nodes = np.expm1(np.asarray(nodes))
    vals = jnp.asarray(np.exp(-z_nodes) * (1.0 + z_nodes))
    table = {"ln1pz": nodes, "values": vals}
    h_hi = float(nodes[-1] - nodes[-2])
    z_in_band = float(np.expm1(float(nodes[-1]) + 0.25 * h_hi))
    g = float(jax.grad(lambda z: interp_z_table(table, z, extrapolate=True))(z_in_band))
    assert np.isfinite(g) and g != 0.0


@pytest.mark.contract
def test_boundary_node_derivative_is_not_halved() -> None:
    """At an outermost node the derivative equals the one-sided derivative from inside.

    ``jnp.clip`` splits its derivative in half at a tie, so the boundary node
    lost half its gradient. The where-form clamp keeps it.
    """
    lookup = next(lk for lk in FITTED_LOOKUPS if lk.lookup_id == "kd18.agn_log_mbh")
    f = lookup.build()
    g_bound = _grad_at(f, lookup.hull[1])
    g_inside = _grad_at(f, lookup.hull[1] - 1e-7)
    assert g_bound == pytest.approx(g_inside, rel=1e-5)


_LOOKUP_SYMBOLS = (
    "pchip_interp_local(",
    "interp_z_table(",
    "z_bracket(",
    "interp_nd_pchip(",
    "interp_nd_triweight(",
)


@pytest.mark.contract
def test_new_clamping_lookup_sites_are_censused() -> None:
    """Every src file that calls a lookup helper is named in the census (guards new sites).

    Call-site files that appear here but not in CENSUS are not evaluated by the
    sweep; they must be added to CENSUS with a verdict, or the test fails.
    """
    censused_files = set(CENSUS)
    unlisted = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text()
        rel = path.relative_to(_SRC).as_posix()
        uses_lookup = any(re.search(r"\b" + re.escape(sym), text) for sym in _LOOKUP_SYMBOLS)
        if uses_lookup and rel not in censused_files:
            unlisted.append(rel)
    assert not unlisted, f"lookup call sites missing from CENSUS: {unlisted}"

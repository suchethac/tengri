# SPDX-License-Identifier: BSD-3-Clause
"""Grid-support policy at build time: refuse explicit overhang, resolve FREE (#2626, #2244).

A template-backed component clips a parameter onto its grid's edge node
(``jnp.clip`` is flat there), so a value outside the grid is bit-identical to
the edge and has a zero gradient. The policy for every component registered in
:data:`~tengri.components.grid_support.GRID_SUPPORT`:

1. A ``Fixed`` value outside the support is **refused** at build, naming the
   parameter, value, component and extent.
2. A ``FREE`` parameter whose prior is declared by the registry **resolves to
   the backend's support** (owner ruling 2026-10-14), so the prior equals the
   extent and the summary reads ``[resolved to <backend> grid: lo..hi]``.
3. An explicit user prior that extends past the support is **refused**.
4. An explicit prior inside the support is unchanged.

:data:`~tengri.components.grid_support.EXTRAPOLATING_SUPPORT` (Cue, a neural
emulator) keeps the warning and never raises.

The parametrization runs over every ``(selector, name, parameter)`` the build
grammar accepts. Two exclusions are documented in :data:`_NOT_IN_GRAMMAR`.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry
from tengri._data_setup import find_data_str
from tengri.components.agn.blocks.runner import RecipeWarning
from tengri.components.grid_support import EXTRAPOLATING_SUPPORT, GRID_SUPPORT, grid_support
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import GridSupportWarning, ParameterError
from tengri.parameters import parse_groups
from tengri.parameters.priors import Uniform
from tests._cb19_grid import write_synthetic_cb19_grid

pytestmark = pytest.mark.contract

#: ``(selector, name, parameter)`` the grammar does not accept as a key, so no
#: build can set it. MAPPINGS (stellar and AGN) is registered as experimental
#: (``describe('mappings')``: "refuses until #2082"); its neb group rejects
#: ``logU`` and ``logZ_gas``. Nothing here can be refused or resolved for a key
#: the user cannot write, so these rows are excluded rather than failed.
_NOT_IN_GRAMMAR = frozenset(
    {
        ("neb", "mappings", "neb_logU"),
        ("neb", "mappings_agn", "neb_logU"),
    }
)

#: Off-support Fixed value, where the brief's ``hi + 0.5 * width`` would fall
#: outside the registry's own physical range and so fail for a different reason.
_OFF_SUPPORT_OVERRIDE = {
    ("dust.emission", "astrodust", "dust_lgU"): 6.5,  # physical [0, 7]
    ("neb", "cue", "neb_logU"): -0.5,  # physical [-5, 0]
    ("neb", "cue", "gas_logn"): 4.5,  # physical [0, 5]
}

#: Explicit overhang that stays inside the registry's physical range, so the
#: refusal is the grid's and not the registry's.
_OVERHANG_OVERRIDE = {("dust.emission", "astrodust", "dust_lgU"): Uniform(0.5, 6.5)}

#: An inside-support explicit prior that also satisfies the registry's range.
_INNER_OVERRIDE = {("dust.emission", "astrodust", "dust_lgU"): Uniform(0.5, 5.5)}

#: Cue's optional knobs carry no registry default, so the grammar refuses a
#: bare FREE for them (an explicit prior is required). That refusal is the
#: policy for these rows: FREE has no declared prior to resolve.
_FREE_REFUSED = frozenset(
    {("neb", "cue", "gas_logn"), ("neb", "cue", "gas_logco"), ("neb", "cue", "gas_logno")}
)

#: Where the registry's physical range is narrower than the grid, the resolved
#: FREE prior is ``declared intersect physical`` rather than the full support.
_PHYSICAL_FLOOR = {("dust.emission", "astrodust", "dust_lgU"): (0.0, 6.0)}

_TOL = 1e-6

_NEB_KEY = {"neb_logU": "logU", "neb_logZ_gas": "logZ_gas"}
_CUE_KEYS = {"gas_logn": "gas_logn", "gas_logco": "gas_logco", "gas_logno": "gas_logno"}


def _cases():
    """Yield ``(selector, name, parameter)`` for every registered support row.

    A component whose grid is not installed (``grid_support`` returns ``{}``)
    still yields one row, marked skip with the reason, so the absence is
    visible in the report rather than silently absent from collection.
    """
    for selector, name in GRID_SUPPORT:
        support = grid_support(selector, name)
        if not support:
            reason = f"{selector}={name}: grid not installed (grid_support returns {{}})"
            yield pytest.param(
                selector,
                name,
                None,
                marks=pytest.mark.skip(reason=reason),
                id=f"{selector}-{name}-absent",
            )
            continue
        for pname in support:
            if (selector, name, pname) in _NOT_IN_GRAMMAR:
                continue
            yield pytest.param(selector, name, pname, id=f"{selector}-{name}-{pname}")


def _support_or_skip(selector: str, name: str) -> dict[str, tuple[float, float]]:
    """The installed support of one component; skip only when its grid is absent."""
    support = grid_support(selector, name)
    if not support:
        pytest.skip(f"{selector}={name}: grid not installed (grid_support returns {{}})")
    return support


def _key(selector: str, name: str, pname: str) -> tuple[str, str]:
    """``(group, key)`` the build grammar uses for one parameter."""
    if selector == "agn.disc":
        return "agn", pname[len("agn_") :] if pname.startswith("agn_") else pname
    if selector == "dust.emission":
        return "dust_emission", pname[len("dust_") :]
    if pname in _NEB_KEY:
        return "neb", _NEB_KEY[pname]
    return "neb", _CUE_KEYS[pname]


def _build(selector: str, name: str, pname: str, value):
    """Build a spec with ``pname`` set to ``value`` through the grammar."""
    group, key = _key(selector, name, pname)
    body = {"type": name, key: value}
    if selector == "agn.disc":
        spec = {"type": "composable", "disc": body}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecipeWarning)
            return parse_groups(sfh={"type": "dpl"}, agn=spec, redshift=0.1)
    return parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=0.1,
        **{group: body},
    )


def _is_extrapolating(selector: str, name: str) -> bool:
    return (selector, name) in EXTRAPOLATING_SUPPORT


def _width(extent: tuple[float, float]) -> float:
    return extent[1] - extent[0]


@pytest.mark.parametrize(("selector", "name", "pname"), _cases())
def test_fixed_outside_support_is_refused_or_warned(selector, name, pname):
    """(i) A Fixed value past the support raises, naming parameter, value, component, extent."""
    support = _support_or_skip(selector, name)
    lo, hi = support[pname]
    value = _OFF_SUPPORT_OVERRIDE.get((selector, name, pname), hi + 0.5 * _width((lo, hi)))
    assert value > hi, "the probe value must lie past the support"

    if _is_extrapolating(selector, name):
        with pytest.warns(GridSupportWarning):
            _build(selector, name, pname, Fixed(value))
        return

    with pytest.raises(ParameterError) as info:
        _build(selector, name, pname, Fixed(value))
    message = str(info.value)
    assert pname in message
    assert name in message
    assert f"{lo:g}" in message and f"{hi:g}" in message


@pytest.mark.parametrize(("selector", "name", "pname"), _cases())
def test_free_resolves_to_support_extent(selector, name, pname):
    """(ii)/(v) FREE builds, its prior equals the support extent, summary says so."""
    support = _support_or_skip(selector, name)
    lo, hi = support[pname]
    if (selector, name, pname) in _FREE_REFUSED:
        with pytest.raises(ValueError, match="explicit prior"):
            _build(selector, name, pname, FREE)
        return
    spec = _build(selector, name, pname, FREE)
    got_lo, got_hi = spec.get_distribution(pname).bounds

    expected = _PHYSICAL_FLOOR.get((selector, name, pname), (lo, hi))
    assert abs(got_lo - expected[0]) < _TOL
    assert abs(got_hi - expected[1]) < _TOL
    assert f"resolved to {name} grid" in spec.summary_str()


@pytest.mark.parametrize(("selector", "name", "pname"), _cases())
def test_explicit_prior_past_support_is_refused_or_warned(selector, name, pname):
    """(iii) An explicit Uniform reaching past the support raises, naming the extent."""
    support = _support_or_skip(selector, name)
    lo, hi = support[pname]
    prior = _OVERHANG_OVERRIDE.get((selector, name, pname), Uniform(lo - 1.0, hi + 1.0))

    if _is_extrapolating(selector, name):
        with pytest.warns(GridSupportWarning):
            _build(selector, name, pname, prior)
        return

    with pytest.raises(ParameterError) as info:
        _build(selector, name, pname, prior)
    message = str(info.value)
    assert pname in message
    assert f"{lo:g}" in message and f"{hi:g}" in message


@pytest.mark.parametrize(("selector", "name", "pname"), _cases())
def test_explicit_prior_inside_support_is_unchanged(selector, name, pname):
    """(iv) An explicit Uniform inside the support builds bit-exact and is not resolved."""
    support = _support_or_skip(selector, name)
    lo, hi = support[pname]
    inner = _INNER_OVERRIDE.get(
        (selector, name, pname), Uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = _build(selector, name, pname, inner)
    # Only this parameter's findings matter: a registry default elsewhere that
    # overhangs a narrower grid is a separate, pre-existing report.
    own = [
        w for w in caught if issubclass(w.category, GridSupportWarning) and pname in str(w.message)
    ]
    assert own == []
    got = spec.get_distribution(pname).bounds
    assert got == pytest.approx((inner.bounds[0], inner.bounds[1]), abs=0.0)
    assert "resolved to" not in spec.summary_str()


# --------------------------------------------------------------------------
# Gradient through the lookup: the knob must be live (owner ruling 2026-10-14).
# --------------------------------------------------------------------------

_CLOUDY_GRID = find_data_str("cloudy_grid_prsc.h5")
_CUE_WEIGHTS = find_data_str("cue_weights.npz")


def _synthetic_ssp() -> SSPData:
    """Small synthetic SSP; only Q_H and the line/continuum shape matter here."""
    wave = jnp.logspace(2.0, 5.0, 400)
    ages = jnp.linspace(-3.0, 1.14, 15)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    flux = jnp.broadcast_to((5000.0 / wave) ** 2, (3, 15, 400))
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
    )


def _model_with_logzgas_free(neb: dict):
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.SEDModel.build(
            ssp_data=_synthetic_ssp(),
            observation=obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            neb={**neb, "logZ_gas": FREE, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.0),
        )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("cloudy", id="cloudy"),
        pytest.param("cue", id="cue"),
        pytest.param("cb19", id="cb19"),
    ],
)
def test_logzgas_gradient_is_live_at_interior_point(name, tmp_path):
    """(vi) d(total photometry)/d(neb_logZ_gas) is finite and non-zero inside the support.

    CB19 is built from the synthetic grid the suite writes (``write_synthetic_cb19_grid``):
    the shipped ``cb19_templates.h5`` is the flat placeholder (#924), which the
    backend refuses to build from. Its support still comes from ``grid_support``.
    """
    support = _support_or_skip("neb", name)
    lo, hi = support["neb_logZ_gas"]
    mid = 0.5 * (lo + hi)

    neb = {"type": name}
    if name == "cloudy":
        if _CLOUDY_GRID is None:
            pytest.skip("cloudy: cloudy_grid_prsc.h5 is not reachable from this checkout")
        neb["grid"] = _CLOUDY_GRID
    elif name == "cb19":
        neb["grid"] = str(write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5"))
    elif _CUE_WEIGHTS is None:
        pytest.skip("cue: cue_weights.npz is not reachable from this checkout")
    model = _model_with_logzgas_free(neb)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    params["neb_logZ_gas"] = jnp.asarray(mid)

    def total(p):
        return jnp.sum(model.predict_photometry(p))

    grad = float(jax.grad(total)(params)["neb_logZ_gas"])
    assert jnp.isfinite(grad)
    assert grad != 0.0, (
        f"neb={name}: d(photometry)/d(neb_logZ_gas) is exactly zero at {mid:g}; "
        "the knob is dead through this backend's lookup"
    )


def test_direct_flat_build_refuses_fixed_overhang(tmp_path):
    """(i) on the flat ``Parameters(...)`` path: the same refusal, from its own provenance.

    CB19 is used with the synthetic grid the suite writes: the flat cloudy build
    needs a grid file that the hermetic pytest locator does not reach.
    """
    from tengri.parameters.parameters import Parameters

    grid = write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5")
    lo, hi = grid_support("neb", "cb19")["neb_logU"]
    with (
        warnings.catch_warnings(),
        pytest.raises(ParameterError, match=rf"neb_logU.*\[{lo:g}, {hi:g}\]"),
    ):
        warnings.simplefilter("ignore")
        Parameters(
            mean_sfh_type="dpl",
            nebular="cb19",
            nebular_cb19_grid_path=str(grid),
            neb_logU=Fixed(hi + 0.5),
            redshift=Fixed(0.1),
        )

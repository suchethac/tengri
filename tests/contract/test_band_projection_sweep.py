# SPDX-License-Identifier: BSD-3-Clause
"""One band-projection contract: every emitter x convention x redshift mode (#3505, M1a).

The band integral must read one :class:`~tengri.utils.band_projection.BandSpec` (filters,
convention, sub-band scheme) in every table builder and every runtime projection. Three
checks pin that class:

* the sweep: for each emitter, each filter convention and each redshift mode, the
  ``WavePrecomp`` photometry agrees with the exact path per band at the LUT's documented
  bound, with fallback warnings promoted to errors;
* the canary: each registered band-table provider's tables change between the photon and
  energy conventions on a sloped SED, so a provider that ignores the spec fails;
* the AST check: outside the named public entry points where a user first declares it, no
  function parameter and no dataclass field named ``convention`` in ``src/tengri`` has a default.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FeaturePrecomp,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.config.exceptions import (
    ComponentDataNotAvailableWarning,
    CorruptEnergyBalanceWarning,
    DeadPrecomputeAxisWarning,
    FilterCoverageWarning,
    GridSupportWarning,
    PrecompBiasWarning,
    SSPMassRemainingFallbackWarning,
    ZTableCoverageWarning,
)
from tengri.forward.precompute.band_table import BandTable
from tengri.forward.precompute.registry import BAND_TABLE_PROVIDERS, resolve_band_table_provider
from tengri.observation.photometry import FilterConvention
from tests.contract import _band_table_cases as cases

pytestmark = pytest.mark.contract

#: Documented bound of the photometry LUT against the exact path (relative, per band).
LUT_RTOL = 6e-3

CONVENTIONS = (FilterConvention.BESSELL, FilterConvention.ENERGY)
REDSHIFTS = {"fixed": Fixed(0.1), "free": Uniform(0.05, 0.3)}
_Z_EVAL = 0.1

_DUST_ATT = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
_SINGLE = {"type": "single_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
_DEFAULT_BANDS = ["sdss_g", "sdss_r", "sdss_z", "2mass_ks"]

#: Emitter -> group kwargs, SSP kind, bands and approximation.
EMITTERS: dict[str, dict] = {
    "stellar": {"groups": {}, "ssp": "wide"},
    "nebular_grid": {
        "groups": {"neb": {"type": "cue", "all_params": Fixed(DEFAULT)}},
        "ssp": "fsps",
        "bands": ["sdss_r"],
        "approx": (WavePrecomp(), FeaturePrecomp()),
    },
    "cloudy_grid": {"groups": {}, "ssp": "cloudy", "bands": ["sdss_g", "sdss_r"]},
    "dust_two_component": {"groups": {"dust_attenuation": _DUST_ATT}, "ssp": "wide"},
    "dust_single_screen": {"groups": {"dust_attenuation": _SINGLE}, "ssp": "wide"},
    "dust_emission": {
        "groups": {
            "dust_attenuation": _DUST_ATT,
            "dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)},
        },
        "ssp": "wide",
        "bands": ["wise_w3", "sdss_r"],
    },
    "agn": {
        "groups": {
            "agn": {
                "type": "composable",
                "all_params": Fixed(DEFAULT),
                "disc": {"type": "multicolor"},
                "torus": {"type": "simple"},
                "agn_log_lbol": Fixed(11.0),
            }
        },
        "ssp": "wide",
    },
    "radio": {
        "groups": {
            "dust_attenuation": _DUST_ATT,
            "radio": {
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "all_params": Fixed(DEFAULT),
            },
        },
        "ssp": "wide",
    },
    "xray": {"groups": {"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}}, "ssp": "wide"},
}

#: Cells that cannot run, keyed by (emitter, convention value), each with the reason it gives.
SKIPS: dict[tuple[str, str], str] = {}


#: Extra spectral steepening of the synthetic SSP: its L_nu is multiplied by
#: ``(5000 A / lambda) ** STEEP_INDEX``, so it falls as lambda^-(2 + STEEP_INDEX) and a band
#: integral under the photon and energy weights differs by more than the LUT bound.
STEEP_INDEX = 2.0


def _steepen(ssp):
    return ssp._replace(ssp_flux=ssp.ssp_flux * (5000.0 / ssp.ssp_wave) ** STEEP_INDEX)


def _ssp(kind, request, physical=False):
    if kind == "wide":
        # The float32 arm needs a physically normalized grid (the plain one is not).
        return _steepen(
            request.getfixturevalue(
                "synthetic_ssp_wide_physical" if physical else "synthetic_ssp_wide"
            )
        )
    if kind == "fsps":
        return request.getfixturevalue("ssp_data_fsps")
    from tests._cloudy_grid import synthetic_ionizing_ssp

    return synthetic_ionizing_ssp()


def _build(request, name, convention, z_name, approx, physical=False):
    """Build one cell; the caller owns the warning filter."""
    cfg = EMITTERS[name]
    bands = cfg.get("bands") or _DEFAULT_BANDS
    obs = Observation(photometry=Photometry.from_names(bands, convention=convention))
    groups = dict(cfg["groups"])
    if cfg["ssp"] == "cloudy":
        groups["neb"] = {
            "type": "cloudy",
            "all_params": Fixed(DEFAULT),
            "grid": cases._synthetic_cloudy_grid(),
        }
    groups.setdefault("neb", {"type": "none"})
    groups.setdefault("dust_attenuation", {"type": "none"})
    return SEDModel.build(
        ssp_data=_ssp(cfg["ssp"], request, physical),
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=REDSHIFTS[z_name],
        approx=approx,
        **groups,
    )


#: Warning classes a precompute path raises when it declines or falls back (enumerated from
#: ``tengri.config.exceptions``): a cell that reaches one of them has silently left the path it
#: names, so it is an error here, not an advisory.
_FALLBACK_WARNINGS = (
    DeadPrecomputeAxisWarning,
    ZTableCoverageWarning,
    FilterCoverageWarning,
    PrecompBiasWarning,
    ComponentDataNotAvailableWarning,
    GridSupportWarning,
    CorruptEnergyBalanceWarning,
    SSPMassRemainingFallbackWarning,
)
#: Messages that name a decline or fallback whatever their class.
_FALLBACK_MESSAGE = r".*(fall(s|ing)? back|declin|per-call path|exact path).*"


def _fallbacks_are_errors():
    """Error on the fallback and decline warnings; leave configuration advisories alone."""
    warnings.simplefilter("default")
    for category in _FALLBACK_WARNINGS:
        warnings.filterwarnings("error", category=category)
    warnings.filterwarnings("error", category=RuntimeWarning)
    warnings.filterwarnings("error", message=_FALLBACK_MESSAGE)


#: Band-table slots that must hold a built table in each cell (the others decline by design:
#: the AGN and CLOUDY tables are built at one fixed redshift, and the stellar, dust-screen
#: and nebular-grid cells have no band-table provider). A provider that starts declining under
#: one convention changes this set and fails the cell.
_ENGAGED = {
    ("cloudy_grid", "fixed"): {("nebular", "band_table")},
    ("dust_emission", "fixed"): {("dust_emission", "emission_band_response")},
    ("dust_emission", "free"): {("dust_emission", "emission_band_response")},
    ("agn", "fixed"): {("agn", "band_table")},
    ("radio", "fixed"): {("radio", "term_band_response")},
    ("radio", "free"): {("radio", "term_band_response")},
    ("xray", "fixed"): {("xray", "term_band_response")},
    ("xray", "free"): {("xray", "term_band_response")},
}


def _engaged(model):
    return {
        (comp, slot)
        for comp, slots in model._band_tables.items()
        for slot, entry in slots.items()
        if isinstance(entry, BandTable)
    }


def _phot(model, z_name):
    params = {} if z_name == "fixed" else {"redshift": _Z_EVAL}
    return np.asarray(model.predict_photometry(params), dtype=np.float64)


@pytest.mark.parametrize("z_name", list(REDSHIFTS))
@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
@pytest.mark.parametrize("name", list(EMITTERS))
def test_waveprecomp_matches_the_exact_path(request, name, convention, z_name):
    reason = SKIPS.get((name, convention.value))
    if reason:
        pytest.skip(reason)
    approx = EMITTERS[name].get("approx", WavePrecomp())
    with warnings.catch_warnings():
        _fallbacks_are_errors()
        exact = _phot(_build(request, name, convention, z_name, None), z_name)
        lut_model = _build(request, name, convention, z_name, approx)
        lut = _phot(lut_model, z_name)
    assert _engaged(lut_model) == _ENGAGED.get((name, z_name), set()), (
        f"{name}/{convention.value}/{z_name}: band tables built {_engaged(lut_model)}"
    )
    assert np.all(np.isfinite(lut)), f"{name}: non-finite LUT photometry {lut}"
    rel = np.abs(lut / np.where(exact == 0.0, 1.0, exact) - 1.0)
    assert np.all(rel < LUT_RTOL), f"{name}/{convention.value}/{z_name}: per-band {rel}"


# ----------------------------------------------------------------- convention canary

#: Band-table provider -> parity case that engages it; every registry key needs one.
_CANARY_CASES = {
    "dust_emission": "dust_fixed_z",
    "agn": "composable_agn",
    "nebular": "cloudy_nebular",
    "shock": "shock",
    "radio": "free_z_radio_xray",
    "xray": "xray_band",
}


def _xray_band_model(convention):
    """An X-ray band beside an optical one, so the X-ray term has flux to project."""
    import jax.numpy as jnp

    from tengri.observation.photometry import FilterCurve

    xw = jnp.linspace(2.0, 12.0, 60)
    ow = jnp.linspace(5000.0, 6000.0, 40)
    curves = (
        FilterCurve(wave=xw, trans=jnp.full_like(xw, 0.5), name="xray_band"),
        FilterCurve(wave=ow, trans=jnp.sin(jnp.linspace(0.0, jnp.pi, 40)), name="optical"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=cases._synthetic_agn_ssp(),
            observation=Observation(photometry=Photometry(filters=curves, convention=convention)),
            approx=WavePrecomp(),
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            neb={"type": "none"},
            xray={"type": "simple", "all_params": Fixed(DEFAULT)},
        )


def _case_model(case, convention):
    """The parity-case model rebuilt with the given convention on its observation."""
    if case == "xray_band":
        return _xray_band_model(convention)
    cases.CONVENTION = convention
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cases.CASES[case]()
    finally:
        cases.CONVENTION = "bessell"


def _leaves(entry):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(entry.arrays)]


def test_every_registered_provider_has_a_canary():
    assert set(_CANARY_CASES) == set(BAND_TABLE_PROVIDERS)


@pytest.mark.parametrize("provider", sorted(BAND_TABLE_PROVIDERS))
def test_band_tables_follow_the_convention(provider):
    models = {c: _case_model(_CANARY_CASES[provider], c) for c in CONVENTIONS}
    slots = resolve_band_table_provider(provider).BAND_SLOTS
    worst = 0.0
    for slot in slots:
        a, b = (models[c]._band_table(provider, slot) for c in CONVENTIONS)
        if not (isinstance(a, BandTable) and isinstance(b, BandTable)):
            continue
        for x, y in zip(_leaves(a), _leaves(b), strict=True):
            if x.shape != y.shape or x.size == 0 or not np.issubdtype(x.dtype, np.floating):
                continue
            x, y = x.astype(np.float64), y.astype(np.float64)
            scale = np.max(np.abs(x))
            if scale > 0.0:
                worst = max(worst, float(np.max(np.abs(x - y)) / scale))
    assert worst > 1e-3, f"{provider}: band tables ignore the convention (max rel {worst:.2e})"


# ------------------------------------------------------------------------- AST check


#: The public entry points where a user declares the convention for the first time, by
#: qualified name (``file:function`` or ``file:Class.field``). Each one states its own default in
#: its documentation: ``from_names`` and ``from_filter_set`` default to Bessell, the
#: ``Photometry.convention`` and ``PhotometryObservationModel.convention`` fields default to
#: Bessell, and ``measure.photometry`` defaults to the convention its ``Photometry`` carries.
#: Everything below them takes the convention from the observation's ``BandSpec``, with no default.
_USER_DECLARATION_ENTRY_POINTS = {
    ("observation/photometry_config.py", "from_names"),
    ("observation/photometry_config.py", "from_filter_set"),
    ("observation/photometry_config.py", "Photometry.convention"),
    ("observation/photometry_model.py", "PhotometryObservationModel.convention"),
    ("measure.py", "photometry"),
}


def _convention_defaults_in(tree):
    """Yield ``(qualified_name, lineno)`` of each defaulted ``convention`` parameter or field."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "convention"
                    and stmt.value is not None
                ):
                    yield f"{node.name}.convention", stmt.lineno
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            a = node.args
            pos = a.posonlyargs + a.args
            defaulted = pos[len(pos) - len(a.defaults) :]
            defaulted += [k for k, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d]
            if any(arg.arg == "convention" for arg in defaulted):
                yield node.name, node.lineno


def _defaulted_conventions():
    root = Path(__file__).resolve().parents[2] / "src" / "tengri"
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        for name, lineno in _convention_defaults_in(ast.parse(path.read_text())):
            if (rel, name) not in _USER_DECLARATION_ENTRY_POINTS:
                found.append(f"{rel}:{lineno} {name}")
    return found


def test_convention_is_required_outside_the_public_entry_points():
    found = _defaulted_conventions()
    assert not found, "convention must be required, never defaulted:\n" + "\n".join(found)


# ------------------------------------------------------------------------------- float32


#: Emitters whose float32 table build fails on origin/main too, measured by running these cells
#: against its sources (the Bessell cells fail identically there): a band table or its
#: input overflows the float32 range ("overflow encountered in cast", or a non-finite CLOUDY
#: band). Strict xfail, so the cell flips to a failure when the defect is fixed.
_F32_ON_MAIN = {
    "radio": "float32 overflow while casting the radio band table (also on origin/main)",
    "dust_emission": "float32 overflow casting the dust-emission table (also on origin/main)",
    "cloudy_grid": "non-finite float32 CLOUDY band table (also on origin/main)",
    "dust_two_component": (
        "the float32 wavelength-table path of this dust model drifts past the bar on main: "
        "2mass_ks is off by 1.1e-2 against 7e-3 in both conventions, while "
        "the float64 table agrees with the exact path to 1.4e-4 (also on origin/main)"
    ),
    "dust_single_screen": (
        "the float32 wavelength-table path of this dust model drifts past the bar on main: "
        "2mass_ks is off by 8.0e-3 against 7e-3 in both conventions, while "
        "the float64 table agrees with the exact path to 1.4e-4 (also on origin/main)"
    ),
}


_F32_DRIFT = frozenset({"dust_two_component", "dust_single_screen"})


@pytest.mark.parametrize("z_name", list(REDSHIFTS))
@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
@pytest.mark.parametrize("name", list(EMITTERS))
def test_float32_waveprecomp_matches_the_float64_exact_path(request, name, convention, z_name):
    """The LUT built and evaluated in pure float32 stays finite and within the LUT bound.

    The float32 arm is built in its own mode (``jax.enable_x64(False)``), as the precision
    tests do; the reference is the float64 exact path of the same cell. The bound is the LUT's
    own (6e-3) plus float32 rounding.
    """
    if name in _F32_ON_MAIN:
        # The overflow cells fail with a warning raised as an error; the drift cells fail on
        # the bound, and only that failure is the expected one.
        raises = AssertionError if name in _F32_DRIFT else None
        request.applymarker(
            pytest.mark.xfail(strict=True, raises=raises, reason=_F32_ON_MAIN[name])
        )
    approx = EMITTERS[name].get("approx", WavePrecomp())
    with warnings.catch_warnings():
        _fallbacks_are_errors()
        exact = _phot(_build(request, name, convention, z_name, None, True), z_name)
        with jax.enable_x64(False):
            lut32 = _phot(_build(request, name, convention, z_name, approx, True), z_name)
    assert np.all(np.isfinite(lut32)), f"{name}: non-finite float32 photometry {lut32}"
    rel = np.abs(lut32 / np.where(exact == 0.0, 1.0, exact) - 1.0)
    assert np.all(rel < LUT_RTOL + 1e-3), f"{name}/{convention.value}/{z_name}: per-band {rel}"

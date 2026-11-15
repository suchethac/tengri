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
* the AST check: no function in ``src/tengri`` has a parameter named ``convention`` with a
  default value.
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
from tengri.components.stellar.component import SFHBeyondOldestTemplateWarning
from tengri.config.exceptions import AdvisoryWarning
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


def _ssp(kind, request):
    if kind == "wide":
        return request.getfixturevalue("synthetic_ssp_wide")
    if kind == "fsps":
        return request.getfixturevalue("ssp_data_fsps")
    from tests._cloudy_grid import synthetic_ionizing_ssp

    return synthetic_ionizing_ssp()


def _build(request, name, convention, z_name, approx):
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
        ssp_data=_ssp(cfg["ssp"], request),
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=REDSHIFTS[z_name],
        approx=approx,
        **groups,
    )


def _fallbacks_are_errors():
    """Promote every warning to an error except configuration advisories the cells trigger."""
    warnings.simplefilter("error")
    for category in (AdvisoryWarning, DeprecationWarning, SFHBeyondOldestTemplateWarning):
        warnings.filterwarnings("ignore", category=category)
    warnings.filterwarnings("ignore", message=".*outside the SSP grid.*")
    warnings.filterwarnings("ignore", message=".*ionizing spectral shape.*")
    warnings.filterwarnings("ignore", message=".*no .zsun. attribute.*")


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
        lut = _phot(_build(request, name, convention, z_name, approx), z_name)
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


#: The three public entry points where a user declares the convention for the first time.
#: Each states its own default in its documentation; everything below them takes the
#: convention from the observation's ``BandSpec`` with no default.
_USER_DECLARATION_ENTRY_POINTS = {
    ("observation/photometry_config.py", "from_names"),
    ("observation/photometry_config.py", "from_filter_set"),
    ("measure.py", "photometry"),
}


def _defaulted_convention_parameters():
    root = Path(__file__).resolve().parents[2] / "src" / "tengri"
    found = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            a = node.args
            pos = a.posonlyargs + a.args
            defaulted = pos[len(pos) - len(a.defaults) :]
            defaulted += [k for k, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d]
            if (str(path.relative_to(root)), node.name) in _USER_DECLARATION_ENTRY_POINTS:
                continue
            if any(arg.arg == "convention" for arg in defaulted):
                found.append(f"{path.relative_to(root)}:{node.lineno} {node.name}")
    return found


def test_no_function_defaults_its_filter_convention():
    found = _defaulted_convention_parameters()
    assert not found, "convention must be required, never defaulted:\n" + "\n".join(found)

# SPDX-License-Identifier: BSD-3-Clause
"""The ``(tau_V, mu)`` dust parameterization in the dict grammar and the flat surface (#3176).

``dust_attenuation={'type': 'two_component', 'parameterization': 'tauv_mu', ...}``
declares ``dust_tau_v`` and ``dust_mu`` in place of the birth-cloud / diffuse pair.
The same choice is reachable on the flat ``Parameters(...)`` surface as
``dust_parameterization='tauv_mu'``. ``chevallard13`` is legal only as the
diffuse-screen law ``law_diff`` (#3178).
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, FREE, Fixed
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract

_SUBSTR_TOTAL = "tau_v = tau_bc + tau_diff"
_SUBSTR_MU = "mu = tau_diff / (tau_bc + tau_diff)"
_SUBSTR_NATIVE = "parameterization='tauv_mu'"


def _atten(**overrides):
    """A two-screen block with power-law screens and the given overrides."""
    base = {
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "all_params": Fixed(DEFAULT),
    }
    base.update(overrides)
    return base


def _groups(**atten_overrides):
    return dict(
        redshift=Fixed(1.0),
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation=_atten(**atten_overrides),
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "none"},
    )


def _tauv_mu_spec(**atten_overrides):
    return parse_groups(
        **_groups(
            parameterization="tauv_mu",
            tau_v=Fixed(1.6),
            mu=Fixed(0.25),
            **atten_overrides,
        )
    )


# ── declarations ───────────────────────────────────────────────────────────


def test_grammar_declares_tau_v_and_mu_not_tau_bc():
    """tauv_mu declares dust_tau_v and dust_mu, and no birth-cloud/diffuse pair."""
    spec = _tauv_mu_spec()
    names = set(spec.all_params)
    assert {"dust_tau_v", "dust_mu"} <= names
    assert "dust_tau_bc" not in names
    assert "dust_tau_diff" not in names
    assert spec.get_distribution("dust_mu") is not None
    with pytest.raises(KeyError):
        spec.get_distribution("dust_tau_bc")


def test_native_parameterization_declares_tau_bc_and_diff_not_tau_v():
    """The default pair is unchanged: no tau_v/mu declared on the native choice."""
    spec = parse_groups(**_groups(tau_bc=Fixed(1.2), tau_diff=Fixed(0.4)))
    names = set(spec.all_params)
    assert {"dust_tau_bc", "dust_tau_diff"} <= names
    assert "dust_tau_v" not in names
    assert "dust_mu" not in names


# ── refusals (grammar surface) ─────────────────────────────────────────────


@pytest.mark.parametrize("native_key", ["tau_bc", "tau_diff", "dust_tau_bc", "dust_tau_diff"])
def test_tauv_mu_refuses_native_depth_keys(native_key):
    """Either spelling of a native depth key is refused under tauv_mu, with the identity."""
    with pytest.raises(ValueError) as exc:
        parse_groups(**_groups(parameterization="tauv_mu", **{native_key: Fixed(1.0)}))
    assert _SUBSTR_TOTAL in str(exc.value)
    assert _SUBSTR_MU in str(exc.value)


@pytest.mark.parametrize("key", ["tau_v", "mu", "dust_tau_v", "dust_mu"])
def test_native_refuses_tauv_mu_keys(key):
    """tau_v / mu inside a native two_component block name the parameterization."""
    with pytest.raises(ValueError, match=_SUBSTR_NATIVE.replace("(", r"\(")):
        parse_groups(**_groups(**{key: Fixed(1.0)}))


@pytest.mark.parametrize("dust_type", ["single_component", "age_binned", "wg00"])
def test_parameterization_refused_off_two_component(dust_type):
    """parameterization names two_component as the only type that reads it."""
    atten = {"type": dust_type, "parameterization": "tauv_mu", "all_params": Fixed(DEFAULT)}
    if dust_type == "single_component":
        atten["law"] = "power_law"
    if dust_type == "age_binned":
        atten["screens"] = [
            {"law": "power_law", "window_log_yr": (6.0, 8.0)},
            {"law": "calzetti", "window_log_yr": (8.0, 10.2)},
        ]
    with pytest.raises(ValueError, match="two_component"):
        parse_groups(
            redshift=Fixed(1.0),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=atten,
        )


def test_unknown_parameterization_value_refused():
    """A value outside DUST_PARAMETERIZATIONS is refused, listing the legal values."""
    with pytest.raises(ValueError, match="tau_bc_diff"):
        parse_groups(**_groups(parameterization="bogus"))


# ── short forms, wildcards, round trip ─────────────────────────────────────


def test_all_params_free_frees_tau_v_and_mu():
    """all_params: FREE frees both tauv_mu parameters and nothing native."""
    spec = parse_groups(
        **_groups(
            parameterization="tauv_mu",
            all_params=FREE,
        )
    )
    assert {"dust_tau_v", "dust_mu"} <= set(spec.free_params)


def test_round_trips_through_to_groups():
    """to_groups emits the parameterization and the tauv_mu keys, and re-parses."""
    spec = _tauv_mu_spec()
    groups = spec.to_groups()
    atten = groups["dust_attenuation"]
    assert atten["parameterization"] == "tauv_mu"
    assert "tau_bc" not in atten
    assert "tau_diff" not in atten
    again = parse_groups(**groups)
    assert again.dust_parameterization == "tauv_mu"
    assert set(again.all_params) == set(spec.all_params)


def test_native_round_trip_does_not_emit_parameterization():
    """The default choice is not emitted, so existing round trips are byte-stable."""
    spec = parse_groups(**_groups(tau_bc=Fixed(1.2), tau_diff=Fixed(0.4)))
    assert "parameterization" not in spec.to_groups()["dust_attenuation"]


# ── flat surface ───────────────────────────────────────────────────────────


def test_flat_surface_matches_grammar():
    """Parameters(dust_parameterization='tauv_mu', ...) declares the grammar's dust names."""
    flat = Parameters(
        dust_model="two_component",
        dust_parameterization="tauv_mu",
        dust_law_bc="power_law",
        dust_law_diff="power_law",
        dust_tau_v=Fixed(1.6),
        dust_mu=Fixed(0.25),
        redshift=Fixed(1.0),
    )
    assert flat.dust_parameterization == "tauv_mu"
    grammar = _tauv_mu_spec()
    flat_dust = {n for n in flat.all_params if n.startswith("dust_")}
    grammar_dust = {n for n in grammar.all_params if n.startswith("dust_")}
    assert flat_dust == grammar_dust


def test_flat_tauv_mu_refuses_native_depth_keys():
    with pytest.raises(ParameterError) as exc:
        Parameters(
            dust_model="two_component",
            dust_parameterization="tauv_mu",
            dust_tau_bc=Fixed(1.0),
            redshift=Fixed(1.0),
        )
    assert _SUBSTR_TOTAL in str(exc.value)
    assert _SUBSTR_MU in str(exc.value)


def test_flat_native_refuses_tauv_mu_keys():
    with pytest.raises(ParameterError, match=_SUBSTR_NATIVE.replace("(", r"\(")):
        Parameters(
            dust_model="two_component",
            dust_tau_v=Fixed(1.0),
            redshift=Fixed(1.0),
        )


def test_flat_unknown_parameterization_refused():
    with pytest.raises(ValueError, match="tauv_mu"):
        Parameters(
            dust_model="two_component",
            dust_parameterization="bogus",
            redshift=Fixed(1.0),
        )


# ── chevallard13 is diffuse-only (#3178) ───────────────────────────────────


@pytest.mark.parametrize(
    "slot",
    ["law", "law_bc", "law_neb"],
)
def test_chevallard13_refused_on_non_diffuse_slot(slot):
    """chevallard13 under law / law_bc / law_neb is refused, naming law_diff."""
    atten = _atten(**{slot: "chevallard13"})
    if slot == "law":
        atten.pop("law_bc")
        atten.pop("law_diff")
    if slot == "law_neb":
        atten["law_neb"] = "chevallard13"
    with pytest.raises(ValueError, match="law_diff"):
        parse_groups(
            redshift=Fixed(1.0),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=atten,
        )


def test_chevallard13_refused_on_single_component():
    with pytest.raises(ValueError, match="law_diff"):
        parse_groups(
            redshift=Fixed(1.0),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "chevallard13",
                "all_params": Fixed(DEFAULT),
            },
        )


def test_chevallard13_refused_on_age_binned():
    with pytest.raises(ValueError, match="law_diff"):
        parse_groups(
            redshift=Fixed(1.0),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "age_binned",
                "screens": [
                    {"law": "chevallard13", "window_log_yr": (6.0, 8.0)},
                    {"law": "calzetti", "window_log_yr": (8.0, 10.2)},
                ],
                "all_params": Fixed(DEFAULT),
            },
        )


def test_chevallard13_refused_on_flat_law_bc():
    with pytest.raises(ValueError, match="law_diff"):
        Parameters(
            dust_model="two_component",
            dust_law_bc="chevallard13",
            dust_law_diff="power_law",
            redshift=Fixed(1.0),
        )


def test_chevallard13_refused_on_flat_single_component():
    with pytest.raises(ValueError, match="law_diff"):
        Parameters(
            dust_model="single_component",
            dust_law_bc="chevallard13",
            dust_tau_v=Fixed(1.0),
            redshift=Fixed(1.0),
        )


def test_chevallard13_accepted_as_law_diff_builds():
    """law_diff='chevallard13' on a two-screen model is the legal spelling."""
    spec = parse_groups(
        **_groups(
            law_bc="power_law",
            law_diff="chevallard13",
            tau_bc=Fixed(1.0),
            tau_diff=Fixed(0.4),
        )
    )
    assert spec.dust_law_diff == "chevallard13"
    assert spec.dust_law_bc == "power_law"

# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp reads the composable AGN's band fluxes from a registry table.

The AGN component builds its whole SED through the block runner on every call and integrates it
through each filter; no WavePrecomp table existed for it. The registry adapter ``composable_agn``
tabulates the band fluxes over the free ``agn_*`` parameters at build, and the component reads
that table instead of integrating. Four things are pinned:

(a) the table is built through the registry and the predict does not integrate the filters;
(b) the table agrees with the exact photometry at three draws inside the priors, within the
    stated accuracy of its family, with finite gradients;
(c) ``agn['axis_grids']`` (``agn_axis_grids``) sets the table's axes (#2471);
(d) exact mode is unchanged: no table, no registry call, the per-filter integrals run.

A family whose table misses its stated accuracy is declined by the build gate or recorded in
``UNWIRED``; a tolerance is never loosened to fit.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

#: Registry key of the adapter that builds the table.
_KEY = "composable_agn"


def _ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 7.0, 1600)
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-2.5, -1.85, -1.2])
    flux = (
        ((5000.0 / wave) ** 2)[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


def _tophat(center_aa, frac_width=0.16, n=40):
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(center_aa * (1.0 - frac_width), center_aa * (1.0 + frac_width), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center_aa)}")


#: Optical to far-IR: the bands the disc and the torus dominate.
_BANDS_AA = (3000.0, 5500.0, 2.2e4, 1.2e5, 2.5e5, 8.0e5)


@dataclasses.dataclass(frozen=True)
class Family:
    """One AGN recipe, its free parameters and the accuracy its table is held to."""

    disc: str
    torus: str
    free: dict
    rtol: float
    atten: dict | None = None


def _free(*names):
    from tengri import Uniform

    # Units are the declared ones: agn_log_lbol is log10(L_bol / L_sun), declared Uniform(8, 14).
    priors = {
        "agn_log_lbol": Uniform(8.0, 14.0),
        "agn_cos_inc": Uniform(0.1, 0.9),
    }
    return {n: priors[n] for n in names}


def _atten_ebv():
    from tengri import Uniform

    return {"law": "prevot_smc", "ebv": Uniform(0.0, 0.5)}


# Accuracy: max relative error of the total photometry, table model against exact model, at
# z = 0.1 over 24 seeded draws (seeds 0-23) of every free parameter, six bands from 3000 A to
# 800 um, at the node counts of ``SEDModel._AGN_TABLE_NODES`` (33 nodes on one axis, 17 on two).
# agn_log_lbol spans its declared prior (8, 14). Measured (kubota_done + skirtor unless noted):
#   1 axis  (lbol)          5.37e-5 at 33, 66, 132 nodes
#   1 axis  (powerlaw+simple) 5.30e-5 at 33, 66, 132 nodes
#   2 axes  (lbol + ebv, prevot_smc) 5.37e-5 at 17, 34, 68 nodes
#   2 axes  (lbol + cos_inc) 5.37e-5 at 17, 34, 68 nodes
# The error is the same at every node count and for every AGN axis: it is the WavePrecomp
# floor of the stellar and dust photometry, which the exact model does not use. The AGN
# table alone, against the per-call WavePrecomp integral (no table), is 2e-16 (lbol), 6e-16
# (lbol + ebv) and 8e-11 (lbol + cos_inc) at 33 / 17 nodes. Each rtol is that measurement
# rounded up to the next power of ten.
FAMILIES: dict[str, Family] = {
    "kubota_done+skirtor/lbol": Family("kubota_done", "skirtor", _free("agn_log_lbol"), rtol=1e-4),
    "powerlaw+simple/lbol": Family("powerlaw", "simple", _free("agn_log_lbol"), rtol=1e-4),
    "kubota_done+skirtor+smc_prevot/lbol+ebv": Family(
        "kubota_done", "skirtor", _free("agn_log_lbol"), rtol=1e-4, atten=_atten_ebv()
    ),
    "kubota_done+skirtor/lbol+cos_inc": Family(
        "kubota_done", "skirtor", _free("agn_log_lbol", "agn_cos_inc"), rtol=1e-4
    ),
}

#: family -> why its table is not engaged (measured accuracy against the stated bound). Empty:
#: every family above meets its rtol at the current node counts.
UNWIRED: dict[str, str] = {}


def _wired():
    return [k for k in FAMILIES if k not in UNWIRED]


def _model(family: Family, approx, *, axis_grids=None, redshift=0.1, agn_extra=None):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel

    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": family.disc},
        "torus": {"type": family.torus},
        **family.free,
    }
    if family.atten is not None:
        agn["atten"] = family.atten
    if axis_grids is not None:
        agn["axis_grids"] = axis_grids
    if agn_extra:
        agn.update(agn_extra)
    return SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(
            photometry=Photometry(filters=tuple(_tophat(c) for c in _BANDS_AA))
        ),
        redshift=Fixed(redshift) if not isinstance(redshift, dict) else redshift["prior"],
        approx=approx,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn=agn,
    )


def _wavepre():
    from tengri import WavePrecomp

    return WavePrecomp()


def _band_table(model) -> dict | None:
    data = model._template_data_for_jit() or {}
    return data.get("agn", {}).get("band_table")


@pytest.fixture
def resolve_spy(monkeypatch):
    from tengri.forward.precompute import registry

    keys: list[str] = []
    real = registry.resolve

    def spy(name):
        keys.append(name)
        return real(name)

    monkeypatch.setattr(registry, "resolve", spy)
    return keys


@pytest.fixture
def integral_spy(monkeypatch):
    from tengri.observation import photometry

    calls: list[int] = []
    real = photometry.lnu_filter_integral

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(photometry, "lnu_filter_integral", spy)
    return calls


@pytest.mark.parametrize("key", _wired())
def test_table_is_built_through_the_registry_and_predict_does_not_integrate(
    key, resolve_spy, integral_spy
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(FAMILIES[key], _wavepre())
    assert _KEY in resolve_spy, "WavePrecomp never asked the registry for the composable AGN"
    assert _band_table(model) is not None, model._agn_band_table_decline
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    before = len(integral_spy)
    jax.block_until_ready(model.predict_photometry(params))
    assert len(integral_spy) == before, "predict integrated the AGN SED through the filters"


@pytest.mark.parametrize("key", _wired())
def test_photometry_graph_drops_the_sed_build(key):
    """Nothing in a photometry predict consumes the SED, so the compiled graph no longer has it."""
    family = FAMILIES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_model = _model(family, _wavepre())
        per_call = _model(family, _wavepre())
    per_call._agn_band_table_cache = None
    params = dict(table_model.spec.sample(jax.random.PRNGKey(0)))

    def flops(model):
        cost = jax.jit(model.predict_photometry).lower(params).compile().cost_analysis()
        return float((cost[0] if isinstance(cost, list) else cost)["flops"])

    assert flops(table_model) < 0.1 * flops(per_call)


@pytest.mark.parametrize("key", _wired())
def test_table_matches_exact_and_has_finite_gradients(key):
    family = FAMILIES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_model = _model(family, _wavepre())
        exact_model = _model(family, None)
    assert _band_table(table_model) is not None, table_model._agn_band_table_decline
    for seed in (11, 12, 13):
        params = dict(exact_model.spec.sample(jax.random.PRNGKey(seed)))
        got = np.asarray(table_model.predict_photometry(params))
        want = np.asarray(exact_model.predict_photometry(params))
        np.testing.assert_allclose(got, want, rtol=family.rtol, atol=0.0)

    free = {k: v for k, v in params.items() if k in table_model.spec.free_params}
    fixed = {k: v for k, v in params.items() if k not in free}

    def loss(theta):
        return jnp.sum(jnp.log(table_model.predict_photometry({**fixed, **theta})))

    grads = jax.grad(loss)(free)
    assert all(bool(jnp.isfinite(g)) for g in grads.values())
    assert any(float(g) != 0.0 for g in grads.values())


def test_agn_axis_grids_sets_the_table_axes():
    """The grids a user gives reach the table (#2471); a grid short of the prior is refused."""
    family = FAMILIES[_wired()[0]]
    nodes = np.linspace(8.0, 14.0, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = _model(family, _wavepre())
        gridded = _model(family, _wavepre(), axis_grids={"agn_log_lbol": nodes})
    default_axis = np.asarray(_band_table(default)["axes"]["agn_log_lbol"])
    gridded_axis = np.asarray(_band_table(gridded)["axes"]["agn_log_lbol"])
    np.testing.assert_array_equal(gridded_axis, nodes)
    assert default_axis.size != nodes.size

    with pytest.warns(UserWarning, match="nodes span"):
        short = _model(family, _wavepre(), axis_grids={"agn_log_lbol": np.linspace(8.5, 13.0, 9)})
    assert _band_table(short) is None
    assert "nodes span" in short._agn_band_table_decline


def test_agn_axis_grids_round_trips_through_the_groups():
    """The grammar carries the grids onto the spec and ``to_groups`` gives them back (#2471)."""
    family = FAMILIES[_wired()[0]]
    nodes = np.linspace(8.0, 14.0, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(family, None, axis_grids={"agn_log_lbol": nodes})
    np.testing.assert_array_equal(model.spec.agn_axis_grids["agn_log_lbol"], nodes)
    np.testing.assert_array_equal(
        model.spec.to_groups()["agn"]["axis_grids"]["agn_log_lbol"], nodes
    )


@pytest.mark.parametrize("key", _wired())
def test_table_matches_the_per_call_integral_path(key):
    """The table reproduces the per-call WavePrecomp integral it replaces, at 1e-10.

    Both models read WavePrecomp, so the stellar and dust photometry are identical in each and
    cancel. Exact mode is not compared here: WavePrecomp's own stellar and dust photometry
    differ from exact by ~5e-5 relative (measured with the AGN switched off), so that comparison
    is not a 1e-10 statement about the AGN table.
    """
    family = FAMILIES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = _model(family, _wavepre())
        per_call = _model(family, _wavepre())
    per_call._agn_band_table_cache = None
    assert _band_table(table) is not None, table._agn_band_table_decline
    assert _band_table(per_call) is None
    params = dict(table.spec.sample(jax.random.PRNGKey(0)))
    np.testing.assert_allclose(
        np.asarray(table.predict_photometry(params)),
        np.asarray(per_call.predict_photometry(params)),
        rtol=1e-10,
    )


@pytest.mark.parametrize("key", _wired())
def test_exact_mode_does_not_touch_the_registry(key, resolve_spy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(FAMILIES[key], None)
    assert _KEY not in resolve_spy
    assert _band_table(model) is None


@pytest.mark.parametrize("key", _wired())
def test_a_declined_gate_leaves_the_exact_path_bit_for_bit(key):
    """A model the table cannot represent (redshift free) is the exact model, with the reason."""
    from tengri import Uniform

    family = FAMILIES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declined = _model(family, _wavepre(), redshift={"prior": Uniform(0.05, 0.2)})
        per_call = _model(family, _wavepre(), redshift={"prior": Uniform(0.05, 0.2)})
    per_call._agn_band_table_cache = None
    assert _band_table(declined) is None
    assert "not Fixed" in declined._agn_band_table_decline
    params = dict(declined.spec.sample(jax.random.PRNGKey(5)))
    # The reference is the same WavePrecomp model with no table: a declined gate must leave that
    # path unchanged. (Against approx=None the WavePrecomp stellar floor, ~5e-5, is not 1e-6.)
    np.testing.assert_allclose(
        np.asarray(declined.predict_photometry(params)),
        np.asarray(per_call.predict_photometry(params)),
        rtol=1e-6,
    )


def test_unwired_keys_name_known_families():
    assert set(UNWIRED) <= set(FAMILIES)


@pytest.mark.parametrize("key", sorted(UNWIRED))
def test_unwired_family_is_not_engaged(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(FAMILIES[key], _wavepre())
    assert _band_table(model) is None, f"{key} is recorded as unwired: {UNWIRED[key]}"


def _fixed_model(approx, *, agn_extra=None):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel

    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "kubota_done"},
        "torus": {"type": "skirtor"},
    }
    if agn_extra:
        agn.update(agn_extra)
    return SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(
            photometry=Photometry(filters=tuple(_tophat(c) for c in _BANDS_AA))
        ),
        redshift=Fixed(0.1),
        approx=approx,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn=agn,
    )


@pytest.mark.parametrize("lum_ratio", [1.0, 0.5])
def test_fully_fixed_agn_is_a_constant_band_row(lum_ratio, integral_spy):
    """A fully Fixed AGN holds one exact row of band fluxes, scaled by agn_lum_ratio.

    The row is one evaluation of the recipe at the fixed agn_log_lbol, through the same
    registry adapter and filter integrals as the table. The whole-model photometry is not
    compared with exact (WavePrecomp's stellar floor); the table is compared with the per-call
    WavePrecomp path, which it reproduces at 1e-10.
    """
    from tengri import Fixed

    extra = {"agn_lum_ratio": Fixed(lum_ratio)} if lum_ratio != 1.0 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = _fixed_model(_wavepre(), agn_extra=extra)
        per_call = _fixed_model(_wavepre(), agn_extra=extra)
    per_call._agn_band_table_cache = None
    assert _band_table(table) is not None, table._agn_band_table_decline
    assert _band_table(per_call) is None
    params = dict(table.spec.sample(jax.random.PRNGKey(0)))
    before = len(integral_spy)
    got = np.asarray(table.predict_photometry(params))
    assert len(integral_spy) == before, "predict integrated the AGN SED through the filters"
    np.testing.assert_allclose(got, np.asarray(per_call.predict_photometry(params)), rtol=1e-10)


def test_fully_fixed_agn_photometry_graph_drops_the_sed_build():
    """The compiled photometry of a fully Fixed AGN has no AGN SED build (flops, not a spy)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = _fixed_model(_wavepre())
        per_call = _fixed_model(_wavepre())
    per_call._agn_band_table_cache = None
    params = dict(table.spec.sample(jax.random.PRNGKey(0)))

    def flops(model):
        cost = jax.jit(model.predict_photometry).lower(params).compile().cost_analysis()
        return float((cost[0] if isinstance(cost, list) else cost)["flops"])

    assert flops(table) < 0.1 * flops(per_call)


#: Torus families whose free torus parameters the table is admitted over (measured table vs
#: per-call WavePrecomp, 24 draws, 33 nodes (1 axis) / 17 nodes per axis (2 axes): silva04
#: 1.0e-15 (T_torus), 2.6e-16 (cos_inc); nenkova 8.1e-16, 2.6e-16; cat3d_wind 8.1e-16, 1.0e-9).
#: fritz is not measured: its grid file in this checkout lacks the norm dataset.
TORUS_AXES = {
    "silva04": ("agn_T_torus", "agn_cos_inc"),
    "nenkova": ("agn_T_torus", "agn_cos_inc"),
    "cat3d_wind": ("agn_T_torus", "agn_cos_inc"),
}


def _torus_model(torus: str, param: str, approx):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    tor = {"type": torus}
    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "kubota_done"},
        "torus": tor,
        "agn_log_lbol": Uniform(8.0, 14.0),
    }
    if param == "agn_T_torus":
        tor["agn_T_torus"] = Uniform(100.0, 1500.0)
    else:
        agn[param] = Uniform(0.1, 0.9)
    return SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(
            photometry=Photometry(filters=tuple(_tophat(c) for c in _BANDS_AA))
        ),
        redshift=Fixed(0.1),
        approx=approx,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn=agn,
    )


@pytest.mark.parametrize("torus", sorted(TORUS_AXES))
@pytest.mark.parametrize("param", ["agn_T_torus", "agn_cos_inc"])
def test_torus_axis_table_matches_per_call_integral_path(torus, param):
    """The torus's free T_torus and cos_inc are tabulated and reproduce the per-call path at 1e-8."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = _torus_model(torus, param, _wavepre())
        per_call = _torus_model(torus, param, _wavepre())
    per_call._agn_band_table_cache = None
    assert _band_table(table) is not None, table._agn_band_table_decline
    params = dict(table.spec.sample(jax.random.PRNGKey(0)))
    np.testing.assert_allclose(
        np.asarray(table.predict_photometry(params)),
        np.asarray(per_call.predict_photometry(params)),
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    "block, reason_key",
    [("fritz", "torus/fritz"), ("cue", "nlr/cue")],
)
def test_unwired_block_is_declined_with_its_reason(block, reason_key):
    """A block the table cannot be built for is declined by the gate, with the recorded reason."""
    from tengri.components.agn.blocks.composable_precompute import UNWIRED_BLOCKS

    assert reason_key in UNWIRED_BLOCKS
    category = reason_key.split("/")[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if category == "torus":
            model = _torus_model(block, "agn_cos_inc", _wavepre())
        else:
            model = _fixed_model(
                _wavepre(), agn_extra={"nlr": {"type": block}, "agn_log_lbol": _free_lbol()}
            )
    assert _band_table(model) is None
    assert f"{reason_key} is not tabulated" in model._agn_band_table_decline


def _free_lbol():
    from tengri import Uniform

    return Uniform(8.0, 14.0)

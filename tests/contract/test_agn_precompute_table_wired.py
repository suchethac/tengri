# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp reads the composable AGN's band fluxes from a registry table.

The AGN component builds its whole SED through the block runner on every call and integrates it
through each filter; no WavePrecomp table existed for it. The registry adapter ``composable_agn``
tabulates the band fluxes over the free ``agn_*`` parameters at build, and the component reads
that table instead of integrating. Five things are pinned:

(a) the table is built through the registry and the predict does not integrate the filters;
(b) the table's AGN band fluxes agree with the exact model's AGN SED, integrated through the
    same filters, at three draws inside the priors, with finite gradients;
(c) ``agn['axis_grids']`` (``agn_axis_grids``) sets the table's axes (#2471);
(d) exact mode is unchanged: no table, no registry call, the per-filter integrals run;
(e) the build measures its own table at the cell centers and declines one that misses
    ``SEDModel._AGN_TABLE_RTOL``: the kubota_done disc over its whole luminosity prior (a jump
    of the disc flux near agn_log_lbol 9.8) and SKIRTOR's inclination (the disc passing behind
    the torus) are declined, with the measured error as the reason.

Every comparison reads the AGN term alone. In this fixture the synthetic stellar population
outshines the AGN by ~1e11 (doubling the table read moves the total photometry by <= 2e-11), so a
total-photometry comparison passes whatever the table returns. A tolerance is never loosened to
fit.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.composable_precompute import UNWIRED_BLOCKS
from tengri.config.exceptions import TengriIOError
from tengri.forward.sed_model import SEDModel

pytestmark = pytest.mark.contract

#: Registry key of the adapter that builds the table.
_KEY = "composable_agn"

#: The bound the build holds every kept table to, at its cell centers.
_AGN_RTOL = SEDModel._AGN_TABLE_RTOL


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


def _free(*names, lbol=(8.0, 14.0)):
    from tengri import Uniform

    # Units are the declared ones: agn_log_lbol is log10(L_bol / L_sun), declared Uniform(8, 14).
    priors = {
        "agn_log_lbol": Uniform(*lbol),
        "agn_cos_inc": Uniform(0.1, 0.9),
    }
    return {n: priors[n] for n in names}


def _atten_ebv():
    from tengri import Uniform

    return {"law": "prevot_smc", "ebv": Uniform(0.0, 0.5)}


# Accuracy: the AGN band fluxes of the table against the exact model's AGN SED integrated
# through the same filters, z = 0.1, six bands from 3000 A to 800 um. The build's cell-center
# gate holds every table it keeps to ``SEDModel._AGN_TABLE_RTOL`` (1e-3), so that is each rtol.
# Measured at the cell centers (the build's own probe): kubota_done over lbol (10.5, 14) passes
# at the 129 one-axis nodes (1.06e-3 at 65, 1.13e-3 at 33); powerlaw+simple with the SMC E(B-V) axis
# passes. powerlaw+simple over lbol alone is rank-1, so the exact term response serves it.
FAMILIES: dict[str, Family] = {
    "kubota_done+skirtor/lbol>10.5": Family(
        "kubota_done", "skirtor", _free("agn_log_lbol", lbol=(10.5, 14.0)), rtol=1e-3
    ),
    "powerlaw+simple/lbol": Family("powerlaw", "simple", _free("agn_log_lbol"), rtol=1e-3),
    "powerlaw+simple+smc_prevot/lbol+ebv": Family(
        "powerlaw", "simple", _free("agn_log_lbol"), rtol=1e-3, atten=_atten_ebv()
    ),
    "kubota_done+skirtor/lbol": Family("kubota_done", "skirtor", _free("agn_log_lbol"), rtol=1e-3),
    "kubota_done+skirtor/lbol+cos_inc": Family(
        "kubota_done", "skirtor", _free("agn_log_lbol", "agn_cos_inc"), rtol=1e-3
    ),
}

_MISSES = "the table misses its 0.001 accuracy"
_RANK1 = "the rank-1 term band response serves this AGN exactly"

#: family -> (the start of the build's decline reason, why).
UNWIRED: dict[str, tuple[str, str]] = {
    "kubota_done+skirtor/lbol": (
        _MISSES,
        "3.9e-1 at the cell centers of 33 nodes over lbol (8, 14): the kubota_done disc flux per "
        "unit luminosity at 3000 A jumps ~4x between lbol 9.80 and 9.95 (mbh 1e8, lambda_Edd "
        "~2e-3), where the corona and warm-zone radii reach their clips at 0.5 and 0.9 R_out",
    ),
    "kubota_done+skirtor/lbol+cos_inc": (
        _MISSES,
        "4.6e-2 at 17 x 17 nodes with lbol (10.5, 14), and 4.2e-1 with a powerlaw disc: SKIRTOR "
        "hides the disc behind the torus at its opening angle, a step in agn_cos_inc (the error "
        "halves per doubling of the cos_inc nodes, 5.1e-2 at 17 and 2.7e-2 at 33)",
    ),
    "powerlaw+simple/lbol": (
        _RANK1,
        "every sub-block is a fixed shape times its amplitude: the term response is exact",
    ),
}


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


# The comparisons below read the AGN band fluxes, never the total photometry. In this fixture
# the synthetic stellar population outshines the AGN by ~1e11 (doubling the table read moves
# the total by <= 2e-11), so a total-photometry comparison passes whatever the table returns.


def _table_row(model, params) -> np.ndarray:
    """Observed-band AGN L_nu [erg/s/Hz] read from the model's table."""
    from tengri.components.agn.component import AGNSEDComponent

    full = {**model.spec.get_fixed_values(), **params}
    return np.asarray(AGNSEDComponent._band_table_phot(full, _band_table(model))[0])


def _per_call_row(model, params) -> np.ndarray:
    """Observed-band AGN L_nu [erg/s/Hz] the component integrates per call.

    ``model`` must have no table (``_agn_band_table_cache = None``), so this is the per-call
    WavePrecomp integral the table replaces, whichever chain the state runs.
    """
    assert _band_table(model) is None, "the per-call reference model still holds a table"
    return np.asarray(model.predict_state(params).derived["agn_phot_lnu_precomp"])


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
        np.testing.assert_allclose(
            _table_row(table_model, params),
            _exact_agn_band(exact_model, params),
            rtol=family.rtol,
            atol=0.0,
        )

    from tengri.components.agn.component import AGNSEDComponent

    free = {k: v for k, v in params.items() if k in table_model.spec.free_params}
    fixed = {
        **table_model.spec.get_fixed_values(),
        **{k: v for k, v in params.items() if k not in free},
    }
    table = _band_table(table_model)

    def loss(theta):
        # The AGN term alone: through the total photometry the stellar light swamps it.
        return jnp.sum(jnp.log(AGNSEDComponent._band_table_phot({**fixed, **theta}, table)[0]))

    grads = jax.grad(loss)(free)
    assert all(bool(jnp.isfinite(g)) for g in grads.values())
    assert any(float(g) != 0.0 for g in grads.values())


def test_agn_axis_grids_sets_the_table_axes():
    """The grids a user gives reach the table (#2471); a grid short of the prior is refused."""
    family = FAMILIES["kubota_done+skirtor/lbol>10.5"]
    nodes = np.linspace(10.5, 14.0, 193)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = _model(family, _wavepre())
        gridded = _model(family, _wavepre(), axis_grids={"agn_log_lbol": nodes})
    default_axis = np.asarray(_band_table(default)["axes"]["agn_log_lbol"])
    gridded_axis = np.asarray(_band_table(gridded)["axes"]["agn_log_lbol"])
    np.testing.assert_array_equal(gridded_axis, nodes)
    assert default_axis.size != nodes.size

    # A user grid short of the prior is a configuration error, refused at build (#2973).
    with pytest.raises(ValueError, match="nodes span"):
        _model(family, _wavepre(), axis_grids={"agn_log_lbol": np.linspace(11.0, 13.0, 9)})


def test_agn_axis_grids_round_trips_through_the_groups():
    """The grammar carries the grids onto the spec and ``to_groups`` gives them back (#2471)."""
    family = FAMILIES["kubota_done+skirtor/lbol>10.5"]
    nodes = np.linspace(10.5, 14.0, 193)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(family, None, axis_grids={"agn_log_lbol": nodes})
    np.testing.assert_array_equal(model.spec.agn_axis_grids["agn_log_lbol"], nodes)
    np.testing.assert_array_equal(
        model.spec.to_groups()["agn"]["axis_grids"]["agn_log_lbol"], nodes
    )


@pytest.mark.parametrize("key", _wired())
def test_table_matches_the_per_call_integral_path(key):
    """The table's AGN row reproduces the per-call WavePrecomp integral it replaces.

    Away from the nodes the table carries its interpolation error, which the build holds to
    ``_AGN_RTOL`` at the cell centers (measured at a random draw: 2.5e-7 for kubota_done over
    lbol (10.5, 14), 1.2e-8 for powerlaw+simple with E(B-V)).
    """
    family = FAMILIES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = _model(family, _wavepre())
        per_call = _model(family, _wavepre())
    per_call._agn_band_table_cache = None
    assert _band_table(table) is not None, table._agn_band_table_decline
    params = dict(table.spec.sample(jax.random.PRNGKey(0)))
    np.testing.assert_allclose(
        _table_row(table, params), _per_call_row(per_call, params), rtol=_AGN_RTOL
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


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_family_is_engaged_unless_recorded_unwired(key):
    """Every family is engaged, or it is recorded in UNWIRED and the build's own probe declines it.

    The UNWIRED families are the anti-vacuity arm of the cell-center gate: a kink (the kubota_done
    disc over its whole luminosity prior) and a step (SKIRTOR's inclination) must be refused with
    the measured error, and the smooth families beside them must be kept.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model(FAMILIES[key], _wavepre())
    if key in UNWIRED:
        prefix, why = UNWIRED[key]
        assert _band_table(model) is None, f"{key} is recorded as unwired: {why}"
        assert model._agn_band_table_decline.startswith(prefix), model._agn_band_table_decline
    else:
        assert _band_table(model) is not None, model._agn_band_table_decline


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
    compared with exact (WavePrecomp's stellar floor); the AGN row is compared with the per-call
    WavePrecomp row, which it reproduces at 1e-10 because the fixed point is the table's node.
    """
    from tengri import Fixed

    extra = {"agn_lum_ratio": Fixed(lum_ratio)} if lum_ratio != 1.0 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = _fixed_model(_wavepre(), agn_extra=extra)
        per_call = _fixed_model(_wavepre(), agn_extra=extra)
    per_call._agn_band_table_cache = None
    assert _band_table(table) is not None, table._agn_band_table_decline
    params = dict(table.spec.sample(jax.random.PRNGKey(0)))
    before = len(integral_spy)
    jax.block_until_ready(table.predict_photometry(params))
    assert len(integral_spy) == before, "predict integrated the AGN SED through the filters"
    np.testing.assert_allclose(
        _table_row(table, params), _per_call_row(per_call, params), rtol=1e-10
    )


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


def _exact_agn_band(model, params) -> np.ndarray:
    """Filter integral of the exact model's own AGN SED, so no stellar term enters."""
    from tengri.observation.photometry import lnu_filter_integral

    state = model.predict_state(params)
    sed = state.derived["sed_agn"]
    return np.asarray(
        [
            lnu_filter_integral(sed, state.wave, f.wave, f.trans, redshift=jnp.asarray(0.1))
            for f in (_tophat(c) for c in _BANDS_AA)
        ]
    )


def _assert_kept_tables_match_exact(table, exact):
    """A kept table matches the exact AGN SED integral within the gate's bound; a declined one
    carries the gate's measured reason (the build never keeps a table it has not measured)."""

    if _band_table(table) is None:
        assert table._agn_band_table_decline.startswith((_MISSES, _RANK1)), (
            table._agn_band_table_decline
        )
        return False
    for seed in (0, 1, 2):
        params = dict(table.spec.sample(jax.random.PRNGKey(seed)))
        np.testing.assert_allclose(
            _table_row(table, params),
            _exact_agn_band(exact, params),
            rtol=_AGN_RTOL,
        )
    return True


#: Torus types the table is built over. Each is tested on its own declared shape axes, beside a
#: powerlaw disc at a Fixed luminosity, so the only free axis is the torus's.
TORUS_TYPES = ("silva04", "nenkova", "cat3d_wind", "simple", "fritz")

#: Shape axes the torus table may be admitted over. Which of these a torus type reads is
#: derived below, never listed by hand: a parameter a type does not declare is refused by
#: ``_reject_inert_agn_subblock_keys`` (main, #985), and the earlier hand list passed only
#: because such keys were silently dropped.
_TORUS_SHAPE_CANDIDATES = ("agn_T_torus", "agn_cos_inc")


def _declared_torus_axes(torus: str) -> tuple[str, ...]:
    """Shape axes ``torus`` declares, in candidate order.

    ``agn_T_torus`` must be in the torus sub-block's own declared set (the one
    ``_reject_inert_agn_subblock_keys`` checks an explicit key against). ``agn_cos_inc``
    is a shared masking knob owned by the top-level ``agn`` group, so it is checked
    against the block's declared reads (``AGN_BLOCK_CONSUMES``) instead, which is what the
    top-level wildcard sources.
    """
    from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES
    from tengri.parameters.agn_ownership import _agn_subblock_declared_params

    reads = AGN_BLOCK_CONSUMES.get(("torus", torus), frozenset())
    sub_owned = _agn_subblock_declared_params("torus", torus) or frozenset()
    axes = []
    for name in _TORUS_SHAPE_CANDIDATES:
        if name not in reads:
            continue
        if name == "agn_T_torus" and name not in sub_owned:
            continue
        axes.append(name)
    return tuple(axes)


TORUS_AXES: dict[str, tuple[str, ...]] = {t: _declared_torus_axes(t) for t in TORUS_TYPES}
TORUS_AXIS_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (torus, name) for torus, axes in TORUS_AXES.items() for name in axes
)


def _torus_model(torus: str, param: str, approx):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    tor = {"type": torus}
    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "powerlaw"},
        "torus": tor,
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


def test_every_torus_type_declares_a_shape_axis():
    """Each torus type in the table has at least one declared shape axis to test."""
    empty = [torus for torus, axes in TORUS_AXES.items() if not axes]
    assert not empty, f"torus types with no declared shape axis: {empty}"


#: Torus axes whose table the build keeps (measured verdicts; the rest are declined at the cell
#: centers: cat3d_wind cos_inc 1.24e-3, fritz cos_inc 4.1e-3, simple T_torus 1.5e-2 at 65 nodes).
#: Pinned so a regression that declines every table fails here rather than passing quietly.
_KEPT_TORUS_AXES = {("silva04", "agn_cos_inc"), ("nenkova", "agn_cos_inc")}


@pytest.mark.parametrize("torus,param", TORUS_AXIS_PAIRS)
def test_torus_axis_table_matches_exact_or_is_declined(torus, param):
    """A torus shape axis is tabulated to the gate's bound against exact, or declined by it."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = _torus_model(torus, param, _wavepre())
            exact = _torus_model(torus, param, None)
    except (FileNotFoundError, TengriIOError) as exc:
        pytest.skip(f"{torus}: model needs data absent from this checkout ({exc})")
    kept = _assert_kept_tables_match_exact(table, exact)
    if (torus, param) in _KEPT_TORUS_AXES:
        assert kept, table._agn_band_table_decline


#: Every disc, nlr, blr, feii, torus and attenuation block that loads a library or a grid,
#: built beside the default disc with agn_log_lbol over (10.5, 14), where that disc is smooth.
#: Each table is either kept and matched to exact, or declined with the gate's measurement.
TEMPLATE_BLOCKS = (
    "disc/grahsp_sbpl",
    "nlr/grahsp",
    "blr/grahsp",
    "feii/grahsp",
    "torus/grahsp",
    "attenuation/grahsp_biatten",
    "torus/fritz",
    "nlr/cue",
    "nlr/synthesizer",
    "nlr/synthesizer_spectra",
    "blr/synthesizer",
    "blr/synthesizer_spectra",
)


#: Blocks whose table the build keeps (measured verdicts). Of the others, disc/grahsp_sbpl is
#: rank-1 (served exactly by the term response) and torus/grahsp is declined (2.1e-3 at 65 nodes).
_KEPT_BLOCKS = {
    "nlr/grahsp",
    "blr/grahsp",
    "feii/grahsp",
    "attenuation/grahsp_biatten",
    "torus/fritz",
    "nlr/cue",
    "nlr/synthesizer",
    "nlr/synthesizer_spectra",
    "blr/synthesizer",
    "blr/synthesizer_spectra",
}


def _block_model(key: str, approx):
    """A composable AGN with the block named by ``key`` (``"<category>/<name>"``), free lum."""
    from tengri import Uniform

    category, name = key.split("/")
    slot = "atten" if category == "attenuation" else category
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fixed_model(
            approx, agn_extra={slot: {"type": name}, "agn_log_lbol": Uniform(10.5, 14.0)}
        )


@pytest.mark.parametrize("key", TEMPLATE_BLOCKS)
def test_template_block_table_matches_exact_or_is_declined(key):
    """Each template-backed block builds its table (no hand-kept exclusion list), and the table
    the build keeps matches the exact AGN SED. Data absent from this checkout skips by name."""
    try:
        table = _block_model(key, _wavepre())
        exact = _block_model(key, None)
    except (FileNotFoundError, TengriIOError) as exc:
        pytest.skip(f"{key}: model needs data absent from this checkout ({exc})")
    kept = _assert_kept_tables_match_exact(table, exact)
    if key in _KEPT_BLOCKS:
        assert kept, table._agn_band_table_decline


@pytest.mark.parametrize("key", sorted(UNWIRED_BLOCKS))
def test_unwired_block_is_declined_with_its_reason(key):
    """Every UNWIRED_BLOCKS entry is declined by the build gate, which records the stated reason.

    Data-driven over ``UNWIRED_BLOCKS``: a new entry is covered without editing this test. A
    model whose block needs data that is absent from this checkout cannot be built, and that
    entry is skipped by name; every other entry must build and be declined.
    """
    try:
        model = _block_model(key, _wavepre())
    except (FileNotFoundError, TengriIOError) as exc:
        pytest.skip(f"{key}: model needs data absent from this checkout ({exc})")
    assert _band_table(model) is None
    assert model._agn_band_table_decline == f"{key} is not tabulated: {UNWIRED_BLOCKS[key]}"

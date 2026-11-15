# SPDX-License-Identifier: BSD-3-Clause
"""Runtime redshift class sweep (#3505).

Redshift is a runtime input of the forward model for every representation (exact
and tabulated) and every way of supplying it: a promoted fixed-z model, a free
redshift, a catalog range, a call-time override on a fixed-z model, and a vmapped
per-galaxy traced z. One compiled program must serve several redshifts, and its
output at each z must equal a model built fixed at that z.
"""

from __future__ import annotations

import warnings
from functools import cache, lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.inference.fitter import _model_at_override_redshift

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ZS = (0.1, 0.3, 0.8)
Z_PLACEHOLDER = 0.1
Z_RANGE = (0.05, 1.0)
RTOL = {"exact": 1e-10, "wp": 3e-3}
REPRESENTATIONS = ("exact", "wp")
STRUCTURES = ("plain", "rich")
MODES = ("promoted", "free", "catalog", "override", "promoted_vmap")


@lru_cache(maxsize=1)
def _ssp():
    return load_ssp_data(SSP_PATH)


@lru_cache(maxsize=1)
def _obs():
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


def _approx(representation, catalog=False):
    if representation == "exact":
        return None
    if catalog:
        return WavePrecomp(catalog_z_range=Z_RANGE)
    return WavePrecomp()


def _build(representation, structure, redshift, *, catalog=False):
    kw = {}
    if structure == "rich":
        kw["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=_obs(),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=redshift,
            approx=_approx(representation, catalog=catalog),
            **kw,
        )


@cache
def _reference(representation, structure, z):
    """Photometry of a model built fixed at ``z`` (the ground truth for the sweep)."""
    model = _build(representation, structure, Fixed(z))
    return np.asarray(model.predict_photometry({}))


def _runtime_model(representation, structure, mode):
    if mode == "promoted":
        base = _build(representation, structure, Fixed(Z_PLACEHOLDER))
        return base.with_runtime_redshift(Z_RANGE if representation == "wp" else None)
    if mode == "promoted_vmap":
        return _runtime_model(representation, structure, "promoted")
    if mode == "free":
        return _build(representation, structure, Uniform(*Z_RANGE))
    if mode == "catalog":
        return _build(representation, structure, Fixed(Z_PLACEHOLDER), catalog=True)
    if mode == "override":
        # Exact only: a tabulated override needs a table spanning its own z, so it
        # gets one table per override value (see test_tabulated_override_spans_its_z).
        base = _build(representation, structure, Fixed(Z_PLACEHOLDER))
        models = [_model_at_override_redshift(base, {"redshift": z}) for z in ZS]
        assert all(m is models[0] for m in models), (
            "one promotion must serve every override redshift; got a distinct model per z"
        )
        return models[0]
    raise ValueError(mode)


def _cases():
    for representation in REPRESENTATIONS:
        for structure in STRUCTURES:
            for mode in MODES:
                if mode == "catalog" and representation == "exact":
                    continue
                if mode == "override" and representation == "wp":
                    continue
                if mode == "promoted_vmap":
                    continue
                yield representation, structure, mode
    yield "wp", "plain", "promoted_vmap"
    yield "exact", "plain", "promoted_vmap"


@pytest.mark.parametrize(("representation", "structure", "mode"), list(_cases()))
def test_one_trace_serves_every_redshift(representation, structure, mode):
    """One jit trace serves three redshifts; each output equals a fixed-z build."""
    model = _runtime_model(representation, structure, mode)
    traces = []

    if mode == "promoted_vmap":

        def program(zs):
            traces.append(1)
            return jax.vmap(lambda z: model.predict_photometry({"redshift": z}))(zs)

        got = np.asarray(jax.jit(program)(jnp.asarray(ZS)))
        outs = list(got)
    else:

        def program(z):
            traces.append(1)
            return model.predict_photometry({"redshift": z})

        jitted = jax.jit(program)
        outs = [np.asarray(jitted(jnp.asarray(z))) for z in ZS]

    assert len(traces) == 1, f"{mode}: {len(traces)} traces for {len(ZS)} redshifts"
    for z, got in zip(ZS, outs, strict=True):
        np.testing.assert_allclose(
            got,
            _reference(representation, structure, z),
            rtol=RTOL[representation],
            atol=0.0,
            err_msg=f"{representation}/{structure}/{mode} at z={z}",
        )


def test_override_promotes_once_and_serves_every_redshift():
    """A call-time override on a fixed-z model no longer raises and reuses one model."""
    base = _build("exact", "plain", Fixed(Z_PLACEHOLDER))
    promoted = _model_at_override_redshift(base, {"redshift": ZS[0]})
    assert promoted is not base
    assert _model_at_override_redshift(base, {"redshift": ZS[2]}) is promoted
    for z in ZS:
        got = np.asarray(promoted.predict_photometry({"redshift": z}))
        np.testing.assert_allclose(got, _reference("exact", "plain", z), rtol=RTOL["exact"])


def test_tabulated_override_spans_its_z():
    """A tabulated override promotion's table brackets the override redshift."""
    base = _build("wp", "plain", Fixed(Z_PLACEHOLDER))
    promoted = _model_at_override_redshift(base, {"redshift": 0.3})
    got = np.asarray(promoted.predict_photometry({"redshift": 0.3}))
    np.testing.assert_allclose(got, _reference("wp", "plain", 0.3), rtol=RTOL["wp"])


def test_runtime_z_defaults_to_declared_redshift():
    """A runtime model with no z in params uses its declared redshift (#3505 default)."""
    promoted = _build("exact", "plain", Fixed(Z_PLACEHOLDER)).with_runtime_redshift()
    got = np.asarray(promoted.predict_photometry({}))
    np.testing.assert_allclose(
        got, _reference("exact", "plain", Z_PLACEHOLDER), rtol=RTOL["exact"]
    )


def test_non_runtime_fixed_model_still_refuses_call_time_override():
    """The refusal stays for a Fixed redshift that is not promoted (#2296)."""
    from tengri.config.exceptions import ParameterError

    base = _build("exact", "plain", Fixed(Z_PLACEHOLDER))
    with pytest.raises(ParameterError):
        base.predict_photometry({"redshift": 0.3})


def test_tabulated_promotion_needs_a_range():
    """A tabulated fixed-z model must name its redshift table span to be promoted."""
    base = _build("wp", "plain", Fixed(Z_PLACEHOLDER))
    with pytest.raises(ValueError, match="z_range"):
        base.with_runtime_redshift()


def test_exact_promotion_rejects_a_table_range():
    """An exact model has no redshift table, so a range is refused, not ignored."""
    base = _build("exact", "plain", Fixed(Z_PLACEHOLDER))
    with pytest.raises(ValueError, match="z_range=None"):
        base.with_runtime_redshift(Z_RANGE)


# Bit-identity pin (#3505): a Fixed-z model that is NOT promoted keeps the fast path.
# Values are photometry [erg/s/cm2/Hz] computed on origin/main before this change,
# as float64 hex, at redshifts 0.1 (plain and rich structures) and 0.1 with WavePrecomp.
_PIN = {
    ("exact", "plain", 0.1): [
        "0x1.eeaad1813c405p-92",
        "0x1.a602837635024p-91",
        "0x1.28addff427fe6p-90",
    ],
    ("exact", "rich", 0.1): [
        "0x1.eec012a2d4d8bp-92",
        "0x1.a621c19a74dc0p-91",
        "0x1.28cdbd620f124p-90",
    ],
    ("wp", "rich", 0.1): [
        "0x1.eec677c37d44bp-92",
        "0x1.a6250caadc550p-91",
        "0x1.28c989d37249ap-90",
    ],
}


@pytest.mark.parametrize(("representation", "structure", "z"), list(_PIN))
def test_fixed_unpromoted_model_is_bit_identical_to_main(representation, structure, z):
    """A Fixed model that is not promoted is bit-identical to origin/main (#3505)."""
    got = np.asarray(_build(representation, structure, Fixed(z)).predict_photometry({}))
    expected = np.array([float.fromhex(v) for v in _PIN[(representation, structure, z)]])
    np.testing.assert_array_equal(got, expected)


# ── Component cells: every table a component bakes follows the runtime z ────────
#
# Each cell builds one model with a single redshift-dependent component, with every
# other parameter Fixed, so its output at z is fully determined by z. The runtime
# model (promoted from a Fixed placeholder) evaluated at three redshifts under ONE
# jit trace must equal a model built Fixed at each z, in both representations.

import tempfile
from pathlib import Path

from tengri import LineList

_FSPS_PATH = Path(__file__).resolve().parents[2] / "data" / "fsps_prsc_miles_chabrier.h5"
_Z_HIGH = (3.2, 3.8, 4.5)
_Z_RANGE_HIGH = (3.0, 5.0)
_LINE_NAMES = ("Hbeta", "OIII_5007", "Halpha")
_BANDS_OPT = ["sdss_g", "sdss_r", "sdss_i"]
_BANDS_IR = ["sdss_r", "wise_w1", "wise_w3", "wise_w4"]
_BANDS_RADIO = ["sdss_r", "wise_w3", "wise_w4", "herschel_250"]
_TAU = {"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)}
_AGN = {
    "type": "composable",
    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
    "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
    "blr": {"type": "none"},
    "nlr": {"type": "none"},
    "feii": {"type": "none"},
    "atten": {"type": "none"},
}
#: cell -> (bands, build kwargs, redshifts, table range)
_COMPONENT_CELLS = {
    "nebular_cue": (
        _BANDS_OPT,
        {"neb": {"type": "cue", "all_params": Fixed(DEFAULT)}, "dust_attenuation": _TAU},
        ZS,
        Z_RANGE,
    ),
    "dust_emission": (
        _BANDS_IR,
        {
            "dust_attenuation": {
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            "dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)},
        },
        ZS,
        Z_RANGE,
    ),
    "agn": (_BANDS_IR, {"agn": _AGN, "dust_attenuation": {"type": "none"}}, ZS, Z_RANGE),
    "radio": (
        _BANDS_RADIO,
        {
            "radio": {
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "all_params": Fixed(DEFAULT),
            },
            "dust_attenuation": _TAU,
        },
        ZS,
        Z_RANGE,
    ),
    "xray": (
        _BANDS_RADIO,
        {"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}, "dust_attenuation": _TAU},
        ZS,
        Z_RANGE,
    ),
    "shock_mappings": (
        _BANDS_OPT,
        {"shock": {"type": "mappings", "all_params": Fixed(DEFAULT)}, "dust_attenuation": _TAU},
        ZS,
        Z_RANGE,
    ),
    "igm_high_z": (
        ["sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        {"igm": {"type": "inoue14"}, "dust_attenuation": {"type": "none"}},
        _Z_HIGH,
        _Z_RANGE_HIGH,
    ),
}
_COMPONENT_CASES = [(cell, rep) for cell in _COMPONENT_CELLS for rep in REPRESENTATIONS]


@cache
def _fsps():
    if not _FSPS_PATH.is_file():
        pytest.skip(f"needs {_FSPS_PATH}")
    return load_ssp_data(str(_FSPS_PATH))


def _component_model(cell, representation, redshift, z_range=None):
    bands, kw, _, _ = _COMPONENT_CELLS[cell]
    approx = None if representation == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=_fsps(),
            observation=Observation(photometry=Photometry.from_names(bands)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(redshift),
            approx=approx,
            **kw,
        )
        return model if z_range is None else model.with_runtime_redshift(z_range)


def _runtime_outputs(model, zs, observable):
    traces = []

    def program(z):
        traces.append(1)
        return observable(model, {"redshift": z})

    jitted = jax.jit(program)
    outs = [np.asarray(jitted(jnp.asarray(z))) for z in zs]
    assert len(traces) == 1, f"{len(traces)} traces for {len(zs)} redshifts"
    return outs


def _photometry(model, params):
    return model.predict_photometry(params)


@pytest.mark.parametrize(("cell", "representation"), _COMPONENT_CASES)
def test_component_photometry_follows_runtime_z(cell, representation):
    """Runtime-z photometry of a component cell equals a Fixed-z build at each z."""
    _, _, zs, z_range = _COMPONENT_CELLS[cell]
    placeholder = zs[0]
    model = _component_model(
        cell, representation, placeholder, z_range if representation == "wp" else None
    )
    if representation == "exact":
        model = model.with_runtime_redshift(None)
    for z, got in zip(zs, _runtime_outputs(model, zs, _photometry), strict=True):
        ref = np.asarray(_component_model(cell, representation, z).predict_photometry({}))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"{cell}/{representation} z={z}"
        )


@cache
def _line_waves():
    cat = LineList.default_optical()
    return jnp.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in _LINE_NAMES])


def _line_fluxes(model, params):
    return model.predict_line_fluxes(params, target_wavelengths=_line_waves())


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_nebular_line_fluxes_follow_runtime_z(representation):
    """Cue line fluxes at a runtime z equal a Fixed-z build's, at three redshifts."""
    cell = "nebular_cue"
    placeholder = ZS[0]
    _line_waves()  # built outside the jit trace
    model = _component_model(
        cell, representation, placeholder, Z_RANGE if representation == "wp" else None
    )
    if representation == "exact":
        model = model.with_runtime_redshift(None)
    for z, got in zip(ZS, _runtime_outputs(model, ZS, _line_fluxes), strict=True):
        ref = np.asarray(_line_fluxes(_component_model(cell, representation, z), {}))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"lines/{representation} z={z}"
        )


def _grid_model(redshift, z_range=None):
    from tengri import FeaturePrecomp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=_fsps(),
            observation=Observation(photometry=Photometry.from_names(_BANDS_OPT)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            met={"logzsol": Fixed(0.0)},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Uniform(-4.0, -1.0),
                "logZ_gas": Uniform(-1.5, 0.3),
            },
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            redshift=Fixed(redshift),
            approx=(WavePrecomp(), FeaturePrecomp(n_grid=4)),
        )
        return model if z_range is None else model.with_runtime_redshift(z_range)


def test_nebular_grid_photometry_follows_runtime_z():
    """The per-Q_H nebular grid's band photometry follows a runtime z (tabulated bound)."""
    free = {"neb_logU": -2.5, "neb_logZ_gas": 0.0}
    model = _grid_model(ZS[0], Z_RANGE)
    assert model._nebular_grid_table is not None, "no nebular grid: the cell tests nothing"
    traces = []

    def program(z):
        traces.append(1)
        return model.predict_photometry({**free, "redshift": z})

    jitted = jax.jit(program)
    for z in ZS:
        got = np.asarray(jitted(jnp.asarray(z)))
        ref = np.asarray(_grid_model(z).predict_photometry(free))
        np.testing.assert_allclose(got, ref, rtol=RTOL["wp"], atol=0.0, err_msg=f"grid z={z}")
    assert len(traces) == 1


def test_nebular_grid_line_fluxes_follow_runtime_z():
    """Line fluxes served from the per-Q_H grid follow a runtime z (tabulated bound)."""
    free = {"neb_logU": -2.5, "neb_logZ_gas": 0.0}
    _line_waves()
    model = _grid_model(ZS[0], Z_RANGE)
    outs = _runtime_outputs(model, ZS, lambda m, p: _line_fluxes(m, {**free, **p}))
    for z, got in zip(ZS, outs, strict=True):
        ref = np.asarray(_line_fluxes(_grid_model(z), free))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL["wp"], atol=0.0, err_msg=f"grid lines z={z}"
        )


# ── Catalog entry points: N galaxies at N redshifts, one promoted model ────────
#
# Every catalog-level entry point derives one table span from the catalog's redshifts
# and promotes the model once. A second run on the same catalog adds no lowering, and
# the model the engine fits predicts at each galaxy's z what a Fixed-z build does.

import jax.monitoring as _jax_monitoring

from tengri.forward import convenience
from tengri.forward.sed_model import SEDModel as _SEDModel

CATALOG_ZS = (0.12, 0.31, 0.58, 0.77)
_TRUTH_LOGM = 9.5
_LOWERING = "/jax/core/compile/jaxpr_to_mlir_module_duration"
#: Programs a fit builds afresh around its result, whatever the redshift: the MAP
#: stationarity derivatives (one per fit, keyed on that fit's loss object), the posterior
#: summary maps and the result conversion. They are not the forward or loss programs the
#: promotion governs, so the second-run bar counts everything else.
_POST_FIT_PROGRAMS = frozenset({"jit(live)", "jit(mapped)", "jit(_convert_one)"})


def _mass_model(representation, redshift):
    approx = None if representation == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=_obs(),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=Fixed(redshift),
            approx=approx,
        )


def _catalog_rows(representation):
    rows = []
    for z in CATALOG_ZS:
        phot = np.asarray(
            _mass_model(representation, z).predict_photometry(
                {"sfh_dpl_log_total_mass": _TRUTH_LOGM}
            )
        )
        row = {"z": z}
        row.update({f"f{j}": float(v) for j, v in enumerate(phot)})
        row.update({f"e{j}": float(0.05 * v) for j, v in enumerate(phot)})
        rows.append(row)
    return rows


def _run_catalog(entry, representation, rows, base, tmp_path):
    approx = None if representation == "exact" else "auto"
    if entry == "catalog_fitter":
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal as CatalogFitter

        n_b = len(_obs().photometry.filters)
        galaxies = [
            {
                "flux_obs": jnp.asarray([r[f"f{j}"] for j in range(n_b)]),
                "noise": jnp.asarray([r[f"e{j}"] for j in range(n_b)]),
                "redshift": jnp.asarray(r["z"]),
            }
            for r in rows
        ]
        return (
            CatalogFitter(base, galaxies, approx=approx)
            .run("mcmc_nuts", key=jax.random.PRNGKey(0), n_warmup=8, n_samples=8, verbose=False)
            .posteriors
        )
    n_b = len(_obs().photometry.filters)
    kw = {"output_dir": str(tmp_path)} if entry == "fit_batch_checkpoint" else {}
    return convenience.fit_batch(
        base,
        rows,
        flux_cols=[f"f{j}" for j in range(n_b)],
        err_cols=[f"e{j}" for j in range(n_b)],
        redshift_col="z",
        method="map",
        n_steps=20,
        verbose=False,
        approx=approx,
        **kw,
    )


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("entry", ("catalog", "catalog_fitter", "fit_batch_checkpoint"))
def test_catalog_promotes_once_and_a_rerun_adds_no_lowering(entry, representation, tmp_path):
    """One promotion for the whole catalog; a second run compiles nothing new."""
    base = _mass_model(representation, Z_PLACEHOLDER)
    rows = _catalog_rows(representation)
    promotions = []
    real = _SEDModel._promote_to_runtime_redshift

    promoted_models = []

    def _count(self, z_range):
        promotions.append(z_range)
        promoted_models.append(real(self, z_range))
        return promoted_models[-1]

    lowerings = []

    def _listener(name, _duration, **kw):
        if name == _LOWERING and kw.get("fun_name") not in _POST_FIT_PROGRAMS:
            lowerings.append(kw.get("fun_name"))

    _SEDModel._promote_to_runtime_redshift = _count
    _jax_monitoring.register_event_duration_secs_listener(_listener)
    try:
        first = _run_catalog(entry, representation, rows, base, tmp_path / "a")
        after_first = len(lowerings)
        n_promotions = len(promotions)
        second = _run_catalog(entry, representation, rows, base, tmp_path / "b")
    finally:
        _SEDModel._promote_to_runtime_redshift = real
        _jax_monitoring.unregister_event_duration_listener(_listener)

    assert n_promotions == 1, f"{entry}/{representation}: promotions {promotions}"
    if representation == "exact":
        assert promotions == [None], f"{entry}: an exact model got a table span {promotions}"
    assert len(promotions) == 1, "the second run promoted again instead of reusing the first"
    assert len(lowerings) == after_first, (
        f"{entry}/{representation}: the second run lowered {lowerings[after_first:]}"
    )
    assert len(first) == len(second) == len(CATALOG_ZS)
    inner = promoted_models[0]
    if representation == "exact":
        assert not inner._approx["wave_precomp"], f"{entry}: the exact model was tabulated"
        assert inner._catalog_z_range is None, f"{entry}: a table was built for an exact model"
    params = {"sfh_dpl_log_total_mass": _TRUTH_LOGM}
    for z in CATALOG_ZS:
        got = np.asarray(inner.predict_photometry({**params, "redshift": z}))
        ref = np.asarray(_mass_model(representation, z).predict_photometry(params))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"{entry}/{representation}"
        )


@cache
def _cloudy_grid_path():
    """The synthetic CLOUDY grid, written once per process (xdist workers do not share it)."""
    from tests._cloudy_grid import write_synthetic_cloudy_grid

    return write_synthetic_cloudy_grid(Path(tempfile.mkdtemp(prefix="tengri_sweep_")) / "grid.h5")


def _cloudy_model(representation, redshift, z_range=None):
    from tests._cloudy_grid import synthetic_ionizing_ssp

    grid = _cloudy_grid_path()
    approx = None if representation == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ionizing_ssp(),
            observation=Observation(
                photometry=Photometry.from_names(["sdss_g", "sdss_r", "galex_fuv", "2mass_ks"])
            ),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "cloudy", "all_params": Fixed(DEFAULT), "grid": str(grid)},
            redshift=Fixed(redshift),
            approx=approx,
        )
        return model if z_range is None else model.with_runtime_redshift(z_range)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_cloudy_nebular_photometry_follows_runtime_z(representation):
    """The CLOUDY-grid band tables follow a runtime z (synthetic grid, tabulated bound)."""
    free = {"sfh_dpl_log_total_mass": 10.0}
    model = _cloudy_model(representation, ZS[0], Z_RANGE if representation == "wp" else None)
    if representation == "exact":
        model = model.with_runtime_redshift(None)
    outs = _runtime_outputs(model, ZS, lambda m, p: m.predict_photometry({**free, **p}))
    for z, got in zip(ZS, outs, strict=True):
        ref = np.asarray(_cloudy_model(representation, z).predict_photometry(free))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"cloudy/{representation} z={z}"
        )


# ── Wide tabulated spans with a Lyman break inside ─────────────────────────────
#
# The runtime-z table adds no error of its own: it agrees with a Fixed-z WavePrecomp
# build at every redshift. Against the EXACT build the tabulated path keeps WavePrecomp's
# own in-band error on a band the IGM has suppressed to ~1e-4 of its neighbors (u at
# z = 3.5: 4.9e-3; u at z = 4.95: 1.3e-2). That error is the same in a plain Fixed-z
# WavePrecomp build and does not move with n_z or n_subbands, so it is not a node
# placement matter; the strict xfail tracks it against the 3e-3 bar.

_UGRI = ["sdss_u", "sdss_g", "sdss_r", "sdss_i"]
_WIDE_SPANS = {
    "low": ((0.05, 3.55), (0.05, 1.0, 2.296, 3.502, 3.55)),
    "high": ((2.15, 4.95), (2.15, 3.2, 3.502, 4.2, 4.95)),
}


def _ugri_model(redshift, approx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=Observation(photometry=Photometry.from_names(_UGRI)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=Fixed(redshift),
            approx=approx,
        )


@pytest.mark.parametrize("span_name", sorted(_WIDE_SPANS))
@pytest.mark.parametrize("reference", ["tabulated", "exact"])
def test_wide_span_tabulation_at_break_crossings(span_name, reference, request):
    """Runtime z on a wide span, at its ends and the Lyman-break crossings, vs a Fixed-z build."""
    if reference == "exact":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="WavePrecomp in-band error on a Lyman-suppressed u band (4.9e-3 at "
                "z=3.5, 1.3e-2 at z=4.95), identical in a Fixed-z WavePrecomp build",
            )
        )
    span, zs = _WIDE_SPANS[span_name]
    runtime = _ugri_model(zs[0], WavePrecomp()).with_runtime_redshift(span)
    for z in zs:
        ref_approx = WavePrecomp() if reference == "tabulated" else None
        ref = np.asarray(_ugri_model(z, ref_approx).predict_photometry({}))
        got = np.asarray(runtime.predict_photometry({"redshift": z}))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL["wp"], atol=0.0, err_msg=f"{span_name} span, z={z}"
        )


# ── Band tables that bake the build redshift ───────────────────────────────────
#
# The CLOUDY and MAPPINGS band tables are built at one redshift, so a runtime-redshift
# model refuses them and takes the per-call path (same values, slower). The refusal is
# reported once per model; the engaged cells are strict xfails that track the debt.

_BAKED_TABLE_CASES = {
    "cloudy": ("nebular", "build_cloudy_nebular"),
    "shock": ("shock", "build_shock"),
}


def _engaged_after_a_call(case, *, runtime):
    from tests.contract import _band_table_cases as cases

    _, builder = _BAKED_TABLE_CASES[case]
    model = getattr(cases, builder)()
    if runtime:
        model = model.with_runtime_redshift(Z_RANGE)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    if runtime:
        params["redshift"] = ZS[1]
    model.predict_photometry(params)
    return model, model.band_table(_BAKED_TABLE_CASES[case][0], "band_table") is not None


@pytest.mark.parametrize("case", sorted(_BAKED_TABLE_CASES))
def test_baked_band_table_engages_at_a_fixed_redshift(case):
    """The table is engaged on the Fixed-z build, so the runtime cell below is not vacuous."""
    assert _engaged_after_a_call(case, runtime=False)[1]


@pytest.mark.xfail(
    strict=True,
    reason="band table bakes the build redshift; "
    "needs a redshift axis (#3505 table-axes mechanism)",
)
@pytest.mark.parametrize("case", sorted(_BAKED_TABLE_CASES))
def test_baked_band_table_engages_under_runtime_z(case):
    """Under a runtime redshift the table should still be engaged (tracked debt)."""
    assert _engaged_after_a_call(case, runtime=True)[1]


@pytest.mark.parametrize("case", sorted(_BAKED_TABLE_CASES))
def test_a_declined_band_table_is_reported_once(case, caplog):
    """The refusal of a band table is logged once per model and slot, with its reason."""
    import logging

    with caplog.at_level(logging.INFO, logger="tengri.forward.sed_model"):
        model, _ = _engaged_after_a_call(case, runtime=True)
        model.predict_photometry(
            {**dict(model.spec.sample(jax.random.PRNGKey(1))), "redshift": ZS[2]}
        )
    name = _BAKED_TABLE_CASES[case][0]
    msgs = [
        r.getMessage() for r in caplog.records if f"band table {name}/band_table" in r.getMessage()
    ]
    assert len(msgs) == 1, msgs
    assert "redshift is a runtime input" in msgs[0]


# ── Fits: the fitted mass follows the runtime redshift ─────────────────────────
#
# A forward evaluation that follows z (every cell above) does not make a fit follow it:
# the mass profiled out of the likelihood is read back through its own forward pass, and
# a pass at the declared redshift rescales the mass by (d_L(z_declared) / d_L(z))^2 while
# the loss, which never sees the rescaling, stays the exact fit's. Shape and loss cells
# cannot see that, so these cells compare the recovered mass with the mass of a fit built
# fixed at the galaxy's z, and with the truth the mock was drawn at, at redshifts that
# span the table and sit close to both of its edges.

FIT_ZS = (0.06, 0.32, 0.94)
FIT_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
FIT_TRUTH = {"sfh_dpl_log_total_mass": 10.0, "sfh_dpl_tau_gyr": 3.0, "dust_tau_diff": 0.3}
#: Atol [dex] between a runtime-z fit and the fixed-z fit of the same mock. The table's
#: flux error is below 3e-3 per band (RTOL), at most 1.3e-3 dex of mass at fixed colors;
#: the measured gap is 1e-4 dex at all three FIT_ZS. A baked distance gives at least
#: 1.07 dex over FIT_ZS.
FIT_LOGM_VS_EXACT = 0.005
#: Atol [dex] on the truth: the noiseless fixed-z fit itself sits up to 0.05 dex off it
#: at z = 0.94, where the age-mass-dust degeneracy and the priors shift the optimum.
FIT_LOGM_VS_TRUTH = 0.1


def _fit_model(representation, redshift):
    approx = None
    if representation != "exact":
        approx = WavePrecomp(catalog_z_range=Z_RANGE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=Observation(photometry=Photometry.from_names(FIT_BANDS)),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.5),
                "tau_gyr": Uniform(0.5, 10.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Uniform(0.0, 1.5),
            },
            neb={"type": "ssp"},
            redshift=redshift,
            approx=approx,
        )


@cache
def _fit_mock(z):
    flux = np.asarray(_fit_model("exact", Fixed(z)).predict_photometry(FIT_TRUTH))
    return flux, 0.05 * flux


def _map_logm(representation, z, *, runtime):
    """MAP log mass of the mock at ``z``, by an override or by a build fixed at ``z``."""
    from tengri import ForwardModel
    from tengri.inference.fitter import Fitter

    flux, noise = _fit_mock(z)
    base = _fit_model(representation, Fixed(Z_PLACEHOLDER if runtime else z))
    fitter = Fitter(
        ForwardModel.build(sed=base),
        flux,
        noise,
        approx=None if representation == "exact" else "auto",
        params_override={"redshift": z} if runtime else None,
    )
    assert fitter._profile_mass_resolved, "the cell must exercise the profiled-mass fit"
    post = fitter.run("map", verbose=False, key=jax.random.PRNGKey(0))
    return float(post.params["sfh_dpl_log_total_mass"])


@cache
def _fixed_z_logm(z):
    return _fixed_z_map(z)[0]


@pytest.mark.parametrize("z", FIT_ZS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_map_mass_at_a_runtime_redshift_matches_the_fixed_z_fit(representation, z):
    """A MAP fit at an override z returns the mass of a fit built fixed at that z."""
    got = _map_logm(representation, z, runtime=True)
    ref = _fixed_z_logm(z)
    assert abs(got - ref) < FIT_LOGM_VS_EXACT, (
        f"{representation} z={z}: logM {got:.3f} vs fixed-z fit {ref:.3f}"
    )
    truth = FIT_TRUTH["sfh_dpl_log_total_mass"]
    assert abs(got - truth) < FIT_LOGM_VS_TRUTH, f"{representation} z={z}: logM {got:.3f}"


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_catalog_masses_at_per_galaxy_redshifts_match_the_fixed_z_fits(representation):
    """Catalog.fit with a redshift column returns each galaxy's fixed-z mass at its own z."""
    from tengri import Catalog, ForwardModel

    fluxes = [_fit_mock(z) for z in FIT_ZS]
    n_b = len(FIT_BANDS)
    table = {"z": np.asarray(FIT_ZS)}
    for j in range(n_b):
        table[f"f{j}"] = np.asarray([f[0][j] for f in fluxes])
        table[f"e{j}"] = np.asarray([f[1][j] for f in fluxes])
    fwd = ForwardModel.build(sed=_fit_model(representation, Fixed(Z_PLACEHOLDER)))
    cat = Catalog(
        fwd,
        table,
        flux_unit="cgs_fnu",
        redshift_col="z",
        flux_cols=[f"f{j}" for j in range(n_b)],
        err_cols=[f"e{j}" for j in range(n_b)],
    )
    kw = {"approx": None} if representation == "exact" else {}
    fit = cat.fit(key=jax.random.PRNGKey(0), method="map", properties=("stellar_mass",), **kw)
    got = np.log10(np.asarray(fit["stellar_mass"]))
    ref = np.asarray([_fixed_z_logm(z) for z in FIT_ZS])
    np.testing.assert_allclose(
        got, ref, atol=FIT_LOGM_VS_EXACT, err_msg=f"{representation}: catalog logM at z={FIT_ZS}"
    )


# ── Samplers: the reinserted mass follows the runtime redshift ─────────────────
#
# Under a sampler the profiled mass is drawn per posterior sample after the run, through
# the reinsertion program and the batched catalog paths, which assemble their own
# prediction parameters. One NUTS chain of 100 warmup and 100 draws per fit.

SAMPLER_ZS = (0.06, 0.94)
SAMPLER_KW = {"n_warmup": 100, "n_samples": 100, "n_chains": 1}
#: Atol [dex] on a posterior median against the fixed-z fit's. Two NUTS fits of the same
#: mock at different seeds differ by 0.02 dex in the median, and a table fit and the
#: exact one by up to 0.05 dex on one seed (measured at z = 0.94); a baked distance gives
#: at least 1.07 dex over SAMPLER_ZS.
SAMPLER_LOGM_ATOL = 0.1


def _median_logm(posterior):
    return float(np.median(np.asarray(posterior.samples["sfh_dpl_log_total_mass"])))


def _nuts_logm(representation, z, *, runtime):
    from tengri import ForwardModel
    from tengri.inference.fitter import Fitter

    flux, noise = _fit_mock(z)
    base = _fit_model(representation, Fixed(Z_PLACEHOLDER if runtime else z))
    fitter = Fitter(
        ForwardModel.build(sed=base),
        flux,
        noise,
        approx=None if representation == "exact" else "auto",
        params_override={"redshift": z} if runtime else None,
    )
    assert fitter._profile_mass_resolved, "the cell must exercise the profiled-mass fit"
    post = fitter.run("mcmc_nuts", verbose=False, key=jax.random.PRNGKey(0), **SAMPLER_KW)
    return _median_logm(post)


@cache
def _fixed_z_nuts_logm(z):
    return _nuts_logm("exact", z, runtime=False)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_nuts_mass_at_a_runtime_redshift_matches_the_fixed_z_fit(representation):
    """NUTS at an override z reinserts the mass of a fit built fixed at that z."""
    z = SAMPLER_ZS[1]
    got = _nuts_logm(representation, z, runtime=True)
    ref = _fixed_z_nuts_logm(z)
    assert abs(got - ref) < SAMPLER_LOGM_ATOL, (
        f"{representation} z={z}: logM {got:.3f} vs fixed-z fit {ref:.3f}"
    )


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_catalog_nuts_masses_at_per_galaxy_redshifts_match_the_fixed_z_fits(representation):
    """Catalog.fit with NUTS reinserts each galaxy's fixed-z mass at its own z."""
    from tengri import Catalog, ForwardModel

    fluxes = [_fit_mock(z) for z in SAMPLER_ZS]
    n_b = len(FIT_BANDS)
    table = {"z": np.asarray(SAMPLER_ZS)}
    for j in range(n_b):
        table[f"f{j}"] = np.asarray([f[0][j] for f in fluxes])
        table[f"e{j}"] = np.asarray([f[1][j] for f in fluxes])
    fwd = ForwardModel.build(sed=_fit_model(representation, Fixed(Z_PLACEHOLDER)))
    cat = Catalog(
        fwd,
        table,
        flux_unit="cgs_fnu",
        redshift_col="z",
        flux_cols=[f"f{j}" for j in range(n_b)],
        err_cols=[f"e{j}" for j in range(n_b)],
    )
    kw = {"approx": None} if representation == "exact" else {}
    ref = np.asarray([_fixed_z_nuts_logm(z) for z in SAMPLER_ZS])
    jax.clear_caches()  # the references' compiled programs are not needed by the catalog fit
    fit = cat.fit(key=jax.random.PRNGKey(0), method="mcmc_nuts", **SAMPLER_KW, **kw)
    got = np.asarray([_median_logm(p) for p in fit.posteriors])
    np.testing.assert_allclose(
        got, ref, atol=SAMPLER_LOGM_ATOL, err_msg=f"{representation}: catalog NUTS at {SAMPLER_ZS}"
    )
    # Derived properties and posterior photometry of the same posteriors are evaluated at
    # each galaxy's own redshift: an extensive one (SFR) and an intensive one (the
    # mass-weighted age, capped by the age of the Universe at z), then the exact band
    # photometry against the noiseless mock it was fit to.
    for z, post, (flux, _noise) in zip(SAMPLER_ZS, fit.posteriors, fluxes):
        for name in PROPERTY_NAMES_FIT:
            value = float(np.median(np.asarray(post.properties[name])))
            np.testing.assert_allclose(
                value,
                _fixed_z_map(z)[1][name],
                rtol=PROPERTY_RTOL_SAMPLED,
                err_msg=f"{representation} z={z}: catalog NUTS {name}",
            )
        bands = np.median(np.asarray(post.observables(n_draws=32)), axis=0)
        np.testing.assert_allclose(
            bands,
            flux,
            rtol=PROPERTY_RTOL_SAMPLED,
            err_msg=f"{representation} z={z}: catalog NUTS posterior photometry",
        )


# ── Derived properties and the remaining fit routes ────────────────────────────
#
# A derived property of a posterior (the star formation rate over the last 100 Myr, whose
# window is capped by the age of the Universe at the galaxy's z) is evaluated at the
# redshift of the galaxy that was fit, through the same assembly of parameters as the
# profiled mass.

#: Rtol on a derived property of a runtime-z fit against the fixed-z fit's, for fits that
#: agree to 1e-4 dex in mass (MAP) or to the sampling noise above (samplers).
PROPERTY_RTOL_MAP = 0.02
PROPERTY_RTOL_SAMPLED = 0.25


def _catalog_table(zs):
    fluxes = [_fit_mock(z) for z in zs]
    n_b = len(FIT_BANDS)
    table = {"z": np.asarray(zs)}
    for j in range(n_b):
        table[f"f{j}"] = np.asarray([f[0][j] for f in fluxes])
        table[f"e{j}"] = np.asarray([f[1][j] for f in fluxes])
    return table


def _fit_catalog(representation, zs, method, **kwargs):
    from tengri import Catalog, ForwardModel

    n_b = len(FIT_BANDS)
    fwd = ForwardModel.build(sed=_fit_model(representation, Fixed(Z_PLACEHOLDER)))
    cat = Catalog(
        fwd,
        _catalog_table(zs),
        flux_unit="cgs_fnu",
        redshift_col="z",
        flux_cols=[f"f{j}" for j in range(n_b)],
        err_cols=[f"e{j}" for j in range(n_b)],
    )
    kw = {"approx": None} if representation == "exact" else {}
    return cat.fit(key=jax.random.PRNGKey(0), method=method, **kwargs, **kw)


def _fixed_z_map_post(z):
    from tengri import ForwardModel
    from tengri.inference.fitter import Fitter

    flux, noise = _fit_mock(z)
    fitter = Fitter(
        ForwardModel.build(sed=_fit_model("exact", Fixed(z))), flux, noise, approx=None
    )
    return fitter.run("map", verbose=False, key=jax.random.PRNGKey(0))


PROPERTY_NAMES_FIT = ("sfr_100myr", "mass_weighted_age_gyr")


@cache
def _fixed_z_map(z):
    """One fixed-z MAP fit: its log mass and the properties the cells compare."""
    post = _fixed_z_map_post(z)
    props = {n: float(np.asarray(post.properties[n])) for n in PROPERTY_NAMES_FIT}
    return float(post.params["sfh_dpl_log_total_mass"]), props


def _fixed_z_sfr(z):
    return _fixed_z_map(z)[1]["sfr_100myr"]


@pytest.mark.parametrize("z", FIT_ZS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_map_property_at_a_runtime_redshift_matches_the_fixed_z_fit(representation, z):
    """The SFR of a MAP fit at an override z is the fixed-z fit's, not the placeholder's."""
    from tengri import ForwardModel
    from tengri.inference.fitter import Fitter

    flux, noise = _fit_mock(z)
    fitter = Fitter(
        ForwardModel.build(sed=_fit_model(representation, Fixed(Z_PLACEHOLDER))),
        flux,
        noise,
        approx=None if representation == "exact" else "auto",
        params_override={"redshift": z},
    )
    post = fitter.run("map", verbose=False, key=jax.random.PRNGKey(0))
    got = float(np.asarray(post.properties["sfr_100myr"]))
    np.testing.assert_allclose(got, _fixed_z_sfr(z), rtol=PROPERTY_RTOL_MAP)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_catalog_laplace_masses_and_properties_match_the_fixed_z_fits(representation):
    """A sequential-fallback sampler reinserts each galaxy's fixed-z mass and properties."""
    names = ("stellar_mass", "sfr_100myr", "mass_weighted_age_gyr")
    fit = _fit_catalog(representation, FIT_ZS, "laplace", properties=names)
    mass = np.log10(np.asarray(fit["stellar_mass"]))
    ref = np.asarray([_fixed_z_logm(z) for z in FIT_ZS])
    np.testing.assert_allclose(
        mass, ref, atol=FIT_LOGM_VS_EXACT * 10, err_msg=f"{representation}: laplace logM"
    )
    for name in names[1:]:
        got = np.ravel(np.asarray(fit[name]))[: len(FIT_ZS)]
        want = np.asarray([_fixed_z_map(z)[1][name] for z in FIT_ZS])
        np.testing.assert_allclose(
            got, want, rtol=PROPERTY_RTOL_SAMPLED, err_msg=f"{representation}: laplace {name}"
        )


def test_catalog_reinsertion_compiles_once_for_galaxies_at_different_redshifts():
    """The profiled-mass reinsertion is one program for the catalog, not one per galaxy."""
    lowerings = []

    def _listener(name, _duration, **kw):
        if name == _LOWERING and kw.get("fun_name") == "jit(_reinsert)":
            lowerings.append(kw.get("fun_name"))

    _jax_monitoring.register_event_duration_secs_listener(_listener)
    try:
        _fit_catalog("wp", (0.06, 0.32, 0.94), "mcmc_nuts", n_warmup=5, n_samples=5, n_chains=1)
    finally:
        _jax_monitoring.unregister_event_duration_listener(_listener)
    assert len(lowerings) == 1, f"{len(lowerings)} reinsertion programs for 3 galaxies"


PROPERTY_NAMES = ("stellar_mass", "sfr_100myr", "mass_weighted_age_gyr")


@pytest.mark.parametrize("z", FIT_ZS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_properties_at_fixed_parameters_follow_the_runtime_redshift(representation, z):
    """Properties at one parameter point equal a Fixed-z build's, extensive and intensive."""
    base = _fit_model(representation, Fixed(Z_PLACEHOLDER))
    runtime = base if representation == "wp" else base.with_runtime_redshift(None)
    got = runtime.predict_properties({**FIT_TRUTH, "redshift": z}, names=PROPERTY_NAMES)
    ref = _fit_model("exact", Fixed(z)).predict_properties(FIT_TRUTH, names=PROPERTY_NAMES)
    for name in PROPERTY_NAMES:
        np.testing.assert_allclose(
            np.asarray(got[name]),
            np.asarray(ref[name]),
            rtol=RTOL["exact"] if representation == "exact" else RTOL["wp"],
            err_msg=f"{representation} z={z}: {name}",
        )

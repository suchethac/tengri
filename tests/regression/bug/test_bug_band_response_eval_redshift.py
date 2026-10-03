# SPDX-License-Identifier: BSD-3-Clause
"""Fast-path response tables follow the EVALUATION redshift, not a build-time one.

Under ``WavePrecomp(catalog_z_range=...)`` the model's redshift is a ``Fixed``
placeholder while every evaluation runs at its own runtime z (the fitter hands it
in through ``fixed_values``). A free redshift is the same situation: each sample
has its own z. Three fast-path responses used to be baked at the spec's redshift
and gated only on "redshift is not free", so a catalog fit read them at the
placeholder:

* the dust-IR per-filter response ``R = int template * filters`` (``L_ir x R``),
* the radio and X-ray per-term responses, and
* the energy-balance LUT of an attenuation law that reads the redshift
  (``narayanan_z``).

Measured on the unfixed code, two-component calzetti + dale2014 + tsnorm-free
photometry, ``catalog_z_range=(0.05, 2)`` with a ``Fixed(0.05)`` placeholder,
against a ``Fixed(z)`` exact build (relative error):

    z      WISE W4   WISE W3   sdss r
    0.5     0.573     0.505    <= 3.5e-3
    1.0     0.496     0.904    <= 3.5e-3
    1.5     0.980     2.98     <= 3.5e-3

Free-z models were not wrong, they were *refused* (the gates sent them to the
exact per-call path). The responses are now tabulated over the model's redshift
range, uniform in ln(1+z), and interpolated at the traced z, so a free redshift
uses the fast path too.

Each test compares against an independent reference: a model built at ``Fixed(z)``
with ``approx=None``, which never touches any table.
"""

from __future__ import annotations

import warnings

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SSPData,
    Uniform,
    WavePrecomp,
)
from tengri.observation.filters import load_tophat_filter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Redshift grid nodes for the real-SSP models: enough to hold the table's own
#: interpolation error under the budgets below while keeping the stellar z-table
#: build affordable in a test.
N_Z = 120

#: Evaluation redshifts: three on the design grid and an off-grid midpoint.
Z_EVAL = (0.5, 1.0, 1.5, 0.77)

#: A catalog window that starts at the placeholder, as the design measurement does.
Z_RANGE = (0.05, 2.0)
PLACEHOLDER = 0.05

#: 2-10 keV, lambda = 12.398 / E[keV] = 1.24-6.20 A (observed frame).
XRAY_BAND = load_tophat_filter(3.72, 4.96, name="xray_2_10kev")
#: 1.4 GHz, lambda = c / nu = 2.14e9 A (21 cm).
RADIO_BAND = load_tophat_filter(2.14e9, 2.14e8, name="radio_1p4ghz")

#: Budget on the WavePrecomp fast path against the exact path. The fast path's own
#: residual (stellar quadrature, the dust Taylor projection) is 7.8e-4 at W3/W4 at a
#: plain Fixed z; the table adds its interpolation (<= 2e-4 at 250 nodes). The pre-fix
#: error was 0.5-3.0, so 5e-3 separates them by 100x.
BUDGET_VS_EXACT = 5e-3

#: The catalog/free-z model against a plain ``Fixed(z)`` WavePrecomp model: the same
#: fast path with no redshift table involved, so only the table's interpolation and
#: the shared stellar z-table differ.
BUDGET_VS_FIXED_FAST = 1.5e-3


def _quiet_build(**kwargs) -> SEDModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _observation(*names: str, extra: tuple = ()) -> Observation:
    optical = Photometry.from_names(list(names))
    return Observation(photometry=Photometry(filters=(*tuple(optical.filters), *extra)))


def _emitter_model(z, approx) -> SEDModel:
    """Dale IR (radio-tail-free grid) + radio + X-ray on the tracked FSPS SSP at ``z``."""
    from tengri import load_ssp

    return _quiet_build(
        ssp_data=load_ssp(),
        observation=_observation("sdss_r", "wise_w4", "wise_w3", extra=(XRAY_BAND, RADIO_BAND)),
        redshift=z,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "dust_tau_bc": Uniform(0.0, 2.0),
            "dust_tau_diff": Uniform(0.0, 1.0),
        },
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        },
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        approx=approx,
    )


_TAU = {"dust_tau_bc": jnp.asarray(0.6), "dust_tau_diff": jnp.asarray(0.3)}


def _exact_and_fast_fixed(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Photometry of a plain ``Fixed(z)`` model: (exact, WavePrecomp)."""
    exact = _emitter_model(Fixed(z), None)
    fast = _emitter_model(Fixed(z), WavePrecomp(n_z=N_Z))
    return (
        np.asarray(exact.predict_photometry(dict(_TAU))),
        np.asarray(fast.predict_photometry(dict(_TAU))),
    )


def _catalog_photometry(model: SEDModel, z: float) -> np.ndarray:
    """Photometry at runtime redshift ``z`` through the jitted observables.

    This is the channel the fitter uses: the spec's redshift stays a placeholder
    and the evaluation redshift rides ``fixed_values``.
    """
    fixed = {**model.spec.get_fixed_values(), "redshift": jnp.asarray(z)}
    fn = model._get_or_build_predict_observables_jit()
    observables = fn(dict(_TAU), fixed, *model._resolve_threaded_data(None, None, None))
    return np.asarray(observables.phot_fnu)


def _free_photometry(model: SEDModel, z: float) -> np.ndarray:
    return np.asarray(model.predict_photometry({**_TAU, "redshift": jnp.asarray(z)}))


@pytest.fixture(scope="module")
def catalog_model() -> SEDModel:
    return _emitter_model(Fixed(PLACEHOLDER), WavePrecomp(catalog_z_range=Z_RANGE, n_z=N_Z))


@pytest.fixture(scope="module")
def free_model() -> SEDModel:
    return _emitter_model(Uniform(*Z_RANGE), WavePrecomp(n_z=N_Z))


@pytest.fixture(scope="module")
def references() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    return {z: _exact_and_fast_fixed(z) for z in Z_EVAL}


#: Band positions in the model's filter order.
_R, _W4, _W3, _XRAY, _RADIO = range(5)


def _rel(a: np.ndarray, b: np.ndarray, idx) -> float:
    return float(np.max(np.abs(a[list(idx)] / b[list(idx)] - 1.0)))


def _assert_bands(got, references, z, idx, label):
    exact, fast = references[z]
    vs_exact = _rel(got, exact, idx)
    vs_fast = _rel(got, fast, idx)
    assert vs_exact < BUDGET_VS_EXACT, (
        f"{label} z={z}: {vs_exact:.3e} from the exact Fixed(z) build "
        f"(budget {BUDGET_VS_EXACT:.0e}); the response is not on the evaluation redshift."
    )
    assert vs_fast < BUDGET_VS_FIXED_FAST, (
        f"{label} z={z}: {vs_fast:.3e} from the Fixed(z) WavePrecomp build "
        f"(budget {BUDGET_VS_FIXED_FAST:.0e})."
    )


# ── T1: dust-IR response under catalog_z_range ────────────────────────


@pytest.mark.parametrize("z", Z_EVAL)
def test_dust_ir_response_follows_the_catalog_runtime_redshift(catalog_model, references, z):
    """W3/W4/r at a runtime z match a Fixed(z) build, not the placeholder's response."""
    got = _catalog_photometry(catalog_model, z)
    _assert_bands(got, references, z, (_R, _W4, _W3), "catalog dust-IR")


# ── T2: the same under a free redshift, on the fast path ──────────────


def test_a_free_redshift_engages_the_dust_ir_band_response(free_model):
    """A free z must use the fast path: the table exists and spans the prior."""
    table = free_model._dust_band_response_cache
    assert table is not None, (
        "a free redshift left the dust-IR band response off: the model fell back to "
        "the per-call dense filter integral"
    )
    assert table["ln1pz"].shape[0] == N_Z
    z_lo, z_hi = float(jnp.expm1(table["ln1pz"][0])), float(jnp.expm1(table["ln1pz"][-1]))
    assert z_lo <= Z_RANGE[0] and z_hi >= Z_RANGE[1], (z_lo, z_hi)


@pytest.mark.parametrize("z", Z_EVAL)
def test_dust_ir_response_follows_a_free_redshift(free_model, references, z):
    got = _free_photometry(free_model, z)
    _assert_bands(got, references, z, (_R, _W4, _W3), "free-z dust-IR")


# ── T3: radio and X-ray term responses ────────────────────────────────


def test_the_emitter_bands_are_not_vacuous(references):
    """The radio and X-ray bands must be carried by their emitters, or T3 proves nothing."""
    exact, _ = references[1.0]
    assert np.all(np.isfinite(exact)) and np.all(exact > 0.0), exact


def test_radio_and_xray_responses_are_engaged_under_both_redshift_modes(catalog_model, free_model):
    for label, model in (("catalog", catalog_model), ("free", free_model)):
        for name in ("radio", "xray"):
            table = getattr(model, f"_{name}_term_response_cache")
            assert table is not None, f"{label} z: the {name} term response is off"
            assert table["ln1pz"].shape[0] == N_Z, (label, name)


@pytest.mark.parametrize("z", Z_EVAL)
def test_radio_and_xray_responses_follow_the_catalog_runtime_redshift(
    catalog_model, references, z
):
    got = _catalog_photometry(catalog_model, z)
    _assert_bands(got, references, z, (_XRAY, _RADIO), "catalog radio/X-ray")


@pytest.mark.parametrize("z", Z_EVAL)
def test_radio_and_xray_responses_follow_a_free_redshift(free_model, references, z):
    got = _free_photometry(free_model, z)
    _assert_bands(got, references, z, (_XRAY, _RADIO), "free-z radio/X-ray")


# ── Catalog engines: the shared loss inherits the evaluation-z tables ─

#: A small catalog spanning the window.
CATALOG_Z = (0.3, 0.8, 1.3, 1.9)


def _catalog_data() -> tuple[np.ndarray, np.ndarray]:
    """Per-galaxy photometry from an exact Fixed(z_i) build, plus 2 % noise."""
    flux = np.stack(
        [
            np.asarray(_emitter_model(Fixed(z), None).predict_photometry(dict(_TAU)))
            for z in CATALOG_Z
        ]
    )
    return flux, 0.02 * flux


@pytest.fixture(scope="module")
def catalog_data():
    return _catalog_data()


def _exact_losses(catalog_data) -> np.ndarray:
    """Single-galaxy Fixed(z_i) exact-build loss at the shared start point."""
    from tengri.inference.fitter import Fitter
    from tengri.inference.loss_functions import build_loss_fn

    flux, noise = catalog_data
    out = []
    for i, z in enumerate(CATALOG_Z):
        fitter = Fitter(_emitter_model(Fixed(z), None), flux[i], noise[i])
        init = fitter._initialize_unbounded(jax.random.PRNGKey(3))
        out.append(float(build_loss_fn(fitter)(init, dict(fitter._data_args))))
    return np.asarray(out)


def test_the_sequential_catalog_engine_loss_follows_each_galaxys_redshift(
    catalog_model, catalog_data
):
    """``CatalogFitter``'s sequential path: ``Fitter(params_override={"redshift": z_i})``."""
    from tengri.inference.fitter import Fitter
    from tengri.inference.loss_functions import build_loss_fn

    flux, noise = catalog_data
    exact = _exact_losses(catalog_data)
    for i, z in enumerate(CATALOG_Z):
        fitter = Fitter(catalog_model, flux[i], noise[i], params_override={"redshift": z})
        init = fitter._initialize_unbounded(jax.random.PRNGKey(3))
        data_args = dict(fitter._data_args)
        assert "redshift" in data_args, "the runtime redshift must ride data_args"
        loss = float(build_loss_fn(fitter)(init, data_args))
        assert abs(loss - exact[i]) < LOSS_RTOL * abs(exact[i]), (
            f"galaxy {i} (z={z}): loss {loss:.4f} vs exact Fixed(z) {exact[i]:.4f}"
        )
    assert catalog_model._dust_band_response_cache is not None, "the fast path fell off"


#: Relative budget on the loss (chi^2/2 + prior) at a shared start point, 2 % noise.
#: The fast path's own residual (<= 1e-3 in flux) moves it ~2e-3 relative; the
#: pre-fix W3/W4 error (0.5-3.0 in flux) moves a galaxy's loss by 10^2-10^4 absolute.
LOSS_RTOL = 1e-2


def test_the_batched_catalog_engine_loss_follows_each_galaxys_redshift(
    catalog_model, catalog_data
):
    """The vmapped batch engine: one compiled loss, a per-galaxy traced redshift."""
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
    from tengri.inference.backends.mcmc.catalog import _make_substitute
    from tengri.inference.fitter import Fitter

    flux, noise = catalog_data
    exact = _exact_losses(catalog_data)
    template = Fitter(catalog_model, flux[0], noise[0])
    init = template._initialize_unbounded(jax.random.PRNGKey(3))
    log_post, _unravel, init_flat, template_args = _get_flat_logdensity(template, init)
    substitute = _make_substitute(template_args, True, False)
    presence = jnp.ones_like(jnp.asarray(flux))

    def loss_one(data, noise_i, presence_i, z):
        return -log_post(init_flat, substitute(data, noise_i, presence_i, z, None, None))

    losses = np.asarray(
        jax.vmap(loss_one)(jnp.asarray(flux), jnp.asarray(noise), presence, jnp.asarray(CATALOG_Z))
    )
    np.testing.assert_allclose(losses, exact, rtol=LOSS_RTOL)
    assert catalog_model._dust_band_response_cache is not None, "the fast path fell off"


def test_the_batched_map_warm_start_runs_with_the_tables_threaded(catalog_model, catalog_data):
    """``build_catalog_map_init`` vmapped over galaxies gives a finite start for each."""
    from tengri.inference.backends.mcmc.catalog import build_catalog_map_init
    from tengri.inference.fitter import Fitter

    flux, noise = catalog_data
    template = Fitter(catalog_model, flux[0], noise[0])
    map_init = build_catalog_map_init(template, n_steps=3, thread_redshift=True)
    init = template._initialize_unbounded(jax.random.PRNGKey(3))
    start, _ = jax.flatten_util.ravel_pytree(init)
    n_gal = flux.shape[0]
    warm = jax.vmap(map_init)(
        jnp.tile(start, (n_gal, 1)),
        jnp.asarray(flux),
        jnp.asarray(noise),
        jnp.ones_like(jnp.asarray(flux)),
        jnp.asarray(CATALOG_Z),
        jnp.zeros((n_gal, 1)),
        jnp.ones((n_gal, 1)),
    )
    assert np.all(np.isfinite(np.asarray(warm)))


# ── T4: the energy-balance LUT of a z-reading attenuation law ─────────


@pytest.fixture(scope="module")
def uv_ssp() -> SSPData:
    """A smooth power-law SSP cube, so the dust curve is the only structure."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1200)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-30, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def ir_obs() -> Observation:
    """Optical bands that feed the energy balance plus the 100 um band it heats."""

    def _tophat(center: float) -> FilterCurve:
        wave = jnp.linspace(center * 0.84, center * 1.16, 40)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, 40)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _law_model(uv_ssp, ir_obs, z, approx, law="narayanan_z") -> SEDModel:
    return _quiet_build(
        ssp_data=uv_ssp,
        observation=ir_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": law,
            "dust_tau_bc": Uniform(0.0, 1.0),
            "dust_tau_diff": Uniform(0.0, 1.5),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=z,
        approx=approx,
    )


_LUT_Z = (2.0, 3.1, 4.0, 5.3)
_LUT_RANGE = (0.05, 6.0)
_LUT_TAU = {"dust_tau_bc": jnp.asarray(0.0), "dust_tau_diff": jnp.asarray(0.6)}
#: The 100 um band carries L_ir. Pre-fix, the LUT baked at the placeholder z = 0.05
#: drifted 1.1e-2 at z = 2 and 1.7e-1 at z = 6; the LUT's own residual is ~6e-4.
LUT_BUDGET = 5e-3


@pytest.fixture(scope="module")
def lut_catalog(uv_ssp, ir_obs) -> SEDModel:
    return _law_model(
        uv_ssp,
        ir_obs,
        Fixed(PLACEHOLDER),
        WavePrecomp(catalog_z_range=_LUT_RANGE, n_z=48),
    )


@pytest.fixture(scope="module")
def lut_free(uv_ssp, ir_obs) -> SEDModel:
    return _law_model(uv_ssp, ir_obs, Uniform(*_LUT_RANGE), WavePrecomp(n_z=48))


def _ir_band_error(got, exact_model) -> float:
    ref = np.asarray(exact_model.predict_photometry(dict(_LUT_TAU)))
    return abs(float(got[-1] / ref[-1] - 1.0))


def _z_tabulated(model) -> bool:
    lut = model._energy_balance_lut_cache
    return lut is not None and getattr(lut, "ln1pz", None) is not None


@pytest.mark.parametrize("z", _LUT_Z)
def test_the_energy_balance_lut_follows_the_catalog_runtime_redshift(
    lut_catalog, uv_ssp, ir_obs, z
):
    got = _catalog_lut_photometry(lut_catalog, z)
    err = _ir_band_error(got, _law_model(uv_ssp, ir_obs, Fixed(z), None))
    assert err < LUT_BUDGET, (
        f"z={z}: the 100 um band is {err:.3e} from the Fixed(z) exact path "
        f"(budget {LUT_BUDGET:.0e}); the LUT was built on the placeholder's curve."
    )
    assert _z_tabulated(lut_catalog), "narayanan_z must engage a z-tabulated LUT"


def _catalog_lut_photometry(model: SEDModel, z: float) -> np.ndarray:
    fixed = {**model.spec.get_fixed_values(), "redshift": jnp.asarray(z)}
    fn = model._get_or_build_predict_observables_jit()
    out = fn(dict(_LUT_TAU), fixed, *model._resolve_threaded_data(None, None, None))
    return np.asarray(out.phot_fnu)


@pytest.mark.parametrize("z", _LUT_Z)
def test_the_energy_balance_lut_follows_a_free_redshift(lut_free, uv_ssp, ir_obs, z):
    got = np.asarray(lut_free.predict_photometry({**_LUT_TAU, "redshift": jnp.asarray(z)}))
    err = _ir_band_error(got, _law_model(uv_ssp, ir_obs, Fixed(z), None))
    assert err < LUT_BUDGET, f"z={z}: free-z 100 um band {err:.3e} from exact"
    assert _z_tabulated(lut_free), (
        "a free redshift must keep the LUT on narayanan_z, tabulated over z"
    )


def test_a_law_that_reads_no_redshift_keeps_the_single_curve_lut(uv_ssp, ir_obs):
    """kriek_conroy ignores z: its LUT has no z axis, whatever the redshift mode."""
    model = _law_model(
        uv_ssp,
        ir_obs,
        Fixed(PLACEHOLDER),
        WavePrecomp(catalog_z_range=_LUT_RANGE, n_z=48),
        law="kriek_conroy",
    )
    lut = model._energy_balance_lut_cache
    assert lut is not None and getattr(lut, "ln1pz", None) is None


def test_the_z_tabulated_lut_agrees_with_the_single_curve_builder_at_a_node(uv_ssp):
    """On a node, the factorized z-builder equals the per-node single-curve LUT."""
    from tengri.components.dust.energy_balance_precompute import (
        build_energy_balance_lut,
        build_energy_balance_lut_over_z,
    )

    ages = 10.0**uv_ssp.ssp_lg_age_gyr * 1e9
    tau_bc, tau_diff = jnp.linspace(0.0, 1.0, 5), jnp.linspace(0.0, 1.5, 6)
    zs = jnp.asarray([0.4, 2.2])
    kw = dict(law_bc="narayanan_z", law_diff="narayanan_z", f_obscuration=0.1)
    tabulated = build_energy_balance_lut_over_z(
        uv_ssp.ssp_flux,
        uv_ssp.ssp_wave,
        ln1pz=jnp.log1p(zs),
        params_at_z=lambda z: ({"redshift": z}, {"redshift": z}),
        tau_bc_grid=tau_bc,
        tau_diff_grid=tau_diff,
        **kw,
    )
    for i, z in enumerate(np.asarray(zs)):
        single = build_energy_balance_lut(
            uv_ssp.ssp_flux,
            uv_ssp.ssp_wave,
            bc_params={"redshift": float(z)},
            diff_params={"redshift": float(z)},
            tau_bc_grid=tau_bc,
            tau_diff_grid=tau_diff,
            **kw,
        )
        np.testing.assert_allclose(np.asarray(tabulated.G[i]), np.asarray(single.G), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(tabulated.B), np.asarray(single.B), rtol=1e-12)


#: A z-tabulated LUT is capped at 31 redshift nodes here, the count the cap allows on a
#: real SSP with a 24 x 24 optical-depth grid. The accuracy claim is for that regime.
_CAPPED_NODES = 31
#: Half the 5e-3 band budget: the LUT is one factor in the IR band beside the stellar
#: z-table and the R table, so its interpolation may not spend the whole budget.
_SWEEP_BUDGET = 2.5e-3


def _sweep_redshifts() -> np.ndarray:
    """Dense sweep over the window, plus both sides of every law breakpoint."""
    breaks = np.arange(1.0, 6.0)
    near = np.concatenate([breaks - 1e-3, breaks, breaks + 1e-3, [4.7, 5.7, 0.05, 6.0]])
    return np.unique(np.concatenate([np.linspace(0.05, 6.0, 200), near]))


def test_the_z_tabulated_lut_holds_its_budget_across_the_whole_redshift_range(
    uv_ssp, ir_obs, monkeypatch
):
    """L_abs from the capped z-LUT vs the exact curve at 200+ redshifts, kinks included.

    ``narayanan_z`` is piecewise in z (table nodes at integers), so a z axis that
    ignores that spends its nodes where the curve is flat and misses the segments
    where it moves: 4.1e-3 in L_abs near z = 4.8 with 31 uniform-plus-breakpoint
    nodes (1.1e-3 with the budget spent evenly per segment).
    The reference is the same transmission built directly at each redshift.
    """
    from tengri.components.dust.energy_balance_precompute import (
        _lut_contract,
        build_energy_balance_lut_over_z,
    )

    n_met, n_age = uv_ssp.ssp_flux.shape[:2]
    monkeypatch.setattr(SEDModel, "_EB_LUT_MAX_ELEMENTS", _CAPPED_NODES * n_met * n_age * 24 * 24)
    model = _law_model(
        uv_ssp, ir_obs, Fixed(PLACEHOLDER), WavePrecomp(catalog_z_range=_LUT_RANGE, n_z=200)
    )
    lut = model._energy_balance_lut_cache
    assert lut is not None and lut.ln1pz.shape[0] <= _CAPPED_NODES
    z_nodes = np.expm1(np.asarray(lut.ln1pz))
    for b in np.arange(1.0, 6.0):
        assert np.min(np.abs(z_nodes - b)) < 1e-9, f"no node on breakpoint z={b}"

    tb = float(lut.tau_bc_grid[10])
    td = float(lut.tau_diff_grid[12])
    zs = _sweep_redshifts()
    ages = 10.0**uv_ssp.ssp_lg_age_gyr * 1e9
    truth = build_energy_balance_lut_over_z(
        uv_ssp.ssp_flux,
        uv_ssp.ssp_wave,
        ln1pz=jnp.log1p(jnp.asarray(zs)),
        params_at_z=lambda z: ({"redshift": z}, {"redshift": z}),
        law_bc="narayanan_z",
        law_diff="narayanan_z",
        f_obscuration=0.0,
        tau_bc_grid=jnp.asarray([tb]),
        tau_diff_grid=jnp.asarray([td]),
    )
    weights = jnp.ones((n_met, n_age)) / (n_met * n_age)
    worst = 0.0
    for i, z in enumerate(zs):
        exact = float(jnp.sum(weights * (truth.B - truth.G[i, :, :, 0, 0])))
        got = float(
            _lut_contract(lut, weights, jnp.asarray(tb), jnp.asarray(td), redshift=jnp.asarray(z))
        )
        worst = max(worst, abs(got / exact - 1.0))
    assert worst < _SWEEP_BUDGET, (
        f"L_abs error {worst:.3e} over the sweep (budget {_SWEEP_BUDGET})"
    )


@pytest.mark.parametrize("z", (0.999, 1.001, 3.999, 4.7, 5.7, 5.999))
def test_the_ir_band_holds_its_budget_at_the_law_kinks(lut_catalog, uv_ssp, ir_obs, z):
    got = _catalog_lut_photometry(lut_catalog, z)
    err = _ir_band_error(got, _law_model(uv_ssp, ir_obs, Fixed(z), None))
    assert err < LUT_BUDGET, f"z={z}: IR band {err:.3e} from exact"


# ── T5: catalog models still share one compile signature ──────────────


def test_catalog_models_share_a_signature_and_a_grid_not_keyed_on_the_placeholder(uv_ssp, ir_obs):
    """Three Fixed-z catalog models: one compile signature, one z grid."""
    models = [
        _law_model(
            uv_ssp,
            ir_obs,
            Fixed(z0),
            WavePrecomp(catalog_z_range=(0.05, 2.0), n_z=24),
            law="calzetti",
        )
        for z0 in (0.1, 0.7, 1.4)
    ]
    signatures = {m.compile_signature() for m in models}
    assert len(signatures) == 1, "placeholder redshifts must not split the compile signature"

    tables = [m._dust_band_response_cache for m in models]
    assert all(t is not None for t in tables)
    for table in tables[1:]:
        np.testing.assert_array_equal(np.asarray(table["ln1pz"]), np.asarray(tables[0]["ln1pz"]))
        np.testing.assert_allclose(
            np.asarray(table["values"]), np.asarray(tables[0]["values"]), rtol=1e-12
        )
    # ...and the threaded data has one pytree structure, so one compiled kernel serves all.
    structures = {jax.tree_util.tree_structure(m._template_data_for_jit()) for m in models}
    assert len(structures) == 1

# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp tabulates a component through the registry when its shape parameters are free.

The exact band-response tables (``_dust_emission_band_response``,
``_additive_term_band_response``) need every shape parameter fixed. With a shape parameter free
(the usual fit) the component used to integrate the filters on every call. The registry
adapters tabulate the band fluxes over those shape axes at build, so WavePrecomp reads a table
there instead. Three things are pinned per family:

(a) the free-shape model builds the registry lookup and the predict does not integrate;
(b) the table agrees with the exact photometry at three draws inside the prior, within the
    adapter's stated accuracy, with finite gradients;
(c) with every shape parameter fixed the exact table is still used and the registry is not
    asked for the family.

A family whose table misses its stated accuracy stays unwired and is recorded in ``UNWIRED``.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

#: family -> "measured X vs stated Y" for a family whose table misses its stated accuracy.
_TRIWEIGHT = "triweight smoother over a handful of linear-flux nodes"


def _missed(measured: str, detail: str) -> str:
    return (
        f"measured {measured} (vs the 1e-3 the analytic adapters assert): {detail}; {_TRIWEIGHT}"
    )


_UNMEASURED = f"not measured separately, same adapter module as dale2014; {_TRIWEIGHT}"
UNWIRED: dict[str, str] = {
    "dale2014": _missed("1.6e-2", "30 draws, alpha_dale free, z = 0.05, 80-500 um"),
    "draine_li2007": _missed("4.3e-1", "30 draws, umin and qpah free, z = 0.05"),
    "dl07": "same adapter as draine_li2007",
    "draine_li2014": _UNMEASURED,
    "astrodust": _UNMEASURED,
    "themis": _UNMEASURED,
    "bosa": _UNMEASURED,
    "casey2012": (
        "measured 2.7e-3 (100 seeded draws, all four axes free, z = 0.05, 100-500 um, worst at "
        "alpha_mir ~ 1.15, T ~ 30 K) vs the adapter's 1e-3 contract; the 4-D lookup is also "
        "2x slower than the per-call integral (2.6 vs 1.25 ms per jitted predict)"
    ),
}


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


def _tophat(center_aa, frac_width=0.18, n=48):
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(center_aa * (1.0 - frac_width), center_aa * (1.0 + frac_width), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center_aa)}")


#: 100 um to 500 um: the bands the thermal dust emission dominates.
_FAR_IR_AA = (1.0e5, 1.6e5, 2.5e5, 5.0e5)


@dataclasses.dataclass(frozen=True)
class Family:
    """One family's builder and its stated accuracy."""

    key: str
    build: Callable[..., object]
    rtol: float
    stated: str


#: Free shape parameters per analytic dust model: the declared priors of the table axes.
_DUST_SHAPE_PRIORS = {
    "modified_blackbody": {"T": (20.0, 80.0), "beta_ir": (1.0, 2.5)},
    "graybody": {"T": (20.0, 80.0), "beta_ir": (1.0, 2.5), "lambda_0_um": (50.0, 500.0)},
    "casey2012": {
        "T": (20.0, 80.0),
        "beta_ir": (1.0, 2.5),
        "alpha_mir": (1.0, 3.0),
        "lambda_0_um": (50.0, 500.0),
    },
}


def _dust_model(model_key: str, approx, *, free: bool, ssp=None):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    emission = {"type": model_key, "all_params": Fixed(DEFAULT)}
    if free:
        for name, (lo, hi) in _DUST_SHAPE_PRIORS[model_key].items():
            emission[name] = Uniform(lo, hi)
    return SEDModel.build(
        ssp_data=ssp if ssp is not None else _ssp(),
        observation=Observation(
            photometry=Photometry(filters=tuple(_tophat(c) for c in _FAR_IR_AA))
        ),
        redshift=Fixed(0.05),
        approx=approx,
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.5,
        },
        dust_emission=emission,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
    )


# Accuracy: the analytic dust adapters assert 1e-3 against the exact closure at random points
# inside the declared priors (dust_analytic_precompute.build_lookup). Measured here through the
# model at z = 0.05 in the 100-500 um bands, max over 100 seeded draws with every table axis
# free: modified_blackbody 3.5e-4, graybody 3.6e-4, casey2012 2.7e-3 (see UNWIRED).
FAMILIES: dict[str, Family] = {
    key: Family(
        key,
        lambda approx, *, free, _key=key: _dust_model(_key, approx, free=free),
        rtol=1e-3,
        stated=stated,
    )
    for key, stated in (
        ("modified_blackbody", "3.5e-4"),
        ("graybody", "2.9e-4"),
        ("casey2012", "8.0e-4"),
    )
}


def _wired():
    return [k for k in FAMILIES if k not in UNWIRED]


def _shape_table(model) -> dict | None:
    data = model._template_data_for_jit() or {}
    return data.get("dust_ir", {}).get("emission_shape_table")


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
    real = photometry.lnu_filter_integral_batch

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(photometry, "lnu_filter_integral_batch", spy)
    return calls


@pytest.mark.parametrize("key", _wired())
def test_free_shape_builds_registry_lookup_and_predict_does_not_integrate(
    key, resolve_spy, integral_spy
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FAMILIES[key].build(_wavepre(), free=True)
    assert key in resolve_spy, f"WavePrecomp never asked the registry for {key}"
    assert _shape_table(model) is not None
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    before = len(integral_spy)
    jax.block_until_ready(model.predict_photometry(params))
    assert len(integral_spy) == before, "predict integrated the filters per call"


@pytest.mark.parametrize("key", _wired())
def test_free_shape_matches_exact_and_has_finite_gradients(key):
    family = FAMILIES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_model = family.build(_wavepre(), free=True)
        exact_model = family.build(None, free=True)
    assert _shape_table(table_model) is not None, "the registry table was not engaged"
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


def test_unwired_keys_are_registered_adapters():
    from tengri.forward.precompute import registry

    assert set(UNWIRED) <= set(registry.registered_components())


@pytest.mark.parametrize("key", sorted(set(UNWIRED) & set(FAMILIES)))
def test_unwired_family_is_not_engaged(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FAMILIES[key].build(_wavepre(), free=True)
    assert _shape_table(model) is None, f"{key} is recorded as unwired: {UNWIRED[key]}"


@pytest.mark.parametrize("key", _wired())
def test_fixed_shape_keeps_exact_table_and_skips_registry(key, resolve_spy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FAMILIES[key].build(_wavepre(), free=False)
    data = model._template_data_for_jit() or {}
    assert "emission_band_response" in data.get("dust_ir", {})
    assert "emission_shape_table" not in data.get("dust_ir", {})
    assert key not in resolve_spy


def _wavepre():
    from tengri import WavePrecomp

    return WavePrecomp()


def test_threaded_term_responses_follow_the_discovered_emitter_list(
    synthetic_ssp, synthetic_tophat_obs, monkeypatch
):
    """The emitters whose term response is threaded are the ones the build loop discovered."""
    from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp
    from tengri.forward import sed_model

    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "other_params": Fixed(DEFAULT),
        },
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}, "all_params": Fixed(DEFAULT)},
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    chain = model._cached_component_chain
    discovered = sed_model._chain_implements_emission_terms(chain)
    threaded = model._template_data_for_jit() or {}
    assert sorted(k for k in ("radio", "xray") if "term_band_response" in threaded.get(k, {})) == (
        discovered
    )

    monkeypatch.setattr(sed_model, "_chain_implements_emission_terms", lambda chain: ["radio"])
    fresh = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "other_params": Fixed(DEFAULT),
        },
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}, "all_params": Fixed(DEFAULT)},
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    data = fresh._template_data_for_jit() or {}
    assert "term_band_response" in data.get("radio", {})
    assert "xray" not in data


# ── Radio term families: shape-only tables (#2324) ──────────────────────────────────────────
#: registry key -> (free shape parameter, its prior, where the parameter is declared).
RADIO_SHAPE_PRIORS: dict[str, tuple[str, tuple[float, float]]] = {
    "radio_synchrotron": ("radio_alpha_sf", (0.5, 1.0)),
    "radio_freefree": ("radio_alpha_ff", (-0.2, 0.0)),
    "radio_agn_jet": ("radio_alpha_agn", (0.4, 1.2)),
}

#: Radio bands from 3 mm to 30 cm: the bands the radio terms dominate.
_RADIO_AA = (3.0e7, 1.0e8, 3.0e8)


def _radio_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 12.0, 1600)
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


def _radio_model(key: str, approx, *, free: bool):
    """A model whose radio ``key`` term is live: a radio-loud AGN supplies ``L_agn_bol``."""
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    name, (lo, hi) = RADIO_SHAPE_PRIORS[key]
    agn_radio = {"type": "powerlaw", "radio_loudness": Fixed(8.0)}
    radio = {"sf": {"type": "bell2003"}, "agn": agn_radio, "all_params": Fixed(DEFAULT)}
    if free:
        if name.startswith("radio_alpha_agn"):
            agn_radio[name] = Uniform(lo, hi)
        else:
            radio[name] = Uniform(lo, hi)
    return SEDModel.build(
        ssp_data=_radio_ssp(),
        observation=Observation(
            photometry=Photometry(filters=tuple(_tophat(c) for c in _RADIO_AA))
        ),
        redshift=Fixed(0.05),
        approx=approx,
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.5,
        },
        dust_emission={"type": "modified_blackbody", "all_params": Fixed(DEFAULT)},
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        agn={
            "type": "composable",
            "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
            "blr": {"type": "analytic", "all_params": Fixed(DEFAULT), "agn_log_lbol": 13.0},
            "nlr": {"type": "none"},
            "feii": {"type": "none"},
            "atten": {"type": "none"},
        },
        radio=radio,
    )


def _term_shape_table(model) -> dict | None:
    data = model._template_data_for_jit() or {}
    return data.get("radio", {}).get("term_shape_table")


def _radio_term_photometry(model, params) -> np.ndarray:
    """The radio emission photometry alone, eager, from the component's derived output."""
    state = model.predict_state(params, template_data=model._template_data_for_jit())
    return np.asarray(state.derived["radio_phot_lnu_precomp"])


def _dense_radio_term_photometry(model, params, monkeypatch) -> np.ndarray:
    """The same photometry with the shape-table branch off, so the term is integrated per call.

    Call this on a model that has not yet predicted: a compiled predict keeps whichever branch
    it traced first.
    """
    from tengri.components.radio import component as radio_component

    monkeypatch.setattr(radio_component, "_term_shape_response", lambda *a, **k: None)
    return _radio_term_photometry(model, params)


@pytest.mark.parametrize("key", sorted(RADIO_SHAPE_PRIORS))
def test_radio_free_shape_builds_term_table_and_predict_does_not_integrate(
    key, resolve_spy, integral_spy
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _radio_model(key, _wavepre(), free=True)
    assert key in resolve_spy, f"WavePrecomp never asked the registry for {key}"
    assert _term_shape_table(model) is not None
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    before = len(integral_spy)
    jax.block_until_ready(model.predict_photometry(params))
    assert len(integral_spy) == before, "predict integrated the filters per call"


@pytest.mark.parametrize("key", sorted(RADIO_SHAPE_PRIORS))
def test_radio_free_shape_term_matches_dense_precomp_within_1e3(key, integral_spy, monkeypatch):
    """The radio term from its shape table against the same model's per-call integral."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_model = _radio_model(key, _wavepre(), free=True)
        dense_model = _radio_model(key, _wavepre(), free=True)
    assert _term_shape_table(table_model) is not None, "the registry table was not engaged"
    for seed in (11, 12, 13):
        params = dict(table_model.spec.sample(jax.random.PRNGKey(seed)))
        got = _radio_term_photometry(table_model, params)
        before = len(integral_spy)
        want = _dense_radio_term_photometry(dense_model, params, monkeypatch)
        assert len(integral_spy) > before, "the dense reference did not integrate per call"
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("key", sorted(RADIO_SHAPE_PRIORS))
def test_radio_free_shape_term_has_finite_nonzero_gradients(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _radio_model(key, _wavepre(), free=True)
    params = dict(model.spec.sample(jax.random.PRNGKey(11)))
    free = {k: v for k, v in params.items() if k in model.spec.free_params}
    fixed = {k: v for k, v in params.items() if k not in free}

    def loss(theta):
        state = model.predict_state(
            {**fixed, **theta}, template_data=model._template_data_for_jit()
        )
        return jnp.sum(jnp.log(state.derived["radio_phot_lnu_precomp"]))

    grads = jax.grad(loss)(free)
    assert all(bool(jnp.isfinite(g)) for g in grads.values())
    assert any(float(g) != 0.0 for g in grads.values())


@pytest.mark.parametrize("key", sorted(RADIO_SHAPE_PRIORS))
def test_radio_fixed_shape_keeps_exact_table_and_skips_registry(key, resolve_spy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _radio_model(key, _wavepre(), free=False)
    data = model._template_data_for_jit() or {}
    assert "term_band_response" in data.get("radio", {})
    assert "term_shape_table" not in data.get("radio", {})
    assert key not in resolve_spy


# ── X-ray term families: shape-only tables (#2324) ──────────────────────────────────────────
#: registry key -> the free shape parameters of its family, each with its prior.
XRAY_SHAPE_PRIORS: dict[str, dict[str, tuple[float, float]]] = {
    "xray_xrb": {"xray_gamma_hmxb": (1.7, 2.3), "xray_gamma_lmxb": (1.4, 1.9)},
    "xray_corona": {"xray_gamma_agn": (1.5, 2.3)},
    "xray_corona_lopez24": {"xray_gamma_agn": (1.5, 2.3)},
}

#: X-ray bands at 2 to 20 Angstrom (about 0.6 to 6 keV): the bands the X-ray terms dominate.
_XRAY_AA = (2.0, 6.0, 20.0)


def _xray_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(-1.0, 12.0, 2000)
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


def _agn_composable() -> dict:
    """A composable AGN supplying the corona's disc luminosity."""
    from tengri import DEFAULT, Fixed

    return {
        "type": "composable",
        "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
        "blr": {"type": "analytic", "all_params": Fixed(DEFAULT), "agn_log_lbol": 13.0},
        "nlr": {"type": "none"},
        "feii": {"type": "none"},
        "atten": {"type": "none"},
    }


def _xray_model(key: str, approx, *, free: bool):
    """A model whose X-ray term family ``key`` is live (the corona families carry an AGN)."""
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    xray_type = "lopez24" if key == "xray_corona_lopez24" else "yang20"
    xray = {"type": xray_type, "all_params": Fixed(DEFAULT)}
    if free:
        for name, (lo, hi) in XRAY_SHAPE_PRIORS[key].items():
            xray[name] = Uniform(lo, hi)
    return SEDModel.build(
        ssp_data=_xray_ssp(),
        observation=Observation(
            photometry=Photometry(filters=tuple(_tophat(c) for c in _XRAY_AA))
        ),
        redshift=Fixed(0.05),
        approx=approx,
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.5,
        },
        dust_emission={"type": "modified_blackbody", "all_params": Fixed(DEFAULT)},
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        # The XRB family has no AGN: its terms then dominate, so its gradients are measurable.
        **({} if key == "xray_xrb" else {"agn": _agn_composable()}),
        xray=xray,
    )


def _xray_term_table(model) -> dict | None:
    data = model._template_data_for_jit() or {}
    return data.get("xray", {}).get("term_shape_table")


def _xray_photometry(model, params) -> np.ndarray:
    """The X-ray emission photometry alone, eager, from the component's derived output."""
    state = model.predict_state(params, template_data=model._template_data_for_jit())
    return np.asarray(state.derived["xray_phot_lnu_precomp"])


def _dense_xray_photometry(model, params, monkeypatch) -> np.ndarray:
    """The same photometry with the shape-table branch off. Call before the model has predicted."""
    from tengri.components.xray import component as xray_component

    monkeypatch.setattr(xray_component, "_term_shape_response", lambda *a, **k: None)
    return _xray_photometry(model, params)


@pytest.mark.parametrize("key", sorted(XRAY_SHAPE_PRIORS))
def test_xray_free_shape_builds_term_table_and_predict_does_not_integrate(
    key, resolve_spy, integral_spy
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _xray_model(key, _wavepre(), free=True)
    assert key in resolve_spy, f"WavePrecomp never asked the registry for {key}"
    assert _xray_term_table(model) is not None
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    before = len(integral_spy)
    jax.block_until_ready(model.predict_photometry(params))
    assert len(integral_spy) == before, "predict integrated the filters per call"


@pytest.mark.parametrize("key", sorted(XRAY_SHAPE_PRIORS))
def test_xray_free_shape_term_matches_dense_precomp_within_1e3(key, integral_spy, monkeypatch):
    """The X-ray term from its shape tables against the same model's per-call integral."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_model = _xray_model(key, _wavepre(), free=True)
        dense_model = _xray_model(key, _wavepre(), free=True)
    assert _xray_term_table(table_model) is not None, "the registry table was not engaged"
    for seed in (11, 12, 13):
        params = dict(table_model.spec.sample(jax.random.PRNGKey(seed)))
        got = _xray_photometry(table_model, params)
        before = len(integral_spy)
        want = _dense_xray_photometry(dense_model, params, monkeypatch)
        assert len(integral_spy) > before, "the dense reference did not integrate per call"
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("key", sorted(XRAY_SHAPE_PRIORS))
def test_xray_free_shape_term_has_finite_nonzero_gradients(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _xray_model(key, _wavepre(), free=True)
    params = dict(model.spec.sample(jax.random.PRNGKey(11)))
    free = {k: v for k, v in params.items() if k in model.spec.free_params}
    fixed = {k: v for k, v in params.items() if k not in free}

    def loss(theta):
        state = model.predict_state(
            {**fixed, **theta}, template_data=model._template_data_for_jit()
        )
        return jnp.sum(jnp.log(state.derived["xray_phot_lnu_precomp"]))

    grads = jax.grad(loss)(free)
    assert all(bool(jnp.isfinite(g)) for g in grads.values())
    assert any(float(g) != 0.0 for g in grads.values())


@pytest.mark.parametrize("key", sorted(XRAY_SHAPE_PRIORS))
def test_xray_fixed_shape_keeps_exact_table_and_skips_registry(key, resolve_spy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _xray_model(key, _wavepre(), free=False)
    data = model._template_data_for_jit() or {}
    assert "term_band_response" in data.get("xray", {})
    assert "term_shape_table" not in data.get("xray", {})
    assert key not in resolve_spy

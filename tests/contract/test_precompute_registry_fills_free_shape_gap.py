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
    "dale2014": _missed("7.0e-2", "30 draws, alpha_dale free, z = 0.05, 8-500 um"),
    "draine_li2007": _missed("4.3e-1", "30 draws, umin and qpah free, z = 0.05"),
    "dl07": "same adapter as draine_li2007",
    "draine_li2014": _UNMEASURED,
    "astrodust": _UNMEASURED,
    "themis": _UNMEASURED,
    "bosa": _UNMEASURED,
    "radio_synchrotron": _missed("2.3e-2", "50 draws of alpha_sf, 0.3-10 GHz bands, z = 0.05"),
    "radio_freefree": _missed("1.6e-2", "50 draws of alpha_ff"),
    "radio_agn_jet": _missed("7.5e-2", "50 draws of alpha_agn"),
    "xray_xrb": _missed("4.0e-2", "20 draws of both photon indices, 0.5-10 keV"),
    "xray_corona_lopez24": _missed("5.5e-2", "20 draws, same bands"),
    "xray_corona": f"not measured like for like (alpha_ox axis is converted); {_TRIWEIGHT}",
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

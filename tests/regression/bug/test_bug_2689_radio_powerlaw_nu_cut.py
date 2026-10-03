"""
Regression test for issue #2689: the power-law AGN radio jet honors radio_log_nu_cut.

The jet is ``L_5GHz (nu / 5 GHz)^-alpha exp(-nu / nu_cut)`` with
``nu_cut = 10^radio_log_nu_cut`` (Martinez-Ramirez et al. 2024 Eq. 2 cutoff;
AGNfitter-rx SPL jet).  Covered here:

* the cutoff is threaded to the kernel (absolute match to the formula);
* the default is bit-identical to an explicit 13.0;
* the cutoff has a correct autodiff gradient, in float64 and in float32;
* every AGN radio key is accepted by a model iff that model reads it;
* ``all_params: FREE`` on the power-law jet frees the cutoff.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, load_ssp
from tengri.components.radio import radio
from tengri.components.radio._params import PARAMS
from tengri.components.radio.component import (
    AGN_RADIO_MODELS,
    RadioSEDComponent,
    RadioSEDComponentConfig,
)
from tengri.parameters.groups import _RADIO_AGN_PARAM_NAMES
from tengri.protocols.component import declared_default
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

_NU_REF = radio._NU_REF_AGN_HZ
_GRID_GHZ = (1.0, 5.0, 30.0, 100.0, 300.0)
#: Brings L_nu [erg/s/Hz] to O(1) so a float32 reverse pass (L_nu * nu ~ 1e41) cannot overflow.
_SCALE = 1e-28


@pytest.fixture(scope="module")
def ssp():
    return load_ssp()


def _build(ssp, agn_radio):
    """Radio-only model; ``agn_radio`` is the ``radio['agn']`` dict."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"all_params": Fixed(DEFAULT), "agn": agn_radio},
        redshift=Fixed(0.0),
    )


@pytest.fixture(scope="module")
def sed_sf(ssp):
    """Radio SED with the AGN jet switched off (star formation + free-free)."""
    return np.asarray(_build(ssp, {"type": "none"}).predict_state({}).derived["sed_radio"])


def _node(nu, ghz):
    return int(np.argmin(np.abs(nu - ghz * 1e9)))


def _l5ghz(state, radio_loudness):
    """The jet's 5 GHz luminosity from the published AGN luminosities."""
    l_b = float(state.derived["L_4400_intrinsic"])
    if not l_b > 0.0:
        l_b = float(state.derived["L_agn_bol"]) / (5.15 * 6.818e14)
    return l_b * 10.0**radio_loudness


@pytest.mark.parametrize("cut", [11.0, 13.0, 40.0])
def test_agn_jet_matches_formula_at_grid_nodes(ssp, sed_sf, cut):
    """AGN term == L_5GHz (nu/5GHz)^-alpha exp(-nu/10^cut) at exact grid nodes, rtol 1e-6."""
    m = _build(ssp, {"radio_log_nu_cut": Fixed(cut), "all_params": Fixed(DEFAULT)})
    state = m.predict_state({})
    nu = C_AA / np.asarray(state.wave)
    term = np.asarray(state.derived["sed_radio"]) - sed_sf
    alpha = declared_default(PARAMS, "radio_alpha_agn")
    l5 = _l5ghz(state, declared_default(PARAMS, "radio_loudness"))
    assert l5 > 0.0
    for ghz in _GRID_GHZ:
        i = _node(nu, ghz)
        expect = l5 * (nu[i] / _NU_REF) ** (-alpha) * np.exp(-nu[i] / 10.0**cut)
        np.testing.assert_allclose(term[i], expect, rtol=1e-6)


def test_default_is_bit_identical_to_cut_13(ssp):
    """Leaving the cutoff out of the dict is bit-identical to writing Fixed(13.0)."""
    default = _build(ssp, {"all_params": Fixed(DEFAULT)}).predict_state({})
    explicit = _build(ssp, {"radio_log_nu_cut": Fixed(13.0), "all_params": Fixed(DEFAULT)})
    np.testing.assert_array_equal(
        np.asarray(default.derived["sed_radio"]),
        np.asarray(explicit.predict_state({}).derived["sed_radio"]),
    )
    assert declared_default(PARAMS, "radio_log_nu_cut") == 13.0


def test_wildcard_frees_cutoff_on_powerlaw(ssp):
    """``all_params: FREE`` on the default (power-law) jet frees the cutoff, not DPL knobs."""
    m = _build(ssp, {"all_params": FREE})
    free = set(m.spec.free_params)
    assert "radio_log_nu_cut" in free
    assert not free & {"radio_alpha_thin", "radio_alpha_thick", "radio_log_nu_t"}


def test_nu_cut_gradient_float64(ssp, sed_sf):
    """jax.grad w.r.t. a FREE cutoff of the 300 GHz jet == L (nu/nu_cut) ln10 (rtol 1e-4)."""
    m = _build(ssp, {"radio_log_nu_cut": FREE, "all_params": Fixed(DEFAULT)})
    assert "radio_log_nu_cut" in m.spec.free_params
    state0 = m.predict_state({"radio_log_nu_cut": 13.0})
    nu = C_AA / np.asarray(state0.wave)
    i = _node(nu, 300.0)

    def jet(log_cut):
        return m.predict_state({"radio_log_nu_cut": log_cut}).derived["sed_radio"][i] - sed_sf[i]

    cut0 = 13.0
    grad_ana = float(jet(jnp.asarray(cut0))) * (nu[i] / 10.0**cut0) * np.log(10.0)
    assert np.isfinite(grad_ana) and grad_ana != 0.0
    np.testing.assert_allclose(float(jax.grad(jet)(jnp.asarray(cut0))), grad_ana, rtol=1e-4)
    eps = 1e-4  # forward-model cross-check
    fd = (float(jet(jnp.asarray(cut0 + eps))) - float(jet(jnp.asarray(cut0 - eps)))) / (2 * eps)
    np.testing.assert_allclose(fd, grad_ana, rtol=1e-3)


def _kernel(fn, wave, cut):
    return fn(wave, 0.0, log_nu_cut=cut, log_L_agn_bol=jnp.asarray(45.0, wave.dtype))


@pytest.mark.parametrize("fn", [radio.radio_agn, radio.radio_agn_dpl], ids=["powerlaw", "dpl"])
def test_nu_cut_float32_value_and_gradient(fn):
    """Float32: no overflow of 10^cut, value equals float64, gradient finite (0 at cut=40)."""
    wave64 = jnp.asarray(np.geomspace(3e5, 3e8, 40))  # 1 THz .. 1 GHz in Angstrom
    ref = {c: np.asarray(_kernel(fn, wave64, jnp.asarray(c))) for c in (11.0, 13.0, 40.0)}
    with jax.enable_x64(False):
        wave = jnp.asarray(np.geomspace(3e5, 3e8, 40), dtype=jnp.float32)
        assert wave.dtype == jnp.float32
        for c in (11.0, 13.0, 40.0):
            cut = jnp.asarray(c, dtype=jnp.float32)
            val = _kernel(fn, wave, cut)
            assert val.dtype == jnp.float32
            assert np.all(np.isfinite(np.asarray(val)))
            np.testing.assert_allclose(
                np.asarray(val), ref[c], rtol=1e-5, atol=1e-9 * ref[c].max()
            )
            g = jax.grad(lambda x: jnp.sum(_kernel(fn, wave, x) * _SCALE))(cut)
            assert g.dtype == jnp.float32
            assert np.isfinite(float(g))
        g40 = jax.grad(lambda x: jnp.sum(_kernel(fn, wave, x) * _SCALE))(
            jnp.asarray(40.0, jnp.float32)
        )
        scale = float(jnp.sum(_kernel(fn, wave, jnp.asarray(13.0, jnp.float32)) * _SCALE))
        assert abs(float(g40)) < 1e-6 * scale


# ── the issue's sweep: every AGN radio key against every AGN radio model ──────────────


def _perturbed(name):
    """A value distinct from the declared default and inside the physical range."""
    return declared_default(PARAMS, name) + 0.3


def _read_set(model, wave):
    """Names of the declared radio_* parameters the model's AGN jet actually reads."""
    comp = RadioSEDComponent(RadioSEDComponentConfig(agn_radio_model=model))
    base = {d.name: jnp.asarray(declared_default(PARAMS, d.name)) for d in PARAMS}
    base["redshift"] = jnp.asarray(0.0)
    kw = dict(
        L_ir=jnp.asarray(0.0),
        L_agn_bol=jnp.asarray(1e45),
        L_4400_intrinsic=jnp.asarray(1e30),
        log_mstar=jnp.asarray(10.0),
    )
    ref = np.asarray(comp.emission_terms(base, wave, **kw)["agn"])
    changed = set()
    for d in PARAMS:
        probe = {**base, d.name: jnp.asarray(_perturbed(d.name))}
        if not np.array_equal(np.asarray(comp.emission_terms(probe, wave, **kw)["agn"]), ref):
            changed.add(d.name)
    return changed


@pytest.mark.parametrize("model", [m for m in AGN_RADIO_MODELS if m != "none"])
def test_agn_radio_key_accepted_iff_model_reads_it(ssp, model):
    """Each declared radio.agn key builds on a model iff that model's jet reads it."""
    wave = jnp.asarray(np.geomspace(3e5, 3e8, 40))
    reads = _read_set(model, wave)
    assert "radio_loudness" in reads  # the probe is live
    # What the model reads must be routable (nothing read is unreachable) ...
    assert reads <= _RADIO_AGN_PARAM_NAMES
    # ... and every routable key is accepted exactly when read.
    for name in sorted(_RADIO_AGN_PARAM_NAMES):
        agn = {"type": model, name: Fixed(_perturbed(name)), "all_params": Fixed(DEFAULT)}
        if name in reads:
            _build(ssp, agn)
        else:
            with pytest.raises(ValueError, match=name.removeprefix("radio_")) as exc:
                _build(ssp, agn)
            assert "not read by" in str(exc.value) and repr(model) in str(exc.value)


def test_refusal_names_the_models_that_read_the_key(ssp):
    """A refused key's message says which model does read it."""
    with pytest.raises(ValueError) as exc:
        _build(ssp, {"type": "powerlaw", "radio_alpha_thin": Fixed(-1.0)})
    assert "Models that do read the key: dpl" in str(exc.value)


def test_typo_key_gets_unknown_key_error(ssp):
    """A key no AGN radio model declares is an ordinary unknown key, not a 'read by' refusal."""
    with pytest.raises(ValueError) as exc:
        _build(ssp, {"type": "powerlaw", "radio_log_nu_cutt": Fixed(12.0)})
    msg = str(exc.value)
    assert "radio_log_nu_cutt" in msg and "not read by" not in msg
    assert "read by:" not in msg


def test_signature_default_matches_declaration():
    """The kernel's log_nu_cut default is the declared default (single source)."""
    for fn in (radio.radio_agn, radio.radio_agn_dpl):
        assert inspect.signature(fn).parameters["log_nu_cut"].default == declared_default(
            PARAMS, "radio_log_nu_cut"
        )

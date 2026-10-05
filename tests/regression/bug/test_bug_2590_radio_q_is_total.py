# SPDX-License-Identifier: BSD-3-Clause
"""Bell (2003)'s q_IR is a TOTAL 1.4 GHz calibration; thermal emission is counted once (#2590).

Bell (2003, ApJ 586, 794) Eq. 1 defines q on the *total* 1.4 GHz flux and Sect. 4 gives
that total as (n + 0.1) eta psi: about 90 % non-thermal and 10 % thermal. The default
``bell2003`` block put the whole total in the synchrotron term and added the separately
normalized Murphy et al. (2011) free-free term (+13.4 % at 1.4 GHz, +265 % at 100 GHz); with
a free-free-bearing nebular backend the nebular continuum sat on top of the full-q
synchrotron the same way.

Three spellings, every expectation written out here from the literature:

* ``freefree`` unset (default): q is the TOTAL. Synchrotron = (1 - f_th) x total, with
  f_th the Murphy share at 1.4 GHz, whichever component supplies the thermal term (the
  radio block's, or the nebular continuum under the #2346 auto-rule).
* ``freefree=False`` explicit: q calibrates the non-thermal term alone (CIGALE's
  convention): full-q synchrotron, nothing subtracted, no radio-block thermal term.
* ``freefree=True`` explicit: as unset, with the radio block's own thermal term.

total at nu_ref = L_IR / (3.75e12 10^q)                         (Bell Eq. 1)
thermal term    = (1/4.6e-28) (T_e/1e4)^0.45 (nu/GHz)^alpha_ff 3.88e-44 L_IR  (Murphy Eqs. 11, 4)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.radio._params import PARAMS as _RADIO_PARAMS
from tengri.components.radio.component import (
    SF_RADIO_MODELS,
    RadioSEDComponent,
    RadioSEDComponentConfig,
)
from tengri.config.exceptions import ConfigError
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18  # Angstrom/s
_L_IR = 1.0e44  # erg/s
_FREQS = (1.4e9, 5.0e9, 3.0e10, 1.0e11)
_FREQS5 = (1.5e8, 1.4e9, 5.0e9, 3.0e10, 1.0e11)  # the AGNfitter-rX comparison set
_Q_TYPES = tuple(m for m in SF_RADIO_MODELS if m != "none")

#: spelling -> (include_freefree, q_is_total) as the factory resolves it
_SPELLINGS = {
    "unset": (True, True),
    "unset_nebular": (False, True),  # #2346 auto-rule: thermal term comes from the nebular
    "false": (False, False),  # explicit freefree=False: q is non-thermal
}


def _radio_params() -> dict[str, float]:
    out = {d.name: declared_default(_RADIO_PARAMS, d.name) for d in _RADIO_PARAMS}
    return out | {"redshift": 0.0}


def _terms(sf, spelling, freqs, *, params=None, dtype=jnp.float64, **inputs):
    """Radio ``sf`` / ``ff`` terms at exactly ``freqs`` [Hz] for one galaxy."""
    ff, q_total = _SPELLINGS[spelling]
    cfg = RadioSEDComponentConfig(
        sfr_mode=sf, include_freefree=ff, q_is_total=q_total, agn_radio_model="none"
    )
    p = {k: jnp.asarray(v, dtype) for k, v in (params or _radio_params()).items()}
    wave = jnp.asarray(_C_AA / np.asarray(freqs, dtype=np.float64), dtype)
    kw = {
        "L_ir": jnp.asarray(_L_IR, jnp.float64),
        "L_agn_bol": jnp.asarray(0.0, dtype),
        "L_4400_intrinsic": jnp.asarray(0.0, dtype),
        "log_mstar": jnp.asarray(10.0, dtype),
    }
    kw.update(inputs)
    return RadioSEDComponent(config=cfg).emission_terms(p, wave, **kw)


def _sf_ff(sf, spelling, freqs, **kw):
    t = _terms(sf, spelling, freqs, **kw)
    return np.asarray(t["sf"], np.float64), np.asarray(t["ff"], np.float64)


# ---- literature formulas, written out in the test ---------------------------------


def _bell_total_ref(q, l_ir=_L_IR):
    return l_ir / (3.75e12 * 10.0**q)


def _murphy_ff(nu, l_ir=_L_IR, t_e=1.0e4, alpha_ff=-0.1):
    sfr = 3.88e-44 * l_ir  # Msun/yr
    return (1.0 / 4.6e-28) * (t_e / 1.0e4) ** 0.45 * (np.asarray(nu) / 1.0e9) ** alpha_ff * sfr


def _expected_sf_ff(freqs, spelling, q=2.64, alpha=0.8, t_e=1.0e4, alpha_ff=-0.1):
    nu = np.asarray(freqs, np.float64)
    total_ref = _bell_total_ref(q)
    subtract = float(_murphy_ff(1.4e9, t_e=t_e, alpha_ff=alpha_ff)) if _SPELLINGS[spelling][1] else 0.0
    sf = (total_ref - subtract) * (nu / 1.4e9) ** (-alpha)
    ff = _murphy_ff(nu, t_e=t_e, alpha_ff=alpha_ff) if _SPELLINGS[spelling][0] else 0.0 * nu
    return sf, ff


def _f_th(q=2.64, t_e=1.0e4, alpha_ff=-0.1):
    return float(_murphy_ff(1.4e9, t_e=t_e, alpha_ff=alpha_ff)) / _bell_total_ref(q)


# ---- component level: the three spellings -----------------------------------------


@pytest.mark.parametrize("spelling", list(_SPELLINGS))
@pytest.mark.parametrize("q", [2.3, 2.64, 3.0])
@pytest.mark.parametrize("alpha", [0.7, 0.8, 1.0])
def test_terms_follow_the_literature_shapes(spelling, q, alpha):
    params = _radio_params() | {"radio_q_ir": q, "radio_alpha_sf": alpha}
    sf, ff = _sf_ff("bell2003", spelling, _FREQS5, params=params)
    want_sf, want_ff = _expected_sf_ff(_FREQS5, spelling, q, alpha)
    np.testing.assert_allclose(sf, want_sf, rtol=1e-9)
    np.testing.assert_allclose(ff, want_ff, rtol=1e-9)


def test_total_at_1p4ghz_is_the_calibration_with_the_thermal_term_in_the_radio_block():
    sf, ff = _sf_ff("bell2003", "unset", [1.4e9])
    assert sf[0] + ff[0] == pytest.approx(_bell_total_ref(2.64), rel=1e-3), (
        f"L(1.4 GHz) / Bell total = {(sf[0] + ff[0]) / _bell_total_ref(2.64):.4f}"
    )


def test_explicit_false_is_the_nonthermal_calibration():
    """Full-q synchrotron, nothing subtracted, no thermal term (CIGALE's convention)."""
    sf, ff = _sf_ff("bell2003", "false", _FREQS)
    np.testing.assert_allclose(sf, _bell_total_ref(2.64) * (np.asarray(_FREQS) / 1.4e9) ** -0.8)
    assert np.all(ff == 0.0)


def test_nebular_spelling_takes_a_fixed_share_off_the_synchrotron():
    """Auto-rule (thermal term from the nebular): sync = (1 - f_th) total; no radio thermal."""
    sf, ff = _sf_ff("bell2003", "unset_nebular", [1.4e9])
    assert np.all(ff == 0.0)
    assert sf[0] == pytest.approx((1.0 - _f_th()) * _bell_total_ref(2.64), rel=1e-9)


def test_nebular_spelling_keeps_the_synchrotron_nonnegative_when_the_dust_is_gone():
    """The share scales with L_IR, so a dust-poor galaxy is not driven negative."""
    for l_ir in (1.0e38, 1.0e44):
        sf, _ = _sf_ff("bell2003", "unset_nebular", [1.4e9], L_ir=jnp.asarray(l_ir))
        assert sf[0] == pytest.approx((1.0 - _f_th()) * _bell_total_ref(2.64, l_ir), rel=1e-9)


@pytest.mark.parametrize("alpha_ff", [-0.3, -0.1, 0.0])
@pytest.mark.parametrize("t_e", [5.0e3, 1.5e4])
def test_thermal_share_follows_alpha_ff_and_t_e(alpha_ff, t_e):
    params = _radio_params() | {"radio_alpha_ff": alpha_ff, "radio_T_e": t_e}
    sf, ff = _sf_ff("bell2003", "unset", [1.4e9, 5.0e9], params=params)
    want_sf, want_ff = _expected_sf_ff([1.4e9, 5.0e9], "unset", t_e=t_e, alpha_ff=alpha_ff)
    np.testing.assert_allclose(sf, want_sf, rtol=1e-9)
    np.testing.assert_allclose(ff, want_ff, rtol=1e-9)
    assert sf[0] + ff[0] == pytest.approx(_bell_total_ref(2.64), rel=1e-9)


def test_issue_table_rederived_from_the_two_spectral_shapes():
    """L_nu(on) / L_nu(off) at 1.4, 5, 30, 100 GHz (was 1.134 / 1.326 / 2.141 / 3.650)."""
    nu = np.asarray(_FREQS)
    f = _f_th()
    off = _bell_total_ref(2.64) * (nu / 1.4e9) ** -0.8
    on = _bell_total_ref(2.64) * (1.0 - f) * (nu / 1.4e9) ** -0.8 + _murphy_ff(nu)
    sf_on, ff_on = _sf_ff("bell2003", "unset", _FREQS)
    sf_off, _ = _sf_ff("bell2003", "false", _FREQS)
    np.testing.assert_allclose((sf_on + ff_on) / sf_off, on / off, rtol=1e-9)
    assert ((sf_on + ff_on) / sf_off)[0] == pytest.approx(1.0, abs=1e-3)


# ---- bell2003_split: the AGNfitter-rX construction, unchanged ------------------------


def _split_cfg(**kw):
    return RadioSEDComponentConfig(sfr_mode="bell2003_split", agn_radio_model="none", **kw)


def test_bell2003_split_is_the_agnfitter_construction_at_five_frequencies():
    """Bell total split 90 % / 10 % into alpha = 0.75 and 0.10 power laws (Martinez-Ramirez+2024)."""
    comp = RadioSEDComponent(config=_split_cfg())
    p = {k: jnp.asarray(v) for k, v in _radio_params().items()}
    wave = jnp.asarray(_C_AA / np.asarray(_FREQS5))
    t = comp.emission_terms(
        p,
        wave,
        L_ir=jnp.asarray(_L_IR),
        L_agn_bol=jnp.asarray(0.0),
        L_4400_intrinsic=jnp.asarray(0.0),
        log_mstar=jnp.asarray(10.0),
    )
    x = np.asarray(_FREQS5) / 1.4e9
    want = _bell_total_ref(2.64) * (0.9 * x**-0.75 + 0.1 * x**-0.10)
    np.testing.assert_allclose(np.asarray(t["sf"]), want, rtol=1e-12)
    assert np.all(np.asarray(t["ff"]) == 0.0)


def test_bell2003_split_keeps_its_config_contract():
    assert _split_cfg().include_freefree is False
    with pytest.raises(ConfigError, match="bell2003_split"):
        _split_cfg(include_freefree=True)


# ---- the whole q-taking class: float32 and gradients --------------------------------


@pytest.mark.parametrize("sf", _Q_TYPES)
def test_float32_finite_with_overflowing_linear_l_ir(sf):
    """float32: the linear L_IR (~1e44) is inf; the log companion carries the amplitude."""
    spelling = "unset"
    t = _terms(
        sf,
        spelling if sf != "bell2003_split" else "false",
        [*_FREQS, 1.5e8],
        dtype=jnp.float32,
        L_ir=jnp.asarray(jnp.inf, jnp.float32),
        log_L_ir=jnp.asarray(np.log10(_L_IR), jnp.float32),
    )
    for key in ("sf", "ff"):
        assert np.all(np.isfinite(np.asarray(t[key]))), f"{sf}: {key} not finite in float32"
    assert float(np.asarray(t["sf"])[0]) > 0.0


@pytest.mark.parametrize("sf", ["bell2003", "delvecchio2021", "mccheyne2022"])
@pytest.mark.parametrize("nu", [1.4e9, 5.0e9])
def test_grad_wrt_q_finite_and_nonzero(sf, nu):
    q_key = {"delvecchio2021": "radio_delv_q0", "mccheyne2022": "radio_mcch_q0"}.get(
        sf, "radio_q_ir"
    )
    base = _radio_params()

    def flux(q):
        t = _terms(sf, "unset", [nu], params=base | {q_key: q})
        return (t["sf"] + t["ff"])[0]

    g = float(jax.grad(flux)(jnp.asarray(base[q_key])))
    assert np.isfinite(g) and g != 0.0, f"{sf}: d flux / d {q_key} = {g}"


def test_grad_wrt_spectral_index_finite_and_nonzero():
    """Off the reference frequency: at nu_ref the index drops out identically."""
    base = _radio_params()

    def flux(a):
        t = _terms("bell2003", "unset", [5.0e9], params=base | {"radio_alpha_sf": a})
        return (t["sf"] + t["ff"])[0]

    g = float(jax.grad(flux)(jnp.asarray(base["radio_alpha_sf"])))
    assert np.isfinite(g) and g != 0.0, f"d flux / d alpha_sf = {g}"


# ---- the declared box: the synchrotron share cannot go negative ----------------------


def _q_star(t_e, alpha_ff=-0.1):
    """q at which Bell's total equals the Murphy thermal luminosity at 1.4 GHz."""
    ff_per_lir = 3.88e-44 / 4.6e-28 * (t_e / 1.0e4) ** 0.45 * 1.4**alpha_ff
    return -np.log10(3.75e12 * ff_per_lir)


def test_limit_function_matches_the_constants_written_here():
    from tengri.components.radio.radio import radio_q_total_limit

    for t_e in (5.0e3, 1.0e4, 2.0e4):
        assert radio_q_total_limit(t_e, -0.1) == pytest.approx(_q_star(t_e), abs=1e-12)
    assert _q_star(1.0e4) == pytest.approx(3.5145, abs=1e-4)
    assert _q_star(2.0e4) == pytest.approx(3.379, abs=1e-3)


@pytest.mark.parametrize("q", [1.8, 3.0, 3.37])
@pytest.mark.parametrize("t_e", [5.0e3, 2.0e4])
@pytest.mark.parametrize("alpha", [0.5, 1.2])
@pytest.mark.parametrize("spelling", ["unset", "unset_nebular"])
def test_every_corner_of_the_declared_box_has_nonnegative_synchrotron(spelling, q, t_e, alpha):
    """The declared free prior of radio_q_ir and radio_T_e: synchrotron >= 0 at 150 MHz, 1.4 GHz."""
    params = _radio_params() | {"radio_q_ir": q, "radio_T_e": t_e, "radio_alpha_sf": alpha}
    sf, ff = _sf_ff("bell2003", spelling, [1.5e8, 1.4e9], params=params)
    assert np.all(sf >= 0.0) and np.all(sf + ff >= 0.0)


def test_declared_q_prior_stays_inside_the_limit():
    from tengri.components.radio._params import PARAMS

    q_decl = next(d for d in PARAMS if d.name == "radio_q_ir").free_prior
    t_decl = next(d for d in PARAMS if d.name == "radio_T_e").free_prior
    assert q_decl.bounds[1] <= _q_star(t_decl.bounds[1])


def _build_box(ssp, **radio):
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        radio={"sf": radio.pop("sf", {"type": "bell2003"}), "agn": {"type": "none"}, **radio},
        redshift=tengri.Fixed(0.0),
    )


def test_a_widened_q_prior_is_refused_at_build(ssp_data_fsps):
    with pytest.raises(ConfigError, match=r"q_\* = 3\.5145"):
        _build_box(ssp_data_fsps, radio_q_ir=tengri.Uniform(1.8, 3.6))


def test_hot_gas_with_the_default_q_prior_is_refused_naming_the_largest_q(ssp_data_fsps):
    q_hi = 3.37
    with pytest.raises(ConfigError, match=r"radio_q_ir <= 3\.3"):
        _build_box(
            ssp_data_fsps,
            radio_q_ir=tengri.Uniform(1.8, q_hi + 0.1),
            radio_T_e=tengri.Uniform(5.0e3, 2.0e4),
        )


def test_the_declared_free_box_and_the_nonthermal_spelling_build(ssp_data_fsps):
    _build_box(ssp_data_fsps, all_params=tengri.FREE)
    # freefree=False calibrates the non-thermal term: no subtraction, no limit
    _build_box(
        ssp_data_fsps, sf={"type": "bell2003", "freefree": False}, radio_q_ir=tengri.Uniform(1.8, 3.6)
    )


# ---- the public path: total SED = radio + nebular --------------------------------------

_NEBULAR = {
    "none": {"type": "none"},
    "cue": {"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
}
_FF = {"unset": None, "true": True, "false": False}


def _public(ssp, neb, ff, *, radio=True):
    sf = {"type": "bell2003", "all_params": tengri.Fixed(tengri.DEFAULT)}
    if ff is not None:
        sf["freefree"] = ff
    kw = {
        "ssp_data": ssp,
        "sfh": {
            "type": "delayed",
            "tau_gyr": tengri.Fixed(1.0),
            "age_gyr": tengri.Fixed(5.0),
            "log_total_mass": tengri.Fixed(10.5),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "dust_attenuation": {
            "law": "calzetti",
            "type": "two_component",
            "tau_bc": tengri.Fixed(0.0),
            "tau_diff": tengri.Fixed(1.0),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        "dust_emission": {"type": "dl14", "all_params": tengri.Fixed(tengri.DEFAULT)},
        "neb": _NEBULAR[neb],
        "redshift": tengri.Fixed(0.0),
    }
    if radio:
        kw["radio"] = {
            "sf": sf,
            "radio_q_ir": tengri.Fixed(2.64),
            "radio_alpha_sf": tengri.Fixed(0.8),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        }
    return tengri.SEDModel.build(**kw).predict_state({})


def _l_nu_at(state, key, freqs):
    nu = _C_AA / np.asarray(state.wave)
    r = np.asarray(state.derived[key])
    o = np.argsort(nu)
    return np.array(
        [np.exp(np.interp(np.log(f), np.log(nu[o]), np.log(np.maximum(r[o], 1e-300)))) for f in freqs]
    )


@pytest.fixture(scope="module")
def cue_only_share(ssp_data_fsps):
    """The nebular continuum alone (no radio block) at the four frequencies, over the calibration."""
    st = _public(ssp_data_fsps, "cue", None, radio=False)
    l_ir = float(np.asarray(st.derived["L_ir"]))
    return _l_nu_at(st, "sed_nebular", _FREQS) / _bell_total_ref(2.64, l_ir)


@pytest.mark.parametrize("neb", ["none", "cue"])
@pytest.mark.parametrize("ff", ["unset", "true", "false"])
def test_public_total_sed_over_the_calibration(ssp_data_fsps, cue_only_share, neb, ff):
    """radio + nebular at 1.4, 5, 30, 100 GHz over L_IR / (3.75e12 10^q), in all six cells.

    nebular none: unset and True are the Bell total shared with the Murphy term (1.0 at
    1.4 GHz); False is the full-q synchrotron. cue: unset takes (1 - f_th) off the
    synchrotron and the nebular share is added; False keeps the full-q synchrotron.
    cue + True is the explicit second thermal term #2346 allows (its test pins that the
    override works): here too the radio block's Murphy share is taken off the synchrotron.
    """
    st = _public(ssp_data_fsps, neb, _FF[ff])
    l_ir = float(np.asarray(st.derived["L_ir"]))
    cal = _bell_total_ref(2.64, l_ir)
    x = np.asarray(_FREQS) / 1.4e9
    f_th = float(_murphy_ff(1.4e9, l_ir)) / cal
    murphy = np.asarray(_murphy_ff(np.asarray(_FREQS), l_ir)) / cal
    full = x**-0.8
    nebular = cue_only_share if neb == "cue" else 0.0 * x
    if ff == "false":
        want = full + nebular
    elif ff == "true":
        want = (1.0 - f_th) * full + murphy + nebular
    else:  # unset
        thermal = nebular if neb == "cue" else murphy
        want = (1.0 - f_th) * full + thermal
    got = _l_nu_at(st, "sed_radio", _FREQS)
    if neb == "cue":
        got = got + _l_nu_at(st, "sed_nebular", _FREQS)
    np.testing.assert_allclose(got / cal, want, rtol=3e-3)
    if neb == "none" and ff != "false":
        assert got[0] / cal == pytest.approx(1.0, abs=2e-3)
    if neb == "cue" and ff == "unset":
        # 1.4 GHz total equals the calibration to within Cue's free-free share minus Murphy's
        assert got[0] / cal == pytest.approx(1.0 - f_th + cue_only_share[0], abs=3e-3)

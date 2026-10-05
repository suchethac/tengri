# SPDX-License-Identifier: BSD-3-Clause
"""Bell (2003)'s q_IR is a TOTAL 1.4 GHz calibration; free-free must not be added on top (#2590).

Bell (2003, ApJ 586, 794) Eq. 1 defines q on the *total* 1.4 GHz luminosity and
Sect. 4 parameterises that total as (n + 0.1) eta psi: about 90 % non-thermal and
10 % thermal at 1.4 GHz (Condon 1992). The default ``bell2003`` block put the whole
calibrated total in the synchrotron term and then added the independently
normalized Murphy et al. (2011) free-free term, counting the thermal emission twice
(+13.4 % at 1.4 GHz, +265 % at 100 GHz).

Every expectation below is written out here from the literature, never read back
from the code under test:

* total at nu_ref  =  L_IR / (3.75e12 * 10^q)                        (Bell 2003 Eq. 1)
* thermal term     =  (1/4.6e-28) (T_e/1e4)^0.45 (nu/GHz)^alpha_ff * 3.88e-44 L_IR
                                                      (Murphy et al. 2011 Eqs. 4, 11)
* synchrotron term =  (total at nu_ref - thermal at nu_ref) (nu/nu_ref)^-alpha

so that with the free-free term on or off, the total at nu_ref equals the calibration.
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
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18  # Angstrom/s
_L_IR = 1.0e44  # erg/s (a typical 3e10 Lsun galaxy)
_NU_REF = {"bell2003": 1.4e9, "bell2003_split": 1.4e9, "delvecchio2021": 1.4e9}
_NU_REF_MCCHEYNE = 1.5e8
_FREQS = (1.4e9, 5.0e9, 3.0e10, 1.0e11)
_BELL_FAMILY = ("bell2003", "bell2003_split")
#: every star-formation radio type that takes a q (the registry's own list, minus "none")
_Q_TYPES = tuple(m for m in SF_RADIO_MODELS if m != "none")
#: the Bell-family cells of the matrix: the ones where q calibrates the total and the
#: block adds a separate thermal term (the defect cells)
_L_STAR = 3.0e28  # Bell (2003) Eq. 3, erg/s/Hz


def _nu_ref(sf: str) -> float:
    return _NU_REF_MCCHEYNE if sf == "mccheyne2022" else 1.4e9


def _radio_params() -> dict[str, float]:
    out = {d.name: declared_default(_RADIO_PARAMS, d.name) for d in _RADIO_PARAMS}
    return out | {"redshift": 0.0}


def _terms(sf: str, freefree: bool, freqs, *, params=None, dtype=jnp.float64, **inputs):
    """Radio ``sf`` / ``ff`` terms at exactly ``freqs`` [Hz] for one galaxy."""
    cfg = RadioSEDComponentConfig(sfr_mode=sf, include_freefree=freefree, agn_radio_model="none")
    comp = RadioSEDComponent(config=cfg)
    p = {k: jnp.asarray(v, dtype) for k, v in (params or _radio_params()).items()}
    wave = jnp.asarray(_C_AA / np.asarray(freqs, dtype=np.float64), dtype)
    kw = {
        "L_ir": jnp.asarray(_L_IR, jnp.float64),
        "L_agn_bol": jnp.asarray(0.0, dtype),
        "L_4400_intrinsic": jnp.asarray(0.0, dtype),
        "log_mstar": jnp.asarray(10.0, dtype),
    }
    kw.update(inputs)
    return comp.emission_terms(p, wave, **kw)


def _total(sf, freefree, freqs, **kw):
    t = _terms(sf, freefree, freqs, **kw)
    return np.asarray(t["sf"] + t["ff"], dtype=np.float64)


# ---- literature formulas, written out in the test ---------------------------------


def _bell_total_ref(q: float, l_ir: float = _L_IR) -> float:
    """Bell (2003) Eq. 1 at the calibration frequency [erg/s/Hz]."""
    return l_ir / (3.75e12 * 10.0**q)


def _murphy_ff(nu, l_ir: float = _L_IR, t_e: float = 1.0e4, alpha_ff: float = -0.1):
    """Murphy et al. (2011) free-free: SFR (Eq. 4) through the Eq. 11 calibration [erg/s/Hz]."""
    sfr = 3.88e-44 * l_ir  # Msun/yr
    return (1.0 / 4.6e-28) * (t_e / 1.0e4) ** 0.45 * (np.asarray(nu) / 1.0e9) ** alpha_ff * sfr


def _expected_bell_total(freqs, q: float, alpha: float, freefree: bool):
    """Synchrotron (+ Murphy thermal) such that the total at 1.4 GHz is Bell's total."""
    nu = np.asarray(freqs, dtype=np.float64)
    total_ref = _bell_total_ref(q)
    thermal_ref = float(_murphy_ff(1.4e9)) if freefree else 0.0
    sync = (total_ref - thermal_ref) * (nu / 1.4e9) ** (-alpha)
    thermal = _murphy_ff(nu) if freefree else 0.0
    return sync + thermal


# ---- the defect cells --------------------------------------------------------------


@pytest.mark.parametrize("sf", _BELL_FAMILY)
@pytest.mark.parametrize("freefree", [True, False], ids=["ff_on", "ff_off"])
def test_total_at_1p4ghz_is_the_bell_calibration(sf, freefree):
    """Total L_nu(1.4 GHz) = L_IR / (3.75e12 10^q), thermal share included (not 1 + f_th)."""
    got = _total(sf, freefree, [1.4e9])[0]
    want = _bell_total_ref(2.64)
    assert got == pytest.approx(want, rel=1e-3), (
        f"{sf} ff={freefree}: L(1.4 GHz) = {got:.6e} vs Bell total {want:.6e} "
        f"(ratio {got / want:.4f}); the calibrated total already holds the thermal ~10 %"
    )


@pytest.mark.parametrize("sf", _BELL_FAMILY)
def test_issue_table_rederived_from_the_two_spectral_shapes(sf):
    """L_nu(on) / L_nu(off) at 1.4, 5, 30, 100 GHz from the two shapes written here.

    off: the calibrated total as one power law.  on: (total - thermal(1.4)) (nu/1.4)^-alpha
    plus the Murphy thermal term.  Before the fix the ratio was 1.134 / 1.326 / 2.141 / 3.650.
    """
    q, alpha = 2.64, 0.8
    nu = np.asarray(_FREQS)
    off = _bell_total_ref(q) * (nu / 1.4e9) ** (-alpha)
    f_th = float(_murphy_ff(1.4e9)) / _bell_total_ref(q)
    on = _bell_total_ref(q) * (1.0 - f_th) * (nu / 1.4e9) ** (-alpha) + _murphy_ff(nu)
    want = on / off
    got = _total(sf, True, _FREQS) / _total(sf, False, _FREQS)
    np.testing.assert_allclose(got, want, rtol=1e-3)
    assert got[0] == pytest.approx(1.0, abs=1e-3), f"1.4 GHz ratio {got[0]:.4f} (was 1.134)"
    # the thermal term is flat, the synchrotron steep: the ratio still grows with frequency,
    # by the thermal share only (3.516 at 100 GHz, not the 3.650 that included the double count)
    assert np.all(np.diff(got) > 0.0)


@pytest.mark.parametrize("freefree", [True, False], ids=["ff_on", "ff_off"])
def test_bell2003_split_is_the_same_path_bit_for_bit(freefree):
    a = _terms("bell2003", freefree, _FREQS)
    b = _terms("bell2003_split", freefree, _FREQS)
    for key in ("sf", "ff"):
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))


def test_bell2003_split_default_builds_the_thermal_term_like_bell2003():
    cfg = RadioSEDComponentConfig(sfr_mode="bell2003_split")
    assert cfg.include_freefree is True
    assert RadioSEDComponentConfig(sfr_mode="bell2003_split", include_freefree=True)


@pytest.mark.parametrize("sf", _BELL_FAMILY)
@pytest.mark.parametrize("q", [2.3, 2.64, 3.0])
@pytest.mark.parametrize("alpha", [0.7, 0.8, 1.0])
def test_total_follows_the_two_shapes_across_q_and_alpha(sf, q, alpha):
    params = _radio_params() | {"radio_q_ir": q, "radio_alpha_sf": alpha}
    got = _total(sf, True, _FREQS, params=params)
    np.testing.assert_allclose(got, _expected_bell_total(_FREQS, q, alpha, True), rtol=1e-3)
    off = _total(sf, False, _FREQS, params=params)
    np.testing.assert_allclose(off, _expected_bell_total(_FREQS, q, alpha, False), rtol=1e-3)


@pytest.mark.parametrize("sf", _BELL_FAMILY)
def test_hot_gas_thermal_share_follows_murphy(sf):
    """The thermal share removed from the synchrotron is the one Murphy's term adds."""
    params = _radio_params() | {"radio_T_e": 1.5e4}
    got = _total(sf, True, [1.4e9], params=params)[0]
    assert got == pytest.approx(_bell_total_ref(2.64), rel=1e-3)


# ---- the whole q-taking class ------------------------------------------------------


@pytest.mark.parametrize("sf", _Q_TYPES)
@pytest.mark.parametrize("freefree", [True, False], ids=["ff_on", "ff_off"])
def test_every_q_type_hits_its_own_total_at_its_own_nu_ref(sf, freefree):
    """Bell family: the total equals the q calibration.  Delvecchio/McCheyne: the
    calibrated total times Bell's non-thermal fraction n(L) (Eq. 3, applied by design to
    these two) plus the thermal term, i.e. what the block was already built to emit."""
    nu_ref = _nu_ref(sf)
    params = _radio_params()
    if sf == "delvecchio2021":
        q = params["radio_delv_q0"] - (10.0 - 10.0) * params["radio_delv_mass_slope"]
    elif sf == "mccheyne2022":
        q = params["radio_mcch_q0"] + params["radio_mcch_mass_slope"] * (10.0 - 10.0)
    else:
        q = params["radio_q_ir"]
    total_ref = _bell_total_ref(q)
    got = _total(sf, freefree, [nu_ref])[0]
    if sf in _BELL_FAMILY:
        want = total_ref
    else:
        n_l = 0.9 if total_ref >= _L_STAR else 0.9 * (total_ref / _L_STAR) ** 0.3
        want = n_l * total_ref + (float(_murphy_ff(nu_ref)) if freefree else 0.0)
    assert got == pytest.approx(want, rel=1e-3)


@pytest.mark.parametrize("sf", _Q_TYPES)
def test_float32_finite_with_overflowing_linear_l_ir(sf):
    """float32: the linear L_IR (~1e44) is inf; the log companion carries the amplitude."""
    t = _terms(
        sf,
        True,
        [*_FREQS, 1.5e8],
        dtype=jnp.float32,
        L_ir=jnp.asarray(jnp.inf, jnp.float32),
        log_L_ir=jnp.asarray(np.log10(_L_IR), jnp.float32),
    )
    for key in ("sf", "ff"):
        assert np.all(np.isfinite(np.asarray(t[key]))), f"{sf}: {key} not finite in float32"
    assert float(np.asarray(t["sf"])[0]) > 0.0
    # float32 total at the calibration agrees with float64 to float32 resolution
    got = float((t["sf"] + t["ff"])[0])
    want = float(_total(sf, True, [1.4e9])[0])
    assert got == pytest.approx(want, rel=2e-4)


@pytest.mark.parametrize("sf", _Q_TYPES)
@pytest.mark.parametrize("nu", [1.4e9, 5.0e9])
def test_grad_wrt_q_finite_and_nonzero(sf, nu):
    q_key = {"delvecchio2021": "radio_delv_q0", "mccheyne2022": "radio_mcch_q0"}.get(
        sf, "radio_q_ir"
    )
    base = _radio_params()

    def flux(q):
        t = _terms(sf, True, [nu], params=base | {q_key: q})
        return (t["sf"] + t["ff"])[0]

    g = float(jax.grad(flux)(jnp.asarray(base[q_key])))
    assert np.isfinite(g) and g != 0.0, f"{sf}: d flux / d {q_key} = {g}"


@pytest.mark.parametrize("sf", _BELL_FAMILY)
def test_grad_wrt_spectral_index_finite_and_nonzero(sf):
    """Off the reference frequency: at nu_ref the index drops out identically.

    Delvecchio / McCheyne carry their own cited fixed indices (ruling R8) and do not read
    ``radio_alpha_sf``, so the Bell family is the class that has this parameter.
    """
    base = _radio_params()

    def flux(a):
        t = _terms(sf, True, [5.0e9], params=base | {"radio_alpha_sf": a})
        return (t["sf"] + t["ff"])[0]

    g = float(jax.grad(flux)(jnp.asarray(base["radio_alpha_sf"])))
    assert np.isfinite(g) and g != 0.0, f"{sf}: d flux / d alpha_sf = {g}"


# ---- the public path: nebular off and on (the #2346 auto-rule must keep working) -----


def _model(ssp, sf_block, neb):
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": tengri.Fixed(1.0),
            "age_gyr": tengri.Fixed(5.0),
            "log_total_mass": tengri.Fixed(10.5),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_attenuation={
            "law": "calzetti",
            "type": "two_component",
            "tau_bc": tengri.Fixed(0.0),
            "tau_diff": tengri.Fixed(1.0),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_emission={"type": "dl14", "all_params": tengri.Fixed(tengri.DEFAULT)},
        neb=neb,
        radio={
            "sf": sf_block,
            "radio_q_ir": tengri.Fixed(2.64),
            "radio_alpha_sf": tengri.Fixed(0.8),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        redshift=tengri.Fixed(0.0),
    )


def _l_nu_at(state, freq):
    nu = _C_AA / np.asarray(state.wave)
    r = np.asarray(state.derived["sed_radio"])
    o = np.argsort(nu)
    return float(np.exp(np.interp(np.log(freq), np.log(nu[o]), np.log(np.maximum(r[o], 1e-300)))))


_NEB = {
    "none": {"type": "none"},
    "cue": {"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
}


@pytest.mark.parametrize("neb", ["none", "cue"])
@pytest.mark.parametrize("sf_type", _BELL_FAMILY)
def test_public_path_radio_total_is_the_bell_total(ssp_data_fsps, neb, sf_type):
    """Default build: radio L_nu(1.4 GHz) = L_IR/(3.75e12 10^q) with the nebular off and on.

    Nebular on: the #2346 auto-rule drops the radio thermal term (the nebular continuum
    owns it), so the synchrotron carries the whole calibrated total; nebular off: the
    synchrotron carries total - thermal and the Murphy term the rest.  L_IR is read from
    the model's own dust state.
    """
    state = _model(
        ssp_data_fsps, {"type": sf_type, "all_params": tengri.Fixed(tengri.DEFAULT)}, _NEB[neb]
    ).predict_state({})
    l_ir = float(np.asarray(state.derived["L_ir"]))
    got = _l_nu_at(state, 1.4e9)
    assert got == pytest.approx(_bell_total_ref(2.64, l_ir), rel=2e-3)


@pytest.mark.parametrize("neb", ["none", "cue"])
def test_public_path_explicit_freefree_pin_subtracts_the_thermal_share(ssp_data_fsps, neb):
    """freefree=True pinned explicitly (also with the nebular on): the sum is still the total."""
    sf = {"type": "bell2003", "freefree": True, "all_params": tengri.Fixed(tengri.DEFAULT)}
    state = _model(ssp_data_fsps, sf, _NEB[neb]).predict_state({})
    l_ir = float(np.asarray(state.derived["L_ir"]))
    assert _l_nu_at(state, 1.4e9) == pytest.approx(_bell_total_ref(2.64, l_ir), rel=2e-3)


def test_public_path_on_off_ratio_matches_the_issue_table_rederived(ssp_data_fsps):
    """The issue's reproducer: ratio on/off at 1.4, 5, 30, 100 GHz, expected from the shapes."""
    on = _model(
        ssp_data_fsps,
        {"type": "bell2003", "all_params": tengri.Fixed(tengri.DEFAULT)},
        _NEB["none"],
    ).predict_state({})
    off = _model(
        ssp_data_fsps,
        {"type": "bell2003", "freefree": False, "all_params": tengri.Fixed(tengri.DEFAULT)},
        _NEB["none"],
    ).predict_state({})
    l_ir = float(np.asarray(on.derived["L_ir"]))
    nu = np.asarray(_FREQS)
    f_th = float(_murphy_ff(1.4e9, l_ir)) / _bell_total_ref(2.64, l_ir)
    want = (1.0 - f_th) + _murphy_ff(nu, l_ir) / (
        _bell_total_ref(2.64, l_ir) * (nu / 1.4e9) ** -0.8
    )
    got = np.array([_l_nu_at(on, f) / _l_nu_at(off, f) for f in _FREQS])
    np.testing.assert_allclose(got, want, rtol=3e-3)

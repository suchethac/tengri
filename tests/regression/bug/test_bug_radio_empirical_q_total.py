# SPDX-License-Identifier: BSD-3-Clause
"""Delvecchio+2021 and McCheyne+2022 q_IR are TOTAL-radio calibrations (#2590 follow-up).

Both relations are fitted to observed total radio luminosity against total IR luminosity,
so the calibrated total at the relation's own reference frequency is

    L_tot(nu_ref) = L_IR / (3.75e12 Hz x 10^q(M*, z)),                 (Delvecchio Eq. 1)

and the emitted synchrotron is what remains after the Murphy+2011 free-free term at the same
frequency is taken out. With the free-free term on the radio block the public SED
(``sed_radio`` + ``sed_nebular``) therefore reproduces the calibration at nu_ref, for every
(M*, z, L_IR). The earlier construction multiplied the total by the Bell (2003) n(L)
suppression on top of a q(M*, z) that already carries the mass and luminosity trend,
which counted that correction twice and missed the calibration by -11 % to +12 %.

Every expectation in this file is written from the paper equation, never by calling the
code under test:

* Delvecchio+2021, arXiv:2010.05510, Eq. 5 (multi-parametric q_IR(M*, z) fit, 1.4 GHz,
  AGN-corrected total radio), coefficients 2.646 +/- 0.024, -0.023 +/- 0.008, 0.148 +/- 0.013.
* McCheyne+2022, A&A 662, A100, Sect. 5.2 joint fit (150 MHz): q_TIR = 1.98 (1+z)^0.02
  - 0.22 (log M* - 10.45), valid for z < 0.4 and M* > 10^10.45. The text was read from the
  paper's Leiden accepted manuscript as quoted in the tengri issue thread (#2805); the
  public path below grids M* outside that validity range as a formula check only.
* Murphy+2011 Eqs. 4 and 11 for the free-free term (3.88e-44 Msun/yr per erg/s of L_IR,
  2.174e27 erg/s/Hz per Msun/yr at 1 GHz, nu^-0.1 at T_e = 1e4 K).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.config.exceptions import ConfigError
from tengri.utils.physics_constants import L_SUN as _L_SUN

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18  # Angstrom/s
_LOG_M_GRID = (9.5, 10.0, 10.5, 11.0)
_Z_GRID = (0.0, 1.0)
_LOG_TOTAL_MASS_OFFSET = 0.2246  # log_total_mass - log_mstar at age 5 Gyr, measured


# (sfr_mode, paper q(M*, z), nu_ref [Hz], synchrotron index used by the mode)
def _edge(x, lo, hi):
    """Hold x at the calibrated edge outside [lo, hi] (the relation is not extrapolated)."""
    return min(max(x, lo), hi) if hi is not None else max(x, lo)


_MODES = {
    # Delvecchio+2021 Eq. 5, held on the Sect. 2 domain 8 < log M* < 12, 0.1 < z < 4.5.
    "delvecchio2021": (
        lambda logm, z: (
            2.646 * (1.0 + _edge(z, 0.1, 4.5)) ** (-0.023)
            - 0.148 * (_edge(logm, 8.0, 12.0) - 10.0)
        ),
        1.4e9,
    ),
    # McCheyne+2022 Sect. 5.2 joint fit, pivot log M* = 10.45, held on M* > 10^10.45, z < 0.4.
    "mccheyne2022": (
        lambda logm, z: (
            1.98 * (1.0 + _edge(z, 0.0, 0.4)) ** 0.02 - 0.22 * (_edge(logm, 10.45, None) - 10.45)
        ),
        1.5e8,
    ),
}


# ---- the literature, written out here ----------------------------------------------


def _total_ref(q, l_ir):
    """L_tot at the calibration frequency [erg/s/Hz] (Delvecchio Eq. 1)."""
    return l_ir / (3.75e12 * 10.0**q)


def _murphy_ff(nu, l_ir):
    """Murphy+2011 free-free L_nu [erg/s/Hz] at T_e = 1e4 K, alpha_ff = -0.1."""
    sfr = 3.88e-44 * l_ir  # Murphy Eq. 4, Msun/yr
    return (1.0 / 4.6e-28) * (nu / 1.0e9) ** (-0.1) * sfr  # Murphy Eq. 11, inverted


# ---- public path --------------------------------------------------------------------

_NEBULAR = {
    "none": {"type": "none"},
    "cue": {"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
}


def _build(ssp, mode, *, log_total_mass, age, z, neb):
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": tengri.Fixed(1.0),
            "age_gyr": tengri.Fixed(age),
            "log_total_mass": tengri.Fixed(log_total_mass),
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
        radio={
            "sf": {"type": mode, "all_params": tengri.Fixed(tengri.DEFAULT)},
            "agn": {"type": "none"},
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        neb=_NEBULAR[neb],
        redshift=tengri.Fixed(z),
    ).predict_state({})


def _l_nu_at(state, key, nu):
    """Log-interpolate a rest-frame L_nu [erg/s/Hz] component at frequency ``nu`` [Hz]."""
    wave_nu = _C_AA / np.asarray(state.wave, dtype=np.float64)
    vals = np.asarray(state.derived[key], dtype=np.float64)
    order = np.argsort(wave_nu)
    return float(
        np.exp(
            np.interp(
                np.log(nu),
                np.log(wave_nu[order]),
                np.log(np.maximum(vals[order], 1e-300)),
            )
        )
    )


def _radio_ir_input(state):
    """The IR luminosity [erg/s] the radio block receives, read from the published state."""
    # The windowed IR input is published as radio_L_ir_input [L_sun] when present.
    if "radio_L_ir_input" in state.derived:
        return float(np.asarray(state.derived["radio_L_ir_input"])) * _L_SUN
    return float(np.asarray(state.derived["L_ir"]))


def _public_state(ssp, mode, neb, log_total_mass, age, z):
    """Build the public model for one mode and return (state, logM*, L_IR,in)."""
    st = _build(ssp, mode, log_total_mass=log_total_mass, age=age, z=z, neb=neb)
    return st, float(np.asarray(st.derived["log_mstar"])), _radio_ir_input(st)


def _cases():
    for mode in _MODES:
        for logm in _LOG_M_GRID:
            for z in _Z_GRID:
                for age in (1.0, 5.0):
                    yield mode, logm, z, age


@pytest.mark.parametrize(("mode", "logm", "z", "age"), list(_cases()))
def test_radio_block_alone_reproduces_the_total_calibration(ssp_data_fsps, mode, logm, z, age):
    """No nebular: sed_radio at nu_ref is the paper's total, thermal term included."""
    q_of, nu_ref = _MODES[mode]
    st, got_logm, l_ir = _public_state(
        ssp_data_fsps, mode, "none", logm + _LOG_TOTAL_MASS_OFFSET, age, z
    )
    assert got_logm == pytest.approx(logm, abs=0.3)  # realized M*; the offset depends on age
    total = _total_ref(q_of(got_logm, z), l_ir)
    got = _l_nu_at(st, "sed_radio", nu_ref)
    assert got / total == pytest.approx(1.0, abs=1e-3), (
        f"{mode} logM*={got_logm:.3f} z={z} L_IR={l_ir:.3g}: total/calibration = {got / total:.5f}"
    )


@pytest.mark.parametrize(("mode", "logm", "z", "age"), list(_cases()))
def test_radio_block_with_cue_nebular_is_the_total_minus_the_murphy_share(
    ssp_data_fsps, mode, logm, z, age
):
    """Cue on: the radio block keeps (1 - f_th) of the calibration, f_th the Murphy share."""
    q_of, nu_ref = _MODES[mode]
    st, got_logm, l_ir = _public_state(
        ssp_data_fsps, mode, "cue", logm + _LOG_TOTAL_MASS_OFFSET, age, z
    )
    total = _total_ref(q_of(got_logm, z), l_ir)
    murphy = _murphy_ff(nu_ref, l_ir)
    got = _l_nu_at(st, "sed_radio", nu_ref)
    assert got == pytest.approx(total - murphy, rel=1e-3)


@pytest.mark.parametrize(("mode", "logm", "z", "age"), list(_cases()))
def test_radio_plus_cue_nebular_is_the_fixed_share_residual_band(
    ssp_data_fsps, mode, logm, z, age
):
    """The Cue free-free share, measured over the 32 cases, sets the residual band.

    The radio block removes Murphy's fixed share f_th of the calibration (asserted by the
    test above); the nebular backend adds its own free-free share L_ff,neb / L_cal. The sum is
    therefore (1 - f_th) + L_ff,neb / L_cal, and the residual from the calibration is
    L_ff,neb / L_cal - f_th. Cue's share follows the ionizing photon rate per unit L_IR, so it
    falls with stellar age (0.115 at 1 Gyr, 0.061 at 5 Gyr for delvecchio2021 at log M* 10,
    z 0) and moves with z. Measured over the 32 cases (16 per mode), L_ff,neb / L_cal lies in
    [0.0125, 0.136]; the Murphy share f_th is 0.094 to 0.161 for delvecchio2021 (1.4 GHz) and
    0.027 to 0.063 for mccheyne2022 (150 MHz). The band below is the measured range, widened
    by 0.001 on each side.
    """
    q_of, nu_ref = _MODES[mode]
    st, got_logm, l_ir = _public_state(
        ssp_data_fsps, mode, "cue", logm + _LOG_TOTAL_MASS_OFFSET, age, z
    )
    total = _total_ref(q_of(got_logm, z), l_ir)
    neb_ff = _l_nu_at(st, "sed_nebular", nu_ref)
    assert 0.0115 <= neb_ff / total <= 0.137


# ---- negative synchrotron: the refusal evaluates the declared box ----------------------


def _build_radio_box(ssp, mode, sf=None, log_total_mass=None, z=None, neb=None):
    """Build one radio-only box. ``sf`` overrides the radio ``sf`` group; the SFH mass and
    redshift default to Fixed(DEFAULT) and Fixed(0). ``neb`` adds a nebular backend."""
    return tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "const",
            "log_total_mass": log_total_mass or tengri.Fixed(tengri.DEFAULT),
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        radio={"sf": {"type": mode, **(sf or {})}, "agn": {"type": "none"}},
        redshift=z if z is not None else tengri.Fixed(0.0),
        **({"neb": neb} if neb is not None else {}),
    )


@pytest.mark.parametrize("mode", list(_MODES))
def test_declared_free_box_of_the_new_modes_builds(ssp_data_fsps, mode):
    _build_radio_box(ssp_data_fsps, mode, sf={"all_params": tengri.FREE})


def test_delvecchio_q0_reaching_log_mstar_8_is_refused_naming_the_corner(ssp_data_fsps):
    """Review reproducer: q0 up to 3.25 with a formed-mass prior reaching low mass."""
    with pytest.raises(ConfigError, match=r"radio_delv_q0=3\.25, log M\* = .*q_\* = "):
        _build_radio_box(
            ssp_data_fsps,
            "delvecchio2021",
            sf={"radio_delv_q0": tengri.Uniform(1.0, 3.25)},
            log_total_mass=tengri.Uniform(8.0, 12.5),
        )


def test_mccheyne_q0_above_the_limit_is_refused_at_the_domain_edge(ssp_data_fsps):
    """McCheyne is held at z = 0.4 and M* = 10^10.45 outside its domain, so the worst corner
    is q0 (1.4)^0.02 at the domain edge. q0 = 3.5 is above q* = 3.4175 (150 MHz) there."""
    with pytest.raises(ConfigError, match=r"radio_mcch_q0=3\.5, log M\* = .*q_\* = "):
        _build_radio_box(
            ssp_data_fsps,
            "mccheyne2022",
            sf={"radio_mcch_q0": tengri.Uniform(1.0, 3.5)},
            log_total_mass=tengri.Uniform(8.0, 12.5),
        )


def test_mccheyne_low_mass_and_high_z_are_held_so_the_declared_box_builds(ssp_data_fsps):
    """Positive control: the held worst corner (q0 3.0 at z = 0.4, M* = 10^10.45) is below q*."""
    _build_radio_box(
        ssp_data_fsps,
        "mccheyne2022",
        sf={"radio_mcch_q0": tengri.Uniform(1.0, 3.0)},
        log_total_mass=tengri.Uniform(8.0, 12.5),
        z=tengri.Uniform(0.0, 20.0),
    )


def test_delvecchio_q0_with_a_galaxy_mass_box_above_the_fit_builds(ssp_data_fsps):
    """Positive control: the same q0 support with the mass box inside the fit sample builds."""
    _build_radio_box(
        ssp_data_fsps,
        "delvecchio2021",
        sf={"radio_delv_q0": tengri.Uniform(1.8, 3.25)},
        log_total_mass=tengri.Uniform(10.0, 12.0),
    )


def _declared_q0_bounds(q_name):
    """The (low, high) support of a q0 parameter's declared free prior."""
    from tengri.components.radio._params import PARAMS

    decl = next(d for d in PARAMS if d.name == q_name)
    return decl.free_prior.bounds


@pytest.mark.parametrize("neb", ["none", "cue"])
@pytest.mark.parametrize("mode", list(_MODES))
def test_no_built_corner_of_the_box_has_negative_sed_radio(ssp_data_fsps, mode, neb):
    """Every corner of a small (q0, log M*, z) grid that builds gives sed_radio >= 0.

    With a free-free nebular (``cue``) the radio block's own free-free is off, so
    ``sed_radio`` is the synchrotron term alone and a negative synchrotron shows directly.
    Without a nebular the radio block's free-free can hide it, so both are checked.
    """
    q_name = {"delvecchio2021": "radio_delv_q0", "mccheyne2022": "radio_mcch_q0"}[mode]
    # The corners are the declared free-prior bounds of q0, read from PARAMS.
    q_lo, q_hi = _declared_q0_bounds(q_name)
    neb_group = _NEBULAR[neb]
    built = 0
    for q0 in (q_lo, q_hi):
        for logm in (8.0, 9.5, 11.5):  # 8.0: the lowest log M* the fit sample reaches
            for z in (0.0, 1.0):
                try:
                    model = _build_radio_box(
                        ssp_data_fsps,
                        mode,
                        sf={q_name: tengri.Fixed(q0)},
                        log_total_mass=tengri.Fixed(logm + _LOG_TOTAL_MASS_OFFSET),
                        z=tengri.Fixed(z),
                        neb=neb_group,
                    )
                except ConfigError:
                    continue  # refused at build: the corner is outside the valid box
                built += 1
                st = model.predict_state({})
                wnu = _C_AA / np.asarray(st.wave, dtype=np.float64)
                sed = np.asarray(st.derived["sed_radio"], dtype=np.float64)
                band = (wnu > 5.0e7) & (wnu < 2.0e10)
                assert sed[band].min() >= 0.0, (
                    f"{mode} neb={neb} q0={q0} logM*={logm} z={z}: "
                    f"min sed_radio {sed[band].min():.3e}"
                )
    assert built >= 1


# ---- domain hold: the relations are held at the calibrated edges, never extrapolated ---------

_WAVE_1P4 = jnp.array([_C_AA / 1.4e9])
_WAVE_150MHZ = jnp.array([_C_AA / 1.5e8])
_L_IR = 1.0e44  # erg/s


def _delv_L(log_mstar, z):
    from tengri.components.radio import radio_sfr_delvecchio2021

    return float(
        radio_sfr_delvecchio2021(
            _WAVE_1P4, _L_IR, log_mstar=log_mstar, redshift=z, apply_suppression=False
        )[0]
    )


def _mcch_L(log_mstar, z):
    from tengri.components.radio import radio_sfr_mccheyne2022

    return float(
        radio_sfr_mccheyne2022(
            _WAVE_150MHZ, _L_IR, log_mstar=log_mstar, redshift=z, apply_suppression=False
        )[0]
    )


@pytest.mark.parametrize("logm", [6.5, 7.0, 7.9])
def test_delvecchio_below_the_mass_domain_is_held_at_log_mstar_8(logm):
    """Below 10^8 the relation is held at the sample edge (q independent of M* there)."""
    assert _delv_L(logm, 1.0) == pytest.approx(_delv_L(8.0, 1.0), rel=1e-12)


@pytest.mark.parametrize("z", [0.0, 0.05, 5.0, 20.0])
def test_delvecchio_outside_the_redshift_domain_is_held_at_the_edge(z):
    """Outside 0.1 < z < 4.5 the relation is held at the nearest redshift edge."""
    edge = 0.1 if z < 0.1 else 4.5
    assert _delv_L(10.0, z) == pytest.approx(_delv_L(10.0, edge), rel=1e-12)


@pytest.mark.parametrize("logm", [9.0, 10.0, 10.45])
def test_mccheyne_below_the_mass_domain_is_held_at_10_45(logm):
    assert _mcch_L(logm, 0.2) == pytest.approx(_mcch_L(10.45, 0.2), rel=1e-12)


@pytest.mark.parametrize("z", [0.5, 1.0, 20.0])
def test_mccheyne_above_the_redshift_domain_is_held_at_0_4(z):
    assert _mcch_L(11.0, z) == pytest.approx(_mcch_L(11.0, 0.4), rel=1e-12)


@pytest.mark.parametrize("logm,z", [(9.0, 1.0), (10.5, 2.0), (11.5, 0.3)])
def test_inside_the_domain_the_delvecchio_total_is_the_paper_eq5(logm, z):
    """Inside the domain the total is L_IR / (3.75e12 10^q) with Eq. 5 q, unchanged."""
    q = 2.646 * (1.0 + z) ** (-0.023) - 0.148 * (logm - 10.0)
    assert _delv_L(logm, z) == pytest.approx(_L_IR / (3.75e12 * 10.0**q), rel=1e-9)


@pytest.mark.parametrize("logm,z", [(10.8, 0.1), (11.2, 0.3), (12.0, 0.4)])
def test_inside_the_domain_the_mccheyne_total_is_the_joint_fit(logm, z):
    q = 1.98 * (1.0 + z) ** 0.02 - 0.22 * (logm - 10.45)
    assert _mcch_L(logm, z) == pytest.approx(_L_IR / (3.75e12 * 10.0**q), rel=1e-9)


@pytest.mark.parametrize("log_mstar,z", [(7.0, 0.0), (9.0, 1.0), (12.5, 5.0)])
def test_delvecchio_gradient_is_finite_and_zero_outside_the_domain(log_mstar, z):
    import jax

    from tengri.components.radio import radio_sfr_delvecchio2021

    def total(lm, zz):
        return jnp.sum(
            radio_sfr_delvecchio2021(
                _WAVE_1P4, _L_IR, log_mstar=lm, redshift=zz, apply_suppression=False
            )
        )

    g_m, g_z = jax.grad(total, argnums=(0, 1))(jnp.asarray(log_mstar), jnp.asarray(z))
    assert np.isfinite(float(g_m)) and np.isfinite(float(g_z))
    if log_mstar < 8.0 or log_mstar > 12.0:
        assert float(g_m) == 0.0
    if z < 0.1 or z > 4.5:
        assert float(g_z) == 0.0


@pytest.mark.parametrize("log_mstar,z", [(7.0, 0.0), (10.0, 0.2), (11.0, 1.0)])
def test_mccheyne_gradient_is_finite_and_zero_outside_the_domain(log_mstar, z):
    import jax

    from tengri.components.radio import radio_sfr_mccheyne2022

    def total(lm, zz):
        return jnp.sum(
            radio_sfr_mccheyne2022(
                _WAVE_150MHZ, _L_IR, log_mstar=lm, redshift=zz, apply_suppression=False
            )
        )

    g_m, g_z = jax.grad(total, argnums=(0, 1))(jnp.asarray(log_mstar), jnp.asarray(z))
    assert np.isfinite(float(g_m)) and np.isfinite(float(g_z))
    if log_mstar < 10.45:
        assert float(g_m) == 0.0
    if z > 0.4:
        assert float(g_z) == 0.0


def test_delvecchio_warns_when_the_declared_support_leaves_its_domain(ssp_data_fsps):
    """A formed-mass prior below 10^8 is held at the edge and announced at build."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_radio_box(
            ssp_data_fsps,
            "delvecchio2021",
            sf={"radio_delv_q0": tengri.Fixed(2.646)},
            log_total_mass=tengri.Uniform(7.5, 12.0),
            z=tengri.Fixed(1.0),
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("held at the sample edge" in m and "log M* down to" in m for m in msgs), msgs


def test_mccheyne_warns_when_the_redshift_support_leaves_its_domain(ssp_data_fsps):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_radio_box(
            ssp_data_fsps,
            "mccheyne2022",
            sf={"radio_mcch_q0": tengri.Fixed(1.98)},
            log_total_mass=tengri.Fixed(11.0),
            z=tengri.Uniform(0.0, 1.0),
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("held at the sample edge" in m and "z up to 1" in m for m in msgs), msgs


def test_no_domain_warning_inside_the_calibrated_domain(ssp_data_fsps):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_radio_box(
            ssp_data_fsps,
            "delvecchio2021",
            sf={"radio_delv_q0": tengri.Fixed(2.646)},
            log_total_mass=tengri.Uniform(9.0, 11.5),
            z=tengri.Fixed(1.0),
        )
        _build_radio_box(
            ssp_data_fsps,
            "mccheyne2022",
            sf={"radio_mcch_q0": tengri.Fixed(1.98)},
            log_total_mass=tengri.Uniform(10.8, 11.5),
            z=tengri.Fixed(0.2),
        )
    assert not [w for w in caught if "held at the sample edge" in str(w.message)]

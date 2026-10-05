# SPDX-License-Identifier: BSD-3-Clause
r"""One inclination and one opening angle: torus screen, Fritz library and polar cone.

``agn_cos_inc`` is the single inclination of every torus (:math:`i` from the polar axis). The
dust-free polar cone of a torus has half-angle :math:`\theta_c`: :math:`90^\circ - {\rm oa}` for
SKIRTOR (``agn_oa_skirtor`` is the elevation from the equatorial plane, Stalevski et al. 2016
[1]_), and ``agn_fritz_oa`` itself for the Fritz et al. (2006) library [2]_ (full opening
:math:`\Theta = 180^\circ - 2\theta_c`). The sightline is Type 1 when :math:`i < \theta_c`, that
is :math:`\cos i > \cos\theta_c`, for the disc screen and the polar-dust mask alike, and both
weights are the same logistic in :math:`\cos i` of width 0.025 about :math:`\cos\theta_c`, so
they sum to one. The Fritz library's own elevation :math:`\psi = 90^\circ - i` is derived from
the inclination, and an explicit ``agn_fritz_psy`` is refused.

The Fritz optical depth axis is :math:`\tau_{9.7}`; the screen needs :math:`\tau_V`. Fritz et al.
(2006) Section 3 give :math:`\tau(9.7) = 0.1` for :math:`N_{\rm H} = 9.0\times 10^{21}`
cm\ :sup:`-2`, "an optical extinction of :math:`A_V = 2.3`", so :math:`A_V/\tau_{9.7} = 23` and
:math:`\tau_V = 0.4\ln 10\,A_V = 21.2\,\tau_{9.7}`.

The polar cone's solid-angle share is :math:`g(\Phi) = 7/18 - \sin^2\Phi/6 - 2\sin^3\Phi/9`
(SKIRTOR; :math:`f_{\rm cone} = 18 g/7` against the bolometric disc) with
:math:`\Phi` the torus half-opening angle from the equator, and :math:`1 - \cos\theta_c` for the
Fritz library (CIGALE ``skirtor2016`` and ``fritz2006`` [3]_). On the CIGALE-tied path the
polar graybody sits inside the unit-integral normalization, so the disc is divided by
:math:`1 + l_{\rm ext}` exactly as the torus is. The absorbed power is
:math:`L_\nu(1 - e^{-\tau})`, whose derivative at :math:`E(B-V) = 0` is the one-sided one.

References
----------
.. [1] M. Stalevski et al., MNRAS, 458, 2288 (2016). arXiv:1602.06954.
.. [2] J. Fritz, A. Franceschini and E. Hatziminaoglou, MNRAS, 366, 767 (2006).
   arXiv:astro-ph/0511428.
.. [3] M. Boquien et al., A&A, 622, A103 (2019). arXiv:1811.03094.
"""

from __future__ import annotations

import inspect
import re

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn.blocks.atten import (
    polar_dust_attenuation_block,
    polar_dust_reemission_lnu,
)
from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.components.agn.polar_dust import polar_dust_extinction
from tengri.components.dust.attenuation import smc
from tengri.utils.physics_constants import C_AA
from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.asarray(np.geomspace(500.0, 1.0e9, 4000))  # [A]
_WAVE_TIE = jnp.asarray(np.geomspace(8.0, 1.0e8, 3000))  # [A] covers the SKIRTOR library axis
_HALVES = (20.0, 40.0, 60.0)  # Fritz dust-free cone half-angles = the grid axis
_WIDTH = 0.025  # the one Type-1/2 width in cos i
_V = 5500.0  # [A]
_LOG_LBOL = 12.0

_BASE = dict(
    agn_disc_block="schartmann2005",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_norm="independent",
)
_POLAR_PARAMS = dict(agn_polar_T=100.0, agn_polar_beta=1.6)


def _cos(i_deg):
    return float(np.cos(np.radians(i_deg)))


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _power(component, wave=_WAVE):
    """``int L_nu dnu`` of an ``L_nu`` array [erg/s], in float64."""
    nu = C_AA / np.asarray(wave, dtype=np.float64)
    order = np.argsort(nu)
    return float(np.trapezoid(np.asarray(component, dtype=np.float64)[order], nu[order]))


def _at(component, wave_aa, wave=_WAVE):
    return float(np.interp(wave_aa, np.asarray(wave), np.asarray(component)))


def _run(i_deg=None, *, wave=_WAVE, cos_inc=None, **params):
    """``compose_l_nu(..., return_components=True)`` -> ``{component: ndarray}`` [erg/s/Hz]."""
    cos = _cos(i_deg) if cos_inc is None else cos_inc
    kwargs = {**_BASE, **params, "agn_cos_inc": cos}
    _, comps = compose_l_nu(wave, _LOG_LBOL, return_components=True, **kwargs)
    return {k: np.asarray(v) for k, v in comps.items()}


def _fritz(i_deg, half, tau=0.1, **extra):
    return _run(
        i_deg,
        agn_torus_block="fritz",
        agn_attenuation_block=extra.pop("agn_attenuation_block", "none"),
        agn_fritz_oa=half,
        agn_fritz_tau=tau,
        agn_torus_frac=0.3,
        **extra,
    )


def _skirtor(i_deg, oa, tau=7.0, **extra):
    return _run(
        i_deg,
        agn_torus_block="skirtor",
        agn_attenuation_block=extra.pop("agn_attenuation_block", "none"),
        agn_oa_skirtor=oa,
        agn_tau_skirtor=tau,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_torus_frac=0.3,
        **extra,
    )


def _screen_weight(run, i_deg, tau_a, tau_b):
    """Type-2 weight of the torus screen at ``i_deg`` as the runner applies it.

    The screen is ``exp(-tau_V k/k_V w)``, so two optical depths at one inclination give
    ``ln(L_a/L_b) = -(tau_a - tau_b) w`` at V with the disc itself canceling; dividing by the
    same ratio at ``i = 89`` deg (``w = 1`` there) removes the unit of ``tau_V`` and leaves ``w``.
    """

    def ln_ratio(i):
        return np.log(_at(run(i, tau_a)["disc"], _V) / _at(run(i, tau_b)["disc"], _V))

    return ln_ratio(i_deg) / ln_ratio(89.0)


def _fritz_screen_weight(i_deg, half):
    return _screen_weight(lambda i, tau: _fritz(i, half, tau), i_deg, 0.1, 0.3)


def _skirtor_screen_weight(i_deg, oa):
    return _screen_weight(lambda i, tau: _skirtor(i, oa, tau), i_deg, 3.0, 7.0)


_EBV = 0.3
_EXT_V = float(np.exp(-0.921 * _EBV * 2.93 * np.asarray(smc(jnp.asarray([_V])))[0]))


def _polar_weight(run, i_deg):
    """Type-1 weight of the polar mask at ``i_deg`` as the runner applies it.

    ``factor = 1 - w (1 - exp(-tau))`` multiplies the screened disc, so the ratio of the disc
    with and without polar reddening at one inclination is ``factor`` at V.
    """
    with_dust = _at(run(i_deg, _EBV)["disc"], _V)
    without = _at(run(i_deg, 0.0)["disc"], _V)
    return (1.0 - with_dust / without) / (1.0 - _EXT_V)


def _fritz_polar_weight(i_deg, half):
    return _polar_weight(
        lambda i, ebv: _fritz(
            i,
            half,
            agn_attenuation_block="polar_dust",
            agn_polar_ebv=ebv,
            **_POLAR_PARAMS,
        ),
        i_deg,
    )


def _skirtor_polar_weight(i_deg, oa, **extra):
    return _polar_weight(
        lambda i, ebv: _skirtor(
            i,
            oa,
            agn_attenuation_block="polar_dust",
            agn_polar_ebv=ebv,
            **_POLAR_PARAMS,
            **extra,
        ),
        i_deg,
    )


# ----------------------------------------------------------------------------------
# 1. one inclination, one Type-1/2 boundary
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("half", _HALVES)
def test_fritz_disc_screen_type1_limit_is_the_half_angle(half):
    """Type 1 iff i < half: the screen weight is ``sigmoid((cos half - cos i)/0.025)``.

    The weight is 0.5 at i = half (it was at 90 - half: 70 / 50 / 30 deg for half 20 / 40 / 60).
    """
    cos_half = _cos(half)
    for i in (half - 15.0, half - 5.0, half, half + 5.0, half + 15.0):
        expected = _sigmoid((cos_half - _cos(i)) / _WIDTH)
        got = _fritz_screen_weight(i, half)
        assert got == pytest.approx(expected, abs=2e-3), (
            f"half={half}: Fritz screen Type-2 weight at i={i:g} deg is {got:.4f}, "
            f"the Type-1 limit i = half gives {expected:.4f}"
        )


@pytest.mark.parametrize("half", _HALVES)
def test_fritz_polar_mask_type1_limit_is_the_half_angle(half):
    """The polar mask follows the Fritz half-angle: Type-1 weight 0.5 at i = half."""
    cos_half = _cos(half)
    for i in (half - 15.0, half, half + 15.0):
        expected = _sigmoid((_cos(i) - cos_half) / _WIDTH)
        got = _fritz_polar_weight(i, half)
        assert got == pytest.approx(expected, abs=2e-3), (
            f"half={half}: polar Type-1 weight at i={i:g} deg is {got:.4f}, "
            f"expected {expected:.4f}"
        )


@pytest.mark.parametrize("oa", (10.0, 40.0, 70.0))
def test_skirtor_polar_mask_follows_the_torus_angle(oa):
    """With ``agn_polar_oa`` unset the cone's boundary is the torus's own, i = 90 - oa."""
    cos_limit = np.sin(np.radians(oa))
    for i in (90.0 - oa - 12.0, 90.0 - oa, 90.0 - oa + 12.0):
        i = float(np.clip(i, 0.0, 90.0))
        expected = _sigmoid((_cos(i) - cos_limit) / _WIDTH)
        got = _skirtor_polar_weight(i, oa)
        assert got == pytest.approx(expected, abs=2e-3), (
            f"oa={oa}: polar Type-1 weight at i={i:g} deg is {got:.4f}, expected {expected:.4f}"
        )


@pytest.mark.parametrize("i_deg", (30.0, 45.0, 50.0, 55.0, 60.0, 70.0))
def test_polar_and_torus_weights_are_complementary(i_deg):
    """One angle, one width: polar Type-1 weight + screen Type-2 weight = 1 at every i."""
    oa = 40.0
    w_polar = _skirtor_polar_weight(i_deg, oa)
    w_screen = _skirtor_screen_weight(i_deg, oa)
    assert w_polar + w_screen == pytest.approx(1.0, abs=2e-3), (
        f"i={i_deg:g} deg: polar Type-1 {w_polar:.4f} + torus-screen Type-2 {w_screen:.4f}"
    )


def test_explicit_polar_oa_overrides_the_torus_angle():
    """A positive ``agn_polar_oa`` is the one angle; it need not equal the torus's."""
    got = _skirtor_polar_weight(50.0, 70.0, agn_polar_oa=40.0)  # polar boundary 90 - 40 = 50
    assert got == pytest.approx(0.5, abs=2e-3)


# ----------------------------------------------------------------------------------
# 2. the Fritz library elevation is derived from the one inclination
# ----------------------------------------------------------------------------------
def test_fritz_psy_is_derived_from_the_inclination():
    """psi = 90 deg - i (cos i = sin psi), held to the grid extent [0.001, 89.99]."""
    from tengri.components.agn.fritz import fritz_psy_from_cos_inc

    for i in (5.0, 20.0, 45.0, 60.0, 80.0):
        assert float(fritz_psy_from_cos_inc(_cos(i))) == pytest.approx(90.0 - i, abs=1e-6)
    assert float(fritz_psy_from_cos_inc(0.0)) == pytest.approx(0.001)
    # Face-on is capped 0.08 deg short of the pole (float32-safe arccos slope), inside the
    # last node interval of the library axis.
    assert 89.9 < float(fritz_psy_from_cos_inc(1.0)) <= 89.99


def test_fritz_torus_template_follows_the_inclination():
    """The library torus SED changes with i (it was keyed to a separate, default type-2 psi)."""
    face_on = _fritz(10.0, 60.0, tau=6.0)["torus"]
    edge_on = _fritz(80.0, 60.0, tau=6.0)["torus"]
    ratio = _at(edge_on, 3.0e4) / _at(face_on, 3.0e4)
    assert abs(np.log(ratio)) > 0.5, (
        f"Fritz torus L_nu at 3 um, i = 80 over i = 10 deg: {ratio:.3f}; the library "
        f"template does not follow the inclination"
    )


@pytest.mark.parametrize("half", _HALVES)
def test_fritz_disc_power_follows_the_inclination(half):
    """Disc power is Type 1 below i = half and screened above it.

    tau_V = 21 at tau_9.7 = 1; the infrared part of the disc passes the screen.
    """
    type1 = _power(_fritz(half - 12.0, half, tau=1.0)["disc"])
    type2 = _power(_fritz(half + 12.0, half, tau=1.0)["disc"])
    assert type2 < 0.35 * type1, (
        f"half={half}: disc power {type1:.4e} at i = {half - 12:g} deg, {type2:.4e} at "
        f"i = {half + 12:g} deg: no Type-1/2 switch at i = half"
    )


@pytest.fixture(scope="module")
def _model_inputs(synthetic_ssp_wide, synthetic_tophat_obs):
    return synthetic_ssp_wide, synthetic_tophat_obs


def _fritz_model(inputs, **torus):
    ssp, obs = inputs
    agn = {
        "type": "composable",
        "norm": "independent",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "fritz", "agn_torus_frac": Fixed(0.3), **torus},
        "agn_log_lbol": Fixed(11.0),
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        agn=agn,
        redshift=Fixed(0.1),
    )


def test_explicit_agn_fritz_psy_is_refused(_model_inputs):
    """``agn_fritz_psy`` is retired; the message names ``agn_cos_inc`` and the mapping."""
    with pytest.raises(ValueError, match=r"agn_cos_inc.*psy = 90 deg - i"):
        _fritz_model(_model_inputs, agn_fritz_psy=Fixed(50.1))


def test_fritz_model_builds_on_the_one_inclination(_model_inputs):
    model = _fritz_model(_model_inputs, agn_fritz_oa=Fixed(40.0))
    assert "agn_fritz_psy" not in model.spec.free_params
    state = model.predict_state({})
    assert np.all(np.isfinite(np.asarray(state.derived["sed_agn_torus"])))


# ----------------------------------------------------------------------------------
# 3. the Fritz grid axis and its documentation
# ----------------------------------------------------------------------------------
def _grid_axis(name):
    with h5py.File(DATA_DIR / "fritz2006_torus_grid.h5", "r") as f:
        return np.asarray(f[f"fritz2006/{name}"])


def test_fritz_oa_outside_the_grid_is_refused_at_build(_model_inputs):
    for bad in (70.0, 80.0, 100.0, 140.0, 10.0):
        with pytest.raises(ValueError, match="Fritz2006 grid extent"):
            _fritz_model(_model_inputs, agn_fritz_oa=Fixed(bad))


def test_fritz_sed_oa_outside_the_grid_is_refused():
    from tengri.components.agn.fritz import fritz_sed

    wave = jnp.asarray(np.geomspace(1.0e3, 1.0e6, 200))
    for bad in (80.0, 100.0, 140.0):
        with pytest.raises(ValueError, match="agn_fritz_oa"):
            fritz_sed(wave, agn_log_lbol=0.0, agn_torus_frac=1.0, agn_fritz_oa=bad)
    for good in _grid_axis("opening_angle_axis"):
        out = fritz_sed(wave, agn_log_lbol=0.0, agn_torus_frac=1.0, agn_fritz_oa=float(good))
        assert np.all(np.isfinite(np.asarray(out)))


def test_docstring_half_angle_lists_equal_the_grid_axis():
    from tengri.components.agn import fritz as FR
    from tengri.components.agn.blocks.torus import fritz_torus_block

    axis = {int(v) for v in _grid_axis("opening_angle_axis")}
    assert axis == {20, 40, 60}
    for obj in (FR.fritz_sed, FR.fritz_sed_from_grid, fritz_torus_block):
        doc = inspect.getdoc(obj)
        match = re.search(
            r"agn_fritz_oa[^\n]*\n(?:[^\n]*\n){0,3}?[^\n]*Allowed(?: values)?:\s*([\d, ]+)", doc
        )
        assert match is not None, f"{obj.__name__}: no 'Allowed' list for agn_fritz_oa"
        listed = {int(v) for v in re.findall(r"\d+", match.group(1))}
        assert listed == axis, f"{obj.__name__}: documented {sorted(listed)}, grid {sorted(axis)}"
        assert "100, 140" not in doc


# ----------------------------------------------------------------------------------
# 4. tau_9.7 is not tau_V
# ----------------------------------------------------------------------------------
def test_fritz_screen_depth_is_tau_v_from_the_stated_ratio():
    """Edge-on, ln(L(b)/L(a)) at V is minus the tau_V difference, tau_V = 0.4 ln10 x 23 tau_9.7."""
    tau_a, tau_b = 0.1, 0.3
    disc_a = _at(_fritz(89.0, 40.0, tau=tau_a)["disc"], _V)
    disc_b = _at(_fritz(89.0, 40.0, tau=tau_b)["disc"], _V)
    expected = -0.4 * np.log(10.0) * 23.0 * (tau_b - tau_a)
    assert np.log(disc_b / disc_a) == pytest.approx(expected, rel=2e-3), (
        f"edge-on optical depth difference at V is {np.log(disc_b / disc_a):.3f}, the Fritz+2006 "
        f"A_V/tau_9.7 = 23 gives {expected:.3f}"
    )


# ----------------------------------------------------------------------------------
# 5. the polar cone's share
# ----------------------------------------------------------------------------------
def _g(oa_deg):
    s = np.sin(np.radians(oa_deg))
    return 7.0 / 18.0 - s**2 / 6.0 - 2.0 * s**3 / 9.0


@pytest.mark.parametrize("i_deg", (0.0, 85.0), ids=("type1", "type2"))
def test_skirtor_polar_share_follows_the_torus_g(i_deg):
    """polar(oa)/polar(40) = g(oa)/g(40): the cone follows ``agn_oa_skirtor`` (was fixed at 45)."""
    power = {
        oa: _power(
            _skirtor(
                i_deg,
                oa,
                agn_attenuation_block="polar_dust",
                agn_polar_ebv=0.1,
                **_POLAR_PARAMS,
            )["polar"]
        )
        for oa in (10.0, 40.0, 70.0)
    }
    for oa in (10.0, 70.0):
        assert power[oa] / power[40.0] == pytest.approx(_g(oa) / _g(40.0), rel=3e-3), (
            f"oa={oa}: polar power ratio to oa=40 is {power[oa] / power[40.0]:.4f}, "
            f"g(oa)/g(40) = {_g(oa) / _g(40.0):.4f}"
        )


@pytest.mark.parametrize("i_deg", (0.0, 85.0), ids=("type1", "type2"))
def test_fritz_polar_share_follows_one_minus_cos_half(i_deg):
    """polar(half)/polar(60) = (1 - cos half)/(1 - cos 60): it was identical for 20 and 60 deg."""
    power = {
        half: _power(
            _fritz(
                i_deg,
                half,
                agn_attenuation_block="polar_dust",
                agn_polar_ebv=0.1,
                **_POLAR_PARAMS,
            )["polar"]
        )
        for half in _HALVES
    }
    for half in (20.0, 40.0):
        expected = (1.0 - _cos(half)) / (1.0 - _cos(60.0))
        assert power[half] / power[60.0] == pytest.approx(expected, rel=3e-3), (
            f"half={half}: polar power ratio to half=60 is {power[half] / power[60.0]:.4f}, "
            f"(1-cos half)/(1-cos 60) = {expected:.4f}"
        )


def test_explicit_polar_oa_fixes_the_cone_share():
    """``agn_polar_oa`` given: the share no longer moves with ``agn_oa_skirtor``."""
    power = [
        _power(
            _skirtor(
                0.0,
                oa,
                agn_attenuation_block="polar_dust",
                agn_polar_ebv=0.1,
                agn_polar_oa=30.0,
                **_POLAR_PARAMS,
            )["polar"]
        )
        for oa in (10.0, 70.0)
    ]
    assert power[0] == pytest.approx(power[1], rel=1e-6)


_TIE = dict(
    agn_disc_block="schartmann2005",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="skirtor",
    agn_attenuation_block="polar_dust",
    agn_norm="cigale_joint",
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_torus_frac=0.5,
    **_POLAR_PARAMS,
)


def _tied(ebv, i_deg=80.0):
    _, comps = compose_l_nu(
        _WAVE_TIE,
        _LOG_LBOL,
        agn_ir_frac=0.3,
        agn_cos_inc=_cos(i_deg),
        agn_polar_ebv=ebv,
        return_components=True,
        **_TIE,
    )
    return {k: np.asarray(v) for k, v in comps.items()}


@pytest.mark.parametrize("ebv", (0.1, 0.5))
def test_polar_power_is_inside_the_disc_normalization(ebv):
    """CIGALE divides disc and dust by ``1 + l_ext`` together: disc(E)/disc(0) = torus(E)/torus(0).

    The sightline is Type 2 for both the cone (i = 80 > 50) so no line-of-sight reddening
    enters the disc. The disc kept its whole ``agn_power x R`` while the dust gave up the
    polar share, so the ratio was 1.
    """
    base, dusty = _tied(0.0), _tied(ebv)
    disc_ratio = _power(dusty["disc"], _WAVE_TIE) / _power(base["disc"], _WAVE_TIE)
    torus_ratio = _power(dusty["torus"], _WAVE_TIE) / _power(base["torus"], _WAVE_TIE)
    assert disc_ratio == pytest.approx(torus_ratio, rel=2e-3), (
        f"E(B-V)={ebv}: disc(E)/disc(0) = {disc_ratio:.4f}, torus(E)/torus(0) = {torus_ratio:.4f}"
    )
    total = _power(dusty["torus"], _WAVE_TIE) + _power(dusty["polar"], _WAVE_TIE)
    assert total == pytest.approx(_power(base["torus"], _WAVE_TIE), rel=2e-3)


# ----------------------------------------------------------------------------------
# 6. extinction laws
# ----------------------------------------------------------------------------------
_LAW_WAVES = np.array(
    [1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 3650.0, 4400.0, 5500.0, 7000.0, 9000.0, 1.2e4, 2.0e4]
)  # [A]


def _published_k(law, wave_aa):
    """A(lambda)/E(B-V) from the published expressions, written out here."""
    wl_nm = wave_aa / 10.0
    if law == "calzetti":  # Calzetti et al. 2000, eqs. (3)-(4), R_V' = 4.05
        short = 2.659 * (-2.156 + 1.509e3 / wl_nm - 0.198e6 / wl_nm**2 + 0.011e9 / wl_nm**3) + 4.05
        long = 2.659 * (-1.857 + 1.040e3 / wl_nm) + 4.05
        return np.where(wl_nm < 630.0, short, long)
    if law == "gaskell":  # Gaskell et al. 2004, appendix, A(B)/A(V) = 1.182
        x = 1.0e3 / wl_nm
        a_av = np.where(
            x < 3.69,
            -0.8175 + 1.5848 * x - 0.3774 * x**2 + 0.0296 * x**3,
            1.3468 + 0.0087 * x,
        )
        return np.maximum(a_av, 0.0) / 0.182
    if law == "bongiorno":  # Bongiorno et al. 2012 power law, as used by CIGALE skirtor2016
        return 1.39 * (wave_aa * 1.0e-4) ** -1.2
    raise AssertionError(law)


@pytest.mark.parametrize("law", ("calzetti", "gaskell", "bongiorno"))
def test_polar_law_k_matches_the_published_curve(law):
    ebv = 0.05
    ones = jnp.ones(_LAW_WAVES.size)
    _, absorbed = polar_dust_extinction(ones, jnp.asarray(_LAW_WAVES), 1.0, 40.0, ebv, law=law)
    k = -np.log1p(-np.asarray(absorbed)) / (0.921 * ebv)
    np.testing.assert_allclose(k, _published_k(law, _LAW_WAVES), rtol=2e-3)


@pytest.mark.parametrize("name", ("bogus", "Calzetti", ""))
def test_unknown_polar_law_raises(name):
    ones = jnp.ones(3)
    wave = jnp.asarray([3000.0, 5500.0, 9000.0])
    with pytest.raises(ValueError, match="not a known polar-dust extinction law"):
        polar_dust_extinction(ones, wave, 1.0, 40.0, 0.1, law=name)
    with pytest.raises(ValueError, match="not a known polar-dust extinction law"):
        polar_dust_attenuation_block(wave, agn_polar_ebv=0.1, agn_polar_law=name)


# ----------------------------------------------------------------------------------
# 7. gradients and float32
# ----------------------------------------------------------------------------------
def _polar_absorbed(ebv, oa=40.0):
    wave = jnp.asarray(np.geomspace(1.0e3, 1.0e6, 1500))
    disc = jnp.exp(-((jnp.log(wave) - jnp.log(3000.0)) ** 2) / 2.0) * 1.0e30
    _, absorbed = polar_dust_reemission_lnu(
        wave, disc, agn_polar_ebv=ebv, agn_polar_oa=oa, return_absorbed=True
    )
    return absorbed


def test_absorbed_power_gradient_at_zero_ebv_is_the_one_sided_derivative():
    """d(absorbed)/dE(B-V) at 0 equals the one-sided difference; a clip at 0 halved it."""
    grad = float(jax.grad(_polar_absorbed)(0.0))
    h = 1.0e-8
    one_sided = float(_polar_absorbed(h) - _polar_absorbed(0.0)) / h
    assert np.isfinite(grad)
    assert grad != 0.0
    assert grad == pytest.approx(one_sided, rel=1e-6), (
        f"AD {grad:.6e} against the one-sided difference {one_sided:.6e} "
        f"(ratio {one_sided / grad:.4f})"
    )


def test_polar_reemission_gradient_at_zero_ebv_through_the_runner():
    def polar_power(ebv):
        comps = compose_l_nu(
            _WAVE,
            _LOG_LBOL,
            agn_cos_inc=_cos(80.0),
            agn_polar_ebv=ebv,
            return_components=True,
            **{
                **_BASE,
                "agn_torus_block": "skirtor",
                "agn_attenuation_block": "polar_dust",
                "agn_torus_frac": 0.3,
                **_POLAR_PARAMS,
            },
        )[1]
        nu = C_AA / _WAVE
        return -jnp.trapezoid(comps["polar"], nu)

    grad = float(jax.grad(polar_power)(0.0))
    h = 1.0e-7
    one_sided = float(polar_power(h) - polar_power(0.0)) / h
    assert np.isfinite(grad) and grad != 0.0
    assert grad == pytest.approx(one_sided, rel=1e-4)


def _disc_power_of_cos(cos_inc, half=40.0):
    _, comps = compose_l_nu(
        _WAVE,
        _LOG_LBOL,
        agn_cos_inc=cos_inc,
        return_components=True,
        **{
            **_BASE,
            "agn_torus_block": "fritz",
            "agn_attenuation_block": "polar_dust",
            "agn_fritz_oa": half,
            "agn_fritz_tau": 0.1,
            "agn_torus_frac": 0.3,
            "agn_polar_ebv": 0.1,
            **_POLAR_PARAMS,
        },
    )
    nu = C_AA / _WAVE
    return -jnp.trapezoid(comps["disc"], nu)


def test_gradient_wrt_cos_inc_across_the_type_boundary_is_finite_and_nonzero():
    """d(disc power)/d(cos i) at i = half is finite and non-zero (the soft edge is kept)."""
    for half in _HALVES:
        grad = float(jax.grad(_disc_power_of_cos)(_cos(half), half))
        assert np.isfinite(grad) and grad != 0.0, f"half={half}: gradient {grad}"


def test_float32_finite_across_the_boundary():
    with jax.enable_x64(False):
        wave = jnp.asarray(np.geomspace(500.0, 1.0e9, 2000), dtype=jnp.float32)
        for i in (10.0, 40.0, 70.0):
            _, comps = compose_l_nu(
                wave,
                _LOG_LBOL,
                agn_cos_inc=jnp.float32(_cos(i)),
                return_components=True,
                **{
                    **_BASE,
                    "agn_torus_block": "fritz",
                    "agn_attenuation_block": "polar_dust",
                    "agn_fritz_oa": 40.0,
                    "agn_fritz_tau": 1.0,
                    "agn_torus_frac": 0.3,
                    "agn_polar_ebv": 0.1,
                    **_POLAR_PARAMS,
                },
            )
            for name, comp in comps.items():
                assert np.all(np.isfinite(np.asarray(comp))), f"i={i}: {name} non-finite"
        grad = jax.grad(lambda c: jnp.sum(_disc_cos_f32(c, wave)))(jnp.float32(_cos(40.0)))
        assert np.isfinite(float(grad)) and float(grad) != 0.0


def _disc_cos_f32(cos_inc, wave):
    _, comps = compose_l_nu(
        wave,
        _LOG_LBOL,
        agn_cos_inc=cos_inc,
        return_components=True,
        **{
            **_BASE,
            "agn_torus_block": "fritz",
            "agn_attenuation_block": "polar_dust",
            "agn_fritz_oa": 40.0,
            "agn_fritz_tau": 1.0,
            "agn_torus_frac": 0.3,
            "agn_polar_ebv": 0.1,
            **_POLAR_PARAMS,
        },
    )
    return comps["disc"]

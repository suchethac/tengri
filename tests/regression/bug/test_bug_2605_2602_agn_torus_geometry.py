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
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
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

    At the grid's largest depth, tau = 10, the screen is deep at V and the infrared passes.
    """
    type1 = _power(_fritz(half - 12.0, half, tau=10.0)["disc"])
    type2 = _power(_fritz(half + 12.0, half, tau=10.0)["disc"])
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
# 4. the polar cone's share
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
    # A(lambda) = E(B-V) k(lambda) mag, transmission 10^(-0.4 A) = exp(-0.4 ln10 A): the
    # coefficient is written here, not read back from the code under test.
    k = -np.log1p(-np.asarray(absorbed)) / (0.4 * np.log(10.0) * ebv)
    np.testing.assert_allclose(k, _published_k(law, _LAW_WAVES), rtol=1e-4)


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


# ----------------------------------------------------------------------------------
# 8. the SMC curve against Pei (1992) Table 4, and the law through the builder
# ----------------------------------------------------------------------------------
# Pei (1992, ApJ 395, 130) Table 4, Small Magellanic Cloud rows, read from the scanned page:
# columns a_i, lambda_i [um], b_i, n_i of
# xi(lambda) = sum_i a_i / [(l/l_i)^n_i + (l_i/l)^n_i + b_i]
# (eq. 20). The installed pcigale's law 0 is the Bongiorno power law, not this curve, so there is
# no pcigale SMC curve at these wavelengths to compare with; the two are reported side by side.
_PEI_SMC = np.array(
    [
        (185.0, 0.042, 90.0, 2.0),
        (27.0, 0.08, 5.50, 4.0),
        (0.005, 0.22, -1.95, 2.0),
        (0.010, 9.7, -1.95, 2.0),
        (0.012, 18.0, -1.80, 2.0),
        (0.030, 25.0, 0.00, 2.0),
    ]
)
_PEI_WAVES = np.array(
    [1216.0, 1500.0, 2000.0, 2500.0, 3000.0, 3650.0, 4400.0, 5500.0, 7000.0, 9000.0, 1.2e4, 2.2e4]
)  # [A]


def _pei_xi(wave_aa):
    a, lam, b, n = _PEI_SMC.T
    x = np.asarray(wave_aa)[:, None] * 1.0e-4
    return np.sum(a / ((x / lam) ** n + (lam / x) ** n + b), axis=1)


def test_polar_smc_curve_has_the_pei_1992_table4_shape():
    """tengri's SMC k(lambda) is Pei's Table 4 SMC curve up to one constant, at 12 wavelengths."""
    ebv = 0.05
    wave = jnp.asarray(_PEI_WAVES)
    _, absorbed = polar_dust_extinction(jnp.ones(wave.size), wave, 1.0, 40.0, ebv, law="smc")
    a_lambda = -np.log1p(-np.asarray(absorbed)) / 0.921  # A(lambda) [mag] = E(B-V) R_V k
    ratio = a_lambda / _pei_xi(_PEI_WAVES)
    np.testing.assert_allclose(ratio / ratio[7], 1.0, atol=1e-3)


def _agn_builder_model(inputs, **agn_extra):
    ssp, obs = inputs
    atten_extra = agn_extra.pop("atten_extra", {})
    agn = {
        "type": "composable",
        "norm": "independent",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "agn_torus_frac": Fixed(0.3), "all_params": Fixed(DEFAULT)},
        "atten": {
            "type": "polar_dust",
            "agn_polar_ebv": Fixed(0.2),
            "all_params": Fixed(DEFAULT),
            **atten_extra,
        },
        "agn_log_lbol": Fixed(11.0),
        "agn_cos_inc": Fixed(1.0),
        "all_params": Fixed(DEFAULT),
        **agn_extra,
    }
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        agn=agn,
        redshift=Fixed(0.1),
    )


def _polar_total(model):
    state = model.predict_state({})
    return float(np.sum(np.asarray(state.derived["sed_agn_polar"])))


def test_polar_law_is_reachable_through_the_builder(_model_inputs, capsys):
    """``agn['polar_law']`` and ``agn['atten']['polar_law']`` both select the curve."""
    default = _polar_total(_agn_builder_model(_model_inputs))
    smc_law = _polar_total(_agn_builder_model(_model_inputs, polar_law="smc"))
    assert default == smc_law
    laws = {}
    for law in ("calzetti", "gaskell", "bongiorno"):
        top = _agn_builder_model(_model_inputs, polar_law=law)
        nested = _agn_builder_model(_model_inputs, atten_extra={"polar_law": law})
        laws[law] = _polar_total(top)
        assert laws[law] == _polar_total(nested)
        assert laws[law] != default, f"polar_law={law!r} left the polar re-emission unchanged"
    assert len(set(laws.values())) == 3
    _agn_builder_model(_model_inputs, polar_law="calzetti").spec.summary()
    assert "polar_law=calzetti" in capsys.readouterr().out


def test_polar_law_round_trips_through_to_groups_in_both_spellings(_model_inputs):
    """Either spelling re-emits as ``agn['polar_law']`` and rebuilds the same model."""
    for law in ("calzetti", "bongiorno"):
        top = _agn_builder_model(_model_inputs, polar_law=law)
        nested = _agn_builder_model(_model_inputs, atten_extra={"polar_law": law})
        for model in (top, nested):
            groups = model.spec.to_groups()
            assert groups["agn"]["polar_law"] == law
            assert "polar_law" not in groups["agn"].get("atten", {})
    default_groups = _agn_builder_model(_model_inputs).spec.to_groups()
    assert "polar_law" not in default_groups["agn"]


def test_unknown_polar_law_is_refused_at_build_with_the_menu(_model_inputs):
    for kwargs in ({"polar_law": "bogus"}, {"atten_extra": {"polar_law": "bogus"}}):
        with pytest.raises(ValueError, match=r"polar_law.*bongiorno"):
            _agn_builder_model(_model_inputs, **kwargs)


# ----------------------------------------------------------------------------------
# 9. the polar law is CIGALE's: Bongiorno above 100 nm, the tabulated SMC shape below
# ----------------------------------------------------------------------------------
# Rows of Draine's public ``kext_albedo_WD_SMCbar_0`` (Weingartner & Draine 2001 SMC-bar model;
# https://www.astro.princeton.edu/~draine/dust/extcurvs/): wavelength [um] and extinction cross
# section per H nucleon [cm^2/H], as printed. The first row, four more below 100 nm, and the row
# at 100 nm and the one after it. The mass extinction coefficient is C_ext/H over the file's
# 3.506e-27 g/H of dust; only the shape (a ratio) enters the law.
_DRAINE_M_DUST_PER_H = 3.506e-27
_SMC_TABLE = (
    (1.000e-03, 1.814e-23),
    (3.255e-03, 6.062e-23),
    (1.059e-02, 1.733e-22),
    (3.447e-02, 2.658e-22),
    (7.413e-02, 6.306e-22),
    (1.000e-01, 4.857e-22),
    (1.006e-01, 4.825e-22),
)


def test_bongiorno_below_100nm_is_the_draine_table_shape():
    """k(lambda < 100 nm) = kappa x k_power(100 nm)/kappa(100 nm), kappa from Draine rows."""
    kappa = [(w, c / _DRAINE_M_DUST_PER_H) for w, c in _SMC_TABLE]  # cm^2/g, scale cancels
    table_100nm = kappa[-2][1]  # the table has a node at exactly 100 nm
    k_100nm = 1.39 * 0.1**-1.2
    nodes = np.array(kappa[:5])  # five nodes below 100 nm
    expected = nodes[:, 1] * k_100nm / table_100nm
    wave = jnp.asarray(nodes[:, 0] * 1.0e4)  # [A]
    _, absorbed = polar_dust_extinction(jnp.ones(5), wave, 1.0, 40.0, 0.01, law="bongiorno")
    k = -np.log1p(-np.asarray(absorbed)) / (0.4 * np.log(10.0) * 0.01)
    np.testing.assert_allclose(k, expected, rtol=1e-4)
    # continuous at 100 nm, and the power law just above it
    above = jnp.asarray([1000.0 * 1.0001, 1500.0, 2000.0, 5500.0])
    _, abs_above = polar_dust_extinction(jnp.ones(4), above, 1.0, 40.0, 0.01, law="bongiorno")
    k_above = -np.log1p(-np.asarray(abs_above)) / (0.4 * np.log(10.0) * 0.01)
    np.testing.assert_allclose(k_above, 1.39 * (np.asarray(above) * 1e-4) ** -1.2, rtol=1e-4)
    assert k_above[0] == pytest.approx(k_100nm, rel=1e-3)


#: Largest |tengri / pcigale - 1| of the sub-100 nm shape, relative to 100 nm, by band of
#: wavelength [um], measured against pcigale 2025.1's ``extFun_SMC.dat`` nodes: 18.9 % at 1 nm,
#: 15.3 % (near 22 nm) between 10 and 35 nm, 5.4 % from 35 to 100 nm. The cause is the public
#: Weingartner & Draine (2001) table against pcigale's own re-computed SMC mixture.
_SHAPE_BANDS = ((1.0e-3, 1.0e-2, 0.195), (1.0e-2, 3.5e-2, 0.155), (3.5e-2, 0.1, 0.055))


def test_sub_100nm_shape_matches_pcigale_to_the_measured_residual():
    """Shape relative to 100 nm against pcigale's tabulated SMC mixture, band by band."""
    pcigale = pytest.importorskip("pcigale")
    path = Path(pcigale.__file__).parent / "sed_modules" / "curves" / "extFun_SMC.dat"
    if not path.exists():
        pytest.skip("pcigale does not ship extFun_SMC.dat")
    table = np.loadtxt(path)
    wave_um, ext = table[:, 0], table[:, 1]
    keep = wave_um < 0.1
    pc_shape = ext[keep] / np.interp(0.1, wave_um, ext)
    _, absorbed = polar_dust_extinction(
        jnp.ones(int(keep.sum())), jnp.asarray(wave_um[keep] * 1.0e4), 1.0, 40.0, 0.01,
        law="bongiorno",
    )  # fmt: skip
    k = -np.log1p(-np.asarray(absorbed)) / (0.4 * np.log(10.0) * 0.01)
    shape = k / (1.39 * 0.1**-1.2)
    residual = np.abs(shape / pc_shape - 1.0)
    for lo, hi, bound in _SHAPE_BANDS:
        band = (wave_um[keep] >= lo) & (wave_um[keep] < hi)
        assert band.any()
        assert residual[band].max() < bound, (
            f"{lo:g}-{hi:g} um: shape residual {residual[band].max():.4f} exceeds {bound}"
        )
        # the bound is the measurement, not slack: the residual reaches within 1 % of it
        assert residual[band].max() > bound - 0.01


# pcigale ``skirtor2016`` (SKIRTOR, t = 7, p = q = 1, R = 20, i = 0, schartmann2005 disc,
# extinction_law = 0, fracAGN = 0.3), measured with the installed pcigale 2025.1:
# (oa [deg], E(B-V), polar graybody / agn_power, disc / agn_power).
_PCIGALE_SHARES = (
    (10.0, 0.03, 0.85875, None),
    (40.0, 0.03, 0.20499, 2.7390),
    (70.0, 0.03, 0.02330, None),
    (40.0, 0.1, 0.36310, 1.4323),
    (40.0, 0.5, 0.47706, 0.4904),
)

#: The residual is the curve's match point and the quadrature grid, both measured on
#: pcigale's own 948-node grid: CIGALE rescales the sub-100 nm table at the last grid node
#: below 100 nm (95.5 nm) and tengri at 100 nm, which moves the absorbed power by -0.41 %,
#: -0.18 % and -0.004 % at E(B-V) 0.03, 0.1 and 0.5; integrating on 948 nodes rather than a
#: dense grid moves it by a further 0.10 %, 0.08 % and 0.04 %. Measured agreement: polar
#: 0.9996 / 0.9974 / 0.9963 / 0.9973 / 0.9989, disc 0.9997 / 1.0031 / 1.0015. The disc's
#: 0.31 % is the Draine WD01 sub-100 nm shape (public table) against pcigale's own SMC mixture.
_POLAR_REL = 1.0e-2
_DISC_REL = 3.2e-3


def _tied_shares(oa, ebv, law="bongiorno"):
    """(polar, disc) / agn_power on the CIGALE-tied SKIRTOR path at i = 0."""
    kwargs = {**_TIE, "agn_oa_skirtor": oa, "agn_polar_law": law}

    def run(e):
        _, comps = compose_l_nu(
            _WAVE_TIE,
            _LOG_LBOL,
            agn_ir_frac=0.3,
            agn_cos_inc=1.0,
            agn_polar_ebv=e,
            return_components=True,
            **kwargs,
        )
        return {k: np.asarray(v) for k, v in comps.items()}

    on, off = run(ebv), run(0.0)
    budget = _power(off["torus"], _WAVE_TIE)
    return _power(on["polar"], _WAVE_TIE) / budget, _power(on["disc"], _WAVE_TIE) / budget


@pytest.mark.parametrize(("oa", "ebv", "polar", "disc"), _PCIGALE_SHARES)
def test_polar_share_and_disc_match_pcigale_with_the_bongiorno_law(oa, ebv, polar, disc):
    got_polar, got_disc = _tied_shares(oa, ebv)
    assert got_polar == pytest.approx(polar, rel=_POLAR_REL), (
        f"oa={oa}, E(B-V)={ebv}: polar/agn_power {got_polar:.5f}, pcigale {polar}"
    )
    if disc is not None:
        assert got_disc == pytest.approx(disc, rel=_DISC_REL), (
            f"oa={oa}, E(B-V)={ebv}: disc/agn_power {got_disc:.4f}, pcigale {disc}"
        )


@pytest.mark.parametrize(
    ("oa", "ebv", "polar", "disc"), _PCIGALE_SHARES[:2] + _PCIGALE_SHARES[3:4]
)
def test_polar_share_matches_the_installed_pcigale(oa, ebv, polar, disc):
    """The same comparison against a live pcigale ``skirtor2016`` run (skipped without it)."""
    pytest.importorskip("pcigale")
    from pcigale.sed import SED
    from pcigale.sed_modules import skirtor2016 as PS

    sed = SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    PS.SKIRTOR2016(
        name="skirtor2016", t=7, pl=1.0, q=1.0, oa=int(oa), R=20, Mcl=0.97, i=0, disk_type=1,
        delta=0, fracAGN=0.3, lambda_fracAGN="0/0", law=0, EBV=ebv, temperature=100.0,
        emissivity=1.6,
    ).process(sed)  # fmt: skip
    power = 0.3 / 0.7
    pc_polar = sed.info["agn.polar_dust_luminosity"] / power
    pc_disc = sed.info["agn.disk_luminosity"] / power
    got_polar, got_disc = _tied_shares(oa, ebv)
    assert pc_polar == pytest.approx(polar, rel=1e-4)
    assert got_polar == pytest.approx(pc_polar, rel=_POLAR_REL)
    if disc is not None:
        assert got_disc == pytest.approx(pc_disc, rel=_DISC_REL)


def _unit_disc():
    wave = jnp.asarray(np.geomspace(1.0e3, 3.0e4, 1500))  # [A], above 100 nm
    return wave, jnp.exp(-((jnp.log(wave) - jnp.log(3000.0)) ** 2) / 2.0)


@pytest.mark.parametrize("half", _HALVES)
def test_fritz_cone_absorbed_power_matches_the_cigale_formula(half):
    """l_ext = (1 - cos half) int disk (1 - 10^(-0.4 k E)) dlambda (CIGALE fritz2006.py:309)."""
    wave, disc = _unit_disc()
    ebv = 0.1
    _, absorbed = polar_dust_reemission_lnu(
        wave,
        disc,
        agn_polar_ebv=ebv,
        agn_polar_oa=90.0 - half,
        agn_polar_geometry="fritz",
        agn_polar_law="bongiorno",
        return_absorbed=True,
    )
    k = 1.39 * (np.asarray(wave) * 1.0e-4) ** -1.2
    integrand = np.asarray(disc) * (1.0 - 10.0 ** (-0.4 * k * ebv))
    expected = (1.0 - np.cos(np.radians(half))) * np.trapezoid(integrand, np.asarray(wave))
    assert float(absorbed) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("oa", (10.0, 40.0, 70.0))
def test_skirtor_cone_absorbed_power_matches_the_cigale_formula(oa):
    """l_ext = g(oa) int disk (1 - 10^(-0.4 k E)) dlambda (CIGALE skirtor2016.py:366-368)."""
    wave, disc = _unit_disc()
    ebv = 0.1
    _, absorbed = polar_dust_reemission_lnu(
        wave,
        disc,
        agn_polar_ebv=ebv,
        agn_polar_oa=oa,
        agn_polar_reference="face_on",
        agn_polar_law="bongiorno",
        return_absorbed=True,
    )
    k = 1.39 * (np.asarray(wave) * 1.0e-4) ** -1.2
    integrand = np.asarray(disc) * (1.0 - 10.0 ** (-0.4 * k * ebv))
    expected = _g(oa) * np.trapezoid(integrand, np.asarray(wave))
    assert float(absorbed) == pytest.approx(expected, rel=1e-4)


# ----------------------------------------------------------------------------------
# 10. agn_polar_oa: 0 is the one sentinel for "follow the torus"
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prior",
    (Uniform(0.0, 80.0), Fixed(-5.0), Uniform(-1.0, 40.0), Fixed(120.0)),
    ids=("reaches_zero", "negative", "negative_support", "above_90"),
)
def test_polar_oa_prior_reaching_zero_is_refused(_model_inputs, prior):
    """A value or prior that reaches <= 0 (other than the default 0) names the way to follow."""
    with pytest.raises(ValueError, match=r"agn_polar_oa.*leave agn_polar_oa unset"):
        _agn_builder_model(_model_inputs, atten_extra={"polar_oa": prior})


def test_polar_oa_default_and_explicit_angles_build(_model_inputs):
    for prior in (Fixed(DEFAULT), Fixed(0.0), Fixed(40.0), Uniform(10.0, 80.0)):
        _agn_builder_model(_model_inputs, atten_extra={"polar_oa": prior})


def test_summary_says_the_cone_follows_the_torus(_model_inputs, capsys):
    _agn_builder_model(_model_inputs).spec.summary()
    row = next(ln for ln in capsys.readouterr().out.splitlines() if "agn_polar_oa" in ln)
    assert "follows torus" in row and "agn_oa_skirtor" in row
    _agn_builder_model(_model_inputs, atten_extra={"polar_oa": Fixed(40.0)}).spec.summary()
    row = next(ln for ln in capsys.readouterr().out.splitlines() if "agn_polar_oa" in ln)
    assert "follows torus" not in row


# ----------------------------------------------------------------------------------
# 11. absorbed power is non-negative for every law, 100 A to 1 mm
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("law", ("smc", "calzetti", "gaskell", "bongiorno"))
@pytest.mark.parametrize("x64", (True, False), ids=("float64", "float32"))
def test_absorbed_power_is_non_negative_for_every_law(law, x64):
    """The Calzetti polynomial is negative beyond 3.1 um (min k = -0.80 at 30 um)."""
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        wave = jnp.asarray(np.geomspace(100.0, 1.0e8, 4000), dtype=dtype)
        for ebv in (0.0, 1.0e-6, 0.05, 0.5):
            _, absorbed = polar_dust_extinction(
                jnp.ones_like(wave), wave, 1.0, 40.0, jnp.asarray(ebv, dtype=dtype), law=law
            )
            assert float(jnp.min(absorbed)) >= 0.0, f"{law} E(B-V)={ebv}: negative absorption"
            assert np.all(np.isfinite(np.asarray(absorbed)))


def test_calzetti_is_zero_beyond_its_polynomial_zero_crossing():
    wave_nm = np.array([400.0, 2000.0, 4000.0, 30000.0, 300000.0])  # zero crossing at 3115 nm
    _, absorbed = polar_dust_extinction(
        jnp.ones(5), jnp.asarray(wave_nm * 10.0), 1.0, 40.0, 0.1, law="calzetti"
    )
    np.testing.assert_array_equal(np.asarray(absorbed)[2:], 0.0)
    assert np.all(np.asarray(absorbed)[:2] > 0.0)


# ----------------------------------------------------------------------------------
# 12. the tied Fritz torus under polar dust, against CIGALE ``fritz2006``
# ----------------------------------------------------------------------------------
# CIGALE ``fritz2006`` (r = 60, tau = 1, beta = -0.5, gamma = 4, opening_angle = 60 (the
# half-angle of the dust-free cone is 60 deg), schartmann2005 disc, law = 0, fracAGN = 0.3).
# Its polar blackbody joins the dust BEFORE ``norm = 1 / int dust`` (fritz2006.py:325-331), so
# the disc, the polar blackbody and the intrinsic accretion power are all scaled by ``norm``:
# on a Type-2 sightline the disc is not extincted (line 305), hence
# ``disk(E) = disk(0) x norm(E)/norm(0) = disk(0) / (1 + l_ext)`` (line 309 gives ``l_ext``).
_FRITZ_NODE = dict(r_ratio=60.0, tau=1.0, beta=-0.5, gamma=4.0, opening_angle=60.0)
_FRITZ_HALF = 60.0  # the node's dust-free cone half-angle [deg]
_FRITZ_TIE = dict(
    agn_disc_block="schartmann2005",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="fritz",
    agn_attenuation_block="polar_dust",
    agn_polar_law="bongiorno",
    agn_norm="cigale_joint",
    agn_fritz_oa=_FRITZ_HALF,
    agn_torus_frac=0.5,
    **_POLAR_PARAMS,
)
_FRITZ_FRAC = 0.3
_FRITZ_TIE_TAU = 1.0  # agn_fritz_tau, the node's tau
#: Measured |b / b_CIGALE - 1|: 2e-6 (E(B-V) = 0), 1.4e-4 at most under polar dust (the torus
#: budget quadrature of tengri against CIGALE's, and the polar share the run itself reports).
_CORONA_REL = 2.0e-4
_FRITZ_WAVE = jnp.asarray(np.geomspace(8.0, 1.0e8, 40000))  # [A] covers the library axis


def _cos_of_psy(psy_deg):
    """``agn_cos_inc`` of the Fritz library viewing elevation (psi = 90 deg - i)."""
    return float(np.sin(np.radians(psy_deg)))


def _pcigale_fritz(psy, ebv):
    """``agn.{disk_luminosity, polar_dust_luminosity, accretion_power}`` per ``agn_power``."""
    pcigale_sed = pytest.importorskip("pcigale.sed")
    fritz_module = pytest.importorskip("pcigale.sed_modules.fritz2006")
    sed = pcigale_sed.SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    fritz_module.Fritz2006(
        name="fritz2006", **_FRITZ_NODE, psy=psy, disk_type=1, delta=0.0, fracAGN=_FRITZ_FRAC,
        lambda_fracAGN="0/0", law=0, EBV=ebv, temperature=100.0, emissivity=1.6,
    ).process(sed)  # fmt: skip
    per = _FRITZ_FRAC / (1.0 - _FRITZ_FRAC)
    keys = ("agn.disk_luminosity", "agn.polar_dust_luminosity", "agn.accretion_power")
    return {k: sed.info[k] / per for k in keys}


def _tied_fritz(psy, ebv, *, disc="schartmann2005", mbh=8.0):
    """Components of the tied Fritz composition at the library elevation ``psy`` [erg/s/Hz]."""
    _, comps = compose_l_nu(
        _FRITZ_WAVE,
        _LOG_LBOL,
        agn_ir_frac=_FRITZ_FRAC,
        agn_cos_inc=_cos_of_psy(psy),
        agn_polar_ebv=ebv,
        agn_log_mbh=mbh,
        return_components=True,
        **{**_FRITZ_TIE, "agn_disc_block": disc},
    )
    return {k: np.asarray(v) for k, v in comps.items() if k != "log_L_agn_los"}


_TYPE2_PSY = (10.1, 20.1)  # i = 79.9 and 69.9 deg, beyond the 60 deg cone half-angle


@pytest.mark.parametrize("psy", _TYPE2_PSY)
@pytest.mark.parametrize("ebv", (0.03, 0.1, 0.5))
def test_fritz_disc_under_polar_dust_is_the_unattenuated_disc_over_one_plus_l_ext(psy, ebv):
    """disc(E) = disc(0) / (1 + l_ext) at every wavelength, ``l_ext`` from CIGALE ``fritz2006``.

    CIGALE's ``polar_dust_luminosity / agn_power`` is ``l_ext / (1 + l_ext)``, so
    ``1 / (1 + l_ext)`` is one minus it, and it is also CIGALE's own disc ratio
    (fritz2006.py:328-331). Measured agreement of the disc ratio: 0.20 % / 0.28 % / 0.04 % at
    E(B-V) = 0.03 / 0.1 / 0.5 (the Weingartner-Draine sub-100 nm shape and the quadrature).
    """
    pc_on, pc_off = _pcigale_fritz(psy, ebv), _pcigale_fritz(psy, 0.0)
    norm_ratio = 1.0 - pc_on["agn.polar_dust_luminosity"]
    assert pc_on["agn.disk_luminosity"] / pc_off["agn.disk_luminosity"] == pytest.approx(
        norm_ratio, rel=1e-6
    )
    on, off = _tied_fritz(psy, ebv), _tied_fritz(psy, 0.0)
    live = off["disc"] > 1.0e-6 * off["disc"].max()
    pointwise = on["disc"][live] / off["disc"][live]
    # one factor, not a reshaping; at psy = 20.1 the polar mask's Type-1 weight is 1.9e-3 and
    # leaves a spread across wavelength of 0.19 % at most, at E(B-V) = 0.5 (2e-6 at psy = 10.1)
    np.testing.assert_allclose(pointwise, pointwise[0], rtol=2.0e-3)
    assert pointwise[0] == pytest.approx(norm_ratio, rel=3.0e-3), (
        f"psy={psy}, E(B-V)={ebv}: disc(E)/disc(0) = {pointwise[0]:.5f}, "
        f"CIGALE 1/(1 + l_ext) = {norm_ratio:.5f}"
    )
    power = _power(on["torus"], _FRITZ_WAVE) + _power(on["polar"], _FRITZ_WAVE)
    assert _power(on["polar"], _FRITZ_WAVE) / power == pytest.approx(
        pc_on["agn.polar_dust_luminosity"], rel=3.0e-3
    )


@pytest.mark.parametrize("psy", _TYPE2_PSY)
@pytest.mark.parametrize("ebv", (0.0, 0.1))
def test_fritz_type2_corona_under_the_torus_screen_matches_cigale_accretion_power(psy, ebv):
    """The X-ray corona behind a Type-2 Fritz torus is ``b H T`` with ``b`` from CIGALE.

    Below the library's 10 A edge the tied disc vanishes and the corona alone is left: the
    model's corona ``H`` times the torus screen ``T`` (the dusty torus's own transmission at
    this sightline, tau_V = 1) times ``b = P_D,tied / ((1 - f) L_acc)``. ``P_D,tied`` is CIGALE
    ``fritz2006``'s ``accretion_power`` (``int AGN1.disk x norm``; the Fritz disc is isotropic,
    hemisphere mean 1), and it carries CIGALE's ``1/(1 + l_ext)`` under polar dust, so the
    corona shares the disc's normalization. Measured ``b`` against that: see the tolerance.

    Under polar dust the ``1/(1 + l_ext)`` factor is taken from this model's own polar share:
    CIGALE's ``fritz2006`` runs a ``schartmann2005`` disc, whose polar share differs from the
    ``kubota_done`` disc's. So this cell tests the corona's single scale, the torus screen on a
    Type-2 sightline and ``b`` against CIGALE's accretion power, not the polar share itself;
    ``test_fritz_disc_under_polar_dust_is_the_unattenuated_disc_over_one_plus_l_ext`` tests
    that against CIGALE.
    """
    from tengri.components.agn.blocks import _protocol as protocol
    from tengri.components.agn.blocks.torus_screen import torus_screen_transmission
    from tengri.utils.physics_constants import L_SUN

    mbh = 8.0
    comps = _tied_fritz(psy, ebv, disc="kubota_done", mbh=mbh)
    wave = np.asarray(_FRITZ_WAVE, dtype=float)
    below = wave < 9.0
    _, h_lam, f_corona = protocol.DISC_SPLIT_BLOCKS["kubota_done"](
        _FRITZ_WAVE, _LOG_LBOL, agn_log_mbh=mbh
    )
    cos_inc = _cos_of_psy(psy)
    screen = np.asarray(
        torus_screen_transmission(
            _FRITZ_WAVE, cos_inc=cos_inc, oa_deg=90.0 - _FRITZ_HALF, tau_v=_FRITZ_TIE_TAU
        )
    )
    tied = comps["disc"] * C_AA / wave**2  # L_lambda [erg/s/A]
    b = tied[below] / (np.asarray(h_lam)[below] * screen[below])
    np.testing.assert_allclose(b, b[0], rtol=1e-9)  # one scale: b H T
    polar = _power(comps["polar"], _FRITZ_WAVE)
    agn_power = _power(comps["torus"], _FRITZ_WAVE) + polar
    # CIGALE scales the intrinsic accretion power by norm(E)/norm(0) = 1 - polar share
    # (fritz2006.py:331, 339); the share is taken from this run, whose kubota_done disc
    # reprocesses differently from CIGALE's schartmann2005 one, and CIGALE's own relation is
    # checked on its schartmann2005 run.
    pc_on, pc_off = _pcigale_fritz(psy, ebv), _pcigale_fritz(psy, 0.0)
    assert pc_on["agn.accretion_power"] / pc_off["agn.accretion_power"] == pytest.approx(
        1.0 - pc_on["agn.polar_dust_luminosity"], rel=1e-6
    )
    accretion = pc_off["agn.accretion_power"] * (1.0 - polar / agn_power)
    expected = accretion * agn_power / ((1.0 - float(f_corona)) * 10.0**_LOG_LBOL * L_SUN)
    assert b[0] == pytest.approx(expected, rel=_CORONA_REL), (
        f"psy={psy}, E(B-V)={ebv}: b = {b[0]:.5f}, CIGALE-derived {expected:.5f}"
    )

# SPDX-License-Identifier: BSD-3-Clause
r"""``agn_log_lbol`` is the accretion power; the line-of-sight luminosity is derived (#2678).

A geometrically thin, optically thick disc radiates :math:`\propto\cos i` from each
face (Kubota & Done 2018, MNRAS, 480, 1247, Sect. 2.1); the hot corona is isotropic
(Sect. 2.2). With :math:`D_\nu` the angle-integrated (two-face) spectral luminosity of
the disc and warm zones, :math:`H_\nu` that of the corona and
:math:`L_{\rm acc} = \int (D_\nu + H_\nu)\,{\rm d}\nu`:

* ``agn_log_lbol`` :math:`= \log_{10}(L_{\rm acc}/L_\odot)` and does not depend on :math:`i`;
* the spectrum an observer at inclination :math:`i` assigns assuming isotropy
  (:math:`4\pi d^2 F_\nu`) is :math:`L_\nu(i) = 2\cos i\,D_\nu + H_\nu`;
* the average of :math:`\int L_\nu(i)\,{\rm d}\nu` over :math:`\cos i \in [0, 1]` is
  :math:`L_{\rm acc}` (since :math:`\langle 2\cos i\rangle = 1`), and at
  :math:`\cos i = 0.5` the line-of-sight power equals :math:`L_{\rm acc}`.

Every expected value below is written out from these formulas, never read from a pin.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.components.agn import disc as disc_module
from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.regression_bug

_SSP_PATH = Path(__file__).resolve().parents[3] / "data" / "fsps_prsc_miles_chabrier.h5"

_LOG_LBOL = 12.0
_L_ACC = 10.0**_LOG_LBOL * L_SUN  # [erg/s]
_INCLINATIONS = (0, 30, 45, 60, 75)  # [deg]
_COS = {i: float(np.cos(np.radians(i))) for i in _INCLINATIONS}
_WIDE = np.geomspace(1.0e-2, 1.0e9, 6001)  # [A] holds every zone of the disc and the corona
_OPTICAL_UV = (5100.0, 1500.0)  # [A]
# [A]; the warm Comptonization tail (a disc zone, so ~cos i) still contributes 1e-4 of the flux
# at 2-4 keV and below 1e-6 above 6 keV, where the corona alone is left
_XRAY_6_10_KEV = 12.398419843 / np.linspace(6.0, 10.0, 5)
_NONE = dict(
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="none",
    agn_attenuation_block="none",
)
#: Measured deviation of the trapezoid of the returned spectrum on ``_WIDE`` from the
#: closed-form bolometric normalization (kubota_done 7.1e-4, multicolor 2.1e-5); the energy
#: checks assert at three times these.
_ENERGY_TOL = {"kubota_done": 3 * 7.1e-4, "multicolor": 3 * 2.1e-5}


def _power(l_nu, wave):
    """``int L_nu dnu`` [erg/s] of an ``L_nu`` [erg/s/Hz] spectrum on ``wave`` [A]."""
    nu = C_AA / np.asarray(wave, dtype=np.float64)
    order = np.argsort(nu)
    return float(np.trapezoid(np.asarray(l_nu, dtype=np.float64)[order], nu[order]))


def _run(disc_block, cos_inc, wave=_WIDE, **kwargs):
    """``compose_l_nu(..., return_components=True)`` with only the disc block active."""
    sed, comps = compose_l_nu(
        jnp.asarray(wave),
        _LOG_LBOL,
        agn_disc_block=disc_block,
        agn_cos_inc=cos_inc,
        return_components=True,
        **{**_NONE, **kwargs},
    )
    return np.asarray(sed), {k: np.asarray(v) for k, v in comps.items()}


def _disc_lnu(disc_block, cos_inc, wave):
    """The disc component ``L_nu`` [erg/s/Hz] at the requested wavelengths [A]."""
    return _run(disc_block, cos_inc, wave)[1]["disc"]


# ----------------------------------------------------------------------------------
# a. kubota_done through the public composable AGN: disc + warm ~ cos i / 0.5
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("wave_aa", _OPTICAL_UV)
@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_kubota_done_disc_and_warm_scale_as_cos_i_over_half(i_deg, wave_aa):
    """L_nu(i) / L_nu(60 deg) = cos i / 0.5 at 5100 and 1500 A, to 1e-3."""
    wave = np.array([wave_aa])
    level = _disc_lnu("kubota_done", _COS[i_deg], wave)[0]
    reference = _disc_lnu("kubota_done", _COS[60], wave)[0]
    assert level / reference == pytest.approx(_COS[i_deg] / 0.5, rel=1e-3)


@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_kubota_done_corona_is_isotropic(i_deg):
    """The corona (6-10 keV, above the warm Comptonization tail) does not depend on i, to 1e-6."""
    level = _disc_lnu("kubota_done", _COS[i_deg], _XRAY_6_10_KEV)
    reference = _disc_lnu("kubota_done", _COS[60], _XRAY_6_10_KEV)
    np.testing.assert_allclose(level, reference, rtol=1e-6, atol=0.0)


# ----------------------------------------------------------------------------------
# b. energy: <int L_nu(i) dnu> over cos i in [0, 1] = L_acc; power(cos i = 0.5) = L_acc
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("disc_block", ["kubota_done", "multicolor"])
def test_sphere_average_power_is_the_accretion_power(disc_block):
    """The 16-node Gauss-Legendre mean over cos i in [0, 1] of the power is L_acc."""
    nodes, weights = np.polynomial.legendre.leggauss(16)
    cos_nodes = 0.5 * (nodes + 1.0)  # map [-1, 1] -> [0, 1]
    powers = np.array([_power(_disc_lnu(disc_block, c, _WIDE), _WIDE) for c in cos_nodes])
    mean_power = float(np.sum(0.5 * weights * powers))
    assert mean_power / _L_ACC == pytest.approx(1.0, abs=_ENERGY_TOL[disc_block])


@pytest.mark.parametrize("disc_block", ["kubota_done", "multicolor"])
def test_line_of_sight_power_at_cos_half_is_the_accretion_power(disc_block):
    """At cos i = 0.5 (D + H) / (D + H): the line-of-sight power equals L_acc."""
    power = _power(_disc_lnu(disc_block, 0.5, _WIDE), _WIDE)
    assert power / _L_ACC == pytest.approx(1.0, abs=_ENERGY_TOL[disc_block])


# ----------------------------------------------------------------------------------
# c. shape quantities do not depend on the inclination
# ----------------------------------------------------------------------------------
def _spy_kubota_done(monkeypatch, cos_inc):
    """Run ``kubota_done_disc`` and capture what the shape helpers return."""
    captured = {}

    def spy(name):
        original = getattr(disc_module, name)

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            captured[name] = out
            return out

        monkeypatch.setattr(disc_module, name, wrapper)

    for name in ("_compute_bh_params", "_compute_zone_radii", "_compute_zone_luminosities"):
        spy(name)
    disc_module.kubota_done_disc(jnp.asarray(_WIDE[::50]), _LOG_LBOL, agn_cos_inc=cos_inc)
    return captured


def test_kubota_done_shape_quantities_are_inclination_independent(monkeypatch):
    """R_hot, R_warm, R_out, T_in, mdot and the bolometric scale agree at i = 0, 75 deg (1e-12)."""
    face_on = _spy_kubota_done(monkeypatch, _COS[0])
    inclined = _spy_kubota_done(monkeypatch, _COS[75])
    for name in ("_compute_bh_params", "_compute_zone_radii"):
        for a, b in zip(face_on[name], inclined[name]):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=0.0)
    scale_face_on, scale_inclined = (
        float(face_on["_compute_zone_luminosities"][1]),
        float(inclined["_compute_zone_luminosities"][1]),
    )
    assert scale_inclined == pytest.approx(scale_face_on, rel=1e-12)


# ----------------------------------------------------------------------------------
# d. multicolor and the relagn grid disc: 2 cos i D_nu
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("wave_aa", _OPTICAL_UV)
@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_multicolor_disc_scales_as_cos_i_over_half(i_deg, wave_aa):
    """The multicolor disc (no corona) is 2 cos i D_nu: ratio to i = 60 deg is cos i / 0.5."""
    wave = np.array([wave_aa])
    level = _disc_lnu("multicolor", _COS[i_deg], wave)[0]
    reference = _disc_lnu("multicolor", _COS[60], wave)[0]
    assert level / reference == pytest.approx(_COS[i_deg] / 0.5, rel=1e-6)


@pytest.fixture(scope="module")
def relagn_grid():
    return disc_module.load_relagn_default_grid()


@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_relagn_grid_disc_scales_as_cos_i_over_half(i_deg, relagn_grid):
    """The grid disc (the shipped grid holds the outer disc alone) is 2 cos i D_nu, 1e-6."""
    wave = jnp.asarray(relagn_grid["wave_grid"])
    level = disc_module.relagn_disc_from_grid(
        relagn_grid, wave, agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS[i_deg]
    )
    reference = disc_module.relagn_disc_from_grid(
        relagn_grid, wave, agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS[60]
    )
    np.testing.assert_allclose(
        np.asarray(level), np.asarray(reference) * _COS[i_deg] / 0.5, rtol=1e-6, atol=0.0
    )


def test_relagn_grid_disc_power_at_cos_half_is_the_accretion_power(relagn_grid):
    """At the stored reference cos i = 0.5 the disc power is L_acc (native grid, 3 x 3.7e-5)."""
    wave = np.asarray(relagn_grid["wave_grid"])
    l_nu = disc_module.relagn_disc_from_grid(
        relagn_grid, jnp.asarray(wave), agn_log_lbol=_LOG_LBOL, agn_cos_inc=0.5
    )
    assert _power(l_nu, wave) / _L_ACC == pytest.approx(1.0, rel=1.2e-4)


# ----------------------------------------------------------------------------------
# e. the derived line-of-sight key
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("disc_block", ["kubota_done", "multicolor"])
@pytest.mark.parametrize("i_deg", [0, 30, 60])
def test_line_of_sight_key_is_the_integral_of_the_direct_emission(disc_block, i_deg):
    """log_L_agn_los = log10 int L_nu(i) dnu of the published disc component, 1e-4 dex."""
    _, comps = _run(disc_block, _COS[i_deg])
    expected = np.log10(_power(comps["disc"], _WIDE))
    assert float(comps["log_L_agn_los"]) == pytest.approx(expected, abs=1e-4)


def _build_model(ssp, i_deg):
    """kubota_done composable AGN alone (no torus, no polar dust, no separate X-ray block)."""
    return SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn={
            "type": "composable",
            "disc": {"type": "kubota_done", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"},
            "atten": {"type": "none"},
            "agn_log_lbol": Fixed(_LOG_LBOL),
            "agn_cos_inc": Fixed(_COS[i_deg]),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )


@pytest.mark.parametrize("i_deg", [0, 30, 60])
def test_published_keys_through_the_model(i_deg):
    """Through SEDModel: L_agn_bol = L_acc at every i; log_L_agn_los is the runner's value."""
    ssp = load_ssp_data(str(_SSP_PATH))
    state = _build_model(ssp, i_deg).predict_state({})
    assert float(state.derived["log_L_agn_bol"]) == pytest.approx(_LOG_LBOL + np.log10(L_SUN))
    assert float(state.derived["L_agn_bol"]) == pytest.approx(_L_ACC, rel=1e-12)
    _, comps = _run("kubota_done", _COS[i_deg])
    assert float(state.derived["log_L_agn_los"]) == pytest.approx(
        float(comps["log_L_agn_los"]), abs=1e-4
    )


# ----------------------------------------------------------------------------------
# f. composition with the torus tie, the screen and the anchors
# ----------------------------------------------------------------------------------
_SKIRTOR = dict(
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="skirtor",
    agn_attenuation_block="none",
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_torus_frac=0.5,
)
_SKIRTOR_WAVE = np.geomspace(8.0, 1.0e8, 3000)  # [A] the SKIRTOR library axis


def _skirtor(disc_block, cos_inc, **kwargs):
    _, comps = compose_l_nu(
        jnp.asarray(_SKIRTOR_WAVE),
        _LOG_LBOL,
        agn_disc_block=disc_block,
        agn_cos_inc=cos_inc,
        return_components=True,
        **{**_SKIRTOR, **kwargs},
    )
    return {k: np.asarray(v) for k, v in comps.items()}


def test_skirtor_tie_removes_the_discs_own_inclination_law():
    """Tied disc power is agn_power x library disk(30)/int(dust): the same for every disc block.

    The tie renormalizes the reweighted shape to ``agn_power x R``, so the 2 cos i the disc
    carries must divide out of it. The schartmann2005 disc has no inclination law of its own
    and is the #2601 statement; kubota_done and multicolor must land on the same power.
    """
    cos30 = _COS[30]
    reference = _power(
        _skirtor("schartmann2005", cos30, agn_ir_frac=0.3, agn_norm="cigale_joint")["disc"],
        _SKIRTOR_WAVE,
    )
    for block in ("kubota_done", "multicolor"):
        tied = _skirtor(block, cos30, agn_ir_frac=0.3, agn_norm="cigale_joint")["disc"]
        assert _power(tied, _SKIRTOR_WAVE) == pytest.approx(reference, rel=3e-3)


def _screen_30deg(wave_aa, oa=40.0, tau_v=7.0):
    """Torus screen at i = 30 deg from its documented formula (``torus_screen.py``)."""
    from tengri.components.dust.attenuation import smc

    w = 0.5 * (1 + np.tanh(0.5 * (np.sin(np.radians(oa)) - _COS[30]) / 0.025))
    k = np.asarray(smc(jnp.asarray(np.append(np.asarray(wave_aa, float), 5500.0))))
    return np.exp(-np.clip(tau_v * k[:-1] / k[-1] * w, 0.0, 50.0))


def test_untied_type1_disc_is_2cos_i_d_nu_times_the_screen():
    """SKIRTOR untied (independent norm), i = 30 deg: disc = 2 cos 30 D_nu x screen."""
    cos30 = _COS[30]
    bare = _disc_lnu("kubota_done", cos30, _SKIRTOR_WAVE)
    half = _disc_lnu("kubota_done", 0.5, _SKIRTOR_WAVE)  # D_nu + H_nu, the cos i = 0.5 spectrum
    screened = _skirtor("kubota_done", cos30, agn_ir_frac=0.0, agn_norm="independent")["disc"]
    np.testing.assert_allclose(screened, bare * _screen_30deg(_SKIRTOR_WAVE), rtol=1e-6)
    # optical/UV is disc dominated: 2 cos 30 D_nu with D_nu the cos i = 0.5 spectrum
    optical = (_SKIRTOR_WAVE > 1000.0) & (_SKIRTOR_WAVE < 2.0e4)
    np.testing.assert_allclose(bare[optical], 2.0 * cos30 * half[optical], rtol=2e-3)


def test_intrinsic_anchors_are_the_documented_30_degree_values():
    """L_2500 / L_4400 intrinsic are the line-of-sight values at 30 deg for every viewing angle."""
    wave = np.array([2500.0, 4400.0])
    expected = _disc_lnu("kubota_done", _COS[30], wave)
    for i_deg in (0, 60, 75):
        _, l2500, l4400, _ = compose_l_nu(
            jnp.asarray(_WIDE),
            _LOG_LBOL,
            agn_disc_block="kubota_done",
            agn_cos_inc=_COS[i_deg],
            return_l2500=True,
            return_components=True,
            **_NONE,
        )
        np.testing.assert_allclose([float(l2500), float(l4400)], expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("i_deg", [0, 45, 75])
def test_broad_lines_and_feii_follow_the_angle_integrated_disc(i_deg):
    """BLR and FeII are powered by the engine, not by the viewing angle: no second 2 cos i.

    Their normalization reads the disc at 5100 A; that disc is taken at cos i = 0.5 (where the
    line-of-sight level is ``D_nu``), so the lines do not move with i, and equal the lines of
    a disc seen at cos i = 0.5.
    """
    blocks = dict(agn_blr_block="analytic", agn_feii_block="boroson_green")
    _, inclined = _run("kubota_done", _COS[i_deg], **blocks)
    _, reference = _run("kubota_done", 0.5, **blocks)
    assert np.max(inclined["lines"]) > 0.0
    np.testing.assert_allclose(inclined["lines"], reference["lines"], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("cos_inc", [0.0, 0.3, 0.9])
def test_polar_dust_reemission_does_not_depend_on_the_viewing_angle(cos_inc):
    """The polar dust re-emits the power it absorbs from the disc, seen from anywhere.

    The budget is read from the angle-integrated disc (``agn_log_lbol``), so the re-emitted
    graybody is the same for every inclination, including the edge-on view where the observed
    ``2 cos i`` disc is zero.
    """
    kwargs = dict(
        agn_torus_block="skirtor",
        agn_attenuation_block="polar_dust",
        agn_norm="independent",
        agn_polar_ebv=0.3,
        agn_polar_oa=40.0,
        agn_polar_T=100.0,
        agn_polar_beta=1.6,
    )
    _, reference = _run("kubota_done", 0.5, **kwargs)
    _, comps = _run("kubota_done", cos_inc, **kwargs)
    assert np.max(comps["polar"]) > 0.0
    np.testing.assert_allclose(comps["polar"], reference["polar"], rtol=1e-9, atol=0.0)


# ----------------------------------------------------------------------------------
# g. float32 against float64; the gradient in cos i
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_float32_matches_float64(i_deg):
    """The kubota_done disc component agrees between float32 and float64 to 1e-4."""
    wave = np.array([5100.0, 1500.0, 2500.0])
    levels = {}
    for x64 in (True, False):
        with jax.enable_x64(x64):
            dtype = jnp.float64 if x64 else jnp.float32
            _, comps = compose_l_nu(
                jnp.asarray(wave, dtype=dtype),
                _LOG_LBOL,
                agn_disc_block="kubota_done",
                agn_cos_inc=jnp.asarray(_COS[i_deg], dtype=dtype),
                return_components=True,
                **_NONE,
            )
            assert comps["disc"].dtype == dtype
            levels[x64] = np.asarray(comps["disc"], dtype=np.float64)
    np.testing.assert_allclose(levels[False], levels[True], rtol=1e-4, atol=0.0)


@pytest.mark.parametrize("cos_inc", [0.05, 0.5, 0.9])
def test_flux_gradient_in_cos_inc_is_2_d_nu(cos_inc):
    """d L_nu(5100 A) / d cos i = 2 D_nu, constant in cos i, finite down to cos i = 0.05."""
    wave = jnp.asarray([5100.0])

    def flux(c):
        return compose_l_nu(
            wave,
            _LOG_LBOL,
            agn_disc_block="kubota_done",
            agn_cos_inc=c,
            return_components=True,
            **_NONE,
        )[1]["disc"][0]

    grad = float(jax.grad(flux)(jnp.asarray(cos_inc)))
    d_nu = float(flux(jnp.asarray(0.5)))  # L_nu(cos i = 0.5) = D_nu (+ a negligible corona)
    assert np.isfinite(grad), "a NaN slope in cos i would break every inclination fit"
    assert grad != 0.0, "a zero slope would mean the inclination has dropped out of the disc"
    assert grad == pytest.approx(2.0 * d_nu, rel=1e-3)


# ----------------------------------------------------------------------------------
# h. catalog comparison
# ----------------------------------------------------------------------------------
def test_line_of_sight_key_exceeds_the_parameter_by_the_measured_factor():
    """At i = 30 deg, log_L_agn_los - agn_log_lbol = log10[(2 cos 30 D + H) / (D + H)].

    D and H follow from the two spectra at cos i = 1 and cos i = 0.05 (the power is
    linear in cos i); the factor is within 0.01 dex of log10(2 cos 30) = 0.2386 because the
    corona carries 6e-3 of the power (measured).
    """
    _, comps = _run("kubota_done", _COS[30])
    p_one = _power(_disc_lnu("kubota_done", 1.0, _WIDE), _WIDE)
    p_low = _power(_disc_lnu("kubota_done", 0.05, _WIDE), _WIDE)
    two_d = (p_one - p_low) / 0.95  # = 2 D  (power slope in cos i)
    h = p_one - two_d
    factor = (_COS[30] * two_d + h) / (0.5 * two_d + h)
    expected = np.log10(factor)
    measured = float(comps["log_L_agn_los"]) - (_LOG_LBOL + np.log10(L_SUN))
    assert measured == pytest.approx(expected, abs=2e-3)
    assert measured == pytest.approx(np.log10(2.0 * _COS[30]), abs=1e-2)

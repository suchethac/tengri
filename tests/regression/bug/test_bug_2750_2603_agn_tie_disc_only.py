# SPDX-License-Identifier: BSD-3-Clause
r"""The SKIRTOR tie normalizes the disc only; the anchors follow fracAGN; the debit is closed form.

Three statements about the composable AGN runner under ``agn_norm="cigale_joint"``:

1. **The tie fixes the disc, not the corona (#2750).** The library ties the thin disc and
   warm zone :math:`D` to ``agn_power x R`` (CIGALE ``skirtor2016``: the library disc is a
   pure disc). A Kubota & Done (2018) corona is not reprocessed by that disc, so it carries
   its own power: with the disc at :math:`P_D = {\tt agn\_power}\,R` the corona is
   :math:`H = (H/D)\,P_D`, the same ratio :math:`P_H/P_D = f/(1-f)` of the closed-form
   accretion budget (:math:`f = H/(D+H)`) the untied model has at the declared
   :math:`M_{\rm BH}` and Eddington ratio. Every quantity below is computed in the test
   from the block, the shipped library (``spectra/disk_emission``, ``spectra/dust_emission``)
   and the tie formula, never read from the runner's internals.
2. **The anchors follow the normalization (#2603).** ``L_2500_intrinsic`` and
   ``L_4400_intrinsic`` (the alpha_ox and radio-loudness anchors) are those of the disc as
   normalized in the model: with ``agn_ir_frac > 0`` they scale with ``agn_power``, and for
   a Fritz torus the disc is tied to the Fritz library the way CIGALE ``fritz2006`` does.
3. **The line debit reads the closed-form disc power (#2743).** Under ``conserving`` the
   fraction of the disc the lines take is ``E_lines / L_acc``, not ``E_lines`` over a grid
   integral of the disc.

References
----------
.. [1] Kubota, A. & Done, C. 2018, MNRAS, 480, 1247.
.. [2] Stalevski, M. et al. 2016, MNRAS, 458, 2288 (SKIRTOR).
.. [3] Fritz, J., Franceschini, A. & Hatziminaoglou, E. 2006, MNRAS, 366, 767.
.. [4] Yang, G. et al. 2020, MNRAS, 491, 740 (X-CIGALE; alpha_ox from the 2500 A anchor).
.. [5] Boquien, M. et al. 2019, A&A, 622, A103 (CIGALE ``skirtor2016``, ``fritz2006``).
"""

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import _protocol as protocol, runner as runner_module
from tengri.components.agn.blocks._protocol import LINE_ENERGY_BLOCKS, resolve_agn_block
from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.components.agn.disc import kubota_done_disc
from tengri.utils.grid_interp import resample_template
from tengri.utils.physics_constants import C_AA, L_SUN
from tests._data_skip import DATA_DIR
from tests._torus_sightline_reference import fritz_transmission, skirtor_transmission

pytestmark = pytest.mark.regression_bug

_GRID_FILE = DATA_DIR / "skirtor_templates_v3.h5"
_FRITZ_FILE = DATA_DIR / "fritz2006_torus_grid.h5"
_LOG_LBOL = 11.0
_COS30 = float(np.cos(np.radians(30.0)))
_TORUS_FRAC = 0.5
_WAVE = jnp.asarray(np.geomspace(8.0, 1.0e8, 40000))  # [A] covers the library axis
_L_ACC = 10.0**_LOG_LBOL * L_SUN  # [erg/s]
_SKIRTOR_CELL = (
    ("tau_97", 7),
    ("p", 1),
    ("q", 1),
    ("opening_angle", 40),
    ("radius_ratio", 20),
)
_BASE = dict(
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_attenuation_block="none",
    agn_norm="cigale_joint",
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_torus_frac=_TORUS_FRAC,
)


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _power(component, wave=_WAVE):
    """``int L_nu dnu`` of a component [erg/s], in float64."""
    nu = C_AA / np.asarray(wave, dtype=np.float64)
    order = np.argsort(nu)
    return float(np.trapezoid(np.asarray(component, dtype=np.float64)[order], nu[order]))


def _run(
    disc, torus="skirtor", *, frac=_TORUS_FRAC, wave=_WAVE, mbh=8.0, cos_inc=_COS30, **overrides
):
    """``compose_l_nu(..., return_components=True)`` on a tied composition (fracAGN = 0.1)."""
    kwargs = {**_BASE, "agn_torus_block": torus, "agn_disc_block": disc, **overrides}
    kwargs["agn_torus_frac"] = frac
    sed, comps = compose_l_nu(
        wave,
        _LOG_LBOL,
        agn_ir_frac=0.1,
        agn_cos_inc=cos_inc,
        agn_log_mbh=mbh,
        return_components=True,
        **kwargs,
    )
    return sed, {k: np.asarray(v) for k, v in comps.items() if k != "log_L_agn_los"}


def _anchors(disc, torus="skirtor", *, frac=_TORUS_FRAC, cos_inc=_COS30, **overrides):
    """``(L_2500_intrinsic, L_4400_intrinsic)`` of the tied composition [erg/s/Hz]."""
    kwargs = {**_BASE, "agn_torus_block": torus, "agn_disc_block": disc, **overrides}
    kwargs["agn_torus_frac"] = frac
    _, l2500, l4400 = compose_l_nu(
        _WAVE,
        _LOG_LBOL,
        agn_ir_frac=0.1,
        agn_cos_inc=cos_inc,
        return_l2500=True,
        **kwargs,
    )
    return float(l2500), float(l4400)


@functools.cache
def _load_library(i_deg):
    """The shipped SKIRTOR cube at the fiducial cell at ``i_deg``, with the face-on record."""
    h5py = pytest.importorskip("h5py")
    if not _GRID_FILE.is_file():
        pytest.skip(f"SKIRTOR v3 grid not found at {_GRID_FILE}")
    with h5py.File(_GRID_FILE, "r") as f:
        axes = {k: np.asarray(f["grid"][k]) for k in f["grid"]}
        wave_aa = np.asarray(f["wavelength"], float)
        cell = tuple(int(np.argmin(np.abs(axes[k] - v))) for k, v in _SKIRTOR_CELL)
        nodes = {}
        for angle in (0, i_deg):
            cos = float(np.cos(np.radians(angle)))
            ic = int(np.argmin(np.abs(axes["cos_inclination"] - cos)))
            assert abs(axes["cos_inclination"][ic] - cos) < 1e-9, "inclination is a grid node"
            nodes[angle] = (
                np.asarray(f["spectra/disk_emission"][(*cell, ic)], float),
                np.asarray(f["spectra/dust_emission"][(*cell, ic)], float),
                float(f["spectra/norm"][(*cell, ic)]),
            )
    disk_0, disk_i, dust_i = nodes[0][0], nodes[i_deg][0], nodes[i_deg][1]
    live = disk_0 > 0.0
    incl = np.divide(disk_i, disk_0, out=np.zeros_like(disk_i), where=live)
    last = disk_i[np.max(np.where(live))] / disk_0[np.max(np.where(live))]
    incl = np.where(live, incl, last)
    # R_faceon = int disk(0) x norm(0)/norm(i) / int dust(i): the face-on library disc per
    # unit agn_power (CIGALE ``skirtor2016``: ``AGN1.disk *= AGN1.norm / skirtor2016.norm``)
    r_faceon = (
        np.trapezoid(disk_0, wave_aa)
        * nodes[0][2]
        / nodes[i_deg][2]
        / np.trapezoid(dust_i, wave_aa)
    )
    return {
        "wave_aa": wave_aa,
        "disk_0": disk_0,
        "dust_30": dust_i,
        "incl": incl,
        "incl_edge": float(last),
        "R_faceon": float(r_faceon),
    }


@pytest.fixture(scope="module")
def library():
    """The shipped SKIRTOR cube at the fiducial cell, at i = 30 deg and face-on."""
    return _load_library(30)


def _screen(wave_aa, i_deg, oa=40.0):
    """Torus transmission of the SKIRTOR library at the fiducial cell (``T = R_n/eta``).

    The library's normalized inclination ratio over the disc anisotropy, X-rays absorbed below
    the library's 10 A edge (``tests/_torus_sightline_reference.py``); it replaces the analytic
    ``exp(-tau_V k(lambda)/k(V) w)`` screen, which the runner no longer applies to a library torus.
    """
    return skirtor_transmission(wave_aa, float(np.cos(np.radians(i_deg))), oa=oa)


def _fritz_screen(wave_aa, psy_deg, oa=60.0, tau=1.0):
    """Torus transmission of the Fritz library (``T = R_n``) at elevation ``psy_deg``."""
    return fritz_transmission(wave_aa, psy_deg, oa=oa, tau=tau)


def _incl_on(library, wave):
    """The library ``disk(i)/disk(0)`` on ``wave`` (the runner's fill rule beyond the axis)."""
    return np.asarray(
        resample_template(
            jnp.asarray(wave),
            jnp.asarray(library["wave_aa"]),
            jnp.asarray(library["incl"]),
            left=0.0,
            right=library["incl_edge"],
        )
    )


def _kubota_parts(wave, mbh):
    """``(D_lambda, H_lambda)`` [erg/s/A] from the public disc: ``L(i) = 2 cos i D + H``."""
    w = jnp.asarray(wave)
    kw = dict(agn_log_mbh=mbh)
    face_on = np.asarray(kubota_done_disc(w, _LOG_LBOL, agn_cos_inc=1.0, **kw))
    edge_on = np.asarray(kubota_done_disc(w, _LOG_LBOL, agn_cos_inc=0.0, **kw))
    to_lambda = C_AA / np.asarray(wave, float) ** 2
    return (face_on - edge_on) / 2.0 * to_lambda, edge_on * to_lambda


def _tie_ratio(library, shape_lambda):
    """CIGALE's ``R = int(shape x int(disk_0) x disk_i/disk_0) / int(dust_i)`` on the library axis.

    ``shape_lambda`` is a callable giving the disc on a wavelength array [A]; it is
    normalized to unit area on the library axis, as CIGALE normalizes its disc.
    """
    wl = library["wave_aa"]
    shape = shape_lambda(wl)
    shape = shape / np.trapezoid(shape, wl)
    int_disk0 = np.trapezoid(library["disk_0"], wl)
    return np.trapezoid(shape * int_disk0 * library["incl"], wl) / np.trapezoid(
        library["dust_30"], wl
    )


def _fine_integral(library, disc_lambda):
    """``int D x disk(i)/disk(0) d lambda`` on the runner's fine grid over the library range."""
    wl = library["wave_aa"]
    fine = np.geomspace(wl[0], wl[-1], runner_module._TIE_FINE_NODES)
    incl = _incl_on(library, fine)
    return float(np.trapezoid(disc_lambda(fine) * incl, fine))


# ----------------------------------------------------------------------------------
# 1. the tie normalizes the disc only (#2750)
# ----------------------------------------------------------------------------------
def _decompose(tied_nu, disc_part, corona, incl):
    """Least-squares ``tied = a (D x incl) + b (H x T)``; returns ``(a, b, residual)``.

    ``tied_nu`` is the published ``L_nu`` of the disc; ``disc_part`` is ``D`` and ``corona`` is
    ``H x T`` (``T`` the torus screen), both ``L_lambda`` on ``_WAVE``.
    """
    tied = tied_nu * C_AA / np.asarray(_WAVE, float) ** 2
    live = incl > 0.0
    basis = np.stack([disc_part[live] * incl[live], corona[live]], axis=1)
    norms = np.linalg.norm(basis, axis=0)
    coef, *_ = np.linalg.lstsq(basis / norms, tied[live], rcond=None)
    coef = coef / norms
    resid = np.max(np.abs(basis @ coef - tied[live])) / np.max(np.abs(tied[live]))
    return float(coef[0]), float(coef[1]), float(resid)


@pytest.mark.parametrize("mbh", [6.0, 8.0, 10.0])
def test_tied_kubota_disc_part_carries_the_library_disc_power(mbh, library):
    """The disc and warm zone carry ``agn_power x R``; the corona is no part of that budget.

    The tied disc spectrum is exactly ``a D x disk(i)/disk(0) + b H x T`` (``T`` the torus
    screen); the power of the
    ``a D`` part on the library range equals the library disc power ``agn_power x R`` to the
    tie's own quadrature (1e-4). Before the change the corona took the share ``x/(1+x)`` of
    that budget (0.15 % / 42 % / 47 % at log M_BH = 6 / 8 / 10).
    """
    _, comps = _run("kubota_done", mbh=mbh)
    d_lam, h_lam = _kubota_parts(_WAVE, mbh)
    incl = _incl_on(library, _WAVE)
    a, _b, resid = _decompose(comps["disc"], d_lam, h_lam * _screen(_WAVE, 30), incl)
    assert resid < 1e-9, f"the tied disc is not (a D x incl + b H x T): residual {resid:.2e}"
    wl = library["wave_aa"]
    in_grid = (np.asarray(_WAVE) >= wl[0]) & (np.asarray(_WAVE) <= wl[-1])
    w_in = np.asarray(_WAVE)[in_grid]
    power_d = a * float(np.trapezoid((d_lam * incl)[in_grid], w_in))
    agn_power = _power(comps["torus"])
    ratio = _tie_ratio(library, lambda w: _kubota_parts(w, mbh)[0])
    assert power_d / (agn_power * ratio) == pytest.approx(1.0, abs=1e-4), (
        f"log M_BH {mbh}: disc part carries {power_d / (agn_power * ratio):.4f} of agn_power x R"
    )


def _corona_coefficients(mbh, i_deg):
    """``(a, b, residual, library)`` of the tied kubota_done disc at ``i_deg``."""
    lib = _load_library(i_deg)
    _, comps = _run("kubota_done", mbh=mbh, cos_inc=float(np.cos(np.radians(i_deg))))
    d_lam, h_lam = _kubota_parts(_WAVE, mbh)
    a, b, resid = _decompose(
        comps["disc"], d_lam, h_lam * _screen(_WAVE, i_deg), _incl_on(lib, _WAVE)
    )
    return a, b, resid, lib


@pytest.mark.parametrize("mbh", [6.0, 8.0, 10.0])
def test_tied_corona_carries_the_model_share_of_the_angle_integrated_disc_power(mbh):
    """``int H d nu = f / (1 - f) x P_D,tied`` in angle-integrated powers, to 1e-5.

    ``f`` is the closed-form corona share of the accretion power, so ``f / (1 - f)`` is the
    ratio of the isotropic corona's power to the two-face disc's power in the untied model.
    ``P_D,tied`` is the tied disc's 4 pi power: the library's face-on disc power
    ``agn_power x R_faceon`` times the mean of its anisotropy
    ``eta(cos i) = cos i (1 + 2 cos i)/3``
    over the viewing hemisphere, ``7/18`` (``skirtor2016.py``; the shipped library follows ``eta``
    to 0.3 % at i <= 40 deg). The corona is ``b x H_model``, so
    ``b f L_acc = f / (1 - f) x (7/18) agn_power R_faceon``. ``agn_power`` is eliminated with the
    tie ``a int(D x incl) = agn_power R``: ``b = a I_D (7/18) (R_faceon / R) / ((1 - f) L_acc)``.
    """
    a, b, resid, lib = _corona_coefficients(mbh, 30)
    assert resid < 1e-9
    _, _, f_corona = protocol.DISC_SPLIT_BLOCKS["kubota_done"](_WAVE, _LOG_LBOL, agn_log_mbh=mbh)
    i_d = _fine_integral(lib, lambda w: _kubota_parts(w, mbh)[0])
    ratio = _tie_ratio(lib, lambda w: _kubota_parts(w, mbh)[0])
    expected = (
        a * i_d * (7.0 / 18.0) * (lib["R_faceon"] / ratio) / ((1.0 - float(f_corona)) * _L_ACC)
    )
    assert b == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("i_deg", [0, 20, 30, 40])
def test_tied_corona_per_unit_intrinsic_disc_power_is_the_same_at_every_type_1_inclination(i_deg):
    """``b f L_acc = f/(1 - f) (7/18) agn_power R_faceon(i)`` at every Type-1 inclination (1e-4).

    This is the isotropy of the corona: per unit INTRINSIC (4 pi) disc power it is the same
    ``f/(1 - f)`` whatever the viewing angle. At fixed ``agn_power`` the intrinsic disc power
    itself, and with it the corona and the anchors, rises with inclination, because the library
    normalizes each viewing-angle record to its own observed dust luminosity (``R_faceon`` carries
    ``norm(0)/norm(i)``); that rise is CIGALE's, and is measured against it in
    ``test_corona_and_anchors_at_fixed_agn_power_rise_with_inclination_as_in_cigale``.
    ``agn_power`` cancels between ``a`` and ``b`` and the cell repeats the runner's own
    quadrature, so the identity holds to round-off (measured 6e-14); the bound is 1e-9.
    """
    a, b, resid, lib = _corona_coefficients(8.0, i_deg)
    assert resid < 1e-9
    _, _, f_corona = protocol.DISC_SPLIT_BLOCKS["kubota_done"](_WAVE, _LOG_LBOL, agn_log_mbh=8.0)
    i_d = _fine_integral(lib, lambda w: _kubota_parts(w, 8.0)[0])
    ratio = _tie_ratio(lib, lambda w: _kubota_parts(w, 8.0)[0])
    expected = (
        a * i_d * (7.0 / 18.0) * (lib["R_faceon"] / ratio) / ((1.0 - float(f_corona)) * _L_ACC)
    )
    assert b == pytest.approx(expected, rel=1e-9)


def test_type_2_corona_is_screened_like_any_source_behind_the_torus():
    """At i = 70 deg the tied corona is ``b H T(70 deg)`` with the torus screen ``T`` (1e-9).

    ``T`` is the library transmission of the torus (``_screen``); the decomposition residual
    vanishes only if the corona carries exactly that.
    """
    _, b, resid, _ = _corona_coefficients(8.0, 70)
    assert resid < 1e-9
    assert b > 0.0


@pytest.mark.parametrize("mbh", [6.0, 8.0, 10.0])
def test_closed_form_corona_share_matches_the_spectrum(mbh):
    """``f = P_H / (P_D + P_H)`` from the radial integrals equals the spectra's powers (3e-3)."""
    wide = np.geomspace(1.0e-3, 1.0e10, 400001)
    d_lam, h_lam = _kubota_parts(wide, mbh)
    p_d, p_h = np.trapezoid(d_lam, wide), np.trapezoid(h_lam, wide)
    _, _, f_corona = protocol.DISC_SPLIT_BLOCKS["kubota_done"](
        jnp.asarray(wide), _LOG_LBOL, agn_log_mbh=mbh
    )
    assert float(f_corona) == pytest.approx(p_h / (p_d + p_h), rel=3e-3)
    assert (p_d + p_h) / _L_ACC == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("disc", ["schartmann2005", "skirtor", "multicolor"])
def test_corona_less_disc_is_the_block_times_the_library_ratio(disc, library):
    """A disc without a corona is unchanged: block shape x ``disk(i)/disk(0)`` x one scalar.

    The scalar is fixed by ``int = agn_power x R`` (1e-3). No split is registered for these
    blocks, so the runner takes the whole spectrum as the disc part.
    """
    assert disc not in protocol.DISC_SPLIT_BLOCKS
    _, comps = _run(disc)
    block = resolve_agn_block("disc", disc)(
        _WAVE, agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS30, agn_log_mbh=8.0
    )
    incl = _incl_on(library, _WAVE)
    live = (incl > 0) & (np.asarray(block) > 0)
    tied_lambda = comps["disc"] * C_AA / np.asarray(_WAVE, float) ** 2
    scalar = tied_lambda[live] / (np.asarray(block)[live] * incl[live])
    np.testing.assert_allclose(scalar, scalar[0], rtol=1e-12)
    ratio = _tie_ratio(library, lambda w: np.asarray(_block_at(disc, w)))
    assert _power(comps["disc"]) / (_power(comps["torus"]) * ratio) == pytest.approx(1.0, abs=3e-3)


def _block_at(disc, w):
    """The disc block on a wavelength array [A] at the model's inclination."""
    return resolve_agn_block("disc", disc)(
        jnp.asarray(w), agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS30, agn_log_mbh=8.0
    )


# ----------------------------------------------------------------------------------
# 2. the anchors follow the normalization (#2603)
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("disc", "torus"),
    [
        ("schartmann2005", "skirtor"),
        ("kubota_done", "skirtor"),
        ("multicolor", "skirtor"),
        ("schartmann2005", "fritz"),
        ("kubota_done", "fritz"),
    ],
)
def test_anchors_scale_with_agn_power(disc, torus):
    """The anchors ``L_2500_intrinsic`` and ``L_4400_intrinsic`` scale as the tied disc does.

    ``agn_power = agn_torus_frac x L_bol`` (the runner's reading of fracAGN) takes the values
    0.05 / 0.1 / 0.3 x L_bol; the anchors move with it to 1e-6 relative to the torus power the
    same models emit. Before the change they stayed at the default ``agn_log_lbol``.
    """
    fracs = (0.05, 0.1, 0.3)
    anchors = [np.asarray(_anchors(disc, torus, frac=f)) for f in fracs]
    powers = [_power(_run(disc, torus, frac=f)[1]["torus"]) for f in fracs]
    for k in (1, 2):
        np.testing.assert_allclose(
            anchors[k] / anchors[0], powers[k] / powers[0], rtol=1e-6, err_msg=f"{disc}/{torus}"
        )


_SKIRTOR_NODE = dict(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97)


def _pcigale_skirtor_anchor(i_deg, frac=0.3):
    """CIGALE ``skirtor2016`` ``intrin_Lnu_2500A_30deg`` per unit ``agn_power`` [W/Hz per W]."""
    pcigale_sed = pytest.importorskip("pcigale.sed")
    skirtor_module = pytest.importorskip("pcigale.sed_modules.skirtor2016")
    sed = pcigale_sed.SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    skirtor_module.SKIRTOR2016(
        name="skirtor2016",
        **_SKIRTOR_NODE,
        i=i_deg,
        disk_type=1,
        delta=0,
        fracAGN=frac,
        lambda_fracAGN="0/0",
        law=0,
        EBV=0.0,
        temperature=100.0,
        emissivity=1.6,
    ).process(sed)
    return sed.info["agn.intrin_Lnu_2500A_30deg"] / (frac / (1.0 - frac))


def test_skirtor_anchor_is_cigale_value_up_to_its_two_quadratures(library):
    """Tied anchor per ``agn_power`` = CIGALE x (grid interpolation) x (library quadrature), 5e-4.

    tengri / CIGALE (``schartmann2005``, i = 30 deg) is 0.9968. It factors into
    0.99871, CIGALE's linear interpolation of its disc between its 948 library nodes at 2500 A
    (it reads ``np.interp(250, AGN1.wl, AGN1.disk)``; tengri evaluates the disc at 2500 A), and
    0.99809, the face-on library disc over the dust on the 136-node library against CIGALE's
    948 nodes (the quadrature offset of ``test_bug_2601``). Both are computed here from the
    database and the library, so the cell fails for a 0.1 % change of the anchor law.
    """
    SimpleDatabase = pytest.importorskip("pcigale.data").SimpleDatabase

    l2500, _ = _anchors("schartmann2005")
    agn_power = _power(_run("schartmann2005")[1]["torus"])
    block = resolve_agn_block("disc", "schartmann2005")

    def disc_at(w):
        return np.asarray(block(jnp.asarray(w), agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS30))

    with SimpleDatabase("skirtor2016") as db:
        e0, e30 = db.get(**_SKIRTOR_NODE, i=0), db.get(**_SKIRTOR_NODE, i=30)
    wl_c = np.asarray(e0.wl) * 10.0
    shape_c = disc_at(wl_c)
    exact_2500 = float(disc_at(np.asarray([2500.0]))[0])
    interpolation = exact_2500 / float(np.interp(2500.0, wl_c, shape_c))
    r_face_cigale = np.trapezoid(e0.disk, wl_c) * e0.norm / e30.norm / np.trapezoid(e30.dust, wl_c)
    quadrature = library["R_faceon"] / r_face_cigale
    got = l2500 / agn_power / _pcigale_skirtor_anchor(30)
    assert got == pytest.approx(interpolation * quadrature, rel=5e-4)


@pytest.mark.parametrize("i_deg", [30, 40, 60, 70, 80])
def test_corona_and_anchors_at_fixed_agn_power_rise_with_inclination_as_in_cigale(i_deg):
    """At fixed ``agn_power`` the anchor and the corona rise with i as CIGALE's anchor (5e-4).

    The library normalizes each viewing-angle record to its own observed dust luminosity, so the
    intrinsic disc per unit ``agn_power`` grows with inclination: CIGALE's
    ``intrin_Lnu_2500A_30deg`` relative to face-on is 1.0291 / 1.0909 / 2.1088 / 2.5884 / 3.1807
    at i = 30 / 40 / 60 / 70 / 80 deg (computed here from the module). tengri's anchor per
    ``agn_power`` follows it to the library's quadrature (0.0 / 0.0 / 0.16 / 0.28 / 0.42 %, the
    136-node library against CIGALE's 948 nodes, hence 6e-3 at i >= 60), and so does the tied
    corona (``b`` at the Type-1 inclinations 30 and 40 deg).
    """
    tol = 5e-4 if i_deg <= 40 else 6e-3
    cigale = _pcigale_skirtor_anchor(i_deg) / _pcigale_skirtor_anchor(0)

    def per_power(angle):
        c = float(np.cos(np.radians(angle)))
        l2500, _ = _anchors("schartmann2005", cos_inc=c)
        return l2500 / _power(_run("schartmann2005", cos_inc=c)[1]["torus"])

    assert per_power(i_deg) / per_power(0) == pytest.approx(cigale, rel=tol)
    if i_deg <= 40:
        _, b, _, _ = _corona_coefficients(8.0, i_deg)
        _, b0, _, _ = _corona_coefficients(8.0, 0)
        assert b / b0 == pytest.approx(cigale, rel=tol)


def test_fritz_disc_follows_the_fritz_tie():
    """With a Fritz torus the disc is tied to ``agn_power x R``: the issue's reproducer.

    The disc/torus power ratio no longer moves with ``agn_log_lbol`` (it was 1.3e10 at the
    default and 1.3e12 at ``agn_log_lbol = 12``) and equals CIGALE ``fritz2006``'s
    ``disk_luminosity / agn_power`` to 1e-4 (Type 1, psi = 70.1 deg) with the same library
    node and the same disc (``schartmann2005``, delta = 0).
    """
    pcigale_sed = pytest.importorskip("pcigale.sed")
    fritz_module = pytest.importorskip("pcigale.sed_modules.fritz2006")
    frac = 0.3
    sed = pcigale_sed.SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    fritz_module.Fritz2006(
        name="fritz2006",
        r_ratio=60.0,
        tau=1.0,
        beta=-0.5,
        gamma=4.0,
        opening_angle=60.0,
        psy=70.1,
        disk_type=1,
        delta=0.0,
        fracAGN=frac,
        lambda_fracAGN="0/0",
        law=0,
        EBV=0.0,
        temperature=100.0,
        emissivity=1.6,
    ).process(sed)
    cigale_ratio = sed.info["agn.disk_luminosity"] / (frac / (1.0 - frac))
    for log_lbol in (10.0, 12.0):
        _, comps = compose_l_nu(
            _WAVE,
            log_lbol,
            agn_ir_frac=frac,
            agn_disc_block="schartmann2005",
            agn_torus_block="fritz",
            agn_cos_inc=_cos_of_psy(70.1),
            agn_fritz_oa=60.0,
            agn_torus_frac=_TORUS_FRAC,
            return_components=True,
            **{k: v for k, v in _BASE.items() if k not in ("agn_torus_frac",)},
        )
        got = _power(comps["disc"]) / _power(comps["torus"])
        assert got == pytest.approx(cigale_ratio, rel=1e-4), f"log_lbol {log_lbol}: {got}"


def _cos_of_psy(psy_deg):
    """``agn_cos_inc`` of the Fritz library viewing elevation ``psy`` (psi = 90 deg - i)."""
    return float(np.sin(np.radians(psy_deg)))


_FRITZ_NODE = dict(r_ratio=60.0, tau=1.0, beta=-0.5, gamma=4.0, opening_angle=60.0)


def _pcigale_fritz(psy, frac=0.3):
    """CIGALE ``fritz2006`` per ``agn_power``: ``(disk_luminosity, anchor, accretion_power)``."""
    pcigale_sed = pytest.importorskip("pcigale.sed")
    fritz_module = pytest.importorskip("pcigale.sed_modules.fritz2006")
    sed = pcigale_sed.SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    fritz_module.Fritz2006(
        name="fritz2006",
        **_FRITZ_NODE,
        psy=psy,
        disk_type=1,
        delta=0.0,
        fracAGN=frac,
        lambda_fracAGN="0/0",
        law=0,
        EBV=0.0,
        temperature=100.0,
        emissivity=1.6,
    ).process(sed)
    per = frac / (1.0 - frac)
    return tuple(
        sed.info[k] / per
        for k in ("agn.disk_luminosity", "agn.intrin_Lnu_2500A_30deg", "agn.accretion_power")
    )


@pytest.mark.parametrize("psy", [0.001, 20.1, 50.1, 70.1, 89.99])
def test_fritz_anchor_is_cigale_value_at_every_viewing_angle(psy):
    """Fritz anchor per ``agn_power`` = CIGALE's ``intrin_Lnu_2500A_30deg`` x 0.99951 (2e-5).

    The library's per-record ``norm(0)/norm(psi)`` (0.5260 .. 1.0 across the five angles) is
    stored with the grid and applied, so the anchor follows CIGALE at every angle, Type 2
    included. The constant 0.99951 is CIGALE's linear interpolation of its disc between its
    library nodes at 2500 A (it reads ``np.interp(250, AGN1.wl, AGN1.disk)``; tengri evaluates
    the disc at 2500 A): computed here as the ratio of the exact disc to the interpolated one
    on the library axis.
    """
    import h5py

    _, anchor_cigale, _ = _pcigale_fritz(psy)
    kwargs = dict(cos_inc=_cos_of_psy(psy), agn_fritz_oa=60.0)
    l2500, _ = _anchors("schartmann2005", "fritz", **kwargs)
    power = _power(_run("schartmann2005", "fritz", **kwargs)[1]["torus"])
    with h5py.File(_FRITZ_FILE, "r") as f:
        wl = np.asarray(f["fritz2006/wavelength_aa"], float)
    block = resolve_agn_block("disc", "schartmann2005")
    shape = np.asarray(block(jnp.asarray(wl), agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS30))
    exact = float(block(jnp.asarray([2500.0]), agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS30)[0])
    interpolation = exact / float(np.interp(2500.0, wl, shape))
    assert l2500 / power / anchor_cigale == pytest.approx(interpolation, rel=2e-5)


@pytest.mark.parametrize("disc_mbh", [6.0, 8.0, 10.0])
def test_tied_corona_behind_a_fritz_torus_carries_the_model_share_of_the_disc_power(disc_mbh):
    """``int H d nu = f/(1 - f) x P_D,tied`` behind a Fritz torus, ``P_D,tied`` from CIGALE.

    The Fritz disc is isotropic, so its 4 pi power is the face-on one: CIGALE's
    ``accretion_power`` per ``agn_power`` (``int AGN1.disk x norm``, the hemisphere mean is 1).
    The corona below the library's 10 A edge, where the library ratio vanishes, is
    ``b H T`` exactly (``T`` the torus screen at the model inclination), and
    ``b = P_D,tied / ((1 - f) L_acc)`` with ``agn_power`` from the torus the model emits (3e-3,
    the torus quadrature on this grid).
    """
    _, _, accretion = _pcigale_fritz(70.1)
    kwargs = dict(agn_fritz_oa=60.0, agn_fritz_tau=1.0)
    _, comps = _run("kubota_done", "fritz", mbh=disc_mbh, cos_inc=_cos_of_psy(70.1), **kwargs)
    below = np.asarray(_WAVE) < 9.0
    _, h_lam = _kubota_parts(_WAVE, disc_mbh)
    screen = _fritz_screen(_WAVE, 70.1)
    tied = comps["disc"] * C_AA / np.asarray(_WAVE, float) ** 2
    b = tied[below] / (h_lam[below] * screen[below])
    np.testing.assert_allclose(b, b[0], rtol=1e-9)
    _, _, f_corona = protocol.DISC_SPLIT_BLOCKS["kubota_done"](
        _WAVE, _LOG_LBOL, agn_log_mbh=disc_mbh
    )
    expected = accretion * _power(comps["torus"]) / ((1.0 - float(f_corona)) * _L_ACC)
    assert b[0] == pytest.approx(expected, rel=3e-3)


def test_fritz_disc_is_independent_of_agn_log_lbol_under_the_tie():
    """The disc/torus power ratio of the tied Fritz model does not move with ``agn_log_lbol``."""
    ratios = []
    for log_lbol in (10.0, 11.0, 12.0):
        _, comps = compose_l_nu(
            _WAVE,
            log_lbol,
            agn_ir_frac=0.3,
            agn_disc_block="schartmann2005",
            agn_torus_block="fritz",
            return_components=True,
            **_BASE,
        )
        ratios.append(_power(comps["disc"]) / _power(comps["torus"]))
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)


# ----------------------------------------------------------------------------------
# 2b. through SEDModel: the published anchors, the X-ray corona, the grammar guard
# ----------------------------------------------------------------------------------
_SSP_FILE = DATA_DIR / "fsps_prsc_miles_chabrier.h5"


def _model(torus="skirtor", disc="schartmann2005", *, free_frac=False, xray=False, **agn_extra):
    """A host plus composable AGN with ``agn_ir_frac`` (fracAGN) as the one AGN input."""
    if not _SSP_FILE.is_file():
        pytest.skip(f"SSP grid not found at {_SSP_FILE}")
    from tengri import DEFAULT, Fixed, SEDModel, Uniform
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    frac = Uniform(0.02, 0.8) if free_frac else Fixed(0.1)
    agn = {
        "type": "composable",
        "disc": {"type": disc, "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus, "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "all_params": Fixed(DEFAULT),
        "agn_ir_frac": frac,
        **agn_extra,
    }
    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_FILE)),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
            "log_total_mass": 10.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
        agn=agn,
        **({"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}} if xray else {}),
    )


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
def test_published_anchors_follow_fracAGN(torus):
    """The anchors per unit ``agn_power`` are constant in fracAGN 0.05-0.7.

    ``agn_power = L_absorbed f / (1 - f)``: the anchors per unit ``agn_power`` are constant
    in f to 1e-6 (they were constant themselves: 8.35e27 erg/s/Hz at every f).
    """
    model = _model(torus, free_frac=True)
    per_power = []
    for f in (0.05, 0.1, 0.3, 0.7):
        derived = model.predict_state({"agn_ir_frac": jnp.asarray(f)}).derived
        agn_power = float(derived["L_absorbed"]) * f / (1.0 - f)
        per_power.append(
            (
                float(derived["L_2500_intrinsic"]) / agn_power,
                float(derived["L_4400_intrinsic"]) / agn_power,
            )
        )
    np.testing.assert_allclose(per_power, np.broadcast_to(per_power[0], (4, 2)), rtol=1e-6)


def test_xray_corona_follows_fracAGN():
    """The alpha_ox X-ray corona is anchored on the tied 2500 A luminosity."""
    model = _model(free_frac=True, xray=True)
    powers = []
    for f in (0.05, 0.3):
        state = model.predict_state({"agn_ir_frac": jnp.asarray(f)})
        powers.append(_power(state.derived["sed_xray"], state.wave))
    assert powers[1] / powers[0] > 3.0, f"X-ray power ratio {powers[1] / powers[0]:.3f}"


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
def test_agn_log_lbol_beside_fracAGN_is_refused(torus):
    """``agn_ir_frac`` plus a user ``agn_log_lbol`` raises for the Fritz torus as for SKIRTOR."""
    from tengri.config.exceptions import ConfigError

    with pytest.raises(ConfigError, match="agn_log_lbol"):
        _model(torus, agn_log_lbol=11.0)


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
@pytest.mark.parametrize("disc", ["schartmann2005", "kubota_done"])
def test_anchor_gradient_in_fracAGN_is_finite_and_nonzero(torus, disc):
    """``d L_2500_intrinsic / d agn_ir_frac`` is finite and non-zero (it was exactly 0)."""
    model = _model(torus, disc, free_frac=True)

    def anchor(f):
        return model.predict_state({"agn_ir_frac": f}).derived["L_2500_intrinsic"]

    grad = float(jax.grad(anchor)(jnp.asarray(0.1)))
    assert np.isfinite(grad)
    assert grad > 0.0, f"{disc}/{torus}: gradient {grad}"


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
@pytest.mark.parametrize("disc", ["schartmann2005", "kubota_done"])
def test_float32_tied_anchors_are_finite_and_match_float64(torus, disc):
    """Pure float32 evaluates the tied anchors and disc finitely, within 1e-3 of float64."""
    wave_np = np.geomspace(8.0, 1.0e8, 3000)
    kwargs = {**_BASE, "agn_torus_block": torus, "agn_disc_block": disc, "agn_log_mbh": 8.0}

    def evaluate(dtype):
        w = jnp.asarray(wave_np, dtype=dtype)
        sed, l2500, l4400 = compose_l_nu(
            w, _LOG_LBOL, agn_ir_frac=0.1, agn_cos_inc=_COS30, return_l2500=True, **kwargs
        )
        return np.asarray(sed, float), float(l2500), float(l4400)

    ref = evaluate(jnp.float64)
    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed32, l2500_32, l4400_32 = evaluate(jnp.float32)
    assert np.all(np.isfinite(sed32)) and np.isfinite(l2500_32) and np.isfinite(l4400_32)
    assert l2500_32 / ref[1] == pytest.approx(1.0, abs=1e-3)
    assert l4400_32 / ref[2] == pytest.approx(1.0, abs=1e-3)
    assert float(np.sum(sed32)) / float(np.sum(ref[0])) == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------------------------
# 3. the conserving line debit reads the closed-form disc power (#2743)
# ----------------------------------------------------------------------------------
_LINES = dict(agn_nlr_block="analytic", agn_blr_block="analytic", agn_feii_block="boroson_green")


def _line_power(disc):
    """Sum of the registered closed-form line powers [erg/s] at ``agn_log_lbol = 11``.

    The FeII power is set by ``lambda L_lambda(5100 A)`` of the disc at the reference
    inclination ``cos i = 0.5``, as the runner reads it.
    """
    block = resolve_agn_block("disc", disc)
    l5100 = (
        float(
            block(jnp.asarray([5100.0]), agn_log_lbol=_LOG_LBOL, agn_cos_inc=0.5, agn_log_mbh=8.0)[
                0
            ]
        )
        * 5100.0
    )
    return sum(
        float(LINE_ENERGY_BLOCKS[(kind, name)](_LOG_LBOL, jnp.asarray(l5100)))
        for kind, name in (("nlr", "analytic"), ("blr", "analytic"), ("feii", "boroson_green"))
    )


def _debit_fraction(disc):
    """The fraction of the disc the conserving ledger removes, read off ``sed_agn_disc``."""
    wave = jnp.asarray(np.geomspace(100.0, 1.0e5, 200))
    kwargs = {
        **_BASE,
        **_LINES,
        "agn_disc_block": disc,
        "agn_torus_block": "none",
        "agn_norm": "conserving",
    }
    _, comps = compose_l_nu(
        wave, _LOG_LBOL, agn_cos_inc=0.5, agn_log_mbh=8.0, return_components=True, **kwargs
    )
    block = resolve_agn_block("disc", disc)(
        wave, agn_log_lbol=_LOG_LBOL, agn_cos_inc=0.5, agn_log_mbh=8.0
    )
    nu_conv = np.asarray(wave) ** 2 / C_AA
    mask = np.asarray(block) > 0
    removed = 1.0 - np.asarray(comps["disc"])[mask] / (np.asarray(block)[mask] * nu_conv[mask])
    assert np.ptp(removed) < 1e-12, "the debit is one scalar on the disc"
    return float(removed[0])


@pytest.mark.parametrize("disc", ["kubota_done", "multicolor"])
def test_line_debit_is_lines_over_the_closed_form_disc_power(disc):
    """``1 - disc_sed / disc_block`` = ``E_lines / L_acc`` to 1e-12."""
    assert disc in protocol.DISC_POWER_BLOCKS
    assert _debit_fraction(disc) == pytest.approx(_line_power(disc) / _L_ACC, rel=1e-12)


@pytest.mark.parametrize(("disc", "bound"), [("kubota_done", 7.1e-4), ("multicolor", 3.3e-5)])
def test_closed_form_power_differs_from_the_grid_integral_by_the_measured_amount(disc, bound):
    """Budget-grid integral of the disc: 1 - 7.26e-4 (kubota_done), 1 - 3.3e-5 (multicolor).

    That is how far the debit moves from the grid evaluation: the lines' share of the disc
    rises by that fraction. ``kubota_done``'s is the corona's cut tail (the 0.99927 of the
    quadrature); the bound is the measured one with 20 % headroom.
    """
    if disc == "kubota_done":
        grid = np.asarray(runner_module._KUBOTA_LEDGER_WAVE)
    else:
        grid = np.asarray(runner_module._LEDGER_WAVE)
    lam = resolve_agn_block("disc", disc)(
        jnp.asarray(grid), agn_log_lbol=_LOG_LBOL, agn_cos_inc=0.5, agn_log_mbh=8.0
    )
    shortfall = 1.0 - float(np.trapezoid(np.asarray(lam), grid)) / _L_ACC
    assert shortfall == pytest.approx(bound, rel=0.2)


def test_tied_fritz_disc_refuses_a_grid_file_without_norm():
    """An older grid file (no ``fritz2006/norm``) raises instead of dropping norm(0)/norm(psi)."""
    from tengri.components.agn.fritz import fritz_disc_dust_ratio, load_fritz_default_grid
    from tengri.config.exceptions import TengriIOError

    # Obtain a real FritzGrid and make a copy with norm=None.
    grid = load_fritz_default_grid()
    grid_no_norm = grid._replace(norm=None)

    # Prepare arguments for fritz_disc_dust_ratio.
    wave = jnp.asarray(np.geomspace(100.0, 1e7, 200))
    disc_lambda_unreddened = jnp.ones_like(wave)
    disc_ext_fac = jnp.ones_like(wave)

    # Verify the positive path still works with the unmodified grid.
    result = fritz_disc_dust_ratio(
        wave,
        disc_lambda_unreddened,
        disc_ext_fac,
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=50.1,
        _template=grid,
    )
    assert result is not None

    # Assert that calling with the grid missing norm raises TengriIOError.
    with pytest.raises(TengriIOError, match="build_fritz2006_grid"):
        fritz_disc_dust_ratio(
            wave,
            disc_lambda_unreddened,
            disc_ext_fac,
            agn_fritz_r_ratio=60.0,
            agn_fritz_tau=1.0,
            agn_fritz_beta=-0.5,
            agn_fritz_gamma=4.0,
            agn_fritz_oa=60.0,
            agn_fritz_psy=50.1,
            _template=grid_no_norm,
        )

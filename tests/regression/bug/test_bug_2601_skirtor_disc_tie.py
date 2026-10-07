# SPDX-License-Identifier: BSD-3-Clause
r"""SKIRTOR disc tie: one inclination factor, one torus screen, one 8 nm cut (#2601).

With ``agn_ir_frac > 0`` the SKIRTOR disc is tied to the torus power as in CIGALE's
``skirtor2016`` module (Stalevski et al. 2012, 2016; Yang et al. 2020): the disc is
``agn_power x analytic shape x disk(i)/disk(0)`` and the library ratio
``disk(i)/disk(0)`` carries both the accretion-disc anisotropy
:math:`\eta(i) = \cos i\,(1 + 2\cos i)/3` and, for sightlines through the torus
(:math:`i > 90^\circ - {\rm oa}`), the torus extinction. Every expected value below
is computed in the test from the shipped SKIRTOR library (``spectra/disk_emission``,
``spectra/dust_emission``), from the published piecewise power-law discs written out below
and from CIGALE's tie (analytic shape x int(disk_0) x disk_i/disk_0, divided by int(dust_i)),
never read from tengri.

The tengri template is a 136-node re-gridding of the CIGALE database (948 nodes) and
the torus block evaluates it through the triweight smoother. The absolute disc level
therefore sits **0.55 %** below CIGALE's at i = 0 (fiducial: t = 7, p = q = 1,
oa = 40, R = 20, ``schartmann2005`` disc), and the offset factorizes exactly:

* 0.99633 = ``int(torus template through the interpolator) / agn_power``, against
  ``int(dust)`` = 1.000000 of the library node (the node-exact PCHIP the tie itself
  reads): the smoother moves 0.37 % of the torus power out of the integral (#2606);
* 0.99809 = ``R`` evaluated on the 136-node library grid over ``R`` on CIGALE's
  948-node grid (4.4246 against 4.4331): the quadrature of the same tabulated
  spectra on a coarser wavelength axis.

The inclination dependence is not affected: ``(tengri/CIGALE)(i) / (tengri/CIGALE)(0)``
stays within 3e-3 for i = 30, 50, 70 and 90 degrees.

References
----------
.. [1] Stalevski et al. 2012, MNRAS, 420, 2756.
.. [2] Stalevski et al. 2016, MNRAS, 458, 2288.
.. [3] Yang et al. 2020, MNRAS, 491, 740.
.. [4] Boquien et al. 2019, A&A, 622, A103 (CIGALE ``skirtor2016``).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import disc_cigale as DC
from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.components.dust.attenuation import smc
from tengri.utils.physics_constants import C_AA, L_SUN
from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.regression_bug

_GRID_FILE = DATA_DIR / "skirtor_templates_v3.h5"
_FRAC = 0.3  # CIGALE fracAGN of the reference runs
_LOG_LBOL = 12.0
_TORUS_FRAC = 0.5  # agn_power = _TORUS_FRAC * L_bol
_PARAMS = dict(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97)
_INCLINATIONS = (0, 30, 50, 70, 90)
_TYPE1 = (0, 30, 50)  # i <= 90 - oa
_TYPE2 = (70, 90)
_COS = {i: float(np.cos(np.radians(i))) for i in _INCLINATIONS}
_ETA = lambda i: np.cos(np.radians(i)) * (1 + 2 * np.cos(np.radians(i))) / 3  # noqa: E731
_WAVE = jnp.asarray(np.geomspace(8.0, 1.0e8, 3000))  # [A] covers the library axis
_DENSE_WAVE = jnp.asarray(np.geomspace(8.0, 1.0e8, 40000))  # [A] dense covering
_RUNNER = dict(
    agn_disc_block="schartmann2005",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="skirtor",
    agn_attenuation_block="none",
    agn_norm="cigale_joint",
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_torus_frac=_TORUS_FRAC,
)
_POLAR = dict(
    agn_attenuation_block="polar_dust",
    agn_polar_oa=40.0,
    agn_polar_T=100.0,
    agn_polar_beta=1.6,
)


# ----------------------------------------------------------------------------------
# the published piecewise power-law discs, written out
# ----------------------------------------------------------------------------------
def _discs(disk_type, delta):
    """Components ``(weight, breaks [nm], slopes)`` of the three published discs.

    ``disk_type`` 0: SKIRTOR disc (Stalevski et al. 2012, 2016), breaks 8, 10, 100, 5000, 1e6 nm,
    slopes 0.2, -1, -1.5 + delta, -4. ``disk_type`` 1: Schartmann et al. (2005) disc, breaks
    8, 50, 125, 1e4, 1e6 nm, slopes 1, -0.2, -1.5 + delta, -4. ``disk_type`` 2: ADAF / thin-disc
    blend (Lopez et al. 2024), weights ``1 - delta`` and ``delta``.
    """
    if disk_type == 0:
        return [(1.0, [8.0, 10.0, 100.0, 5000.0, 1e6], [0.2, -1.0, -1.5 + delta, -4.0])]
    if disk_type == 1:
        return [(1.0, [8.0, 50.0, 125.0, 1e4, 1e6], [1.0, -0.2, -1.5 + delta, -4.0])]
    return [
        (
            1.0 - delta,
            [8.0, 75.0, 300.0, 1100.0, 2700.0, 20000.0, 100000.0, 1e6],
            [0.5, 0.15, 0.45, -0.05, -0.55, -1.5, -4.0],
        ),
        (
            delta,
            [8.0, 50.0, 2000.0 - delta * 1875, 5000.0 - delta * 2000, 1e4, 1e6],
            [9 - 8 * delta, 4.2 - 4.4 * delta, 0.7 - 2.2 * delta, -6.5 + 5 * delta, -4.0],
        ),
    ]


def _density(wl, breaks, slopes):
    """Continuous broken power law on [breaks[0], breaks[-1]), up to a constant.

    ``ln f = int slope d ln(wl)``: the log-amplitude at the start of segment k is the
    cumulative sum of ``slope_j ln(b_{j+1}/b_j)`` over the segments below it.
    """
    wl = np.asarray(wl, dtype=float)
    b = np.log(np.asarray(breaks, dtype=float))
    s = np.asarray(slopes, dtype=float)
    start = np.concatenate([[0.0], np.cumsum(s * np.diff(b))[:-1]])
    k = np.clip(np.searchsorted(np.exp(b), wl, side="right") - 1, 0, s.size - 1)
    inside = (wl >= breaks[0]) & (wl < breaks[-1])
    safe = np.where(inside, wl, np.exp(b[0]))
    return np.where(inside, np.exp(start[k] + s[k] * (np.log(safe) - b[k])), 0.0)


def _fine_area(breaks, slopes):
    """Area of ``_density`` by trapezoid quadrature on a 1e6-node log grid (error below 1e-9)."""
    grid = np.logspace(np.log10(breaks[0]), np.log10(breaks[-1]), 1_000_000)
    return float(np.trapezoid(_density(grid, breaks, slopes), grid))


def _disc_exact(disk_type, delta, wl):
    """Unit-area published disc at ``wl`` [nm], normalized by its resolved area."""
    return sum(
        w * _density(wl, br, sl) / _fine_area(br, sl) for w, br, sl in _discs(disk_type, delta)
    )


def _grid_normalized_disc(disk_type, wl, delta):
    """The published disc normalized by its trapezoid area on ``wl`` (CIGALE's convention)."""
    return sum(
        w * _density(wl, br, sl) / np.trapezoid(_density(wl, br, sl), wl)
        for w, br, sl in _discs(disk_type, delta)
    )


_TENGRI_DISC = {
    0: DC.skirtor_disk_spectrum,
    1: DC.schartmann2005_disk_spectrum,
    2: DC.adaf_disk_spectrum,
}


# ----------------------------------------------------------------------------------
# libraries and references
# ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def library():
    """The shipped SKIRTOR cube at the fiducial cell: disk/dust per inclination node."""
    h5py = pytest.importorskip("h5py")
    if not _GRID_FILE.is_file():
        pytest.skip(f"SKIRTOR v3 grid not found at {_GRID_FILE}")
    with h5py.File(_GRID_FILE, "r") as f:
        axes = {k: np.asarray(f["grid"][k]) for k in f["grid"]}
        wl_nm = np.asarray(f["wavelength"]) / 10.0
        cell = tuple(
            int(np.argmin(np.abs(axes[k] - v)))
            for k, v in (
                ("tau_97", 7),
                ("p", 1),
                ("q", 1),
                ("opening_angle", 40),
                ("radius_ratio", 20),
            )
        )
        nodes = {}
        for i in _INCLINATIONS:
            ic = int(np.argmin(np.abs(axes["cos_inclination"] - _COS[i])))
            assert abs(axes["cos_inclination"][ic] - _COS[i]) < 1e-9, "inclination is a grid node"
            nodes[i] = (
                np.asarray(f["spectra/disk_emission"][(*cell, ic)], float),
                np.asarray(f["spectra/dust_emission"][(*cell, ic)], float),
            )
    return {"wl_nm": wl_nm, "nodes": nodes}


def _tie_ratio(wl, disk_0, disk_i, dust_i, shape):
    """``int(shape x int(disk_0) x disk_i/disk_0) / int(dust_i)`` (CIGALE tie)."""
    live = disk_0 > 0.0  # CIGALE ``nan_to_num``: zero where the face-on disc is zero
    ratio = np.divide(disk_i, disk_0, out=np.zeros_like(disk_i), where=live)
    return np.trapezoid(shape * np.trapezoid(disk_0, wl) * ratio, wl) / np.trapezoid(dust_i, wl)


@pytest.fixture(scope="module")
def library_ratio(library):
    """``R(i)`` from the shipped library: the template-only disc per unit ``agn_power``."""
    wl = library["wl_nm"]
    shape = _grid_normalized_disc(1, wl, 0.0)
    d0 = library["nodes"][0][0]
    return {
        i: _tie_ratio(wl, d0, library["nodes"][i][0], library["nodes"][i][1], shape)
        for i in _INCLINATIONS
    }


@pytest.fixture(scope="module")
def cigale():
    """CIGALE's own disc per unit ``agn_power`` and its database ``R`` (polar dust off)."""
    pytest.importorskip("pcigale")
    from pcigale.data import SimpleDatabase
    from pcigale.sed import SED
    from pcigale.sed_modules import skirtor2016 as ps

    def disc(i):
        sed = SED()
        sed.add_info("dust.luminosity", 1.0, True, unit="W")
        ps.SKIRTOR2016(
            name="skirtor2016",
            **_PARAMS,
            i=i,
            disk_type=1,
            delta=0,
            fracAGN=_FRAC,
            lambda_fracAGN="0/0",
            law=0,
            EBV=0.0,
            temperature=100.0,
            emissivity=1.6,
        ).process(sed)
        return sed.info["agn.disk_luminosity"] / (_FRAC / (1 - _FRAC))

    with SimpleDatabase("skirtor2016") as db:
        e0 = db.get(**_PARAMS, i=0)
        shape = _grid_normalized_disc(1, e0.wl, 0.0)
        formula = {}
        for i in _INCLINATIONS:
            e = db.get(**_PARAMS, i=i)
            formula[i] = _tie_ratio(e0.wl, e0.disk, e.disk, e.dust, shape)
    return {"disc": {i: disc(i) for i in _INCLINATIONS}, "formula": formula, "wl_nm": e0.wl}


# ----------------------------------------------------------------------------------
# runner helpers
# ----------------------------------------------------------------------------------
def _power(component, wave=_WAVE):
    """``int L_nu dnu`` of a component [erg/s], in float64."""
    nu = C_AA / np.asarray(wave, dtype=np.float64)
    order = np.argsort(nu)
    return float(np.trapezoid(np.asarray(component, dtype=np.float64)[order], nu[order]))


def _run(i_deg, *, frac=_FRAC, wave=_WAVE, cos_inc=None, **overrides):
    """``compose_l_nu(..., return_components=True)`` on the fiducial SKIRTOR composition."""
    kwargs = {**_RUNNER, **overrides}
    cos = _COS[i_deg] if cos_inc is None else cos_inc
    sed, comps = compose_l_nu(
        wave, _LOG_LBOL, agn_ir_frac=frac, agn_cos_inc=cos, return_components=True, **kwargs
    )
    return sed, {k: np.asarray(v) for k, v in comps.items()}


def _agn_power_target():
    """``agn_torus_frac x L_bol`` [erg/s]: the budget ``fracAGN/(1-fracAGN) x L_absorbed``."""
    return _TORUS_FRAC * 10.0**_LOG_LBOL * L_SUN


def _screen(wave_aa, i_deg, oa=40.0, tau_v=7.0):
    """Torus screen from its documented formula (``torus_screen.py``).

    ``T = exp(-min(tau, 50))``, ``tau = tau_V k(lambda)/k(V) w``,
    ``w = sigmoid((sin(oa) - cos i)/0.025)`` with the SMC curve normalized at V (5500 A).
    """
    w = 0.5 * (1 + np.tanh(0.5 * (np.sin(np.radians(oa)) - np.cos(np.radians(i_deg))) / 0.025))
    k = np.asarray(smc(jnp.asarray(np.append(np.asarray(wave_aa, float), 5500.0))))
    return np.exp(-np.clip(tau_v * k[:-1] / k[-1] * w, 0.0, 50.0))


# ----------------------------------------------------------------------------------
# 1. library facts and CIGALE formula transcription
# ----------------------------------------------------------------------------------
def test_library_ratio_already_carries_eta(library):
    """``int disk(i)/int disk(0)`` of the physical records equals eta(i) for i <= 40 deg.

    The shipped records are divided by their own ``norm``; the physical ratio restores it.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(_GRID_FILE, "r") as f:
        axes = {k: np.asarray(f["grid"][k]) for k in f["grid"]}
        cell = tuple(
            int(np.argmin(np.abs(axes[k] - v)))
            for k, v in (
                ("tau_97", 7),
                ("p", 1),
                ("q", 1),
                ("opening_angle", 40),
                ("radius_ratio", 20),
            )
        )
        wl = np.asarray(f["wavelength"])
        phys = {}
        for i in (0, 10, 20, 30, 40):
            ic = int(np.argmin(np.abs(axes["cos_inclination"] - np.cos(np.radians(i)))))
            disk = np.asarray(f["spectra/disk_emission"][(*cell, ic)], float)
            phys[i] = np.trapezoid(disk, wl) * float(f["spectra/norm"][(*cell, ic)])
    for i in (10, 20, 30, 40):
        assert phys[i] / phys[0] / _ETA(i) == pytest.approx(1.0, abs=1e-2), (
            f"i={i}: library int disk(i)/int disk(0) / eta(i) = {phys[i] / phys[0] / _ETA(i):.4f}"
        )


def test_literature_formula_reproduces_cigale_r(cigale, library_ratio):
    """The test's ``R(i)`` formula on CIGALE's database equals CIGALE's own disc to 1e-6."""
    for i in _INCLINATIONS:
        assert cigale["formula"][i] == pytest.approx(cigale["disc"][i], rel=1e-6), i
    # the 136-node library reproduces the 948-node one to 0.5 %: the quadrature part of the offset
    for i in _INCLINATIONS:
        assert library_ratio[i] / cigale["formula"][i] == pytest.approx(1.0, abs=6e-3), i


# ----------------------------------------------------------------------------------
# 2. disc power on the tied path
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_tied_disc_equals_template_only_value(i_deg, library_ratio):
    """disc = agn_power x R(i) from the library at Type 1 and Type 2 (one eta, one screen).

    A second eta would lower the disc by 0.789 (30), 0.490 (50), 0.192 (70); a second torus
    screen by 0.642 (50) and 0.694 (70).
    """
    _, comps = _run(i_deg)
    got = _power(comps["disc"]) / _power(comps["torus"])
    assert got == pytest.approx(library_ratio[i_deg], rel=3e-3), (
        f"i={i_deg}: disc/agn_power = {got:.5f}, library template-only {library_ratio[i_deg]:.5f}"
    )


@pytest.mark.parametrize("i_deg", _INCLINATIONS[1:])
def test_inclination_ratio_matches_cigale(i_deg, cigale):
    """(tengri/CIGALE)(i) / (tengri/CIGALE)(0) within 3e-3 (6e-3 at i = 90); level within 1 %.

    The edge-on node carries a larger quadrature offset on the 136-node library axis
    (R_library/R_CIGALE = 0.99499 against 0.99809 face-on) and a larger smoother offset
    (int torus / agn_power = 0.99412 against 0.99633): their ratio is 0.99468 and the absolute
    level 0.98913 (1.2 % at i = 90, 1 % below).
    """
    target = _agn_power_target()
    tengri = {i: _power(_run(i)[1]["disc"]) / target for i in (0, i_deg)}
    rel = {i: tengri[i] / cigale["disc"][i] for i in tengri}
    assert rel[i_deg] / rel[0] == pytest.approx(1.0, abs=6e-3 if i_deg == 90 else 3e-3), (
        f"i={i_deg}: inclination ratio {rel[i_deg] / rel[0]:.5f} "
        f"(tengri/CIGALE {rel[i_deg]:.5f} at i, {rel[0]:.5f} at 0)"
    )
    assert rel[i_deg] == pytest.approx(1.0, abs=1.2e-2 if i_deg == 90 else 1e-2)


@pytest.mark.parametrize("i_deg", [0, 70])
def test_absolute_offset_is_template_interpolation_and_native_grid(i_deg, cigale, library_ratio):
    """tengri/CIGALE = (int torus/agn_power) x (R_library/R_CIGALE) x (R_tengri/R_library).

    At i = 0: 0.99633 x 0.99809 x 1.0001 = 0.9945. The first factor is the torus block's
    triweight smoother against the library node's int(dust) = 1 (#2606); the second is the
    quadrature of the same spectra on the 136-node library axis.
    """
    _, comps = _run(i_deg)
    torus_over_target = _power(comps["torus"]) / _agn_power_target()
    r_tengri = _power(comps["disc"]) / _power(comps["torus"])
    quad = library_ratio[i_deg] / cigale["formula"][i_deg]
    assert torus_over_target == pytest.approx(1.0, abs=6e-3)
    assert quad == pytest.approx(1.0, abs=4e-3)
    assert r_tengri / library_ratio[i_deg] == pytest.approx(1.0, abs=3e-3)
    total = torus_over_target * quad * (r_tengri / library_ratio[i_deg])
    assert total == pytest.approx(
        _power(comps["disc"]) / _agn_power_target() / cigale["disc"][i_deg]
    )
    assert total == pytest.approx(1.0, abs=1e-2)


# ----------------------------------------------------------------------------------
# 3. what the screen reaches
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("i_deg", _TYPE2)
def test_broad_lines_screened_identically_tied_and_untied(i_deg):
    """Broad lines + FeII carry the torus screen on both paths.

    The line SED is identical (to 1e-6) with the disc tied (fracAGN = 0.3) and untied
    (fracAGN = 0), and relative to face-on it equals the documented screen transmission.
    """
    overrides = dict(agn_blr_block="analytic", agn_feii_block="boroson_green")
    tied = _run(i_deg, **overrides)[1]["lines"]
    untied = _run(i_deg, frac=0.0, **overrides)[1]["lines"]
    face_on = _run(0, **overrides)[1]["lines"]
    assert np.max(np.abs(tied - untied)) / np.max(np.abs(untied)) < 1e-6
    wave = np.asarray(_WAVE)
    j = int(np.argmin(np.abs(wave - 5000.0)))
    assert face_on[j] > 0.0
    expected = _screen(wave[j : j + 1], i_deg)[0] / _screen(wave[j : j + 1], 0)[0]
    assert tied[j] / face_on[j] == pytest.approx(expected, rel=1e-6), (
        f"i={i_deg}: lines(i)/lines(0) at {wave[j]:.0f} A = {tied[j] / face_on[j]:.6e}, "
        f"screen transmission ratio {expected:.6e} (T(5000 A) = {_screen([5000.0], i_deg)[0]:.4e})"
    )
    bright = face_on > 1e-3 * face_on.max()
    expected_all = _screen(wave[bright], i_deg) / _screen(wave[bright], 0)
    np.testing.assert_allclose(tied[bright] / face_on[bright], expected_all, rtol=1e-6)


@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_components_sum_to_total(i_deg):
    """disc + torus + lines + polar equals the total SED to 1e-10, with lines and polar on."""
    for overrides in (
        dict(agn_blr_block="analytic", agn_feii_block="boroson_green"),
        dict(**_POLAR, agn_polar_ebv=0.03),
    ):
        sed, comps = _run(i_deg, **overrides)
        total = sum(comps[k] for k in ("disc", "torus", "lines", "polar"))
        assert np.max(np.abs(total - np.asarray(sed))) / np.max(np.abs(np.asarray(sed))) < 1e-10


@pytest.mark.parametrize("i_deg", _INCLINATIONS)
def test_untied_disc_is_screened(i_deg):
    """At fracAGN = 0 the disc is the analytic shape and the torus screen is its obscuration.

    Expected: the face-on disc (screen at i = 0 divided out) times the documented screen at i.
    """
    wave = np.asarray(_WAVE)
    face_on = _run(0, frac=0.0)[1]["disc"] / _screen(wave, 0)
    got = _run(i_deg, frac=0.0)[1]["disc"]
    np.testing.assert_allclose(got, face_on * _screen(wave, i_deg), rtol=1e-9, atol=0.0)
    assert _power(got) > 0.0


# ----------------------------------------------------------------------------------
# 4. polar dust
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("i_deg", _INCLINATIONS)
@pytest.mark.parametrize("ebv", [0.0, 0.03, 0.3])
def test_polar_dust_leaves_the_tie_unchanged(i_deg, ebv):
    """The polar switch keeps the AGN budget and divides disc and torus by one factor.

    ``torus + polar`` carries exactly the polar-off ``agn_power`` (the ``R_faceon``
    bookkeeping closes), E(B-V) = 0 leaves the disc untouched at every inclination, and for
    i > 90 - oa (no line-of-sight reddening) the disc is the polar-off disc divided by
    ``1 + polar share``, the factor the torus carries: CIGALE's joint normalization
    ``1/int(dust + polar)`` scales the disc and the dust together (#2602).
    """
    # The budget closes exactly on the runner's fixed grids and is independent of the
    # caller's; it is summed here on a dense covering grid, since the quadrature error
    # of the 3000-node ``_WAVE`` alone is ~1e-6.
    _, off = _run(i_deg, wave=_DENSE_WAVE)
    _, on = _run(i_deg, wave=_DENSE_WAVE, **_POLAR, agn_polar_ebv=ebv)
    budget = (_power(on["torus"], _DENSE_WAVE) + _power(on["polar"], _DENSE_WAVE)) / _power(
        off["torus"], _DENSE_WAVE
    )
    assert budget == pytest.approx(1.0, abs=1e-6)
    ratio = _power(on["disc"], _DENSE_WAVE) / _power(off["disc"], _DENSE_WAVE)
    if ebv == 0.0:
        assert ratio == pytest.approx(1.0, abs=1e-3)
    elif i_deg in _TYPE2:
        # CIGALE's ``norm = 1/int(dust + polar)`` divides the disc by 1 + l_ext, with
        # l_ext = polar/torus read off this run's own components.
        expected = 1.0 / (
            1.0 + _power(on["polar"], _DENSE_WAVE) / _power(on["torus"], _DENSE_WAVE)
        )
        assert ratio == pytest.approx(expected, rel=1e-6)
    else:
        assert 0.0 < ratio < 1.0 - 1e-3  # Type-1 sightline is reddened by the cone dust


# ----------------------------------------------------------------------------------
# 5. precision and gradient
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("i_deg", [30, 70])
def test_disc_power_float32_matches_float64(i_deg):
    """The tied disc power agrees between float32 and float64 to 1e-4 (relative to the torus)."""
    ratios = {}
    for x64 in (True, False):
        with jax.enable_x64(x64):
            dtype = jnp.float64 if x64 else jnp.float32
            wave = jnp.asarray(np.geomspace(8.0, 1.0e8, 3000), dtype=dtype)
            _, comps = _run(i_deg, wave=wave, cos_inc=jnp.asarray(_COS[i_deg], dtype=dtype))
            assert comps["disc"].dtype == dtype
            ratios[x64] = _power(comps["disc"], wave) / _power(comps["torus"], wave)
    assert np.isfinite(ratios[False]) and ratios[False] > 0.0
    assert ratios[False] == pytest.approx(ratios[True], rel=1e-4)


@pytest.mark.parametrize("cos_inc", [0.8, 0.3])
def test_disc_power_gradient_in_cos_inc(cos_inc):
    """d(disc power)/d(cos i) is finite, non-zero and matches a central difference."""

    def disc_over_torus(c):
        _, comps = compose_l_nu(
            _WAVE, _LOG_LBOL, agn_ir_frac=_FRAC, agn_cos_inc=c, return_components=True, **_RUNNER
        )
        nu = C_AA / _WAVE
        order = jnp.argsort(nu)
        return jnp.trapezoid(comps["disc"][order], nu[order]) / jnp.trapezoid(
            comps["torus"][order], nu[order]
        )

    grad = float(jax.grad(disc_over_torus)(jnp.asarray(cos_inc)))
    h = 1e-4
    fd = (float(disc_over_torus(cos_inc + h)) - float(disc_over_torus(cos_inc - h))) / (2 * h)
    assert np.isfinite(grad) and abs(grad) > 0.0
    assert grad == pytest.approx(fd, rel=2e-2)


# ----------------------------------------------------------------------------------
# 6. the piecewise disc
# ----------------------------------------------------------------------------------
_DISC_CASES = [(0, -0.5), (0, 0.0), (0, 0.5), (1, -0.5), (1, 0.0), (1, 0.5), (2, 0.0), (2, 0.5)]
_MODEL_GRID_NM = np.logspace(1, 8, 2380) / 10.0  # 10 A - 1e8 A, the model's panchromatic axis


@pytest.mark.parametrize(("disk_type", "delta"), _DISC_CASES)
def test_disc_level_at_250nm_matches_cigale(disk_type, delta, cigale):
    """Unit-area level within 2e-3 of CIGALE's formula, on CIGALE's grid and on the model grid.

    The closed-form area makes the level independent of the sampling and of any 1-8 nm
    tail. Compared at the grid node nearest 250 nm (no interpolation between sparse nodes).
    """
    tf = _TENGRI_DISC[disk_type]
    for grid in (cigale["wl_nm"], _MODEL_GRID_NM):
        k = int(np.argmin(np.abs(grid - 250.0)))
        got = float(np.asarray(tf(jnp.asarray(grid), delta=delta))[k])
        assert got / _disc_exact(disk_type, delta, grid[k]) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize(("disk_type", "delta"), _DISC_CASES[:6])
def test_residual_against_cigale_output_is_its_trapezoid_area(disk_type, delta, cigale):
    """CIGALE's own level on its 948-node grid is ``exact / trapz(exact)``; tengri's is ``exact``.

    ``skirtor2016.disk`` divides by the trapezoid area on the database axis, so
    tengri/CIGALE at every node equals that area of the unit-area spectrum: 0.9889 / 0.9914 /
    0.9949 (``disk_type`` 0, delta = -0.5, 0, 0.5) and 1.0036 / 1.0030 / 1.0020
    (``disk_type`` 1), the whole of the distance from 1 (the area is 1 to 1e-6 on a fine axis).
    """
    ps = pytest.importorskip("pcigale.sed_modules.skirtor2016")
    wl = cigale["wl_nm"]
    pcigale_disc = (ps.skirtor_disk, ps.schartmann2005_disk)[disk_type]
    tengri = np.asarray(_TENGRI_DISC[disk_type](jnp.asarray(wl), delta=delta))
    live = pcigale_disc(wl, delta=delta) > 0.0
    area = np.trapezoid(tengri, wl)
    np.testing.assert_allclose(tengri[live] / pcigale_disc(wl, delta=delta)[live], area, rtol=1e-6)
    assert abs(area - 1.0) < 1.2e-2


@pytest.mark.parametrize(("disk_type", "delta"), _DISC_CASES)
def test_disc_is_zero_outside_its_limits_and_unit_area(disk_type, delta):
    """Zero below 8 nm and from 1e6 nm up; integrates to 1 on the model grid; grid independent."""
    tf = _TENGRI_DISC[disk_type]
    spectrum = np.asarray(tf(jnp.asarray(_MODEL_GRID_NM), delta=delta))
    assert np.all(spectrum[_MODEL_GRID_NM < 8.0] == 0.0)
    assert np.all(spectrum[_MODEL_GRID_NM >= 1e6] == 0.0)
    inside = (_MODEL_GRID_NM >= 8.0) & (_MODEL_GRID_NM < 1e6)
    assert np.all(spectrum[inside] > 0.0)
    assert np.trapezoid(spectrum, _MODEL_GRID_NM) == pytest.approx(1.0, abs=2e-3)
    single = float(np.asarray(tf(jnp.asarray([250.0]), delta=delta))[0])
    at_250 = float(np.interp(250.0, _MODEL_GRID_NM, spectrum))
    assert at_250 == pytest.approx(single, rel=2e-3)
    coarse = np.logspace(0.5, 7.5, 97) / 10.0
    coarse = np.unique(np.append(coarse, 250.0))
    on_coarse = np.asarray(tf(jnp.asarray(coarse), delta=delta))
    assert float(on_coarse[np.searchsorted(coarse, 250.0)]) == pytest.approx(single, rel=1e-12)


@pytest.mark.parametrize(("disk_type", "delta"), [(0, 0.0), (1, 0.5), (2, 0.5)])
def test_disc_float32_matches_float64(disk_type, delta):
    """The closed-form normalization stays finite and within 1e-4 in pure float32."""
    tf = _TENGRI_DISC[disk_type]
    with jax.enable_x64(True):
        f64 = np.asarray(tf(jnp.asarray(_MODEL_GRID_NM, dtype=jnp.float64), delta=delta))
    with jax.enable_x64(False):
        f32 = tf(jnp.asarray(_MODEL_GRID_NM, dtype=jnp.float32), delta=jnp.float32(delta))
        assert f32.dtype == jnp.float32
        f32 = np.asarray(f32, dtype=np.float64)
    assert np.all(np.isfinite(f32))
    live = f64 > 1e-3 * f64.max()
    np.testing.assert_allclose(f32[live], f64[live], rtol=1e-4)


@pytest.mark.parametrize(("disk_type", "delta"), _DISC_CASES)
def test_written_out_discs_agree_with_cigale_output(disk_type, delta):
    """CIGALE's disc output equals the test's disc normalized on CIGALE's own axis.

    Measured agreement: 8e-15 relative at every live node (the area on the 948-node axis is
    divided out on both sides); asserted at 1e-12, above the rounding floor.
    """
    ps = pytest.importorskip("pcigale.sed_modules.skirtor2016")
    from pcigale.data import SimpleDatabase

    with SimpleDatabase("skirtor2016") as db:
        wl = db.get(**_PARAMS, i=0).wl
    fn = (ps.skirtor_disk, ps.schartmann2005_disk, ps.adaf_disk)[disk_type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = fn(wl, delta=delta)
    mine = _grid_normalized_disc(disk_type, wl, delta)
    live = theirs > 0.0
    assert np.array_equal(live, mine > 0.0)
    np.testing.assert_allclose(mine[live], theirs[live], rtol=1e-12)


# ----------------------------------------------------------------------------------
# 7. the kubota_done reference of the float32 seam sweep
# ----------------------------------------------------------------------------------
#: float64 rest-frame SED of that model (``fracAGN = 0.1``, i = 30 deg, SKIRTOR torus,
#: ``agn_log_mbh`` 6 / 8 / 10) with the disc multiplied by eta(30 deg) T(lambda):
#: ``(sum, first bin, middle bin, last bin)``, on the model's master grid (the disc's
#: native 0.01 A - 1e8 A axis; the first bin is the 0.01 A node, which now carries the corona).
#: The tie normalizes the disc and warm zone ``2 cos i D_nu`` (both faces, ``c = cos 30``) to
#: ``agn_power x R`` on the SKIRTOR native grid (10 A - 1e8 A); the corona ``H_nu`` rides on top
#: with the angle-integrated share ``P_H = f/(1 - f) P_D,tied`` and the torus screen ``T``
#: (``P_D,tied = (7/18) agn_power R_faceon``, ``f`` the closed-form corona share). Derivation,
#: computed forward from measured inputs and not from these literals: with ``x = I_H/I_D`` the
#: corona-to-disc power ratio in the library range along the line of sight (measured
#: 0.00083 / 0.40815 / 0.50865), ``R_old``, ``R_new`` the tie ratios of the
#: ``D + H`` and ``D`` shapes (measured 3.59271 / 3.60042 / 3.60015 and 3.59274 /
#: 3.60933 / 3.60873), the disc part of the tied disc is multiplied by
#: ``(R_new/R_old)(1 + x)`` = 1.00084 / 1.41163 / 1.51225, and the corona (not in the old disc
#: at this level) is added as ``s_H H_nu T`` with
#: ``s_H = (7/18) agn_power R_faceon / ((1 - f) L_acc)`` =
#: 0.00606 / 0.01845 / 0.01988 (``f`` = 0.00629 / 0.67368 / 0.69723, ``agn_power`` the
#: old tie's own scalar x its in-grid integral / ``R_old``). Applied to the previous model's disc
#: this predicts the entries below to 1e-10 (log M_BH 6), 1e-8 (8) and 1.0e-8 (10) relative
#: (sum, middle and last bin).
_DISC_TIMES_ETA_T_REFERENCE = {
    6.0: (1.5586840342805008e32, 81149915103019.72, 2.8901478844181377e28, 8.775296641338167e21),
    8.0: (
        1.6504080200641076e32,
        2.226186048596356e16,
        3.0968665997787287e28,
        8.825196951058274e21,
    ),
    10.0: (
        1.6769970954584036e32,
        2.3324417803895464e16,
        3.1396733849669233e28,
        9.548716267622313e21,
    ),
}


@pytest.mark.parametrize("agn_log_mbh", [6.0, 8.0, 10.0])
def test_tied_disc_reference_of_the_seam_sweep(agn_log_mbh):
    """``total - disc (1 - eta T)`` of the tied model equals the SED with the disc times ``eta T``.

    The tied disc is ``agn_power x disk(i)/disk(0)``-shaped with no explicit eta and no
    screen; multiplying it by ``eta(30 deg) T(lambda)`` (``T`` from its documented formula)
    changes only the disc term of the SED. The ``kubota_done`` disc has no piecewise
    normalization, so nothing else moves (the identity closes to 1e-15 in a fresh process;
    asserted at 1e-7 here).
    """
    ssp_file = DATA_DIR / "fsps_prsc_miles_chabrier.h5"
    if not ssp_file.is_file():
        pytest.skip(f"SSP grid not found at {ssp_file}")
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    model = SEDModel.build(
        ssp_data=load_ssp_data(str(ssp_file)),
        observation=Observation(photometry=Photometry.from_names(["sdss_r"])),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {
                "type": "kubota_done",
                "all_params": Fixed(DEFAULT),
                "log_mbh": Uniform(6.0, 10.0),
            },
            "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            "norm": "cigale_joint",
            "log_lbol": Fixed(11.0),
            "fracAGN": 0.1,
        },
        redshift=Fixed(0.1),
    )
    params = {
        "sfh_delayed_log_total_mass": jnp.asarray(10.0),
        "agn_log_mbh": jnp.asarray(agn_log_mbh),
    }
    state = model.predict_state(params)
    wave = np.asarray(state.wave)
    total = np.asarray(model.predict(params).rest_sed())
    disc = np.asarray(state.derived["sed_agn_disc"])
    cos30 = np.cos(np.radians(30.0))
    eta = cos30 * (1 + 2 * cos30) / 3
    expected = total - disc * (1.0 - eta * _screen(wave, 30))
    got = (expected.sum(), expected[0], expected[len(expected) // 2], expected[-1])
    np.testing.assert_allclose(got, _DISC_TIMES_ETA_T_REFERENCE[agn_log_mbh], rtol=1e-7)


# ----------------------------------------------------------------------------------
# 8. the intrinsic-luminosity anchors follow the disc normalization
# ----------------------------------------------------------------------------------
_ANCHOR_CASES = [
    ("skirtor", 0, 0.0),
    ("schartmann2005", 1, 0.0),
    ("schartmann2005", 1, 0.5),
    ("adaf_lopez2024", 2, 0.0),
    ("adaf_lopez2024", 2, 0.5),
]


@pytest.mark.parametrize(("block", "disk_type", "delta"), _ANCHOR_CASES)
def test_intrinsic_anchors_equal_the_unit_area_disc(block, disk_type, delta):
    """``L_2500_intrinsic`` and ``L_4400_intrinsic`` are the unit-area published disc x L_bol.

    ``L_nu = lambda^2 L_lambda / c`` of ``L_bol x s(lambda)`` at 2500 and 4400 A, with ``s``
    the resolved-area disc of the test (per nm, divided by 10 per A). The anchors come from
    the disc block at cos i = cos 30 deg and follow neither the tie nor the screen. The 8 nm
    cut raises the ``skirtor`` anchors by 14 % (4.154e29 to 4.736e29 at L_bol = 1e12 L_sun)
    through the area it removes below 10 nm; the other discs move by 0.2 % or less.
    """
    _, l2500, l4400 = compose_l_nu(
        _WAVE,
        _LOG_LBOL,
        agn_disc_block=block,
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="none",
        agn_attenuation_block="none",
        agn_cigale_disk_delta=delta,
        return_l2500=True,
    )
    l_bol = 10.0**_LOG_LBOL * L_SUN
    for got, lam_aa in ((l2500, 2500.0), (l4400, 4400.0)):
        l_lambda = l_bol * _disc_exact(disk_type, delta, lam_aa / 10.0) / 10.0
        expected = l_lambda * lam_aa**2 / C_AA
        assert float(got) == pytest.approx(expected, rel=3e-3)

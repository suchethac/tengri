# SPDX-License-Identifier: BSD-3-Clause
r"""Behind a library torus the disc is dimmed by the library's normalized line of sight.

The untied composable path (``agn_norm="independent"``) multiplied the central engine behind a
SKIRTOR or Fritz (2006) torus by an analytic screen :math:`\exp(-\tau_{9.7}\,k(\lambda)/k_V
\, w(i))`. Three things were wrong with it. The 9.7 um depth was used as :math:`\tau_V`; the
same depth applied at every Type-2 inclination; and the screen carried no scattered light.
Against the libraries' own line-of-sight disc it was off by two orders of magnitude either way.

The transmission is now the library's normalized inclination ratio with the disc anisotropy
divided out,

.. math::

    T(\lambda; i) = \frac{R_n(\lambda; i)}{\eta(i)},\qquad
    R_n = \frac{{\rm disk}_i\,{\rm norm}_i}{{\rm disk}_0\,{\rm norm}_0},

with :math:`\eta = \cos i(1 + 2\cos i)/3` for SKIRTOR and :math:`\eta = 1` for Fritz. The
per-record ``norm`` matters: without it the SKIRTOR :math:`R/\eta` is 11 per cent off unity
at a Type-1 inclination, and the Fritz ratio exceeds the face-on disc. Statements checked
here, each against numbers read from the library files directly (no repository interpolator):

1. **T is** :math:`R_n/\eta` at the library nodes of the issue's SKIRTOR and Fritz cases, to
   1e-3.
2. **The disc factor is finite at i = 90 deg**, and its gradient with respect to
   ``agn_cos_inc`` is finite and non-zero in float32 and float64: the disc takes
   :math:`2\cos i\,R_n/\eta` as a product, not as a quotient by :math:`\eta`.
3. **Face-on is unscreened**, and a disc that carries the ``2 cos i`` law (multicolor)
   behind the torus follows :math:`\cos i\,T` relative to face-on.
4. **The set of discs that carry the 2 cos i law is pinned** to the blocks' own response to
   ``agn_cos_inc``.

References
----------
.. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288 (SKIRTOR).
.. [2] Fritz, J., Franceschini, A. & Hatziminaoglou, E. 2006, MNRAS, 366, 767.
.. [3] Boquien, M. et al. 2019, A&A, 622, A103 (CIGALE ``skirtor2016``, ``fritz2006``).
"""

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.agn.blocks.runner import _COS_LAW_DISC_BLOCKS, compose_l_nu
from tengri.components.agn.blocks.torus_library_transmission import (
    library_torus_sightline,
    library_torus_transmission,
)
from tengri.components.agn.skirtor import _find_skirtor_grid, _load_grid_arrays
from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.regression_bug

_FRITZ_FILE = DATA_DIR / "fritz2006_torus_grid.h5"
_LOG_LBOL = 11.0
_TOL = 1.0e-3
_V_WAVE = 5500.0  # [A] comparison wavelength; moved to the nearest library node below
_SKIRTOR_COS_NODES = {60.0: 0.5, 70.0: 0.3420201433256687, 80.0: 0.17364817766693041}
_SKIRTOR_SITES = [(tau, i) for tau in (3.0, 7.0, 11.0) for i in (60.0, 70.0, 80.0)]
_FRITZ_SITES = [(tau, psi) for tau in (0.1, 1.0, 3.0, 10.0) for psi in (10.1, 30.1, 50.1)]
_SKIRTOR_PARAMS = {
    "agn_tau_skirtor": 7.0,
    "agn_p_skirtor": 1.0,
    "agn_q_skirtor": 1.0,
    "agn_oa_skirtor": 40.0,
    "agn_radius_ratio": 20.0,
}


def _skirtor_raw():
    return _load_grid_arrays(_find_skirtor_grid())


def _skirtor_reference(tau: float, cos_node: float) -> tuple[float, float, float]:
    """(library node wavelength [A], R_n, T) at one SKIRTOR node, read from the file."""
    raw = _skirtor_raw()
    ax = [np.asarray(a) for a in raw["axes"]]
    idx = [
        int(np.argmin(np.abs(ax[0] - tau))),
        int(np.argmin(np.abs(ax[1] - 1.0))),
        int(np.argmin(np.abs(ax[2] - 1.0))),
        int(np.argmin(np.abs(ax[3] - 40.0))),
        int(np.argmin(np.abs(ax[4] - 20.0))),
    ]
    i_cos = int(np.argmin(np.abs(ax[5] - cos_node)))
    i_face = int(np.argmin(np.abs(ax[5] - 1.0)))
    wave = np.asarray(raw["wave"])
    i_w = int(np.argmin(np.abs(wave - _V_WAVE)))
    disk = np.asarray(raw["disk"], dtype=float)
    norm = np.asarray(raw["norm"], dtype=float)
    seen = disk[(*idx, i_cos, i_w)] * norm[(*idx, i_cos)]
    face = disk[(*idx, i_face, i_w)] * norm[(*idx, i_face)]
    ratio = seen / face
    eta = cos_node * (1.0 + 2.0 * cos_node) / 3.0
    return float(wave[i_w]), float(ratio), float(ratio / eta)


def _fritz_reference(tau: float, psi: float) -> tuple[float, float]:
    """(library node wavelength [A], R_n = T) at one Fritz node, read from the file."""
    with h5py.File(_FRITZ_FILE, "r") as f:
        g = f["fritz2006"]
        axes = {k: np.asarray(g[k + "_axis"][:]) for k in ("tau", "opening_angle", "psy")}
        wave = np.asarray(g["wavelength_aa"][:])
        i_tau = int(np.argmin(np.abs(axes["tau"] - tau)))
        i_oa = int(np.argmin(np.abs(axes["opening_angle"] - 20.0)))
        i_psy = int(np.argmin(np.abs(axes["psy"] - psi)))
        i_face = len(axes["psy"]) - 1
        i_w = int(np.argmin(np.abs(wave - _V_WAVE)))
        i_r = int(np.argmin(np.abs(np.asarray(g["r_ratio_axis"][:]) - 60.0)))
        i_beta = int(np.argmin(np.abs(np.asarray(g["beta_axis"][:]) + 0.5)))
        i_gamma = int(np.argmin(np.abs(np.asarray(g["gamma_axis"][:]) - 4.0)))
        pre = (i_r, i_tau, i_beta, i_gamma, i_oa)
        disk = g["disk"]
        norm = g["norm"]
        seen = float(disk[(*pre, i_psy, i_w)]) * float(norm[(*pre, i_psy)])
        face = float(disk[(*pre, i_face, i_w)]) * float(norm[(*pre, i_face)])
        return float(wave[i_w]), seen / face


@pytest.mark.parametrize(("tau", "i_deg"), _SKIRTOR_SITES)
def test_skirtor_transmission_is_normalized_ratio_over_eta(tau, i_deg):
    cos_node = _SKIRTOR_COS_NODES[i_deg]
    wave, _ratio, expected = _skirtor_reference(tau, cos_node)
    got = float(
        library_torus_transmission(
            "skirtor",
            jnp.asarray([wave]),
            cos_inc=cos_node,
            params={**_SKIRTOR_PARAMS, "agn_tau_skirtor": tau},
        )[0]
    )
    assert abs(got - expected) <= _TOL, f"SKIRTOR tau={tau} i={i_deg}: {got:.6f} vs {expected:.6f}"


@pytest.mark.parametrize(("tau", "psi_deg"), _FRITZ_SITES)
def test_fritz_transmission_is_normalized_ratio(tau, psi_deg):
    wave, expected = _fritz_reference(tau, psi_deg)
    got = float(
        library_torus_transmission(
            "fritz",
            jnp.asarray([wave]),
            cos_inc=float(np.sin(np.radians(psi_deg))),
            params={
                "agn_fritz_tau": tau,
                "agn_fritz_oa": 20.0,
                "agn_fritz_r_ratio": 60.0,
                "agn_fritz_beta": -0.5,
                "agn_fritz_gamma": 4.0,
            },
        )[0]
    )
    assert abs(got - expected) <= _TOL, (
        f"Fritz tau={tau} psi={psi_deg}: {got:.6f} vs {expected:.6f}"
    )


def test_norm_makes_the_type1_ratio_the_disc_anisotropy():
    """At the 40 deg node (Type 1, tau 7, oa 40) R_n/eta is unity to 2 per cent; raw is not.

    The stored disc without its ``norm`` has R/eta off unity by more than 5 per cent (the 11 per
    cent of the issue), so the per-record scale is what makes the ratio the anisotropy.
    """
    cos_node = 0.766044443118978
    wave, _ratio, t_norm = _skirtor_reference(7.0, cos_node)
    raw = _skirtor_raw()
    ax = [np.asarray(a) for a in raw["axes"]]
    idx = (
        int(np.argmin(np.abs(ax[0] - 7.0))),
        int(np.argmin(np.abs(ax[1] - 1.0))),
        int(np.argmin(np.abs(ax[2] - 1.0))),
        int(np.argmin(np.abs(ax[3] - 40.0))),
        int(np.argmin(np.abs(ax[4] - 20.0))),
    )
    i_cos = int(np.argmin(np.abs(ax[5] - cos_node)))
    i_face = int(np.argmin(np.abs(ax[5] - 1.0)))
    i_w = int(np.argmin(np.abs(np.asarray(raw["wave"]) - wave)))
    disk = np.asarray(raw["disk"], dtype=float)
    eta = cos_node * (1.0 + 2.0 * cos_node) / 3.0
    raw_over_eta = disk[(*idx, i_cos, i_w)] / disk[(*idx, i_face, i_w)] / eta
    assert abs(t_norm - 1.0) <= 0.02, f"R_n/eta = {t_norm:.4f}"
    assert abs(raw_over_eta - 1.0) >= 0.05, f"raw R/eta = {raw_over_eta:.4f}"


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
def test_face_on_is_unscreened(torus):
    wave = jnp.asarray([2500.0, 5500.0, 1.0e4, 1.0e5])
    sight = library_torus_sightline(torus, wave, cos_inc=1.0, params={})
    np.testing.assert_allclose(np.asarray(sight.transmission), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sight.disc_factor), 2.0, atol=1e-6)


def test_skirtor_type1_sightline_is_nearly_unscreened():
    """At 30 deg (inside the polar cone of oa 40) T is the library's, within 1 per cent of 1.

    The previous Type-1/Type-2 blend gave exactly 1 here by construction (1 +- 1e-4). The
    library at this inclination has R_n/eta = 1.0073 at V (tau 7, oa 40): the library's own
    scattering and its disc law differ from the cos-law by that amount, and T follows it.
    """
    cos_inc = float(np.cos(np.radians(30.0)))
    wave, _, _ = _skirtor_reference(7.0, 0.5)
    t = float(
        library_torus_transmission(
            "skirtor", jnp.asarray([wave]), cos_inc=cos_inc, params=_SKIRTOR_PARAMS
        )[0]
    )
    assert abs(t - 1.0) <= 1.0e-2


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
@pytest.mark.parametrize("x64", [True, False])
def test_disc_factor_at_90_deg_is_finite_with_a_finite_nonzero_gradient(torus, x64):
    params = (
        _SKIRTOR_PARAMS if torus == "skirtor" else {"agn_fritz_oa": 20.0, "agn_fritz_tau": 1.0}
    )
    wave = jnp.asarray([5500.0, 1.0e4])

    def factor(cos_inc):
        return library_torus_sightline(torus, wave, cos_inc=cos_inc, params=params).disc_factor[0]

    with jax.enable_x64(x64):
        c0 = jnp.asarray(0.0)
        value, grad = jax.value_and_grad(factor)(c0)
    # grad-assert: finite-only — the Fritz factor 2 cos i R_n is exactly zero at i = 90 deg
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0, f"{torus} d(disc factor)/d cos_inc = 0 at 90 deg"


def test_composed_multicolor_disc_gradient_at_90_deg():
    """A 2 cos i disc behind the torus has a finite non-zero cos_inc gradient at i = 90 deg."""
    wave = jnp.asarray([5500.0, 1.0e4, 2.0e4])

    def total(cos_inc):
        return jnp.sum(
            compose_l_nu(
                wave,
                _LOG_LBOL,
                agn_disc_block="multicolor",
                agn_nlr_block="none",
                agn_blr_block="none",
                agn_feii_block="none",
                agn_torus_block="skirtor",
                agn_attenuation_block="none",
                agn_norm="independent",
                agn_torus_frac=0.0,
                agn_cos_inc=cos_inc,
                **_SKIRTOR_PARAMS,
            )
        )

    value, grad = jax.value_and_grad(total)(jnp.asarray(0.0))
    assert np.isfinite(float(value)) and float(value) > 0.0
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


def _composed(disc: str, torus: str, cos_inc: float, wave, **torus_params) -> np.ndarray:
    return np.asarray(
        compose_l_nu(
            wave,
            _LOG_LBOL,
            agn_disc_block=disc,
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block=torus,
            agn_attenuation_block="none",
            agn_norm="independent",
            agn_torus_frac=0.0,
            agn_cos_inc=cos_inc,
            **torus_params,
        )
    )


@pytest.mark.parametrize("i_deg", [60.0, 70.0])
def test_isotropic_disc_follows_t_and_cos_law_disc_follows_cos_t(i_deg):
    """Relative to face-on: schartmann2005 (isotropic) goes as T, multicolor as cos i T."""
    cos_node = _SKIRTOR_COS_NODES[i_deg]
    wave_node, _, expected_t = _skirtor_reference(7.0, cos_node)
    wave = jnp.asarray([wave_node])
    t_got = float(
        _composed("schartmann2005", "skirtor", cos_node, wave, **_SKIRTOR_PARAMS)[0]
        / _composed("schartmann2005", "skirtor", 1.0, wave, **_SKIRTOR_PARAMS)[0]
    )
    law_got = float(
        _composed("multicolor", "skirtor", cos_node, wave, **_SKIRTOR_PARAMS)[0]
        / _composed("multicolor", "skirtor", 1.0, wave, **_SKIRTOR_PARAMS)[0]
    )
    assert abs(t_got - expected_t) <= _TOL
    assert abs(law_got - cos_node * expected_t) <= _TOL


def test_cos_law_disc_set_matches_the_blocks_response_to_inclination():
    """A disc is in ``_COS_LAW_DISC_BLOCKS`` iff its spectrum doubles from cos 0.5 to 1."""
    wave = jnp.asarray([2500.0, 5500.0])
    carries_law = set()
    for name, block in AGN_BLOCKS["disc"].items():
        if name == "none":
            continue
        try:
            half = np.asarray(block(wave, agn_log_lbol=11.0, agn_cos_inc=0.5))
            full = np.asarray(block(wave, agn_log_lbol=11.0, agn_cos_inc=1.0))
        except TypeError:
            continue
        if np.all(half > 0.0) and np.allclose(full / half, 2.0, rtol=1e-2):
            carries_law.add(name)
    assert carries_law == set(_COS_LAW_DISC_BLOCKS)


# ----------------------------------------------------------------------------------
# X-rays: Morrison & McCammon (1983) photoelectric absorption, N_H from the library's A_V
# ----------------------------------------------------------------------------------
# Table 2 of Morrison & McCammon (1983, ApJ 270, 119) as published in two independent public
# implementations, which agree on all 42 coefficients and 15 edges: SOXS
# (lynx-x-ray-observatory/soxs, soxs/spectra/foreground_absorption.py, ``wabs_cross_section``)
# and cgm_toolkit (ethlau/cgm_toolkit, cgm_toolkit/xray_emissivity.py, ``wabs``). A third
# transcription (jracusin/Judy-code, idl.lib/grbs/wabs.pro) prints c2 = -476.0 for the
# 0.1 to 0.284 keV segment, against -476.1 in both of the above and in this repository.
_MM83_EDGES = np.array(
    [
        0.03,
        0.1,
        0.284,
        0.4,
        0.532,
        0.707,
        0.867,
        1.303,
        1.84,
        2.471,
        3.21,
        4.038,
        7.111,
        8.331,
        10.0,
    ]
)
_MM83_C0 = np.array(
    [17.3, 34.6, 78.1, 71.4, 95.5, 308.9, 120.6, 141.3, 202.7, 342.7, 352.2, 433.9, 629.0, 701.2]
)
_MM83_C1 = np.array(
    [608.1, 267.9, 18.8, 66.8, 145.8, -380.6, 169.3, 146.8, 104.7, 18.7, 18.7, -2.4, 30.9, 25.2]
)
_MM83_C2 = np.array(
    [-2150.0, -476.1, 4.3, -51.4, -61.1, 294.0, -47.7, -31.5, -17.0, 0.0, 0.0, 0.75, 0.0, 0.0]
)
_HC_KEV_AA = 12.39841984
_NH_PER_AV = 2.21e21  # [cm^-2 mag^-1] Guver & Ozel 2009
_XRAY_WAVE = np.array([_HC_KEV_AA / 2.0, _HC_KEV_AA / 5.0, _HC_KEV_AA / 9.0])  # 2, 5, 9 keV


def _sigma_mm83(energy_kev):
    """Cross-section [cm^2] from the table above; the last segment extends beyond 10 keV."""
    e = np.asarray(energy_kev, dtype=float)
    k = np.clip(np.searchsorted(_MM83_EDGES, e, side="right") - 1, 0, 13)
    return (_MM83_C0[k] + _MM83_C1[k] * e + _MM83_C2[k] * e**2) / e**3 * 1e-24


def mm83_below_edge_reference(wave_aa, edge_aa, t_v, library_value):
    """``library_value`` above ``edge_aa``; below it exp(-sigma N_H), N_H = 2.21e21 A_V cm^-2.

    ``A_V = -2.5 log10 t_v`` floored at zero; ``sigma`` is the Morrison & McCammon (1983)
    table of this file.
    """
    w = np.asarray(wave_aa, float)
    below = w < edge_aa
    a_v = max(-2.5 * np.log10(max(t_v, 1e-30)), 0.0)
    t_x = np.exp(-_sigma_mm83(_HC_KEV_AA / np.where(below, w, 1.0)) * _NH_PER_AV * a_v)
    return np.where(below, t_x, library_value)


def test_mm83_table_matches_the_two_public_implementations():
    from tengri.components.xray.xray import wabs_cross_section

    mid = 0.5 * (_MM83_EDGES[:-1] + _MM83_EDGES[1:])
    energies = np.concatenate([mid, _MM83_EDGES[1:-1] * 1.0001, [12.0, 30.0]])
    got = np.asarray(wabs_cross_section(jnp.asarray(energies)))
    np.testing.assert_allclose(got, _sigma_mm83(energies), rtol=1e-12)


def _type2_sightline(wave, i_deg=75.0, torus="skirtor"):
    params = _SKIRTOR_PARAMS if torus == "skirtor" else {"agn_fritz_oa": 20.0}
    return library_torus_sightline(
        torus, jnp.asarray(wave), cos_inc=float(np.cos(np.radians(i_deg))), params=params
    )


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
def test_type2_xray_transmission_is_photoelectric_absorption_of_the_library_column(torus):
    """exp(-sigma N_H) with N_H = 2.21e21 A_V and A_V = -2.5 log10 T(5500 A), to 1e-6.

    The X-rays are less absorbed than the optical (T_X > T_V) and absorbed (T_X < 1).
    """
    i_deg = 75.0 if torus == "skirtor" else 80.0
    sight = _type2_sightline(_XRAY_WAVE, i_deg, torus)
    t_v = float(_type2_sightline([_V_WAVE], i_deg, torus).transmission[0])
    a_v = -2.5 * np.log10(t_v)
    expected = np.exp(-_sigma_mm83(_HC_KEV_AA / _XRAY_WAVE) * _NH_PER_AV * a_v)
    np.testing.assert_allclose(np.asarray(sight.transmission), expected, rtol=1e-6)
    assert np.all(np.asarray(sight.transmission) < 1.0)
    assert np.all(np.asarray(sight.transmission) > t_v)
    np.testing.assert_allclose(
        np.asarray(sight.disc_factor),
        2.0 * np.cos(np.radians(i_deg)) * np.asarray(sight.transmission),
        rtol=1e-12,
    )


def test_type1_sightline_has_no_xray_column():
    sight = library_torus_sightline(
        "skirtor", jnp.asarray(_XRAY_WAVE), cos_inc=1.0, params=_SKIRTOR_PARAMS
    )
    np.testing.assert_allclose(np.asarray(sight.transmission), 1.0, atol=1e-12)


def test_xray_transmission_gradient_wrt_inclination_is_finite_and_nonzero():
    def t_x(cos_inc):
        return library_torus_sightline(
            "skirtor", jnp.asarray(_XRAY_WAVE), cos_inc=cos_inc, params=_SKIRTOR_PARAMS
        ).transmission[0]

    g = jax.grad(t_x)(jnp.asarray(float(np.cos(np.radians(75.0)))))
    assert np.isfinite(float(g))
    assert float(g) != 0.0

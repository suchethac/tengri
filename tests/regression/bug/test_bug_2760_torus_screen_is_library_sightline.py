# SPDX-License-Identifier: BSD-3-Clause
r"""On the untied path the disc behind a library torus is dimmed by the library's own sightline.

The untied composable path (``agn_norm="independent"``) multiplied the central engine behind a
SKIRTOR or Fritz (2006) torus by an analytic screen :math:`\exp(-\tau_{9.7}\,k(\lambda)/k_V
\, w(i))`. Three things were wrong with it. The 9.7 um depth was used as :math:`\tau_V`; the
same depth applied at every Type-2 inclination; and the screen carried no scattered light.
Against the libraries' own line-of-sight disc it was off by two orders of magnitude either way.

The transmission is now the library's own Type-2 disc over its face-on disc,
:math:`T_\nu(i) = {\rm disk}_{\rm lib}(\lambda; i)/{\rm disk}_{\rm lib}(\lambda; 0)`, which
includes the library's scattered light. Three statements are checked here, each on the
composed runner's disc alone (no emission lines, no attenuation block, no torus debit):

1. **The disc transmission is the library's line-of-sight ratio** at 5500 A for the SKIRTOR
   (oa 40 deg) and Fritz (half-opening 20 deg) grid nodes of the issue's tables, to 1e-3.
2. **The X-ray band is at most the library's shortest-wavelength ratio.** The grid starts at
   10 A; 2 to 10 keV lies at 1.24 to 6.2 A and takes the edge value, with no photoelectric
   model behind it.
3. **Type-1 sightlines are unchanged**: inside the dust-free polar cone the disc is unscreened.

The library ratios are read from the disc column of the shipped grids, not from the screen.

References
----------
.. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288 (SKIRTOR).
.. [2] Fritz, J., Franceschini, A. & Hatziminaoglou, E. 2006, MNRAS, 366, 767.
.. [3] Boquien, M. et al. 2019, A&A, 622, A103 (CIGALE ``skirtor2016``, ``fritz2006``).
"""

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.components.agn.blocks.torus_library_transmission import library_torus_transmission
from tengri.components.agn.skirtor import (
    load_skirtor_disc_atten_grid,
    skirtor_disc_attenuation_from_grid,
)
from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.regression_bug

_FRITZ_FILE = DATA_DIR / "fritz2006_torus_grid.h5"
_LOG_LBOL = 11.0
_TOL = 1.0e-3
_V_WAVE = 5500.0  # [A] SKIRTOR comparison wavelength
_FRITZ_NODE_WAVE = 5623.0  # [A] the Fritz library's own tabulated node near V
_WAVE = jnp.asarray(
    np.union1d(np.geomspace(1.0, 1.0e8, 40000), [_V_WAVE, _FRITZ_NODE_WAVE])
)  # [A] covers the X-ray band; both comparison wavelengths are exact samples
_XRAY_WAVE = np.array([6.2, 3.0, 1.24])  # [A] 2, ~4 and 10 keV
_SKIRTOR_SITES = [(tau, i) for tau in (3.0, 7.0, 11.0) for i in (60.0, 75.0, 90.0)]
_FRITZ_SITES = [(tau, psi) for tau in (0.1, 1.0, 3.0, 10.0) for psi in (10.1, 30.1, 50.1)]


def _disc_only(torus: str, cos_inc: float, **torus_params) -> np.ndarray:
    """The untied composed disc, with the torus alone on the sightline [erg/s/A]."""
    lam = compose_l_nu(
        _WAVE,
        _LOG_LBOL,
        agn_disc_block="schartmann2005",
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
    return np.asarray(lam)


def _skirtor_library_ratio(tau: float, cos_inc: float, wave) -> np.ndarray:
    """SKIRTOR ``disk(i)/disk(0)`` read from the shipped disc column."""
    grid = load_skirtor_disc_atten_grid()
    return np.asarray(
        skirtor_disc_attenuation_from_grid(
            grid,
            jnp.asarray(wave),
            agn_tau_skirtor=tau,
            agn_oa_skirtor=40.0,
            agn_cos_inc=cos_inc,
        )
    )


def _fritz_library_ratio(tau: float, psi: float) -> float:
    """Fritz ``disk(psi)/disk(89.99 deg)`` at its tabulated node, read directly."""
    with h5py.File(_FRITZ_FILE, "r") as f:
        g = f["fritz2006"]
        axes = {k: np.asarray(g[k + "_axis"][:]) for k in ("tau", "opening_angle", "psy")}
        wave = np.asarray(g["wavelength_aa"][:])
        disk = g["disk"]
        i_tau = int(np.argmin(np.abs(axes["tau"] - tau)))
        i_oa = int(np.argmin(np.abs(axes["opening_angle"] - 20.0)))
        i_psy = int(np.argmin(np.abs(axes["psy"] - psi)))
        i_face = len(axes["psy"]) - 1
        i_wave = int(np.argmin(np.abs(wave - _FRITZ_NODE_WAVE)))
        # (r_ratio, tau, beta, gamma, oa, psy, wave): r = 60, beta = -0.5, gamma = 4
        i_r = int(np.argmin(np.abs(np.asarray(g["r_ratio_axis"][:]) - 60.0)))
        i_beta = int(np.argmin(np.abs(np.asarray(g["beta_axis"][:]) + 0.5)))
        i_gamma = int(np.argmin(np.abs(np.asarray(g["gamma_axis"][:]) - 4.0)))
        row = disk[i_r, i_tau, i_beta, i_gamma, i_oa, :, i_wave]
        return float(row[i_psy] / row[i_face])


def _disc_transmission_at(torus: str, cos_inc: float, wave_aa: float, **torus_params) -> float:
    """The untied disc transmission at one wavelength: composed disc over its face-on value."""
    face = _disc_only(torus, 1.0, **torus_params)
    seen = _disc_only(torus, cos_inc, **torus_params)
    i_w = int(np.argmin(np.abs(np.asarray(_WAVE) - wave_aa)))
    return float(seen[i_w] / face[i_w])


@pytest.mark.parametrize(("tau", "i_deg"), _SKIRTOR_SITES)
def test_skirtor_disc_transmission_is_library_line_of_sight(tau, i_deg):
    cos_inc = float(np.cos(np.radians(i_deg)))
    got = _disc_transmission_at(
        "skirtor",
        cos_inc,
        _V_WAVE,
        agn_tau_skirtor=tau,
        agn_oa_skirtor=40.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_radius_ratio=20.0,
    )
    lib = float(_skirtor_library_ratio(tau, cos_inc, [_V_WAVE])[0])
    assert abs(got - lib) <= _TOL, f"SKIRTOR tau={tau} i={i_deg}: {got:.5f} vs library {lib:.5f}"


@pytest.mark.parametrize(("tau", "psi_deg"), _FRITZ_SITES)
def test_fritz_disc_transmission_is_library_line_of_sight(tau, psi_deg):
    cos_inc = float(np.sin(np.radians(psi_deg)))
    got = _disc_transmission_at(
        "fritz",
        cos_inc,
        _FRITZ_NODE_WAVE,
        agn_fritz_tau=tau,
        agn_fritz_oa=20.0,
        agn_fritz_r_ratio=60.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
    )
    lib = _fritz_library_ratio(tau, psi_deg)
    assert abs(got - lib) <= _TOL, f"Fritz tau={tau} psi={psi_deg}: {got:.5f} vs library {lib:.5f}"


def test_skirtor_xray_band_is_at_most_library_shortest_wavelength_ratio():
    """2 to 10 keV on a Type-2 sightline transmits no more than the library's edge ratio.

    The disc has no emission at X-ray wavelengths, so the transmission is read from the
    function the runner multiplies the disc by, not from a disc ratio.
    """
    tau, cos_inc = 7.0, float(np.cos(np.radians(75.0)))
    edge = float(_skirtor_library_ratio(tau, cos_inc, [10.0])[0])
    t_x = np.asarray(
        library_torus_transmission(
            "skirtor",
            jnp.asarray(_XRAY_WAVE),
            cos_inc=cos_inc,
            params={"agn_tau_skirtor": tau, "agn_oa_skirtor": 40.0},
        )
    )
    assert np.all(np.isfinite(t_x))
    # The 1e-5 is the Type-1 sigmoid tail at 75 deg (~2e-7 of the weight), not slack.
    assert np.all(t_x <= edge * (1.0 + 1.0e-5)), f"X-ray {t_x} exceeds the edge ratio {edge}"


def test_skirtor_type1_sightline_is_unscreened():
    """Inside the polar cone (i = 30 deg < 90 - 40 deg) the disc transmission is unity."""
    cos_inc = float(np.cos(np.radians(30.0)))
    got = _disc_transmission_at(
        "skirtor", cos_inc, _V_WAVE, agn_tau_skirtor=7.0, agn_oa_skirtor=40.0
    )
    assert abs(got - 1.0) <= 1.0e-4

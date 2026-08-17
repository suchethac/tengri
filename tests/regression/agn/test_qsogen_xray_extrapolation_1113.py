# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: the QSOgen disc must not extrapolate into the X-ray band (#1113).

QSOgen (Temple, Hewett & Banerji 2021 [1]_) is an *empirical* quasar template
defined over 912-100000 Angstrom. Its continuum is an analytic broken power law
(:func:`_broken_powerlaw_continuum`) with no short-wavelength floor, so evaluated
below the Lyman limit it extrapolates the EUV segment indefinitely toward
lambda -> 0, depositing spurious flux across the entire X-ray band
(nu*L_nu ~ nu^-0.349). Measured before the fix: nu*L_nu(2 keV) was ~18% of the
1450 Angstrom optical continuum.

QSOgen carries no X-ray physics. The physical X-ray source in a tengri AGN is
the alpha_ox corona (:func:`tengri.xray.xray_agn_corona_from_disc`), added as a
separate component that emits over 0.1-10 keV (``wavelength < 124.0`` Angstrom;
see ``components/xray/xray.py``). Letting the QSOgen continuum bleed into that
band double-counts with the corona: the capstone AGN SED sat ~2x above
AGNfitter-rX in the hard X-ray despite the *bare* corona matching to ~1%.

The fix floors the QSOgen continuum to zero for ``wavelength < 124.0`` Angstrom
(the corona's blue edge, ``0.1 keV``), giving a clean disc/corona partition:
the disc owns lambda >= 124 A (optical/UV/EUV), the corona owns lambda < 124 A.
The floor is applied before the bolometric normalization, so the (previously
spurious) X-ray flux no longer dilutes the L_bol budget of the UV-optical.

The sibling empirical/grid discs (``richards2006_disc``, ``slone_netzer``) were
never affected: both zero outside their template range via
``jnp.interp(..., left=0.0, right=0.0)``.

References
----------
.. [1] M. J. Temple, P. C. Hewett & M. Banerji, MNRAS, 508, 737 (2021).
   doi:10.1093/mnras/stab2586
.. [2] A. T. Steffen et al., AJ, 131, 2826 (2006) — alpha_ox definition;
   the corona band is 0.1-10 keV.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.components.agn.blocks.qsogen_blocks import qsogen_continuum_block
from tengri.components.agn.qsogen import compute_qsogen_sed
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

# Corona blue edge (0.1 keV). Matches ``in_band = wavelength < 124.0`` in
# components/xray/xray.py; the disc must carry nothing below it.
_CORONA_EDGE_AA = 124.0
# 2 keV = hc / (2 keV) in Angstrom — where alpha_ox anchors L_2keV and the
# original double-count was measured.
_TWO_KEV_AA = 6.199

# Optical -> hard X-ray grid (0.5 A ~ 24.8 keV to 3 um). geomspace never lands
# exactly on the reference wavelengths, so probe values are interpolated.
_WAVE = np.geomspace(0.5, 3.0e4, 4000)
_NU = _C_AA_PER_S / _WAVE


def _nulnu(**kw) -> np.ndarray:
    return _NU * np.asarray(compute_qsogen_sed(jnp.asarray(_WAVE), **kw))


def _at(nulnu: np.ndarray, lam_aa: float) -> float:
    return float(np.interp(lam_aa, _WAVE, nulnu))


def test_qsogen_negligible_at_2_kev():
    """nu*L_nu(2 keV) must be negligible vs the 1450 A optical continuum.

    Before the fix this ratio was 0.18 (the double-count with the alpha_ox
    corona); after flooring the continuum it is ~0.
    """
    nl = _nulnu(agn_log_lbol=12.0)
    ratio = _at(nl, _TWO_KEV_AA) / _at(nl, 1450.0)
    assert ratio < 1e-3, (
        f"QSOgen nu*L_nu(2 keV)/nu*L_nu(1450 A) = {ratio:.4f} — the disc is "
        "extrapolating into the corona's band (#1113 double-count)."
    )


def test_qsogen_continuum_zero_below_corona_edge():
    """The disc *continuum* (the extrapolating component, and the disc block
    used by the composable AGN) is exactly zero across the corona band.

    Tested via the public ``qsogen_continuum_block`` so both the monolithic and
    composable code paths are covered. Emission-line Gaussian tails leak a
    physically meaningless ~1e-236 at these wavelengths, so the *total* SED is
    checked for negligibility (not bit-zero) in
    ``test_qsogen_total_negligible_below_corona_edge``.
    """
    cont = np.asarray(qsogen_continuum_block(jnp.asarray(_WAVE), 12.0))
    below = _WAVE < _CORONA_EDGE_AA
    assert below.sum() > 20, "grid must sample the sub-124 A band"
    assert np.all(cont[below] == 0.0), (
        f"QSOgen continuum has {(cont[below] != 0).sum()} non-zero samples below "
        f"{_CORONA_EDGE_AA} A — it must not overlap the alpha_ox corona."
    )


def test_qsogen_total_negligible_below_corona_edge():
    """The full disc SED carries negligible nu*L_nu across the corona band —
    every sub-124 A sample is >=10 orders of magnitude below the 1450 A
    optical continuum (before the fix, 2 keV alone was 18% of it)."""
    nl = _nulnu(agn_log_lbol=12.0)
    below = _WAVE < _CORONA_EDGE_AA
    worst = float(np.max(nl[below])) / _at(nl, 1450.0)
    assert worst < 1e-10, (
        f"QSOgen total nu*L_nu below {_CORONA_EDGE_AA} A peaks at {worst:.2e} x the "
        "1450 A continuum — still bleeding into the corona's band (#1113)."
    )


def test_qsogen_preserves_uv_optical_euv():
    """The floor removes only the X-ray: the disc still emits a normal quasar
    SED down to the corona edge, peaking in the UV/optical."""
    lnu = np.asarray(compute_qsogen_sed(jnp.asarray(_WAVE), agn_log_lbol=12.0))
    keep = (_WAVE >= _CORONA_EDGE_AA) & (_WAVE <= 3.0e4)
    assert np.all(np.isfinite(lnu[keep])) and np.any(lnu[keep] > 0.0)
    # EUV just above the edge is retained (the disc's ionizing continuum).
    assert _at(_NU * lnu, 300.0) > 0.0, "EUV (300 A) must survive the floor"
    # Peak of nu*L_nu sits in the UV/optical big blue bump, not the far-IR.
    nl = _NU * lnu
    peak_aa = _WAVE[keep][int(np.argmax(nl[keep]))]
    assert 800.0 < peak_aa < 8000.0, f"nu*L_nu peaks at {peak_aa:.0f} A (want UV/optical)"


def test_qsogen_transition_is_exactly_at_corona_edge():
    """The floor sits precisely at 124 A: zero just below, positive just above.

    Guards against a floor set too aggressively (eating UV) or too weakly
    (leaving hard-X-ray flux)."""
    wave = jnp.asarray([120.0, 130.0])
    cont = np.asarray(qsogen_continuum_block(wave, 12.0))
    assert cont[0] == 0.0, "120 A (inside corona band) must be zero"
    assert cont[1] > 0.0, "130 A (disc EUV) must be positive"


def test_qsogen_uv_optical_grid_unaffected_and_grad_safe():
    """On a UV-optical grid (>=1000 A) the floor is a no-op, and the SED stays
    JIT/grad-safe w.r.t. agn_log_lbol."""
    wave = jnp.geomspace(1000.0, 5.0e4, 500)

    def total(log_lbol):
        return jnp.sum(compute_qsogen_sed(wave, agn_log_lbol=log_lbol))

    val = float(total(12.0))
    grad = float(jax.grad(total)(12.0))
    assert np.isfinite(val) and val > 0.0
    assert np.isfinite(grad) and grad > 0.0, "10x L_bol must brighten the disc"

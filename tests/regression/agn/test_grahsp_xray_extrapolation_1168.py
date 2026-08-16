# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: the GRAHSP disc must not extrapolate into the X-ray band (#1168).

Sibling of #1113 (the QSOgen fix). GRAHSP (Buchner et al. 2024 [1]_) models a
quasar as BBB + emission lines + FeII + Balmer continuum + torus — it has *no*
X-ray physics. Its big-blue-bump :func:`sbpl_bbb` is a smooth bending power law
with only an optional IR cutoff, so on a grid reaching short wavelengths it
extrapolates unbounded into the X-ray. Measured before the fix: nu*L_nu at
2 keV was ~600x the optical continuum — worse than QSOgen.

The physical X-ray source in a tengri AGN is the separately-added alpha_ox
corona (0.1-10 keV, ``wavelength < 124.0`` Angstrom; see
``components/xray/xray.py``). Letting the GRAHSP disc bleed into that band
double-counts with the corona.

The fix floors the *assembled* GRAHSP disc to zero for ``wavelength < 124.0``
Angstrom (= 12.4 nm, the corona's blue edge) at every tengri emission surface
(the composable ``grahsp_sbpl_disc_block``, the monolithic ``compute_grahsp_sed``,
and the ``GRAHSPSEDComponent``). The low-level :func:`sbpl_bbb` primitive is
left untouched: it is validated bit-exactly against upstream ``activatepl.sbpl``
(``test_bbb.py``), so the floor lives in tengri's assembly layer, not in the
shared math. GRAHSP's bolometric normalization is unaffected — it already
integrates only above the Lyman limit (91.2 nm), well above the floor.

References
----------
.. [1] J. Buchner et al., arXiv:2405.19297 (2024).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.components.agn.blocks.grahsp_blocks import grahsp_sbpl_disc_block
from tengri.components.agn.grahsp.bbb import sbpl_bbb
from tengri.components.agn.grahsp.model import compute_grahsp_sed

# Corona blue edge: 124 A = 0.1 keV (matches ``wavelength < 124.0`` in
# components/xray/xray.py); 12.4 nm on the GRAHSP nm grid.
_CORONA_EDGE_AA = 124.0
_CORONA_EDGE_NM = 12.4

# Optical -> hard X-ray grid [Angstrom].
_WAVE_AA = np.geomspace(1.0, 3.0e4, 4000)


def _at(arr, lam_aa):
    return float(np.interp(lam_aa, _WAVE_AA, arr))


def test_grahsp_block_zero_below_corona_edge():
    """The composable disc block (combined with the corona) emits exactly
    nothing across the corona band."""
    lam = np.asarray(grahsp_sbpl_disc_block(jnp.asarray(_WAVE_AA), 45.0))
    below = _WAVE_AA < _CORONA_EDGE_AA
    assert below.sum() > 20, "grid must sample the sub-124 A band"
    assert np.all(lam[below] == 0.0), (
        f"GRAHSP disc block has {(lam[below] != 0).sum()} non-zero samples below "
        f"{_CORONA_EDGE_AA} A — it must not overlap the alpha_ox corona (#1168)."
    )


def test_grahsp_model_negligible_below_corona_edge():
    """The full GRAHSP SED carries negligible flux across the corona band.

    Before the fix, lambda*L_lambda below 124 A dwarfed the optical (the disc
    rose steeply toward short wavelengths); after flooring it is ~0.
    """
    sed = np.asarray(compute_grahsp_sed(jnp.asarray(_WAVE_AA), agn_log_lbol=45.0))
    lam_llam = _WAVE_AA * sed  # proportional to nu*L_nu
    below = _WAVE_AA < _CORONA_EDGE_AA
    worst = float(np.max(lam_llam[below])) / _at(lam_llam, 5100.0)
    assert worst < 1e-6, (
        f"GRAHSP total lambda*L_lambda below {_CORONA_EDGE_AA} A peaks at {worst:.2e}x "
        "the 5100 A continuum — still bleeding into the corona's band (#1168)."
    )


def test_grahsp_preserves_uv_optical():
    """The floor removes only the X-ray: the disc still emits normally down to
    the corona edge and peaks in the UV/optical."""
    lam = np.asarray(grahsp_sbpl_disc_block(jnp.asarray(_WAVE_AA), 45.0))
    keep = (_WAVE_AA >= _CORONA_EDGE_AA) & (_WAVE_AA <= 3.0e4)
    assert np.all(np.isfinite(lam[keep])) and np.any(lam[keep] > 0.0)
    # EUV just above the edge survives (the disc's ionizing continuum).
    assert _at(lam, 300.0) > 0.0, "EUV (300 A) must survive the floor"


def test_grahsp_transition_exactly_at_corona_edge():
    """The block floor sits at 124 A: zero just below, positive just above.

    Passes an explicit ``agn_grahsp_l5100`` so the block skips its auto
    L_bol normalization (which integrates only >=91.2 nm and would divide by
    zero on this sub-Lyman probe grid)."""
    lam = np.asarray(
        grahsp_sbpl_disc_block(jnp.asarray([120.0, 130.0]), 45.0, agn_grahsp_l5100=1.0e44)
    )
    assert lam[0] == 0.0, "120 A (inside corona band) must be zero"
    assert lam[1] > 0.0, "130 A (disc EUV) must be positive"


def test_grahsp_primitive_unchanged_parity_preserved():
    """The upstream-faithful ``sbpl_bbb`` primitive is NOT floored — it must
    still emit below the corona edge so its bit-exact parity with
    ``activatepl.sbpl`` (test_bbb.py) is preserved. The floor lives only in
    tengri's assembly layer."""
    below = sbpl_bbb(
        wave_nm=jnp.array([10.0, 11.0]),  # 100, 110 A — inside the corona band
        l5100=1.0e36,
        uvslope=0.0,
        plslope=-1.7,
        plbendloc_nm=100.0,
        plbendwidth=1.0,
        cutoff_nm=-1.0,
    )
    assert np.all(np.asarray(below) > 0.0), (
        "sbpl_bbb must stay upstream-faithful (nonzero below 124 A); the floor "
        "belongs in the assembly layer, not the parity-tested primitive."
    )


def test_grahsp_block_grad_safe():
    """The floored block stays JIT/grad-safe w.r.t. agn_log_lbol."""
    wave = jnp.geomspace(1000.0, 5.0e4, 400)

    def total(log_lbol):
        return jnp.sum(grahsp_sbpl_disc_block(wave, log_lbol))

    val = float(total(45.0))
    grad = float(jax.grad(total)(45.0))
    assert np.isfinite(val) and val > 0.0
    assert np.isfinite(grad) and grad > 0.0, "10x L_bol must brighten the disc"

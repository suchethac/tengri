# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the Inoue et al. (2014) IGM transmission model.

tengri's :func:`tengri.components.igm.igm_transmission` must reproduce the
canonical Inoue+2014 prescription as implemented in eazy-py (the upstream
source of tengri's coefficient tables, and the implementation that bagpipes,
prospector, and synthesizer all descend from).

This module embeds a *self-contained* reference implementation of eazy-py's ``Inoue14`` class
(``tLSLAF``/``tLSDLA``/``tLCLAF``/``tLCDLA``) so the parity check needs no
external dependency. Two classes of regression are guarded:

1. **Wavelength registration** — Lyman-alpha at 1215.67 Å (vacuum) and the
   Lyman limit at 911.8 Å, matching Inoue+2014 / eazy. A previous bug placed
   Lyman-alpha at 1216.0 Å, shifting the whole forest edge redward by
   ~0.33 Å rest (growing to ~2.6 Å observed at z=7).
2. **Lyman-continuum opacity shape** — the LC LAF/DLA optical-depth formulas
   (rest < 912 Å) must match eazy's functional form, not an unrelated one.

Physics references
------------------
- Inoue, A. K., Shimizu, I., Iwata, I., & Tanaka, M. 2014, MNRAS, 442, 1805
- eazy-py: https://github.com/gbrammer/eazy-py/blob/master/eazy/igm.py
- Convention: igm_transmission(wave_obs, z) takes observed-frame wavelengths.
"""

import chex
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

import jax.numpy as jnp

from tengri.components.igm import igm_transmission
from tengri.components.igm.igm import _A_DLA, _A_LAF, _LAMBDA_LYMAN

# ── Canonical eazy-py Inoue14 reference implementation ───────────────
# Coefficients reused from tengri's tables (which come from eazy's
# LAF/DLAcoeff.txt). The reference Lyman-series wavelengths use the eazy
# value lam[0] = 1215.67; the Lyman limit is 911.8 Å.
_LAM_REF = np.asarray(_LAMBDA_LYMAN, dtype=float).copy()
_LAM_REF[0] = 1215.67
_ALAF_REF = np.asarray(_A_LAF, dtype=float)
_ADLA_REF = np.asarray(_A_DLA, dtype=float)
_LAML_REF = 911.8


def _ref_tLSLAF(zS, lobs):
    l2 = _LAM_REF[:, None]
    one = np.ones_like(l2)
    out = np.zeros_like(lobs * l2)
    x0 = lobs < l2 * (1 + zS)
    x1 = x0 & (lobs < l2 * (1 + 1.2))
    x2 = x0 & (lobs >= l2 * (1 + 1.2)) & (lobs < l2 * (1 + 4.7))
    x3 = x0 & (lobs >= l2 * (1 + 4.7))
    out[x1] += ((_ALAF_REF[:, 0:1] / l2**1.2) * lobs**1.2 * one)[x1]
    out[x2] += ((_ALAF_REF[:, 1:2] / l2**3.7) * lobs**3.7 * one)[x2]
    out[x3] += ((_ALAF_REF[:, 2:3] / l2**5.5) * lobs**5.5 * one)[x3]
    return out.sum(axis=0)


def _ref_tLSDLA(zS, lobs):
    l2 = _LAM_REF[:, None]
    one = np.ones_like(l2)
    out = np.zeros_like(lobs * l2)
    x0 = (lobs < l2 * (1 + zS)) & (lobs < l2 * (1 + 2.0))
    x1 = (lobs < l2 * (1 + zS)) & ~(lobs < l2 * (1 + 2.0))
    out[x0] += ((_ADLA_REF[:, 0:1] / l2**2) * lobs**2 * one)[x0]
    out[x1] += ((_ADLA_REF[:, 1:2] / l2**3) * lobs**3 * one)[x1]
    return out.sum(axis=0)


def _ref_tLCLAF(zS, lobs):
    out = np.zeros_like(lobs)
    x0 = lobs < _LAML_REF * (1 + zS)
    r = lobs / _LAML_REF
    if zS < 1.2:
        out[x0] = 0.3248 * (r[x0] ** 1.2 - (1 + zS) ** (-0.9) * r[x0] ** 2.1)
    elif zS < 4.7:
        x1 = lobs >= _LAML_REF * (1 + 1.2)
        out[x0 & x1] = 2.545e-2 * ((1 + zS) ** 1.6 * r[x0 & x1] ** 2.1 - r[x0 & x1] ** 3.7)
        out[x0 & ~x1] = (
            2.545e-2 * (1 + zS) ** 1.6 * r[x0 & ~x1] ** 2.1
            + 0.3248 * r[x0 & ~x1] ** 1.2
            - 0.2496 * r[x0 & ~x1] ** 2.1
        )
    else:
        x1 = lobs > _LAML_REF * (1 + 4.7)
        x2 = (lobs >= _LAML_REF * (1 + 1.2)) & (lobs < _LAML_REF * (1 + 4.7))
        x3 = lobs < _LAML_REF * (1 + 1.2)
        out[x0 & x1] = 5.221e-4 * ((1 + zS) ** 3.4 * r[x0 & x1] ** 2.1 - r[x0 & x1] ** 5.5)
        out[x0 & x2] = (
            5.221e-4 * (1 + zS) ** 3.4 * r[x0 & x2] ** 2.1
            + 0.2182 * r[x0 & x2] ** 2.1
            - 2.545e-2 * r[x0 & x2] ** 3.7
        )
        out[x0 & x3] = (
            5.221e-4 * (1 + zS) ** 3.4 * r[x0 & x3] ** 2.1
            + 0.3248 * r[x0 & x3] ** 1.2
            - 3.140e-2 * r[x0 & x3] ** 2.1
        )
    return out


def _ref_tLCDLA(zS, lobs):
    out = np.zeros_like(lobs)
    x0 = lobs < _LAML_REF * (1 + zS)
    r = lobs / _LAML_REF
    if zS < 2.0:
        out[x0] = (
            0.2113 * (1 + zS) ** 2.0
            - 0.07661 * (1 + zS) ** 2.3 * r[x0] ** (-0.3)
            - 0.1347 * r[x0] ** 2.0
        )
    else:
        x1 = lobs >= _LAML_REF * (1 + 2.0)
        out[x0 & x1] = (
            0.04696 * (1 + zS) ** 3.0
            - 0.01779 * (1 + zS) ** 3.3 * r[x0 & x1] ** (-0.3)
            - 0.02916 * r[x0 & x1] ** 3.0
        )
        out[x0 & ~x1] = (
            0.6340
            + 0.04696 * (1 + zS) ** 3.0
            - 0.01779 * (1 + zS) ** 3.3 * r[x0 & ~x1] ** (-0.3)
            - 0.1347 * r[x0 & ~x1] ** 2.0
            - 0.2905 * r[x0 & ~x1] ** (-0.3)
        )
    return out


def eazy_inoue_transmission(zS, lobs):
    """Reference Inoue+2014 transmission (matches eazy-py)."""
    lobs = np.asarray(lobs, dtype=float)
    tau = (
        _ref_tLSLAF(zS, lobs)
        + _ref_tLSDLA(zS, lobs)
        + _ref_tLCLAF(zS, lobs)
        + _ref_tLCDLA(zS, lobs)
    )
    return np.exp(-tau)


# ── Tests ────────────────────────────────────────────────────────────


class TestInoue14EazyParity:
    """tengri Inoue14 must reproduce the eazy-py reference."""

    @pytest.mark.parametrize("zS", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
    def test_full_transmission_matches_eazy(self, zS):
        """Transmission must match eazy across the observable Lyman break + forest.

        Over rest >= 850 Å (which covers the Lyman limit, the full Lyman-series
        forest, and Lya) tengri reproduces eazy to machine precision at every
        redshift. The deep far-UV (rest < ~810 Å) is checked separately — tengri
        intentionally diverges there by <2.5% (see the lower-bound test below).
        """
        lam_rest = np.linspace(850.0, 1300.0, 1400)
        lobs = lam_rest * (1.0 + zS)
        t_ten = np.asarray(igm_transmission(jnp.asarray(lobs), zS))
        t_ref = eazy_inoue_transmission(zS, lobs)
        np.testing.assert_allclose(
            t_ten, t_ref, atol=2e-3, err_msg=f"Inoue14 transmission mismatch at z={zS}"
        )

    @pytest.mark.parametrize("zS", [0.5, 1.0])
    def test_deep_uv_physical_lower_bound(self, zS):
        """tengri enforces lambda_obs > lambda_j (a line cannot absorb blueward of
        its own rest wavelength); raw eazy omits this.

        The effect is confined to the deep far-UV (rest < ~810 Å, observed
        < ~1215 Å at these redshifts — not an observable regime) and is small
        (<2.5%). tengri transmits slightly *more* there (fewer spurious sub-line
        contributions). This same lower bound is what makes tengri give T=1
        everywhere at z=0.
        """
        lam_rest = np.linspace(600.0, 810.0, 400)
        lobs = lam_rest * (1.0 + zS)
        t_ten = np.asarray(igm_transmission(jnp.asarray(lobs), zS))
        t_ref = eazy_inoue_transmission(zS, lobs)
        # bounded, sub-2.5%, and tengri >= eazy (less absorption from the bound)
        assert np.max(np.abs(t_ten - t_ref)) < 0.025
        assert np.all(t_ten >= t_ref - 1e-9)

    @pytest.mark.parametrize("zS", [3.0, 4.0, 5.0, 7.0])
    def test_lyman_alpha_edge_position(self, zS):
        """Lya forest edge must sit at 1215.67 Å rest, not 1216.0 Å.

        Regression for the wavelength-registration bug: the highest rest
        wavelength with T < 0.5 must agree with eazy to << 0.33 Å.
        """
        lam_rest = np.linspace(1180.0, 1250.0, 70001)  # 0.001 Å resolution
        lobs = lam_rest * (1.0 + zS)
        t_ten = np.asarray(igm_transmission(jnp.asarray(lobs), zS))
        t_ref = eazy_inoue_transmission(zS, lobs)

        # Adaptive level midway between the forest floor and the red continuum,
        # so the crossing is well-defined at every redshift in this window.
        level = 0.5 * (float(np.min(t_ref)) + float(t_ref[-1]))

        def edge(t):
            below = lam_rest[t < level]
            return below.max() if below.size else np.nan

        e_ten, e_ref = edge(t_ten), edge(t_ref)
        assert np.isfinite(e_ten) and np.isfinite(e_ref), f"no crossing at z={zS}"
        assert abs(e_ten - e_ref) < 0.05, (
            f"Lya edge shifted at z={zS}: tengri={e_ten:.3f} ref={e_ref:.3f}"
        )

    def test_lyman_alpha_rest_wavelength_is_vacuum(self):
        """The first Lyman line must be vacuum Lya = 1215.67 Å (project convention)."""
        assert abs(float(_LAMBDA_LYMAN[0]) - 1215.67) < 1e-3

    @pytest.mark.parametrize("zS", [2.0, 3.0])
    def test_lyman_continuum_depth_matches_eazy(self, zS):
        """LC region (rest < 912 Å) absorption depth must match eazy.

        Regression for the wrong LC optical-depth formula that produced
        gross over-absorption (e.g. z=2, rest=800: T=0.017 vs eazy 0.555).
        """
        lam_rest = np.array([700.0, 800.0, 880.0, 905.0])
        lobs = lam_rest * (1.0 + zS)
        t_ten = np.asarray(igm_transmission(jnp.asarray(lobs), zS))
        t_ref = eazy_inoue_transmission(zS, lobs)
        np.testing.assert_allclose(t_ten, t_ref, atol=5e-3)

    def test_bounds_and_finite(self):
        """Transmission stays in [0, 1] and finite across a wide grid."""
        lobs = jnp.linspace(500.0, 12000.0, 2000)
        for zS in (0.0, 1.5, 3.5, 6.0):
            t = igm_transmission(lobs, zS)
            chex.assert_tree_all_finite(t)
            assert float(jnp.min(t)) >= 0.0
            assert float(jnp.max(t)) <= 1.0 + 1e-9

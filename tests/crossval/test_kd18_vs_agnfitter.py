# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate the Kubota & Done 2018 disc against AGNfitter-rX's KD18 grid.

The reference SEDs are the AGNfitter-rX ``models/BBB/KD18.pickle`` grid, converted
ONCE at build time to the committed HDF5 ``data/agnfitter_bbb_reference.h5`` by
``scripts/build_agnfitter_bbb_reference.py``. This test reads only that HDF5 — no
pickle at test time — so it runs in CI instead of skipping when the upstream clone
is absent (the data-gated-tests-mask-regressions gap, #613).

What is asserted (issue #592 A1): tengri's :func:`kubota_done_disc` must reproduce
the KD18 accretion-disc *shape* in the disc-dominated optical-UV window
(1000 Å – 1 µm) across the (logBHmass, logEddra) grid. The far-IR / X-ray tails are
NOT compared: the two codes treat the Compton hot-corona and seed-photon rollover
differently, a documented difference, not a parity failure.

Measured agreement (peak-normalized, 1000 Å–1 µm window) after PR #903 (ADR-0020)
reparameterization: tengri ``kubota_done_disc`` now derives Eddington ratio from
``agn_log_lbol`` and ``agn_log_mbh``, so test nodes must compute matched luminosities
via ``log_lbol = log_edd + log10(L_Edd / L_sun)``. Measured worst-case log-ratios:
0.142 dex at (logBHmass, logEddra) = (6.0, -1.5), with disc peaks tracking to
within ~35% (worst case 1.49 at 8.0/-1.5, a known warm-Compton divergence at low
accretion rates). Per-node tolerances set to measured maximum × 1.25, rounded to 0.05.

References
----------
.. [1] Kubota & Done 2018, MNRAS, 480, 1247. doi:10.1093/mnras/sty1890
.. [2] Calistro Rivera et al. 2016, ApJ, 833, 98 (AGNfitter). doi:10.3847/1538-4357/833/1/98
"""

from __future__ import annotations

from pathlib import Path

import chex
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_REF_H5 = Path(__file__).resolve().parents[2] / "data" / "agnfitter_bbb_reference.h5"

if not _REF_H5.is_file():
    pytest.skip(
        f"BBB reference HDF5 not found at {_REF_H5} "
        "(build with: python scripts/build_agnfitter_bbb_reference.py)",
        allow_module_level=True,
    )

# Disc-dominated optical-UV window where KD18 and tengri should agree [Å].
_DISC_LO_AA = 1000.0
_DISC_HI_AA = 1.0e4


@pytest.fixture(scope="module")
def kd18_reference() -> dict:
    """AGNfitter-rX KD18 grid from the vendored HDF5 (wavelength [Å], F_nu)."""
    with h5py.File(_REF_H5, "r") as f:
        g = f["kd18"]
        return {
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "sed": np.asarray(g["sed"][:], dtype=np.float64),
            "logBHmass": np.asarray(g["logBHmass"][:], dtype=np.float64),
            "logEddra": np.asarray(g["logEddra"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def kubota_done():
    from tengri.components.agn.disc import kubota_done_disc

    return kubota_done_disc


def _nearest_node(ref: dict, log_mbh: float, log_edd: float) -> int:
    return int(np.argmin((ref["logBHmass"] - log_mbh) ** 2 + (ref["logEddra"] - log_edd) ** 2))


def _matched_log_lbol(log_mbh: float, log_edd: float) -> float:
    """Compute log_lbol that achieves a target Eddington ratio.

    Since PR #903 (ADR-0020), kubota_done_disc derives lambda_Edd from
    agn_log_lbol and agn_log_mbh: lambda_Edd = L_bol / L_Edd.
    This helper computes the log_lbol needed to match a target log_edd value.

    Parameters
    ----------
    log_mbh : float
        Black hole mass parameter [log10(M_sun)].
    log_edd : float
        Target Eddington ratio [log10(lambda_Edd)].

    Returns
    -------
    float
        log_lbol = log10(L_bol / L_sun) that produces the desired lambda_Edd.
    """
    from tengri.components.agn.disc import _eddington_luminosity
    from tengri.utils.physics_constants import L_SUN

    l_edd_erg = float(_eddington_luminosity(log_mbh))
    # lambda_Edd = L_bol / L_Edd, so L_bol = 10^log_edd * L_Edd
    # Then log_lbol = log10(L_bol / L_sun) = log_edd + log10(L_Edd / L_sun)
    return log_edd + np.log10(l_edd_erg / L_SUN)


def _peak_norm(sed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return sed / np.nanmax(sed[mask])


class TestKD18DiscShape:
    """tengri kubota_done_disc vs AGNfitter KD18 in the disc-dominated window."""

    # Spread across the grid, deliberately including the high-mass nodes the
    # previous (lazy, rtol=2.0) test skipped.
    @pytest.mark.parametrize(
        "log_mbh,log_edd,max_logratio,peak_ratio_min,peak_ratio_max",
        [
            (6.0, -1.5, 0.20, 0.7, 1.43),
            (6.0, 0.0, 0.15, 0.7, 1.43),
            (7.43, -0.96, 0.10, 0.7, 1.43),
            (8.0, -1.5, 0.25, 0.55, 1.65),  # warm-Compton proxy at low Edd; guard only
            # needs to exclude a far-IR peak (#592 A1), so keep margin off the
            # measured 1.49 rather than pinning the bound to it.
            (8.0, -0.96, 0.20, 0.7, 1.43),
            (8.0, 0.0, 0.05, 0.7, 1.43),
        ],
    )
    def test_disc_shape_matches(
        self,
        kd18_reference,
        kubota_done,
        log_mbh,
        log_edd,
        max_logratio,
        peak_ratio_min,
        peak_ratio_max,
    ):
        ref = kd18_reference
        idx = _nearest_node(ref, log_mbh, log_edd)
        bh, edd = ref["logBHmass"][idx], ref["logEddra"][idx]
        wave = ref["wavelength"]

        af = ref["sed"][idx]
        # Since PR #903 (ADR-0020), agn_log_ledd is deprecated and ignored.
        # Compute matched log_lbol to achieve the target Eddington ratio.
        log_lbol = _matched_log_lbol(bh, edd)
        tg = np.asarray(kubota_done(jnp.asarray(wave), agn_log_lbol=log_lbol, agn_log_mbh=bh))

        mask = (wave >= _DISC_LO_AA) & (wave <= _DISC_HI_AA) & np.isfinite(af) & (tg > 0)
        assert mask.sum() > 10, "Too few points in disc window — grid axis problem"

        af_n = _peak_norm(af, mask)
        tg_n = _peak_norm(tg, mask)

        # Shape agreement in the disc window.
        log_ratio = np.abs(np.log10(tg_n[mask] / af_n[mask]))
        worst = float(np.nanmax(log_ratio))
        assert worst < max_logratio, (
            f"KD18 shape divergence {worst:.2f} dex > {max_logratio} dex at "
            f"(logBHmass={bh:.2f}, logEddra={edd:.2f}) in {_DISC_LO_AA:.0f}-{_DISC_HI_AA:.0f} A"
        )

        # Peak wavelength must track (guards against the #592 A1 far-IR regression).
        af_peak = wave[mask][np.nanargmax(af[mask])]
        tg_peak = wave[mask][np.nanargmax(tg[mask])]
        peak_ratio = tg_peak / af_peak
        assert peak_ratio_min < peak_ratio < peak_ratio_max, (
            f"KD18 disc peak {tg_peak:.0f} A vs AGNfitter {af_peak:.0f} A "
            f"(ratio {peak_ratio:.2f}) — outside [{peak_ratio_min:.2f}, {peak_ratio_max:.2f})"
        )

    def test_peak_is_optical_not_far_ir(self, kd18_reference, kubota_done):
        """Direct guard on issue #592 A1: the disc must peak in the UV/optical."""
        ref = kd18_reference
        idx = _nearest_node(ref, 8.0, -1.0)
        bh, edd = ref["logBHmass"][idx], ref["logEddra"][idx]
        wave = ref["wavelength"]
        # Since PR #903 (ADR-0020), agn_log_ledd is deprecated and ignored.
        # Compute matched log_lbol to achieve the target Eddington ratio.
        log_lbol = _matched_log_lbol(bh, edd)
        tg = np.asarray(kubota_done(jnp.asarray(wave), agn_log_lbol=log_lbol, agn_log_mbh=bh))
        finite = np.isfinite(tg) & (tg > 0)
        peak = wave[finite][np.nanargmax(tg[finite])]
        assert 1.0e3 < peak < 2.0e4, (
            f"kubota_done peaks at {peak:.0f} A — a disc must peak in the UV/optical, "
            "not the far-IR (issue #592 A1 regression)."
        )


class TestKD18Runtime:
    """JIT/grad/scaling sanity for the runtime callable."""

    def test_luminosity_scaling(self, kubota_done):
        wave = jnp.logspace(3.0, 4.0, 200)
        a = kubota_done(wave, agn_log_lbol=12.0, agn_log_mbh=8.0, agn_log_ledd=-1.0)
        b = kubota_done(wave, agn_log_lbol=13.0, agn_log_mbh=8.0, agn_log_ledd=-1.0)
        chex.assert_tree_all_finite(a)
        ratio = np.asarray(b) / np.asarray(a)
        assert np.median(ratio) > 1.5, "10x L_bol should brighten the disc"

    def test_gradient_flows(self, kubota_done):
        wave = jnp.logspace(3.0, 4.0, 64)

        def total(log_lbol):
            return jnp.sum(
                kubota_done(wave, agn_log_lbol=log_lbol, agn_log_mbh=8.0, agn_log_ledd=-1.0)
            )

        grad = float(jax.grad(total)(12.0))
        assert np.isfinite(grad) and abs(grad) > 0, f"non-finite/zero gradient {grad}"

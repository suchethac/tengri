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

Measured agreement (peak-normalised, 1000 Å–1 µm window): max|log10(tengri/AGNfitter)|
= 0.08 / 0.19 / 0.31 dex at logBHmass = 6.0 / 7.4 / 8.0 (worst at the coolest,
high-mass/low-Eddington node), with disc peaks tracking to within ~20%. The test
asserts < 0.4 dex and peak agreement within 30% — it PASSES on this genuine
agreement but would FAIL loudly if KD18 ever regressed to the far-IR-peaked SED
reported (and since fixed) in issue #592 A1.

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

jax.config.update("jax_enable_x64", True)

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


def _peak_norm(sed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return sed / np.nanmax(sed[mask])


class TestKD18DiscShape:
    """tengri kubota_done_disc vs AGNfitter KD18 in the disc-dominated window."""

    # Spread across the grid, deliberately including the high-mass nodes the
    # previous (lazy, rtol=2.0) test skipped.
    @pytest.mark.parametrize(
        "log_mbh,log_edd,max_logratio",
        [
            (6.0, -1.5, 0.40),
            (6.0, 0.0, 0.40),
            (7.43, -0.96, 0.40),
            (8.0, -1.5, 0.40),
            (8.0, -0.96, 0.40),
            (8.0, 0.0, 0.40),
        ],
    )
    def test_disc_shape_matches(self, kd18_reference, kubota_done, log_mbh, log_edd, max_logratio):
        ref = kd18_reference
        idx = _nearest_node(ref, log_mbh, log_edd)
        bh, edd = ref["logBHmass"][idx], ref["logEddra"][idx]
        wave = ref["wavelength"]

        af = ref["sed"][idx]
        tg = np.asarray(
            kubota_done(jnp.asarray(wave), agn_log_lbol=12.0, agn_log_mbh=bh, agn_log_ledd=edd)
        )

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
        assert 0.7 < tg_peak / af_peak < 1.43, (
            f"KD18 disc peak {tg_peak:.0f} A vs AGNfitter {af_peak:.0f} A "
            f"(ratio {tg_peak / af_peak:.2f}) — outside +/-30%"
        )

    def test_peak_is_optical_not_far_ir(self, kd18_reference, kubota_done):
        """Direct guard on issue #592 A1: the disc must peak in the UV/optical."""
        ref = kd18_reference
        idx = _nearest_node(ref, 8.0, -1.0)
        bh, edd = ref["logBHmass"][idx], ref["logEddra"][idx]
        wave = ref["wavelength"]
        tg = np.asarray(
            kubota_done(jnp.asarray(wave), agn_log_lbol=12.0, agn_log_mbh=bh, agn_log_ledd=edd)
        )
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

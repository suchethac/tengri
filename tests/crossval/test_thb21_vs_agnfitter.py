# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's QSOgen disc against AGNfitter-rX's THB21 template.

AGNfitter-rX ships the Temple, Hewett & Banerji (2021) empirical quasar SED as
``models/BBB/THB21.pickle`` (a single composite including blended emission lines).
tengri's equivalent is the parametric ``qsogen`` model (``components/agn/qsogen.py``),
a superset that reproduces the same Temple+2021 continuum + line backbone.

The reference SED is read from the committed HDF5 ``data/agnfitter_bbb_reference.h5``
(converted once from THB21.pickle by ``scripts/build_agnfitter_bbb_reference.py``) —
no pickle at test time, so this runs in CI rather than skipping (#613).

Measured agreement (UV–optical–NIR, 1200 Å–1.5 µm, peak-normalized): the QSOgen and
THB21 peaks coincide at the Hα bump (6539 Å) and max|log10(qsogen/THB21)| = 0.31 dex
(median 0.03 dex). The test asserts peak coincidence and < 0.5 dex shape agreement.

References
----------
.. [1] Temple, Hewett & Banerji 2021, MNRAS, 508, 737. doi:10.1093/mnras/stab2586
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

# UV-optical-NIR window where the disc continuum + line backbone dominate [Å].
_LO_AA, _HI_AA = 1200.0, 1.5e4


@pytest.fixture(scope="module")
def thb21_reference() -> tuple[np.ndarray, np.ndarray]:
    """AGNfitter THB21 SED from the vendored HDF5: (wavelength [Å], F_nu)."""
    with h5py.File(_REF_H5, "r") as f:
        g = f["thb21"]
        wave = np.asarray(g["wavelength"][:], dtype=np.float64)
        sed = np.asarray(g["sed"][:], dtype=np.float64)[0]
    return wave, sed


@pytest.fixture(scope="module")
def qsogen():
    from tengri.components.agn.qsogen import qsogen as _qsogen

    return _qsogen


class TestTHB21:
    def test_peak_and_shape_match(self, thb21_reference, qsogen):
        """QSOgen reproduces the THB21 SED shape across the UV-optical-NIR window."""
        wave, af = thb21_reference
        tg = np.asarray(qsogen(jnp.asarray(wave), agn_log_lbol=12.0))

        mask = (wave >= _LO_AA) & (wave <= _HI_AA) & np.isfinite(af) & (tg > 0)
        af_n = af / np.nanmax(af[mask])
        tg_n = tg / np.nanmax(tg[mask])

        af_peak = wave[mask][np.nanargmax(af[mask])]
        tg_peak = wave[mask][np.nanargmax(tg[mask])]
        assert 0.95 < tg_peak / af_peak < 1.05, (
            f"QSOgen peak {tg_peak:.0f} A vs THB21 {af_peak:.0f} A (ratio {tg_peak / af_peak:.3f})"
        )

        worst = float(np.nanmax(np.abs(np.log10(tg_n[mask] / af_n[mask]))))
        assert worst < 0.5, (
            f"QSOgen vs THB21 shape divergence {worst:.2f} dex > 0.5 dex in "
            f"{_LO_AA:.0f}-{_HI_AA:.0f} A"
        )

    def test_runtime_luminosity_scaling(self, qsogen):
        wave = jnp.logspace(3.1, 4.0, 200)
        a = qsogen(wave, agn_log_lbol=12.0)
        b = qsogen(wave, agn_log_lbol=13.0)
        chex.assert_tree_all_finite(a)
        assert np.median(np.asarray(b) / np.asarray(a)) > 1.5

    def test_gradient_flows(self, qsogen):
        wave = jnp.logspace(3.1, 4.0, 64)
        grad = float(jax.grad(lambda x: jnp.sum(qsogen(wave, agn_log_lbol=x)))(12.0))
        assert np.isfinite(grad) and abs(grad) > 0

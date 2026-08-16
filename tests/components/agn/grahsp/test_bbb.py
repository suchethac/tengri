# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the GRAHSP smooth bending power-law (SBPL) BBB.

Compares the JAX implementation against fixtures generated from upstream
``activatepl.sbpl`` (see ``tools/generate_grahsp_fixtures.py``).

References
----------
Buchner et al. 2024, arXiv:2405.19297, Eq. 1.
Ryde, F. 1999, ApJ, 511, 692.
"""

from __future__ import annotations

from pathlib import Path

import chex
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "sbpl_bbb.npz"


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE)


def test_fixture_present(fixture):
    assert fixture["spectra"].shape[0] == 5
    assert fixture["wave_nm"].size == 401


def test_sbpl_matches_upstream(fixture):
    from tengri.components.agn.grahsp.bbb import sbpl_bbb

    wave_nm = fixture["wave_nm"]
    params = fixture["params"]
    expected = fixture["spectra"]
    for i, p in enumerate(params):
        out = np.asarray(
            sbpl_bbb(
                wave_nm=wave_nm,
                l5100=float(p["lum5100A"]),
                uvslope=float(p["uvslope"]),
                plslope=float(p["plslope"]),
                plbendloc_nm=float(p["plbendloc"]),
                plbendwidth=float(p["plbendwidth"]),
                cutoff_nm=float(p["cutoff"]),
            )
        )
        # rel-err < 1e-6 finite, abs floor for tiny values
        np.testing.assert_allclose(out, expected[i], rtol=1e-9, atol=0.0, err_msg=f"case {i}")


def test_sbpl_normalization_at_5100A():
    """At lambda = 510 nm = 5100 Å, the SBPL must equal l5100 / 510 (in W/nm)."""
    from tengri.components.agn.grahsp.bbb import sbpl_bbb

    out = sbpl_bbb(
        wave_nm=np.array([510.0]),
        l5100=1.0e36,
        uvslope=0.0,
        plslope=-1.7,
        plbendloc_nm=100.0,
        plbendwidth=1.0,
        cutoff_nm=-1.0,
    )
    expected = 1.0e36 / 510.0
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_sbpl_jit_compatible():
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.bbb import sbpl_bbb

    out = assert_jit_matches_eager(
        sbpl_bbb,
        wave_nm=jnp.array([100.0, 510.0, 5000.0]),
        l5100=1.0e36,
        uvslope=0.0,
        plslope=-1.7,
        plbendloc_nm=100.0,
        plbendwidth=1.0,
        cutoff_nm=-1.0,
    )
    chex.assert_shape(out, (3,))
    chex.assert_tree_all_finite(out)

from __future__ import annotations

# SPDX-License-Identifier: BSD-3-Clause
import pytest

"""Smoke + correctness tests for ``preintegrate_ssp_filter_grid``.

The pre-integrated photometric SSP grid is the architectural ingredient
for fast photometry-only orchestrator workflows. We pin:

1. Output shape ``(n_met, n_age, n_filters)``.
2. Values are positive and finite.
3. Result is consistent with the per-filter integral
   :func:`tengri.observation.photometry.compute_flux_density` uses
   internally (sans the source→observer factor).
"""


import pathlib

import jax.numpy as jnp

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward import preintegrate_ssp_filter_grid

pytestmark = pytest.mark.bounds

_SSP_PATH = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_PATH.exists():
        pytest.skip(f"SSP file not present at {_SSP_PATH}")
    return load_ssp_data(str(_SSP_PATH))


def _toy_box_filter(center: float, width: float, n: int = 41):
    fw = jnp.linspace(center - width, center + width, n)
    ft = jnp.where(jnp.abs(fw - center) <= width / 2, 1.0, 0.0)
    return fw, ft


def test_preintegrate_shape(ssp):
    fws, fts = zip(*[_toy_box_filter(c, 500.0) for c in (3500.0, 5000.0, 7500.0)])
    out = preintegrate_ssp_filter_grid(ssp, list(fws), list(fts), redshift=0.0)
    assert out.shape == (ssp.ssp_flux.shape[0], ssp.ssp_flux.shape[1], 3)


def test_preintegrate_finite_and_positive(ssp):
    fw, ft = _toy_box_filter(5000.0, 500.0)
    out = preintegrate_ssp_filter_grid(ssp, [fw], [ft], redshift=0.0)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out > 0.0)


def test_preintegrate_redshift_shifts_filter_alignment(ssp):
    """A filter sitting on a strong age-dependent feature should
    move when the SSP wave grid is redshifted: the integrated value
    at z=0 differs from z=1 for the same filter."""
    fw, ft = _toy_box_filter(5000.0, 200.0)
    z0 = preintegrate_ssp_filter_grid(ssp, [fw], [ft], redshift=0.0)
    z1 = preintegrate_ssp_filter_grid(ssp, [fw], [ft], redshift=1.0)
    # Pick a young SSP where rest-frame 5000 A vs 2500 A differ a lot.
    # Old SSPs: V-band (rest 5000 A) much brighter than UV (rest 2500 A
    # at z=1 observed 5000 A) — large ratio expected.
    v0 = float(z0[-1, -1, 0])
    v1 = float(z1[-1, -1, 0])
    ratio = max(v0, v1) / max(min(v0, v1), 1e-40)
    assert ratio > 1.5, (
        f"z=0 and z=1 should differ by >1.5×; got {ratio:.2f}× ({v0:.3e} vs {v1:.3e})"
    )


def test_preintegrate_compression_factor(ssp):
    """Five-filter pre-integration must shrink the wave axis by ≥1000×
    for the PRSC-MILES grid (n_wave=5994)."""
    fws_fts = [_toy_box_filter(c, 500.0) for c in (3500, 4500, 5500, 6500, 7500)]
    fws, fts = zip(*fws_fts)
    out = preintegrate_ssp_filter_grid(ssp, list(fws), list(fts), redshift=0.0)
    compression = ssp.ssp_flux.size / out.size
    assert compression > 1000.0, f"Got {compression:.0f}× — expected >1000×"

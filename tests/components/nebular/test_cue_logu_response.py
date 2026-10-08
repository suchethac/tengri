# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``neb_logU`` reaches Cue's network as Cue's own ``gas_logu`` (#2786).

The CIGALE reproduction notebook (section 8b) found tengri's Cue [O III]/Hb nearly
flat in ``neb_logU`` where CIGALE's CLOUDY grid rises 4x. The audit (#2786) found the
wiring correct and the flatness to be the network's response to the *SSP-derived
ionizing shape*: Cue evaluated at its own default young-starburst shape rises
steeply with logU, while the soft shape a 0-100 Myr FSPS MIST+MILES population
fits to saturates. These tests guard both facts.

Reference definitions (Li et al. 2025, ApJ 986, 9, arXiv:2405.04598, Table 1 and the
``cue.utils.logQ`` function of https://github.com/yi-jia-li/cue):
``gas_logu`` is the inner-face ionization parameter at R = 1e19 cm,
``logQ = logU + log10(4 pi) + 2 log10(R) + log10(n_H) + log10(c)`` with
``c = 2.9979e10 cm/s``, trained over logU in [-4, -1].
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.cue import (
    CueBackend,
    _prepare_nn_params,
    predict_all_lines,
)
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

_WEIGHTS = Path("data/cue_weights.npz")

# Cue's own default young-starburst ionizing shape (``cue`` Emulator/README default;
# also ``CueBackend._IONSPEC_DEFAULTS``): a HARD spectrum.
_HARD = dict(
    ionspec_index1=19.7,
    ionspec_index2=5.3,
    ionspec_index3=1.6,
    ionspec_index4=0.6,
    ionspec_logLratio1=3.9,
    ionspec_logLratio2=0.01,
    ionspec_logLratio3=0.2,
)
# The shape tengri derives for the section 8b fiducial (delayed SFH, tau = 300 Myr,
# age = 100 Myr, FSPS MIST+MILES, Z = 0.02): a SOFT spectrum. Inputs, not outputs.
_SOFT = dict(
    ionspec_index1=24.083,
    ionspec_index2=11.125,
    ionspec_index3=7.668,
    ionspec_index4=2.457,
    ionspec_logLratio1=2.497,
    ionspec_logLratio2=0.44,
    ionspec_logLratio3=0.615,
)
_LOGU_GRID = (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0)
_LOG_C_CUE = np.log10(2.9979e10)  # cue.utils.logQ


@pytest.fixture(scope="module")
def backend():
    if not _WEIGHTS.is_file():
        pytest.skip("Cue weights not found")
    return CueBackend(str(_WEIGHTS))


def _ratios(wav, lum):
    w, lv = np.asarray(wav), np.asarray(lum)

    def band(lo, hi):
        return float(lv[(w > lo) & (w < hi)].sum())

    hb = band(4862.0, 4864.0)
    return band(5007.0, 5009.0) / hb, band(3726.0, 3731.0) / hb


def _sweep(backend, shape):
    out = [
        _ratios(
            *backend.predict_nebular_line_luminosities(
                gas_logu=lu, gas_logn=2.0, gas_logz=0.149, gas_logqion=49.1, **shape
            )
        )
        for lu in _LOGU_GRID
    ]
    return np.asarray(out)[:, 0], np.asarray(out)[:, 1]


def test_gas_logu_equals_a_direct_network_call(backend):
    """The backend's low-level path is the network evaluated at Cue's logQ(logU)."""
    for logu in (-3.0, -2.0, -1.5):
        logn = 2.0
        # Independent re-derivation of cue.utils.logQ(logU, R=1e19, lognH=logn).
        logq = logu + np.log10(4.0 * np.pi) + 2.0 * np.log10(1e19) + logn + _LOG_C_CUE
        nn = jnp.asarray([*_SOFT.values(), logq, 10.0**logn, 0.149, 0.0, 0.0], dtype=jnp.float32)
        wd, direct = predict_all_lines(nn, backend.weights, jnp.float32(logq), jnp.float32(49.1))
        wb, viaback = backend.predict_nebular_line_luminosities(
            gas_logu=logu,
            gas_logn=logn,
            gas_logz=0.149,
            gas_logqion=49.1,
            **_SOFT,
            cloudyfsps_only=False,
        )
        # Same lines, same order (sorted vacuum wavelengths).
        np.testing.assert_allclose(np.asarray(wb), np.asarray(wd), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(viaback), np.asarray(direct), rtol=2e-4)
    # _prepare_nn_params is the same vector (guards the logU -> logQ slot).
    vec = _prepare_nn_params(
        *(jnp.float32(v) for v in _SOFT.values()),
        *(jnp.float32(v) for v in (-2.0, 2.0, 0.149, 0.0, 0.0)),
    )
    assert abs(float(vec[7]) - (-2.0 + np.log10(4 * np.pi) + 38.0 + 2.0 + _LOG_C_CUE)) < 1e-4


def test_oiii_rises_and_oii_falls_with_logu_for_a_hard_spectrum(backend):
    """Cue's own default young shape: [O III]/Hb monotone up, [O II]/Hb monotone down.

    This is the response a CLOUDY grid shows at fixed spectrum (higher U raises the
    O++ zone); the network reproduces it, so a flat tengri response cannot come from
    a dead ``gas_logu`` input.
    """
    o3, o2 = _sweep(backend, _HARD)
    assert np.all(np.diff(o3) > 0.0), o3
    assert np.all(np.diff(o2) < 0.0), o2
    assert o3[-1] / o3[2] > 1.5  # log U = -1 vs -3


def test_soft_ionizing_shape_saturates_oiii_response(backend):
    """The SSP-derived soft shape starves O++ photons: [O III]/Hb saturates above
    log U ~ -3 (the #2786 flat response), unlike the hard shape at the same logU.
    """
    o3_soft, _ = _sweep(backend, _SOFT)
    o3_hard, _ = _sweep(backend, _HARD)
    i3, i15 = _LOGU_GRID.index(-3.0), _LOGU_GRID.index(-1.5)
    assert o3_soft[i15] / o3_soft[i3] < 1.3
    assert o3_hard[i15] / o3_hard[i3] > 1.5


def test_high_level_neb_logu_matches_the_low_level_call(backend, ssp_data_fsps):
    """``neb_logU`` through the SSP path is bit-for-bit ``gas_logu`` on the network."""
    be = CueBackend(str(_WEIGHTS), ssp_data=ssp_data_fsps)
    log_z = float(ssp_data_fsps.ssp_lgmet[len(ssp_data_fsps.ssp_lgmet) // 2])
    log_ages = jnp.asarray(ssp_data_fsps.ssp_lg_age_gyr) + 9.0
    i = int(np.argmin(np.abs(np.asarray(log_ages) - 6.3)))
    w = jnp.zeros_like(log_ages).at[i].set(1.0)
    ionspec, logqion = be.get_ionizing_params_at(log_z, float(log_ages[i]))
    gas_logz = log_z - float(LOG10_ZSUN)
    for logu in (-3.0, -2.0, -1.5):
        _, hi = be.predict_nebular_line_luminosities(
            ssp_weights=w,
            ssp_log_ages_yr=log_ages,
            log_z=log_z,
            neb_logU=logu,
            neb_logZ_gas=log_z,
        )
        keys = ("index1", "index2", "index3", "index4", "logLratio1", "logLratio2", "logLratio3")
        _, lo = be.predict_nebular_line_luminosities(
            gas_logu=logu,
            gas_logn=2.0,
            gas_logz=gas_logz,
            gas_logqion=float(logqion),
            **{f"ionspec_{k}": float(v) for k, v in zip(keys, ionspec)},
        )
        np.testing.assert_allclose(np.asarray(hi), np.asarray(lo), rtol=1e-4)

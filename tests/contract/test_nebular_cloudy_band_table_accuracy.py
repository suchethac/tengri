# SPDX-License-Identifier: BSD-3-Clause
"""Accuracy of the age-resolved CLOUDY band table against the per-call path (#2324).

The per-call photometric path runs ``predict_nebular_sed`` and two
``lnu_filter_integral`` passes (observed and rest frame) on every call. The
band table replaces that with a contraction over SSP ages of tabulated
per-age quantities, interpolated in (Z_gas, logU) in log space.

The contract measured here: at five seeded draws of (neb_logU, neb_logZ_gas,
stellar metallicity) in their declared priors, with a delayed-tau SFH and
z = 0.1, the worst relative error of the band table against the per-call path
is below 1e-3, and the build finishes within 60 s.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data_str

pytestmark = pytest.mark.contract

_SSP = find_data_str("fsps_prsc_miles_chabrier.h5")
_GRID = find_data_str("cloudy_grid_prsc.h5")
_SKIP = pytest.mark.skipif(
    _SSP is None or _GRID is None,
    reason="needs fsps_prsc_miles_chabrier.h5 and cloudy_grid_prsc.h5 (set TENGRI_DATA_DIR)",
)

_REDSHIFT = 0.1
_LINE_SIGMA_KMS = 100.0
_TOL = 1e-3
_BUILD_BUDGET_S = 60.0
_N_DRAWS = 5


@pytest.fixture(scope="module")
def setup():
    """SSP, CLOUDY backend, filters, delayed-tau weights and the band table."""
    jax.config.update("jax_enable_x64", True)
    import tengri
    from tengri.components.nebular.cloudy_band_table import build_cloudy_band_table
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.observation.photometry_config import Photometry

    ssp = tengri.load_ssp_data(_SSP)
    backend = CloudyGridBackend(_GRID, ssp_data=ssp)
    wave = jnp.asarray(ssp.ssp_wave)
    phot = Photometry.from_names(
        ["sdss_g", "sdss_r", "sdss_i", "galex_fuv", "galex_nuv", "2mass_ks"]
    )
    fw = [jnp.asarray(x) for x in phot.filter_waves]
    ft = [jnp.asarray(x) for x in phot.filter_trans]

    ages_log = np.asarray(backend._qh_log_age)
    t_gyr = 10 ** (ages_log - 9)
    dt = np.gradient(t_gyr)
    tau = 0.5
    weights = jnp.asarray(t_gyr / tau**2 * np.exp(-t_gyr / tau) * dt * 1e9)

    t0 = time.perf_counter()
    table = build_cloudy_band_table(backend, wave, fw, ft, _REDSHIFT, _LINE_SIGMA_KMS)
    build_s = time.perf_counter() - t0
    return {
        "backend": backend,
        "wave": wave,
        "fw": fw,
        "ft": ft,
        "ages_log": ages_log,
        "weights": weights,
        "table": table,
        "build_s": build_s,
    }


def _per_call_bands(s, log_z, log_z_gas, log_u):
    """Observed and rest band L_nu of the per-call path, shape (2, n_filt)."""
    from tengri.observation.photometry import lnu_filter_integral

    backend = s["backend"]
    sed = backend.predict_nebular_sed(
        ssp_weights=s["weights"],
        ssp_wave=s["wave"],
        ssp_log_ages_yr=jnp.asarray(s["ages_log"]),
        log_z=log_z,
        neb_logU=log_u,
        neb_logZ_gas=log_z_gas,
        neb_fesc=0.0,
        neb_fesc_lya=0.0,
        neb_fdust_frac=0.0,
        line_sigma_kms=_LINE_SIGMA_KMS,
        template_data=None,
    )
    obs = jnp.stack(
        [
            lnu_filter_integral(sed, s["wave"], f, t, redshift=_REDSHIFT)
            for f, t in zip(s["fw"], s["ft"], strict=True)
        ]
    )
    rest = jnp.stack(
        [
            lnu_filter_integral(sed, s["wave"], f, t, redshift=0.0)
            for f, t in zip(s["fw"], s["ft"], strict=True)
        ]
    )
    return np.asarray(jnp.stack([obs, rest]))


def _table_bands(s, log_z, log_z_gas, log_u):
    """Observed and rest band L_nu from the band table, shape (2, n_filt)."""
    from tengri.components.nebular.cloudy_band_table import contract_cloudy_band_table

    backend = s["backend"]
    young = np.asarray(backend._young_idx)
    qh = jnp.stack([backend._get_qh_at(log_z, jnp.asarray(a)) for a in s["ages_log"][young]])
    age_weights = s["weights"][young] * qh
    return np.asarray(contract_cloudy_band_table(s["table"], age_weights, log_z_gas, log_u, 0.0))


@_SKIP
def test_band_table_build_within_budget(setup):
    """The table builds in under the registry budget of 60 s."""
    assert setup["build_s"] < _BUILD_BUDGET_S


@_SKIP
def test_band_table_matches_per_call_path_at_seeded_draws(setup):
    """Worst relative band error over five seeded draws is below 1e-3."""
    from tengri.parameters.translate import LOG10_ZSUN

    rng = np.random.default_rng(20241009)
    worst = 0.0
    for _ in range(_N_DRAWS):
        met = rng.uniform(-1.3, 0.2)
        gas = rng.uniform(-1.3, 0.2)
        log_u = rng.uniform(-5.0, 0.0)
        log_z = met + LOG10_ZSUN
        log_z_gas = gas + LOG10_ZSUN
        exact = _per_call_bands(setup, log_z, log_z_gas, log_u)
        approx = _table_bands(setup, log_z, log_z_gas, log_u)
        scale = np.max(np.abs(exact))
        rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-6 * scale)
        worst = max(worst, float(rel.max()))
    assert np.isfinite(worst)
    assert worst < _TOL, f"worst relative band error {worst:.3e} exceeds {_TOL:.0e}"

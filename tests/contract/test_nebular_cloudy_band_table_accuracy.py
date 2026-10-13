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
from tests._cloudy_grid import synthetic_ionizing_ssp, write_synthetic_cloudy_grid

pytestmark = pytest.mark.contract

#: The real grid is built locally from raw CLOUDY output and is not tracked or
#: downloadable, so only the real-data case gates on it. The synthetic case
#: below carries no such gate and runs on every runner.
_GRID = find_data_str("cloudy_grid_prsc.h5")
_REAL_SKIP = pytest.mark.skipif(
    _GRID is None,
    reason="needs cloudy_grid_prsc.h5 (scripts/convert_fsps_cloudy_grid.py; not downloadable)",
)

#: Observed-frame redshift. At z = 0.2 the Ly-alpha line (1215.67 A) lands at
#: 1459 A, inside the GALEX FUV band in this file's filter set. At z = 0.1 it
#: falls at 1337 A, outside every band, so the escape cases could not fail.
_REDSHIFT = 0.2
#: Index of galex_fuv in the filter list built in ``setup``.
_FUV = 3
#: Relative change in the FUV band between fesc_lya = 0.2 and 0.8 must exceed
#: this. It is 50 times the accuracy bar, so an escape convention that stops
#: reaching the bands fails here before it can pass the contract.
_LYA_SENSITIVITY_MIN = 50 * 1e-3
_LINE_SIGMA_KMS = 100.0
_TOL = 1e-3
_BUILD_BUDGET_S = 60.0
_N_DRAWS = 5


@pytest.fixture(scope="module", params=["synthetic", pytest.param("real", marks=_REAL_SKIP)])
def cloudy_source(request, tmp_path_factory):
    """The SSP and CLOUDY grid the contract runs against.

    ``synthetic`` writes a deterministic CLOUDY-format grid and builds an SSP
    with a physical ionizing flux, so the contract runs with no external data.
    ``real`` takes the bare-stellar SSP from ``download_ssp`` (the same helper
    the CI-run shock test uses) and the locally built CLOUDY grid.
    """
    if request.param == "synthetic":
        grid = write_synthetic_cloudy_grid(tmp_path_factory.mktemp("cloudy") / "grid.h5")
        return {"ssp": synthetic_ionizing_ssp(), "grid": str(grid)}
    import tengri

    ssp_path = tengri.download_ssp("fsps_prsc_miles_chabrier", progress=False)
    return {"ssp": tengri.load_ssp_data(str(ssp_path)), "grid": _GRID}


@pytest.fixture(scope="module")
def setup(cloudy_source):
    """SSP, CLOUDY backend, filters, delayed-tau weights and the band table."""
    jax.config.update("jax_enable_x64", True)
    from tengri.components.nebular.cloudy_band_table import build_cloudy_band_table
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.observation.photometry_config import Photometry

    ssp = cloudy_source["ssp"]
    backend = CloudyGridBackend(cloudy_source["grid"], ssp_data=ssp)
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


def _per_call_bands(s, log_z, log_z_gas, log_u, fesc_lya=0.0):
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
        neb_fesc_lya=fesc_lya,
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


def _table_bands(s, log_z, log_z_gas, log_u, fesc_lya=0.0):
    """Observed and rest band L_nu from the band table, shape (2, n_filt)."""
    from tengri.components.nebular.cloudy_band_table import contract_cloudy_band_table

    backend = s["backend"]
    young = np.asarray(backend._young_idx)
    qh = jnp.stack([backend._get_qh_at(log_z, jnp.asarray(a)) for a in s["ages_log"][young]])
    age_weights = s["weights"][young] * qh
    return np.asarray(
        contract_cloudy_band_table(s["table"], age_weights, log_z_gas, log_u, fesc_lya)
    )


def _worst_relative_error(s, fesc_lya):
    """Worst relative band error of the table over the five seeded draws."""
    from tengri.parameters.translate import LOG10_ZSUN

    rng = np.random.default_rng(20241009)
    worst = 0.0
    for _ in range(_N_DRAWS):
        met = rng.uniform(-1.3, 0.2)
        gas = rng.uniform(-1.3, 0.2)
        log_u = rng.uniform(-5.0, 0.0)
        log_z = met + LOG10_ZSUN
        log_z_gas = gas + LOG10_ZSUN
        exact = _per_call_bands(s, log_z, log_z_gas, log_u, fesc_lya)
        approx = _table_bands(s, log_z, log_z_gas, log_u, fesc_lya)
        scale = np.max(np.abs(exact))
        rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-6 * scale)
        worst = max(worst, float(rel.max()))
    return worst


def test_band_table_build_within_budget(setup):
    """The table builds in under the registry budget of 60 s."""
    assert setup["build_s"] < _BUILD_BUDGET_S


def test_band_table_matches_per_call_path_at_seeded_draws(setup):
    """Worst relative band error over five seeded draws is below 1e-3."""
    worst = _worst_relative_error(setup, fesc_lya=0.0)
    assert np.isfinite(worst)
    assert worst < _TOL, f"worst relative band error {worst:.3e} exceeds {_TOL:.0e}"


# 0.5 cannot separate the conventions, since (1 - f) == f there. A table that
# used f_esc in place of (1 - f_esc) passes at 0.5 and fails at 0.2 and 0.8.
@pytest.mark.parametrize("fesc_lya", [0.2, 0.8])
def test_band_table_matches_per_call_path_with_lya_escape(setup, fesc_lya):
    """The Ly-alpha escape convention is the per-call one at nonzero neb_fesc_lya.

    The table applies Ly-alpha through the same ``apply_lya_escape`` the
    per-call backend uses, so it cannot drift from it. The worst relative band
    error stays under the same 1e-3 bar as the zero-escape case.
    """
    worst = _worst_relative_error(setup, fesc_lya=fesc_lya)
    assert np.isfinite(worst)
    assert worst < _TOL, f"worst relative band error at fesc_lya={fesc_lya} is {worst:.3e}"


def test_lya_escape_moves_the_fuv_band(setup):
    """The per-call FUV band must respond to neb_fesc_lya, or the cases above are inert.

    Measured on the per-call path, not the table: if Ly-alpha stops reaching the
    band, the escape contract cannot tell the convention apart, and this fails.
    """
    lo = _per_call_bands(setup, -1.85, -2.35, -3.0, 0.2)[0, _FUV]
    hi = _per_call_bands(setup, -1.85, -2.35, -3.0, 0.8)[0, _FUV]
    change = abs(hi - lo) / max(abs(lo), 1e-300)
    assert change > _LYA_SENSITIVITY_MIN, (
        f"FUV band changes by {change:.3e} between fesc_lya 0.2 and 0.8; "
        f"it must exceed {_LYA_SENSITIVITY_MIN:.0e}"
    )

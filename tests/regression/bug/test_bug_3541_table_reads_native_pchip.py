# SPDX-License-Identifier: BSD-3-Clause
"""#3541: SpectrumPrecomp matches the exact spectrum at an off-node metallicity.

Since #3528 the stellar component reads the native SSP flux across metallicity by
monotone PCHIP, but the spectrum table ``ssp_on_pixels`` is resampled to the pixels and
then PCHIP-read over its own stored values. PCHIP's slope limiter is nonlinear in the
data, so the two differ (2.7e-2 here; 6e-15 before #3528). The fix (owner ruling
2026-11-15, "Exact: native slopes") is on hold, so this case is an expected failure
until it lands.

The model is the synthetic three-node grid at absolute log Z -1.848 (between the -2
and -1 nodes): ``met_logzsol`` is solar-relative, so 0.0 gives that value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, SpectrumPrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Spectroscopy
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of a table read against the exact read (round-off).
RTOL = 1e-12
_MET_LOGZSOL = 0.0  # absolute log Z -1.848, between the -2 and -1 nodes


def _synthetic_ssp():
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (3, 20, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _spectro_obs():
    wave = np.linspace(5200.0, 7400.0, 2000)
    sp = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resolution=1500.0,
        eline_mode="marginalized",
        eline_catalog=LineList.from_names(["Hbeta", "OIII_4959", "OIII_5007", "Halpha"]),
        eline_prior_sigma=1.0e-14,
    )
    return Observation(spectroscopy=sp)


def _build(*, approx):
    return SEDModel.build(
        ssp_data=_synthetic_ssp(),
        observation=_spectro_obs(),
        approx=approx,
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_frac": Fixed(0.6),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
            "age_kernel": "cic",
        },
        met={"type": "delta", "met_logzsol": Fixed(_MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


@pytest.mark.xfail(
    strict=True,
    reason="#3541: table reads PCHIP of its own resampled values; owner ruling on hold",
)
def test_spectrum_lut_matches_exact_at_off_node_z():
    exact = np.asarray(_build(approx=None).predict_spectrum({}))
    lut = np.asarray(_build(approx=SpectrumPrecomp()).predict_spectrum({}))
    rel = float(np.max(np.abs(lut - exact) / np.maximum(np.abs(exact), 1e-300)))
    assert rel <= RTOL

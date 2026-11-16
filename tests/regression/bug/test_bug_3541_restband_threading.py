# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the rest-band tables are not read off a shared closure built by another model.

The fixed-z LUTs are threaded as runtime inputs (5470bb726) because a jit kernel is shared
between models with the same compile signature: a model compiled after another one read the
first model's closure constants. The rest-band tables (``RestBandPrecomputation``, and its
native-slope sibling) stay on the closure, so this probes the same hazard: two models with
different table contents and different sub-band counts, compiled back to back in each order.
Each model's rest-band photometry must equal the value it gives when it is built alone, and
the exact path (approx=None) to round-off.

Synthetic SSP (no ``data/`` grid needed); the second model's flux is tilted in wavelength, so
its rest-band tables differ in content at the same shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of a table read against the exact read (round-off).
RTOL = 1e-12
_BANDS = ["sdss_g", "sdss_r", "2mass_ks"]
_MET_LOGZSOL = -0.052  # absolute log Z -1.900, between the -2.0 and -1.0 synthetic nodes
_OUTPUTS = ("phot_rest_fnu", "phot_fnu", "mag_absolute")


def _ssp(tilt: float) -> SSPData:
    """Three metallicity nodes, 20 ages, a wavelength tilt of ``tilt`` (table contents)."""
    wave = jnp.geomspace(300.0, 3.0e4, 400)
    flux = jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (3, 20, 400))) * 1e-3 + 1e-5
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux * (wave / 5000.0) ** tilt,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _build(ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
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


def _observables(model):
    obs = model.predict_observables({})
    return {name: np.asarray(getattr(obs, name), dtype=np.float64) for name in _OUTPUTS}


def _max_rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


_CONFIGS = {"base": (_ssp(0.0), 8), "tilted": (_ssp(1.5), 4)}


def _lut(name):
    ssp, n_sub = _CONFIGS[name]
    return _build(ssp, WavePrecomp(band_integration="quadrature", n_subbands=n_sub))


@pytest.mark.parametrize("first,second", [("base", "tilted"), ("tilted", "base")])
def test_rest_band_tables_are_not_read_from_another_models_closure(first, second):
    alone = {name: _observables(_lut(name)) for name in _CONFIGS}
    _observables(_lut(first))  # compile the first model's kernel
    after = _observables(_lut(second))
    for out in _OUTPUTS:
        assert _max_rel(after[out], alone[second][out]) <= RTOL, (first, second, out)


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_rest_band_tables_match_exact_path(name):
    ssp, _ = _CONFIGS[name]
    exact = _observables(_build(ssp, None))
    lut = _observables(_lut(name))
    for out in _OUTPUTS:
        assert _max_rel(lut[out], exact[out]) <= RTOL, (name, out)

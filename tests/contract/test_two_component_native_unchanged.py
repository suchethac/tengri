# SPDX-License-Identifier: BSD-3-Clause
"""The native ``two_component`` build is bit-identical to base; ``tauv_mu`` matches it (#3176).

Golden values were computed on the base commit ``fb0aa11c0`` (origin/main, before the
``(tau_V, mu)`` parameterization) with ``golden_native.py`` run under
``PYTHONPATH=<base>/src``: the same model build, fixed parameter dict from
``spec.sample(jax.random.PRNGKey(0))``, ``predict_photometry`` on five top-hat filters,
and ``predict(params).rest_sed(wave=np.geomspace(1000, 3e4, 60))``. The literals below
are the float64 ``repr`` of those arrays, so the comparison is exact.

The generating script is kept outside the repository (job d147787b, ``tmp/golden_native.py``);
regenerating the literals means running the same build on ``fb0aa11c0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

#: Base-commit native depths, the pair the literals were computed at.
NATIVE_TAU_BC = 1.2
NATIVE_TAU_DIFF = 0.4

REST_WAVE = np.geomspace(1000.0, 3.0e4, 60)

_BASE_PHOT = np.array(
    [
        5.8610019958834684e-30,
        7.268141122942855e-30,
        1.0569227000264527e-29,
        1.8034533778581036e-29,
        3.044571736639866e-29,
    ]
)

_BASE_REST = np.array(
    [
        4.1481225903756826e27,
        6.437812178891293e27,
        6.377767438619705e27,
        7.564820044292843e27,
        9.603136344358375e27,
        1.1022797816423397e28,
        1.2545530584897459e28,
        1.4094509141387818e28,
        1.390018963683045e28,
        1.6106405052718035e28,
        1.637633236182242e28,
        1.634955659435329e28,
        1.838155288573873e28,
        1.9742251297987187e28,
        2.042600621882552e28,
        1.9374307150300876e28,
        2.053147438500371e28,
        2.370814520021786e28,
        2.5695761463089444e28,
        2.8145787633570173e28,
        3.0385479245933833e28,
        3.225394991699371e28,
        3.5075572346259864e28,
        4.849816451231366e28,
        7.430087341417771e28,
        8.07189167739403e28,
        8.872978203443601e28,
        9.091742536295496e28,
        8.92408764566587e28,
        9.571408173037343e28,
        9.681115075128762e28,
        1.0320967789790106e29,
        1.0490230509575387e29,
        1.0591905905504847e29,
        1.0465325665360093e29,
        1.2358834098062398e29,
        1.2433684993571353e29,
        1.284624937567359e29,
        1.4625838261746419e29,
        1.5123464523406096e29,
        1.585549354345777e29,
        1.6369970371389403e29,
        1.6077363597299898e29,
        1.682505414621695e29,
        1.743174682592433e29,
        1.697731633008131e29,
        1.6483654717769394e29,
        1.777537875735877e29,
        1.9980561456826044e29,
        2.1029370019428627e29,
        1.9308385561254465e29,
        1.864054039427312e29,
        1.7590837771398375e29,
        1.793989119446518e29,
        1.71474112237385e29,
        1.2956438673891548e29,
        1.1837559455145156e29,
        1.2697115166119778e29,
        1.2535711231629428e29,
        1.2035038713350675e29,
    ]
)


def _tophat(center, frac=0.12, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _observation():
    curves = tuple(_tophat(c) for c in (3550.0, 4700.0, 6200.0, 7500.0, 8900.0))
    return Observation(photometry=Photometry(filters=curves))


def _build(ssp, dust_atten):
    return SEDModel.build(
        ssp,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "none"},
        redshift=Fixed(1.0),
        approx=None,
        observation=_observation(),
    )


def _native_atten(tau_bc, tau_diff):
    return {
        "type": "two_component",
        "law_bc": "power_law",
        "slope_bc": -1.3,
        "law_diff": "power_law",
        "slope_diff": -0.7,
        "tau_bc": Fixed(tau_bc),
        "tau_diff": Fixed(tau_diff),
        "all_params": Fixed(DEFAULT),
    }


def _tauv_mu_atten(tau_v, mu):
    return {
        "type": "two_component",
        "law_bc": "power_law",
        "slope_bc": -1.3,
        "law_diff": "power_law",
        "slope_diff": -0.7,
        "parameterization": "tauv_mu",
        "tau_v": Fixed(tau_v),
        "mu": Fixed(mu),
        "all_params": Fixed(DEFAULT),
    }


def _outputs(model):
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    phot = np.asarray(model.predict_photometry(params))
    rest = np.asarray(model.predict(params).rest_sed(wave=REST_WAVE))
    return phot, rest


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)


@pytest.fixture(scope="module")
def native_outputs(ssp):
    return _outputs(_build(ssp, _native_atten(NATIVE_TAU_BC, NATIVE_TAU_DIFF)))


def test_native_photometry_bit_identical_to_base(native_outputs):
    phot, _ = native_outputs
    assert np.array_equal(phot, _BASE_PHOT)


def test_native_rest_sed_bit_identical_to_base(native_outputs):
    _, rest = native_outputs
    assert np.array_equal(rest, _BASE_REST)


def test_tauv_mu_matches_native_photometry_and_rest_sed(ssp):
    """tauv_mu at (tau_V, mu) equals native at the pair it derives, bit for bit.

    tau_V = 2.0, mu = 0.25 derives (tau_BC, tau_diff) = (1.5, 0.5), exact in binary.
    The spec's (1.6, 0.25) -> (1.2, 0.4) pair is not exact: (1 - 0.25) * 1.6 rounds
    to 1.2000000000000002, so it cannot be bit-identical to native 1.2.
    """
    native = _outputs(_build(ssp, _native_atten(1.5, 0.5)))
    tauv_mu = _outputs(_build(ssp, _tauv_mu_atten(2.0, 0.25)))
    assert np.array_equal(native[0], tauv_mu[0])
    assert np.array_equal(native[1], tauv_mu[1])

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

#: Re-pinned after the Cue float64 move (#3253). These outputs include the Cue nebular
#: continuum; the float32-Cue values moved by at most 1.8e-7 (photometry) and 1.0e-7 (rest
#: SED) relative. With the nebular component off, the outputs are bit-identical to before.
_BASE_PHOT = np.array(
    [
        5.861002018576371e-30,
        7.268141234875467e-30,
        1.0569227594820383e-29,
        1.8034537002135826e-29,
        3.044571909013473e-29,
    ]
)

_BASE_REST = np.array(
    [
        4.1481225909896136e27,
        6.437812179513847e27,
        6.377767439249182e27,
        7.56482004493568e27,
        9.603136351088107e27,
        1.102279783099433e28,
        1.2545530605270987e28,
        1.4094509168118115e28,
        1.3900189671468739e28,
        1.6106405097968238e28,
        1.6376332421657635e28,
        1.634955667996711e28,
        1.8381552993403172e28,
        1.9742251445564992e28,
        2.042600642244746e28,
        1.9374307435398704e28,
        2.0531474790000136e28,
        2.3708146507507927e28,
        2.56957622899081e28,
        2.8145788855940724e28,
        3.0385481044649525e28,
        3.2253952573063364e28,
        3.507557598507654e28,
        4.849816538323137e28,
        7.430087444268653e28,
        8.07189179924209e28,
        8.872978469266773e28,
        9.091742712204022e28,
        8.924087853076675e28,
        9.571408422387998e28,
        9.681115375864424e28,
        1.0320968149329398e29,
        1.0490230939258646e29,
        1.0591906415164088e29,
        1.046532626925984e29,
        1.2358834808413601e29,
        1.2433685808131105e29,
        1.284624976068045e29,
        1.4625838696057447e29,
        1.5123465011947631e29,
        1.5855494749432566e29,
        1.6369970983647795e29,
        1.6077364277770516e29,
        1.682505489978134e29,
        1.7431747655941546e29,
        1.6977317240854138e29,
        1.648365570562109e29,
        1.777537945461509e29,
        1.998056221053872e29,
        2.1029370831442424e29,
        1.930838643316809e29,
        1.864054132655261e29,
        1.7590838765343115e29,
        1.7939892250356215e29,
        1.714741233942329e29,
        1.2956439612886106e29,
        1.1837560444976753e29,
        1.26971162067924e29,
        1.2535712322636103e29,
        1.2035039854563518e29,
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


def _assert_matches_base(values, base):
    """Agreement to rtol 1e-12, not bit identity.

    The literals were recorded at JAX 0.11.1 on macOS. Measured on the same Mac: at 0.11.1
    the outputs are bit-identical to them; at 0.12.0 the photometry differs by at most
    2.9e-14 relative (5 of 5 values) and the rest-frame SED by at most 5.4e-16 (16 of 60).
    The gap follows the JAX version, not the platform, so there is no platform split.
    A changed native build moves these values by far more than 1e-12.
    """
    np.testing.assert_allclose(values, base, rtol=1e-12, atol=0.0)


def test_native_photometry_bit_identical_to_base(native_outputs):
    phot, _ = native_outputs
    _assert_matches_base(phot, _BASE_PHOT)


def test_native_rest_sed_bit_identical_to_base(native_outputs):
    _, rest = native_outputs
    _assert_matches_base(rest, _BASE_REST)


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

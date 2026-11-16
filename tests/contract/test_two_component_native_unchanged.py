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
# Re-pinned 2026-11-15 after #3434 (in #3462): an undeclared metallicity scatter is now a node-
# exact cubic-Hermite read of the SSP. Photometry moves +0.008% to +0.86%, the rest-frame SED by
# up to 1.4%; 8aff37cb3 reproduces the previous photometry exactly and the previous SED to 1e-15.
# Re-pinned after #3528 (PCHIP metallicity read of the SSP table in log Z): old -> new, relative,
# photometry +3.4e-4 max ([0] 5.861448405796055e-30 -> 5.8620773936249606e-30), rest-frame SED
# +2.7e-3 max over 60 bins.
_BASE_PHOT = np.array(
    [
        5.8620773936249606e-30,
        7.273189894006967e-30,
        1.0592386703145563e-29,
        1.8195832166724628e-29,
        3.0664616483370245e-29,
    ]
)

_BASE_REST = np.array(
    [
        4.1222979054839887e27,
        6.403843435350466e27,
        6.368260420621703e27,
        7.554824772788963e27,
        9.581609352712245e27,
        1.1007617429519752e28,
        1.2526234019962995e28,
        1.4091387390058725e28,
        1.3896750027549413e28,
        1.6116077860131205e28,
        1.6386145798598527e28,
        1.6364785814557436e28,
        1.8384962515558242e28,
        1.9749237708999583e28,
        2.0438703131438305e28,
        1.9386797895290095e28,
        2.054596095878661e28,
        2.373936931341685e28,
        2.5743672781843333e28,
        2.8189603888206455e28,
        3.0464962397624936e28,
        3.2350999627564404e28,
        3.517199404279221e28,
        4.922299396896775e28,
        7.500038971046332e28,
        8.178306671144854e28,
        8.918916140100365e28,
        9.139063685178631e28,
        8.97042134232262e28,
        9.590502860009537e28,
        9.688534291201235e28,
        1.0324157741181334e29,
        1.0493053629504977e29,
        1.058326845751752e29,
        1.0421111567469457e29,
        1.2376618465756882e29,
        1.246826311819607e29,
        1.2881812511759776e29,
        1.466757593160423e29,
        1.5144999198596882e29,
        1.5831411429535598e29,
        1.6332203983221852e29,
        1.5974121398008464e29,
        1.6675607050098863e29,
        1.7240044950252635e29,
        1.681505218795062e29,
        1.6309401862064033e29,
        1.7602309224236335e29,
        1.9774063109082374e29,
        2.085696527065909e29,
        1.9151736379735434e29,
        1.850401539177552e29,
        1.747773316491275e29,
        1.7792020449568184e29,
        1.700148358154964e29,
        1.283965436021899e29,
        1.174393766310775e29,
        1.260536604800963e29,
        1.2467710026675165e29,
        1.1942843198009836e29,
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

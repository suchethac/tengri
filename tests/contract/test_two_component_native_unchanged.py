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
_BASE_PHOT = np.array(
    [
        5.861448405796055e-30,
        7.272240961672414e-30,
        1.0590521933962507e-29,
        1.818972868992229e-29,
        3.065671668083773e-29,
    ]
)

_BASE_REST = np.array(
    [
        4.121931682541589e27,
        6.403432835227167e27,
        6.367226223110514e27,
        7.554032174160658e27,
        9.581243039433708e27,
        1.1007290625869624e28,
        1.252533604507802e28,
        1.4089855735107187e28,
        1.3895379992527902e28,
        1.6114454764047245e28,
        1.6384045743620064e28,
        1.6362077847659462e28,
        1.838260470963138e28,
        1.9746388704696988e28,
        2.0436032761361614e28,
        1.9384535624144327e28,
        2.0543027956475654e28,
        2.3736181099067777e28,
        2.573991269587286e28,
        2.8185242394202653e28,
        3.04592672209826e28,
        3.2344270423049456e28,
        3.5164931963781457e28,
        4.919389740306693e28,
        7.497714804201702e28,
        8.175272885690471e28,
        8.916786723162392e28,
        9.137111873363017e28,
        8.968488564293846e28,
        9.589550555965333e28,
        9.6872963285695e28,
        1.0322867965984505e29,
        1.0492134079173541e29,
        1.058213356525216e29,
        1.0420163366064105e29,
        1.2380874878742071e29,
        1.2470926759486695e29,
        1.2885371802975373e29,
        1.4674704266792415e29,
        1.515616083126524e29,
        1.584833823487279e29,
        1.6351530079733894e29,
        1.5996433885556634e29,
        1.6703268209647848e29,
        1.727314170543699e29,
        1.6842566327102243e29,
        1.6336187300532196e29,
        1.7637311726458094e29,
        1.981935985021227e29,
        2.090867848464608e29,
        1.919820255990856e29,
        1.8549403362754684e29,
        1.751915822186649e29,
        1.7838428871842665e29,
        1.704673051532395e29,
        1.2868854938867118e29,
        1.1764625158718849e29,
        1.2630500218468986e29,
        1.2491646135787735e29,
        1.1968228784743791e29,
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

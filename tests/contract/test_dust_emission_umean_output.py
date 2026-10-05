# SPDX-License-Identifier: BSD-3-Clause
"""Contract: Draine & Li dust emission publishes ``dust_umean`` (#2599, item 3).

CIGALE's ``dl2007`` and ``dl2014`` modules publish ``dust.umean``, the mean
starlight intensity of the model. pcigale 2025.1 computes it in ``_init_code``:
``dl2007.py`` as ``(1 - gamma) umin + gamma ln(umax/umin) / (1/umin - 1/umax)``
(alpha = 2, umax = 1e6) and ``dl2014.py`` for general alpha with umax = 1e7.
tengri publishes the same quantity as the derived key ``dust_umean``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.contract


def _umean_dl07(umin, gamma):
    """pcigale ``dl2007.py`` ``self.umean``, written out (umax = 1e6)."""
    umax = 1.0e6
    return (1.0 - gamma) * umin + gamma * np.log(umax / umin) / (1.0 / umin - 1.0 / umax)


def _umean_dl14(umin, gamma, alpha):
    """pcigale ``dl2014.py`` ``self.umean``, written out (umax = 1e7)."""
    umax = 1.0e7
    umean = (1.0 - gamma) * umin
    if alpha == 1.0:
        umean += gamma * (umax - umin) / np.log(umax / umin)
    elif alpha == 2.0:
        umean += gamma * np.log(umax / umin) / (1.0 / umin - 1.0 / umax)
    else:
        oma, tma = 1.0 - alpha, 2.0 - alpha
        umean += gamma * oma / tma * (umin**tma - umax**tma) / (umin**oma - umax**oma)
    return umean


def _build(ssp, emission, attenuation):
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=attenuation,
        dust_emission={**emission, "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )


_ATTENUATIONS = {
    "two_component": {
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(0.2),
        "tau_diff": Fixed(0.5),
        "all_params": Fixed(DEFAULT),
    },
    "single_component": {
        "type": "single_component",
        "law": "power_law",
        "tau_v": Fixed(0.5),
        "all_params": Fixed(DEFAULT),
    },
}

# (umin, gamma) nodes of the shipped DL07 grid; literals = the pcigale formula above
_DL07_CASES = [
    (1.0, 0.1, 2.2815524373488647),
    (10.0, 0.5, 62.56520297688091),
    (0.1, 0.0, 0.1),
    (25.0, 1.0, 264.92249138968657),
]
# (umin, gamma, alpha); literals = pcigale ``DL2014(...).umean``
_DL14_CASES = [
    (1.0, 0.1, 2.0, 2.511809726276805),
    (1.0, 0.1, 1.0, 62042.962639114805),
    (10.0, 0.5, 1.5, 5005.0),
    (0.1, 0.2, 2.5, 0.13999400000006001),
    (3.0, 1.0, 3.0, 5.99999820000054),
]


@pytest.mark.parametrize("attenuation", sorted(_ATTENUATIONS))
@pytest.mark.parametrize(("umin", "gamma", "expected"), _DL07_CASES)
def test_dl07_umean_equals_the_cigale_formula(
    synthetic_ssp_wide, attenuation, umin, gamma, expected
):
    assert _umean_dl07(umin, gamma) == pytest.approx(
        expected, rel=1e-12
    )  # the literal IS the formula
    model = _build(
        synthetic_ssp_wide,
        {
            "type": "draine_li2007",
            "umin": Fixed(umin),
            "gamma_dl": Fixed(gamma),
            "qpah": Fixed(2.5),
        },
        _ATTENUATIONS[attenuation],
    )
    got = float(model.predict_state({}).derived["dust_umean"])
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("attenuation", sorted(_ATTENUATIONS))
@pytest.mark.parametrize(("umin", "gamma", "alpha", "expected"), _DL14_CASES)
def test_dl14_umean_equals_the_cigale_formula(
    synthetic_ssp_wide, attenuation, umin, gamma, alpha, expected
):
    assert _umean_dl14(umin, gamma, alpha) == pytest.approx(expected, rel=1e-12)
    model = _build(
        synthetic_ssp_wide,
        {
            "type": "draine_li2014",
            "umin": Fixed(umin),
            "gamma_dl": Fixed(gamma),
            "qpah": Fixed(2.5),
            "alpha_dl14": Fixed(alpha),
        },
        _ATTENUATIONS[attenuation],
    )
    got = float(model.predict_state({}).derived["dust_umean"])
    assert got == pytest.approx(expected, rel=1e-9)


def test_umean_matches_pcigale_modules():
    pytest.importorskip("pcigale")
    from pcigale.sed_modules.dl2007 import DL2007
    from pcigale.sed_modules.dl2014 import DL2014

    for umin, gamma, _ in _DL07_CASES:
        m = DL2007(name="dl2007", qpah=2.5, umin=umin, umax=1e6, gamma=gamma)
        assert _umean_dl07(umin, gamma) == pytest.approx(m.umean, rel=1e-12)
    for umin, gamma, alpha, _ in _DL14_CASES:
        m = DL2014(name="dl2014", qpah=2.5, umin=umin, alpha=alpha, gamma=gamma)
        assert _umean_dl14(umin, gamma, alpha) == pytest.approx(m.umean, rel=1e-12)


def test_umean_gradient_is_finite_and_positive_in_gamma():
    """<U> rises with gamma (the power law carries U > U_min): d<U>/dgamma is finite and > 0."""
    from tengri.components.dust.emission.templates.draine_li import _dl_umean

    for umin, alpha, umax in ((1.0, 2.0, 1.0e6), (1.0, 1.5, 1.0e7), (5.0, 1.0, 1.0e7)):

        def umean_of_gamma(gamma, umin=umin, umax=umax, alpha=alpha):
            return _dl_umean(umin, gamma, umax, alpha)

        g = jax.grad(umean_of_gamma)(0.3)
        assert bool(jnp.isfinite(g)) and float(g) > 0.0

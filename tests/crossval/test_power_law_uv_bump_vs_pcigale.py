# SPDX-License-Identifier: BSD-3-Clause
"""Power-law attenuation with a Drude UV bump against pcigale 2025.1.

Reference: ``pcigale.sed_modules.dustatt_powerlaw.alambda_av``, the Charlot & Fall
(2000) power law ``(lambda / 550 nm)**delta`` plus the Drude bump
``uv_bump(lambda; wavelength, width, amplitude)``, the same functional form as
Boquien et al. (2019, A&A 622, A103) section 4.2.

pcigale does not normalize the curve: it applies ``10**(-0.4 * Av * att)`` with
``att(550 nm) = 1 + bump(550 nm)``. tengri normalizes every screen to
k(5500 A) = 1, so ``tau_V`` keeps its meaning. The shape is therefore compared
after dividing the pcigale curve by its own value at 550 nm.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import power_law

pytestmark = pytest.mark.crossval

# Ultraviolet to near-infrared, above the Lyman limit that pcigale zeroes.
WAVE_ANGSTROM = np.geomspace(1000.0, 22000.0, 64)


def _pcigale_normalized_curve(wave_angstrom, delta, center_nm, width_nm, amplitude):
    """pcigale's A(lambda)/A(550 nm) on an Angstrom grid."""
    from pcigale.sed_modules.dustatt_powerlaw import alambda_av

    att = alambda_av(wave_angstrom / 10.0, delta, center_nm, width_nm, amplitude)
    att_v = alambda_av(np.array([550.0]), delta, center_nm, width_nm, amplitude)[0]
    return att / att_v


@pytest.mark.parametrize("delta", [-0.7, -1.3, -0.3])
@pytest.mark.parametrize(
    "amplitude, center_nm, width_nm",
    [(0.75, 217.5, 35.0), (2.0, 217.5, 35.0), (1.0, 230.0, 50.0)],
)
def test_power_law_bump_matches_pcigale_curve(delta, amplitude, center_nm, width_nm):
    expected = _pcigale_normalized_curve(WAVE_ANGSTROM, delta, center_nm, width_nm, amplitude)
    got = power_law(
        jnp.asarray(WAVE_ANGSTROM),
        dust_slope=delta,
        dust_bump_strength=amplitude,
        dust_bump_x0=center_nm / 1000.0,
        dust_bump_gamma=width_nm / 1000.0,
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10, atol=0.0)


def test_bump_defaults_are_pcigale_defaults():
    """The bump center and width default to pcigale's 217.5 nm and 35 nm."""
    import inspect

    from pcigale.sed_modules.dustatt_powerlaw import PowerLawAtt

    params = PowerLawAtt.parameters
    sig = inspect.signature(power_law).parameters
    assert sig["dust_bump_x0"].default * 1000.0 == pytest.approx(
        float(params["uv_bump_wavelength"][2]), rel=0, abs=1e-12
    )
    assert sig["dust_bump_gamma"].default * 1000.0 == pytest.approx(
        float(params["uv_bump_width"][2]), rel=0, abs=1e-12
    )
    assert sig["dust_bump_strength"].default == float(params["uv_bump_amplitude"][2])

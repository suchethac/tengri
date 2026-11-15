# SPDX-License-Identifier: BSD-3-Clause
"""Property tests (hypothesis) for four conservation and limit recipes.

Each test draws parameters from the box the registry declares for the callable
and checks an identity that must hold everywhere in that box, not at a few
hand-picked points. The callables are pure: no SSP grid, no SEDModel.

Tolerances are float64 round-off, stated per test. A property that fails is a
finding and is not loosened here.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

pytestmark = pytest.mark.conservation

from tengri.components.dust._age_mixture import interval_fractions
from tengri.components.dust.foreground import FOREGROUND_LAWS, foreground_transmission
from tengri.components.stellar.sfh.mean_sfh import dpl
from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction
from tengri.cosmology import age_at_z0 as _age_at_z0

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9

_SETTINGS = settings(deadline=None, max_examples=25, derandomize=True)

# Lookback grid for the SFH integral: 0.1 Myr to the age of the universe.
_T_LOOKBACK = np.linspace(1e5, _AGE_UNIV_YR, 2000)

# Observed-frame wavelengths [Angstrom] spanning the foreground laws' ranges.
_WAVE_OBS = np.geomspace(1.2e3, 3.0e4, 400)


@_SETTINGS
@given(
    alpha=st.floats(0.5, 3.0),
    beta=st.floats(0.3, 2.0),
    tau_gyr=st.floats(0.5, 10.0),
    log_total_mass=st.floats(8.0, 12.0),
)
def test_dpl_weights_integrate_to_formed_mass(alpha, beta, tau_gyr, log_total_mass):
    """trapezoid(dpl SFR, lookback) equals 10**log_total_mass over the prior box.

    The mass rescaling is a trapezoid over the same grid, so the identity holds
    to float64 round-off (rtol 1e-10 covers a 2000-point sum).
    """
    sfr = np.asarray(
        dpl(
            jnp.asarray(_T_LOOKBACK),
            alpha=alpha,
            beta=beta,
            tau=tau_gyr * 1e9,
            age=_AGE_UNIV_YR,
            log_total_mass=log_total_mass,
        )
    )
    formed = float(np.trapezoid(sfr, _T_LOOKBACK))
    assert np.isfinite(formed)
    assert formed == pytest.approx(10.0**log_total_mass, rel=1e-10)


@_SETTINGS
@given(
    n_boundary=st.integers(1, 5),
    n_age=st.integers(1, 6),
    data=st.data(),
)
def test_age_bin_fractions_sum_to_one(n_boundary, n_age, data):
    """Interval fractions over all age bins sum to one for every SSP node.

    The fractions are differences of a non-decreasing cumulative in [0, 1], so
    the sum telescopes to one; atol 1e-12 covers float64 round-off.
    """
    columns = [
        sorted(data.draw(st.lists(st.floats(0.0, 1.0), min_size=n_boundary, max_size=n_boundary)))
        for _ in range(n_age)
    ]
    younger = jnp.asarray(np.array(columns).T)
    fractions = np.asarray(interval_fractions(younger))
    assert fractions.shape == (n_boundary + 1, n_age)
    assert np.all(fractions >= -1e-12)
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, rtol=0.0, atol=1e-12)


@_SETTINGS
@given(
    law=st.sampled_from(sorted(FOREGROUND_LAWS)),
    rv=st.floats(2.0, 5.0),
    ebmv=st.floats(0.0, 2.0),
)
def test_foreground_transmission_in_unit_interval_and_unity_at_zero(law, rv, ebmv):
    """The foreground transmission lies in (0, 1] and is exactly 1 at E(B-V) = 0.

    Upper bound 1 + 1e-12 allows a one-ulp round-off above unity; the identity
    at zero reddening is exact (10**0 = 1), so rtol 1e-12 is far above its error.
    """
    trans = np.asarray(foreground_transmission(_WAVE_OBS, ebmv, rv, law))
    assert np.all(np.isfinite(trans))
    assert np.all(trans > 0.0)
    assert np.all(trans <= 1.0 + 1e-12)
    unity = np.asarray(foreground_transmission(_WAVE_OBS, 0.0, rv, law))
    np.testing.assert_allclose(unity, 1.0, rtol=1e-12, atol=0.0)


@_SETTINGS
@given(
    imf=st.sampled_from(["chabrier", "salpeter", "kroupa"]),
    age_gyr=st.floats(1e-3, 13.8),
)
def test_surviving_mass_fraction_in_unit_interval(imf, age_gyr):
    """The surviving-mass fraction at any age in the box lies in (0, 1].

    Upper bound 1 + 1e-12 allows a one-ulp round-off above unity.
    """
    frac = np.asarray(compute_mass_remaining_fraction(jnp.asarray([age_gyr]), imf=imf))
    assert np.all(np.isfinite(frac))
    assert np.all(frac > 0.0)
    assert np.all(frac <= 1.0 + 1e-12)

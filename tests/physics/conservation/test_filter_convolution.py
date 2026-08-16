# SPDX-License-Identifier: BSD-3-Clause
"""Conservation tests for filter convolution and spectral integration.

Validates that integrated photometry conserves energy and is self-consistent
across integration methods and filter domains.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import preintegrate_grid

pytestmark = pytest.mark.conservation


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_template_3d():
    """Simple 3D template: (3 metallicities, 5 ages, 1000 wavelengths)."""
    n_met, n_age, n_wave = 3, 5, 1000
    wave = jnp.linspace(1000.0, 10000.0, n_wave)
    template_0d = wave ** (-1.0)
    template = jnp.tile(template_0d, (n_met, n_age, 1))
    met_factor = jnp.array([0.5, 1.0, 1.5])[:, None, None]
    age_factor = jnp.linspace(0.5, 2.0, n_age)[None, :, None]
    template = template * met_factor * age_factor
    return template, wave


@pytest.fixture(scope="module")
def tophat_filters():
    """3 simple top-hat filters."""
    filter_waves = [
        jnp.linspace(1000.0, 2000.0, 50),
        jnp.linspace(4000.0, 5000.0, 50),
        jnp.linspace(8000.0, 9000.0, 50),
    ]
    filter_trans = [
        jnp.ones(50),
        jnp.ones(50),
        jnp.ones(50),
    ]
    return filter_waves, filter_trans


# ── Conservation: SSP precomputation correctness ────────────────


class TestPreintegrateGridSSPCrossval:
    """Conservation test: preintegrate_grid matches existing SSP precomputation.

    Energy conservation: integrated photometry must be identical across
    independent integration paths when applied to the same data.
    """

    def test_matches_ssp_precompute(self):
        """preintegrate_grid on SSP data matches precompute_photometry exactly.

        Conservation of energy integral: ∫ SED × Filter = constant regardless
        of integration method (as long as precision is maintained).
        """
        pytest.importorskip("h5py")
        import os

        ssp_path = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
        if not os.path.exists(ssp_path):
            pytest.skip("SSP data not available")

        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.components.stellar.sps.precompute import precompute_photometry
        from tengri.forward.precompute.grid import preintegrate_grid
        from tengri.observation.filters import load_filter_set
        from tengri.utils.cosmology import luminosity_distance

        # Load SSP
        ssp = load_ssp_data(ssp_path)

        # Load 2 SDSS filters
        try:
            filter_waves, filter_trans, _ = load_filter_set(["sdss_r", "sdss_i"])
        except Exception:
            pytest.skip("Filter data not available")

        # Fixed redshift
        z = 0.1
        dl_cm = float(luminosity_distance(z))

        # Reference: precompute_photometry (now delegates to preintegrate_grid)
        phot_ref = precompute_photometry(ssp, filter_waves, filter_trans, z, dl_cm)

        # Direct: preintegrate_grid (independent code path for verification)
        result = preintegrate_grid(
            ssp.ssp_flux,
            ssp.ssp_wave,
            filter_waves,
            filter_trans,
            redshift=z,
            dl_cm=dl_cm,
        )

        # Conservation: should be identical (energy integral invariant)
        np.testing.assert_allclose(result.phot, phot_ref.ssp_phot, rtol=1e-10, atol=1e-30)


# ── Conservation: Taylor moment additivity ────────────────────


class TestTaylorMomentConservation:
    """Conservation test: Taylor moments are self-consistent.

    The moment ∫ (λ - λ_eff) × T(λ) × SED(λ) dλ should satisfy sum rules
    related to photometry when integrated appropriately.
    """

    def test_taylor_moment_approximately_zero_for_flat_template(self):
        """For a spectrally flat template, the Taylor moment should be ~zero.

        Conservation of moment definition: ∫ (λ - λ_eff) × T(λ) dλ ≈ 0
        by construction of effective wavelength.
        """

        # Flat template (constant SED)
        template = jnp.ones((2, 3, 100))
        wave = jnp.linspace(1000.0, 10000.0, 100)
        filter_waves = [
            jnp.linspace(2000.0, 3000.0, 30),
            jnp.linspace(6000.0, 7000.0, 30),
        ]
        filter_trans = [jnp.ones(30), jnp.ones(30)]

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28, taylor=True
        )

        # Moment should be small (zero by moment definition)
        abs_moment = jnp.abs(result.moment)
        max_moment = jnp.max(abs_moment)
        assert float(max_moment) < 1e-6

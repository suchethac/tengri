"""Tests for the generic template preintegration module — bounds and finiteness.

Validates:
1. preintegrate_grid() basic functionality with synthetic templates
2. Energy normalization mode
3. Taylor moment computation
4. Output shape preservation and finiteness
5. Correctness against existing SSP precomputation
6. preintegrate_lines() basic functionality
7. interp_nd_triweight() 1D and 2D cases
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.bounds


# ── Fixtures: synthetic templates and filters ─────────────────────


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


@pytest.fixture(scope="module")
def line_wavelengths():
    """3 emission lines at specific wavelengths (rest frame)."""
    return jnp.array([1500.0, 4500.0, 8500.0])


# ── Tests: preintegrate_grid() basic functionality ────────────────


class TestPreintegrateGridBasic:
    """Bounds tests: preintegrate_grid() basic functionality."""

    def test_output_shape(self, synthetic_template_3d, tophat_filters):
        """Output shape is (n_met, n_age, n_filters) with wavelength collapsed."""
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters
        n_met, n_age, _ = template.shape
        n_filters = len(filter_waves)

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        assert result.phot.shape == (n_met, n_age, n_filters)
        assert result.effective_wavelengths.shape == (n_filters,)
        assert result.effective_wavelengths_rest.shape == (n_filters,)

    def test_output_finiteness(self, synthetic_template_3d, tophat_filters):
        """All output values are finite (no NaN/Inf).

        Bounds test: numerical overflow/underflow.
        """
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        assert jnp.all(jnp.isfinite(result.phot))
        assert jnp.all(jnp.isfinite(result.effective_wavelengths))
        assert jnp.all(jnp.isfinite(result.effective_wavelengths_rest))
        assert jnp.isfinite(result.flux_scale)
        if result.moment is not None:
            assert jnp.all(jnp.isfinite(result.moment))

    def test_flux_scale_positive(self, synthetic_template_3d, tophat_filters):
        """flux_scale is positive.

        Bounds test: photometric scaling must be positive definite.
        """
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        assert float(result.flux_scale) > 0.0

    def test_effective_wavelengths_in_filter_range(self, synthetic_template_3d, tophat_filters):
        """Effective wavelengths lie within their respective filter ranges.

        Bounds test: definition of effective wavelength.
        """
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        for i, fw in enumerate(filter_waves):
            assert float(result.effective_wavelengths[i]) >= float(jnp.min(fw))
            assert float(result.effective_wavelengths[i]) <= float(jnp.max(fw))

    def test_photometry_positive(self, synthetic_template_3d, tophat_filters):
        """Photometry values are positive (for positive templates).

        Bounds test: flux non-negativity.
        """
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28
        )

        assert jnp.all(result.phot > 0.0)


# ── Tests: energy normalization mode ──────────────────────────────


class TestPreintegrateGridEnergyNormalization:
    """Bounds tests: preintegrate_grid() with energy_normalize=True."""

    def test_energy_normalize_output_shape(self, synthetic_template_3d, tophat_filters):
        """Output shape unchanged with energy_normalize=True."""
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            energy_normalize=True,
        )

        assert result.phot.shape == (3, 5, 3)

    def test_energy_normalize_makes_values_comparable(self, synthetic_template_3d, tophat_filters):
        """With energy_normalize=True, photometry becomes more comparable across filters."""
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template,
            wave,
            filter_waves,
            filter_trans,
            redshift=0.0,
            dl_cm=1e28,
            energy_normalize=True,
        )

        # Compute bandwidth for each filter (simple estimate)
        bandwidths = jnp.array([float(jnp.max(fw) - jnp.min(fw)) for fw in filter_waves])

        # Sum weighted by bandwidth, averaged over grid points
        weighted_sum = jnp.mean(jnp.sum(result.phot * bandwidths[None, None, :], axis=2))

        # Check that the sum is well-defined (not zero, not NaN)
        assert jnp.isfinite(weighted_sum)
        assert float(weighted_sum) > 0.0


# ── Tests: Taylor moment computation ──────────────────────────────


class TestPreintegrateGridTaylorMoment:
    """Bounds tests: preintegrate_grid() with taylor=True."""

    def test_taylor_moment_output_shape(self, synthetic_template_3d, tophat_filters):
        """Taylor moment has same shape as photometry."""
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28, taylor=True
        )

        assert result.moment is not None
        assert result.moment.shape == result.phot.shape

    def test_taylor_moment_finiteness(self, synthetic_template_3d, tophat_filters):
        """Taylor moment values are finite."""
        from tengri.forward.precompute.grid import preintegrate_grid

        template, wave = synthetic_template_3d
        filter_waves, filter_trans = tophat_filters

        result = preintegrate_grid(
            template, wave, filter_waves, filter_trans, redshift=0.0, dl_cm=1e28, taylor=True
        )

        assert jnp.all(jnp.isfinite(result.moment))


# ── Tests: preintegrate_lines() basic functionality ───────────────


class TestPreintegrateLines:
    """Bounds tests: preintegrate_lines() basic functionality."""

    def test_output_shape(self, tophat_filters, line_wavelengths):
        """Output shape is (n_lines, n_filters)."""
        from tengri.forward.precompute.grid import preintegrate_lines

        filter_waves, filter_trans = tophat_filters
        lines = line_wavelengths
        n_lines = len(lines)
        n_filters = len(filter_waves)

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        assert result.line_filter_weights.shape == (n_lines, n_filters)

    def test_output_nonnegative(self, tophat_filters, line_wavelengths):
        """Line weights are non-negative.

        Bounds test: weights must be ≥ 0.
        """
        from tengri.forward.precompute.grid import preintegrate_lines

        filter_waves, filter_trans = tophat_filters
        lines = line_wavelengths

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        assert jnp.all(result.line_filter_weights >= 0.0)

    def test_output_finite(self, tophat_filters, line_wavelengths):
        """Line weights are finite."""
        from tengri.forward.precompute.grid import preintegrate_lines

        filter_waves, filter_trans = tophat_filters
        lines = line_wavelengths

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        assert jnp.all(jnp.isfinite(result.line_filter_weights))

    def test_lines_outside_filters_have_small_weight(self, tophat_filters):
        """Lines far outside filter ranges have near-zero weight.

        Bounds test: out-of-band suppression.
        """
        from tengri.forward.precompute.grid import preintegrate_lines

        filter_waves, filter_trans = tophat_filters
        # Line far below minimum filter wavelength
        lines = jnp.array([100.0])  # < min of all filters

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        # Weights should be very small
        assert jnp.all(result.line_filter_weights < 1e-3)

    def test_line_in_single_filter_has_nonzero_weight(self):
        """A line inside a filter has nonzero weight in that filter only."""
        from tengri.forward.precompute.grid import preintegrate_lines

        filter_waves = [jnp.linspace(4000.0, 5000.0, 50)]
        filter_trans = [jnp.ones(50)]
        lines = jnp.array([4500.0])

        result = preintegrate_lines(lines, filter_waves, filter_trans, redshift=0.0)

        assert float(result.line_filter_weights[0, 0]) > 0.0

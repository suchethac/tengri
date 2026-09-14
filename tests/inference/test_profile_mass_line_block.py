# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``profile_mass`` with emission-line-flux blocks.

Validates that the profiled quadratic correctly scores line fluxes (which are
linear in the mass amplitude, just as photometry and spectroscopy are).

See ``tengri.inference.mass_profile`` for the math.
``tests/inference/`` is auto-marked ``slow`` (see ``tests/conftest.py``);
run with ``-m slow``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    SEDModel,
    builders,
    generate_mock,
    recipes,
)
from tengri.inference.loss_functions import _resolve_measured_line_defs
from tengri.inference.mass_profile import (
    _LineFluxBlock,
    _predict_full_vector,
    _profile_stats,
)
from tengri.observation import LineFluxData, Observation, Photometry
from tengri.utils.physics_constants import LOG10_ZSUN
from tengri.utils.scale import whiten

pytestmark = pytest.mark.contract

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_MASS_NAME = "sfh_tsnorm_log_total_mass"
_SPEC_WAVE = jnp.linspace(4000.0, 7000.0, 40)

# Line definitions for testing. H-alpha and a few others.
_LINE_WAVELENGTHS = jnp.asarray([6562.8, 4861.3, 5006.8])
_LINE_NAMES = ("Halpha", "Hbeta", "OIII_5007")


def _model_with_nebular(ssp_data):
    """Build a model with photometry and nebular emission (wNE grid required).

    The model is built with photometry and a template LineFluxData that declares
    which lines will be fitted. The actual observed line fluxes are provided
    during fitting.
    """
    # Create a template LineFluxData with placeholder values to declare the
    # lines the model should fit. The actual measured values are swapped in
    # at fit time.
    line_template = LineFluxData.from_dict({name: (1e-16, 1e-17) for name in _LINE_NAMES})

    obs = Observation(
        photometry=Photometry.from_names(_FILTERS),
        line_fluxes=line_template,
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()  # Override to use nebular emission
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        **recipe,
    )


def _mock(model, *, seed: int, snr: float = 30.0):
    """Generate mock data (photometry + line fluxes)."""
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(seed))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=snr)
    return truth, jnp.asarray(mock["flux_obs"]), jnp.asarray(mock["noise"])


class TestLineFluxBlockMeasurement:
    """Measurement of the nebular emission grid to verify fixture."""

    def test_fixture_grid_actually_has_nebular_emission(self, ssp_data_wne):
        """Verify the wNE grid has significant nebular emission (Halpha).

        On a bare (non-wNE) grid, neb={'type':'ssp'} silently returns zero,
        and every line-related test would pass while exercising nothing.
        This gate ensures the fixture works.
        """
        wave = np.asarray(ssp_data_wne.ssp_wave)
        ssp_flux = np.asarray(ssp_data_wne.ssp_flux)
        lgmet = np.asarray(ssp_data_wne.ssp_lgmet)  # log10(Z) absolute, not Z/Zsun
        lg_age_gyr = np.asarray(ssp_data_wne.ssp_lg_age_gyr)

        # The axis order is (n_met, n_age, n_wave) -- asserted, not assumed.
        # Both axes are indexed below and a transposed read still lands on a
        # valid template, so a silent layout change would quietly measure a
        # different population instead of failing. (Measured while writing this:
        # reading it as (n_age, n_met) lands on the highest metallicity at
        # ~0.9 Myr and reports 3.41 rather than the 101.3 below -- still above
        # any sane threshold, so the error is invisible to the assertion.)
        assert ssp_flux.shape == (lgmet.size, lg_age_gyr.size, wave.size), (
            f"unexpected ssp_flux layout {ssp_flux.shape}; expected "
            f"(n_met={lgmet.size}, n_age={lg_age_gyr.size}, n_wave={wave.size})"
        )

        # Youngest age and nearest-solar metallicity, both derived from the
        # arrays rather than hardcoded: ssp_lg_age_gyr ascends on this grid
        # (-3.5 to 1.1), so the youngest bin is argmin, not index -1.
        met_idx = int(np.argmin(np.abs(lgmet - LOG10_ZSUN)))

        def halpha_over_continuum(age_idx: int) -> float:
            """Halpha peak over neighbouring continuum for one (met, age) template."""
            sed = ssp_flux[met_idx, age_idx, :]
            peak = np.max(sed[(wave >= 6562.8 - 8.0) & (wave <= 6562.8 + 8.0)])
            cont_blue = np.mean(sed[(wave >= 6400) & (wave <= 6500)])
            cont_red = np.mean(sed[(wave >= 6650) & (wave <= 6750)])
            return float(peak / (0.5 * (cont_blue + cont_red)))

        ratio = halpha_over_continuum(int(np.argmin(lg_age_gyr)))
        ratio_oldest = halpha_over_continuum(int(np.argmax(lg_age_gyr)))

        # Measured at the youngest bin, nearest-solar metallicity:
        #   wNE  ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0  101.33
        #   bare fsps_prsc_miles_chabrier                              0.99
        # A ~0.3 Myr population is where nebular emission is strongest, so the
        # two grids separate by two orders of magnitude here. The threshold
        # stays well below the wNE value rather than tracking it, since the
        # exact ratio is grid-dependent and only the presence of emission is
        # being asserted.
        assert ratio > 2.0, (
            f"Halpha/continuum ratio {ratio:.2f} at the youngest age bin means this "
            f"grid carries no nebular emission (a bare grid measures ~0.99, this wNE "
            f"grid ~101). Every line flux would be zero and every check here would "
            f"pass while exercising nothing -- see #2362."
        )

        # The emission must be age-dependent, which pins the age axis as the age
        # axis rather than merely pinning the array's shape. The shape assertion
        # above cannot catch a transposed *read*: that still lands on a valid
        # template and still measures strong emission. This does catch it,
        # because only a correctly-read age axis puts an unionizing population
        # at one end. Measured on this grid at nearest-solar metallicity:
        # 101.33 at lg_age -3.50 (0.32 Myr) against 0.999 at +1.10 (12.6 Gyr) --
        # the oldest wNE bin is indistinguishable from a bare grid, since at
        # 12.6 Gyr there is nothing left to ionize.
        assert ratio > 10.0 * ratio_oldest, (
            f"Halpha/continuum is {ratio:.2f} at the youngest age bin and "
            f"{ratio_oldest:.2f} at the oldest; nebular emission that does not fall "
            f"away with age means the age axis is not being read as the age axis."
        )

    def test_profiled_quadratic_matches_chi2_at_reference_mass(self, ssp_data_wne):
        """Profiled chi2(M) matches independent computation at the reference mass.

        This is a lighter-weight test than brute-force integration; it verifies
        that the quadratic formula chi2(a) = chi2_min + A_ref * (a - a_star)^2
        holds for the concatenated data vector (photometry + line fluxes) at
        multiple mass values. Uses the bounds midpoint as the reference mass,
        not 1 Msun, to avoid numerical underflow in float32 (see the comment
        in mass_profile.py above _LINEARITY_TOL).
        """
        model = _model_with_nebular(ssp_data_wne)
        truth, flux, noise = _mock(model, seed=0)

        # Resolve the line definitions through the shipping helper
        # (not by hand-constructing them from wavelengths).
        measured_line_defs = _resolve_measured_line_defs(model, has_line_fluxes=True)
        assert measured_line_defs is not None, (
            "model has no line catalog, so measured_line_defs should resolve "
            "from the observation's line_fluxes"
        )
        assert not model._has_line_catalog(), (
            "this test assumes the model uses a BakedIn nebular backend with no discrete catalog"
        )

        # Build the line flux data
        line_fluxes = model.measure_line_fluxes(truth, measured_line_defs)
        line_noise = jnp.abs(line_fluxes) / 30.0  # SNR = 30

        # Build the LineFluxBlock
        block = _LineFluxBlock(
            obs=line_fluxes,
            err=line_noise,
            waves=_LINE_WAVELENGTHS,
            measured_line_defs=measured_line_defs,
        )

        # Use the bounds midpoint as reference mass, not 0.0 (1 Msun).
        # The mass prior's bounds midpoint is a realistic flux scale for
        # the forward model, avoiding float32 underflow in the unit-mass flux.
        mass_prior = model.spec.get_distribution(_MASS_NAME)
        lo, hi = mass_prior.bounds
        ell_ref = 0.5 * (lo + hi)

        phys = {_MASS_NAME: ell_ref}
        for name in model.spec.free_params:
            if name != _MASS_NAME:
                phys[name] = truth[name]

        # Compute profiled stats
        A_ref, a_star, chi2_min, ell_ref_out = _profile_stats(
            model,
            _MASS_NAME,
            phys,
            flux,
            noise,
            data_type="photometry",
            line_flux_block=block,
        )

        # Verify the quadratic at a few mass values
        # Test that chi2(a) = chi2_min + A_ref * (a - a_star)^2
        ell_ref_val = float(ell_ref_out)
        for ell_test in [ell_ref_val - 0.1, ell_ref_val, ell_ref_val + 0.1]:
            phys_test = {**phys, _MASS_NAME: ell_test}
            pred_test = _predict_full_vector(model, "photometry", phys_test, line_flux_block=block)

            # Compute chi2 independently
            pred_whitened = whiten(pred_test, jnp.concatenate([noise, line_noise]))
            data_all = jnp.concatenate([flux, line_fluxes])
            data_whitened = whiten(data_all, jnp.concatenate([noise, line_noise]))
            chi2_test = jnp.sum((data_whitened - pred_whitened) ** 2)

            # Compute from quadratic formula
            a_test = 10.0 ** (ell_test - ell_ref_val)
            chi2_quad = chi2_min + A_ref * (a_test - a_star) ** 2

            rel_diff = abs(chi2_test - chi2_quad) / abs(chi2_quad)
            assert rel_diff < 1e-5, (
                f"ell={ell_test}: computed={chi2_test:.6f} "
                f"quadratic={chi2_quad:.6f} rel_diff={rel_diff:.3e}"
            )

    def test_block_is_none_reproduces_the_photometry_only_vector(self, ssp_data_wne):
        """Passing line_flux_block=None must be identical to omitting it.

        Validates that the photometry-only path is unchanged and produces
        exactly the same prediction vector.
        """
        model = _model_with_nebular(ssp_data_wne)
        truth, _, _ = _mock(model, seed=42)

        # Build minimal phys dict
        mass_prior = model.spec.get_distribution(_MASS_NAME)
        lo, hi = mass_prior.bounds
        ell_ref = 0.5 * (lo + hi)

        phys = {_MASS_NAME: ell_ref}
        for name in model.spec.free_params:
            if name != _MASS_NAME:
                phys[name] = truth[name]

        # Call with line_flux_block=None
        from tengri.inference.mass_profile import _predict_full_vector

        pred_with_none = _predict_full_vector(model, "photometry", phys, line_flux_block=None)
        pred_without_arg = _predict_full_vector(model, "photometry", phys)

        # Must be exactly equal
        assert jnp.array_equal(pred_with_none, pred_without_arg), (
            f"Passing line_flux_block=None should be identical to omitting the parameter. "
            f"Shapes: {pred_with_none.shape} vs {pred_without_arg.shape}"
        )

        # Also verify the length matches the number of bands
        assert len(pred_with_none) == len(_FILTERS), (
            f"Photometry-only prediction has {len(pred_with_none)} elements "
            f"but expected {len(_FILTERS)} bands"
        )

    def test_predicted_vector_is_photometry_then_lines(self, ssp_data_wne):
        """_predict_full_vector concatenates photometry-then-lines, not lines-then-photometry.

        This test checks the order of concatenation explicitly by computing
        photometry and line fluxes independently and verifying the concatenation
        matches what _predict_full_vector returns.
        """
        model = _model_with_nebular(ssp_data_wne)
        truth, _, _ = _mock(model, seed=99)

        # Resolve measured_line_defs
        measured_line_defs = _resolve_measured_line_defs(model, has_line_fluxes=True)
        assert measured_line_defs is not None

        # Build LineFluxBlock
        line_fluxes = model.measure_line_fluxes(truth, measured_line_defs)
        line_noise = jnp.abs(line_fluxes) / 30.0
        block = _LineFluxBlock(
            obs=line_fluxes,
            err=line_noise,
            waves=_LINE_WAVELENGTHS,
            measured_line_defs=measured_line_defs,
        )

        # Build phys dict at the bounds midpoint
        mass_prior = model.spec.get_distribution(_MASS_NAME)
        lo, hi = mass_prior.bounds
        ell_ref = 0.5 * (lo + hi)

        phys = {_MASS_NAME: ell_ref}
        for name in model.spec.free_params:
            if name != _MASS_NAME:
                phys[name] = truth[name]

        # Compute photometry and line fluxes independently, at the SAME phys
        # the function under test is evaluated at -- not at ``truth``, whose
        # mass differs from the bounds midpoint used here.
        phot = jnp.ravel(model.predict_photometry(phys))
        lines = jnp.ravel(model.measure_line_fluxes(phys, measured_line_defs))

        # Concatenate in the expected order (photometry-then-lines)
        expected_full_vector = jnp.concatenate([phot, lines])

        # Compute via _predict_full_vector with the block
        actual_full_vector = _predict_full_vector(model, "photometry", phys, line_flux_block=block)

        # ``atol=0.0`` is mandatory, not stylistic. Every entry here is a flux
        # of order 1e-29 (F_nu) to 1e-16 (line), and ``jnp.allclose``'s default
        # ``atol=1e-8`` swamps all of them: with the default this assertion
        # holds for ANY pair of vectors, including a reversed concatenation.
        def _rel(a, b):
            return jnp.max(jnp.abs(a - b) / jnp.abs(b))

        assert jnp.allclose(actual_full_vector, expected_full_vector, rtol=1e-8, atol=0.0), (
            "_predict_full_vector does not match the independently computed vector "
            f"(max relative deviation {float(_rel(actual_full_vector, expected_full_vector)):.3e})"
        )

        # Which half is which -- asserted separately so a failure says what moved.
        n_phot = len(_FILTERS)
        assert jnp.allclose(actual_full_vector[:n_phot], phot, rtol=1e-8, atol=0.0), (
            "the leading entries are not the photometry block"
        )
        assert jnp.allclose(actual_full_vector[n_phot:], lines, rtol=1e-8, atol=0.0), (
            "the trailing entries are not the line-flux block"
        )

    def test_model_line_fluxes_are_nonzero(self, ssp_data_wne):
        """Model produces nonzero, physical line fluxes for the test grid.

        This ensures the grid has nebular emission and the model can compute
        line fluxes. If this fails, the fixture or the model is broken.
        """
        model = _model_with_nebular(ssp_data_wne)
        truth, _, _ = _mock(model, seed=7)

        # Resolve measured_line_defs
        measured_line_defs = _resolve_measured_line_defs(model, has_line_fluxes=True)
        assert measured_line_defs is not None

        # Measure line fluxes
        line_fluxes = model.measure_line_fluxes(truth, measured_line_defs)

        # Unpack the three lines in order
        halpha, hbeta, oiii = line_fluxes

        # Report the measured values
        halpha_val = float(halpha)
        hbeta_val = float(hbeta)
        oiii_val = float(oiii)

        # Assertion on Halpha
        assert halpha_val > 0.0, (
            f"Halpha must be positive (emission), got {halpha_val:.4e}. Check fixture grid."
        )

        # Hbeta can be near zero or negative due to Balmer absorption,
        # so we do not assert its sign, only that it exists.
        assert jnp.isfinite(hbeta_val), f"Hbeta is not finite: {hbeta_val}"
        assert oiii_val > 0.0, f"OIII_5007 should be positive, got {oiii_val:.4e}"

        # Measured on this fixture (wNE grid, seed 7), recorded so a future
        # change of grid or backend can be compared against something:
        #     Halpha     +6.5e-16
        #     Hbeta      +1.2e-16
        #     OIII_5007  +2.3e-16
        # The load-bearing assertion is ``halpha_val > 0.0``. On a bare grid the
        # same measurement returns -8.3e-18: with no nebular emission the window
        # sees only the stellar Balmer absorption trough, so Halpha comes out
        # negative rather than merely small. That is what makes the sign, not a
        # magnitude floor, the honest discriminator between the two grids -- and
        # why no relative-magnitude assertion is made here: Halpha is the
        # largest line of the set, so any such bound passes unconditionally.

    def test_line_block_refuses_a_two_dimensional_data_vector(self, ssp_data_wne):
        """_profile_stats must raise when given 2-D data with a line block.

        The line-block path uses jnp.ravel, which silently flattens 2-D input.
        This test ensures we catch that error condition explicitly.
        """
        model = _model_with_nebular(ssp_data_wne)
        truth, flux, noise = _mock(model, seed=5)

        # Resolve measured_line_defs
        measured_line_defs = _resolve_measured_line_defs(model, has_line_fluxes=True)
        assert measured_line_defs is not None

        # Build line block
        line_fluxes = model.measure_line_fluxes(truth, measured_line_defs)
        line_noise = jnp.abs(line_fluxes) / 30.0
        block = _LineFluxBlock(
            obs=line_fluxes,
            err=line_noise,
            waves=_LINE_WAVELENGTHS,
            measured_line_defs=measured_line_defs,
        )

        # Build phys dict
        mass_prior = model.spec.get_distribution(_MASS_NAME)
        lo, hi = mass_prior.bounds
        ell_ref = 0.5 * (lo + hi)

        phys = {_MASS_NAME: ell_ref}
        for name in model.spec.free_params:
            if name != _MASS_NAME:
                phys[name] = truth[name]

        # Reshape data to 2-D (which should fail)
        flux_2d = jnp.reshape(flux, (len(flux), 1))

        # Should raise ValueError mentioning "1-D data vector"
        with pytest.raises(ValueError, match="1-D data vector"):
            _profile_stats(
                model,
                _MASS_NAME,
                phys,
                flux_2d,
                noise,
                data_type="photometry",
                line_flux_block=block,
            )

    def test_line_block_quadratic_survives_float32(self, ssp_data_wne):
        """Profiled quadratic is finite in float32 with line blocks.

        Photometry and line fluxes differ by ~13 orders of magnitude in
        physical units. The profiled quadratic whitens by error before
        squaring, so both arrive as S/N-scale numbers. This test verifies
        the computation is numerically stable in float32.
        """
        model = _model_with_nebular(ssp_data_wne)
        truth, flux, noise = _mock(model, seed=11)

        # Resolve measured_line_defs
        measured_line_defs = _resolve_measured_line_defs(model, has_line_fluxes=True)
        assert measured_line_defs is not None

        # Build line block
        line_fluxes = model.measure_line_fluxes(truth, measured_line_defs)
        line_noise = jnp.abs(line_fluxes) / 30.0
        block = _LineFluxBlock(
            obs=line_fluxes,
            err=line_noise,
            waves=_LINE_WAVELENGTHS,
            measured_line_defs=measured_line_defs,
        )

        # Build phys dict at the bounds midpoint
        mass_prior = model.spec.get_distribution(_MASS_NAME)
        lo, hi = mass_prior.bounds
        ell_ref = 0.5 * (lo + hi)

        phys = {_MASS_NAME: ell_ref}
        for name in model.spec.free_params:
            if name != _MASS_NAME:
                phys[name] = truth[name]

        # Compute reference values in float64 (outside the float32 context)
        _, a_star_f64, _, _ = _profile_stats(
            model,
            _MASS_NAME,
            phys,
            flux,
            noise,
            data_type="photometry",
            line_flux_block=block,
        )

        # Now compute in float32
        with jax.enable_x64(False):
            A_ref_f32, a_star_f32, _, _ = _profile_stats(
                model,
                _MASS_NAME,
                phys,
                flux,
                noise,
                data_type="photometry",
                line_flux_block=block,
            )

        # Check that both A_ref and a_star are finite
        assert jnp.isfinite(A_ref_f32), f"A_ref not finite in float32: {A_ref_f32}"
        assert jnp.isfinite(a_star_f32), f"a_star not finite in float32: {a_star_f32}"

        # Check that a_star agrees with float64 to reasonable tolerance
        rel_error = abs(a_star_f32 - a_star_f64) / abs(a_star_f64)
        assert rel_error < 1e-3, (
            f"a_star differs between float32 and float64: "
            f"f32={a_star_f32:.6e}, f64={a_star_f64:.6e}, rel_error={rel_error:.3e}"
        )

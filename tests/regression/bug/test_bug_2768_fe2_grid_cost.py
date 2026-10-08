# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2768: the FeII template broadening must be cheap under grad.

#2728 made the FeII pseudo-continuum independent of the caller's wavelength grid by carrying
the template on a fixed 138 061-node ln(lambda) lattice (5 km/s step) and broadening it there
with a 2^18-point FFT, all inside the differentiated path. A gradient through ``blr='analytic'``
with a free ``agn_fe2_strength`` paid that lattice and four 2^18-point transforms on every
call, 63 MFLOP of the 90 MFLOP gradient of the ``disc/adaf`` wildcard test (27 MFLOP before
#2728), whatever the disc.

The template does not depend on the width, so its spectrum is a constant, and a Gaussian of
FWHM >= 500 km/s leaves only 6 669 of the 131 073 Fourier bins above 1e-10. The broadened
template is synthesized from those bins on a lattice eight times coarser, read at the two
full-lattice nodes that bracket each caller wavelength, and the R_Fe window flux and the
total power are K-term sums against the same transfer function.

Gradient FLOPs (compiled-graph cost analysis, 1000-node caller grid):

- ``_fe2_pseudo_continuum``, d/d(fwhm, strength): 119 884 544 on main, 7 044 777 here;
  asserted <= 0.15 x the full-lattice path compiled in the same test (measured 0.06).
- ``_fe2_total_power``: 120 463 816 on main, 140 067 here; asserted <= 0.01 x the full-lattice
  path (measured 0.001).
- ``compose_l_nu``, multicolor + BLR + FeII: 205 354 976 on main, 16 547 740 here; asserted
  <= 33 000 000 (2 x measured), there being no in-process reference for a whole composition.

Accuracy is measured against the full-lattice path (``_fe2_internal_grid`` +
``_fe2_broadened_on_grid``), which is what the band-limited path replaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import blr
from tengri.components.agn.blocks.runner import compose_l_nu

pytestmark = pytest.mark.regression_bug

_FWHM_MIN = 500.0  # [km/s] narrowest width the band-limited path represents
_FWHM_MAX = 30000.0  # [km/s]
_N_FWHM_DRAWS = 50
_PROBES_PER_DRAW = 600
_RELATIVE_ACCURACY = 1.0e-6
_PEAK_ACCURACY = 1.0e-8
_NORMALIZATION_ACCURACY = 1.0e-8
_PSEUDO_CONTINUUM_RATIO = 0.15  # measured 0.06 of the full-lattice path, same process
_TOTAL_POWER_RATIO = 0.01  # measured 0.001
_COMPOSITION_FLOP_BUDGET = 33_000_000  # 2 x 16 547 740 measured; 205 354 976 on main
_CALLER_GRID = np.geomspace(1.0e3, 1.0e4, 1000)  # [Angstrom]
_COMPOSITION_GRID = np.geomspace(1.0e3, 1.0e8, 1000)  # [Angstrom]
_STRENGTHS = (0.3, 1.0, 2.5)


def _reference(wave, fwhm):
    """Full-lattice FeII: ``(L_lambda per unit strength, window flux, total)``."""
    grid, _, window_weights, _ = blr._fe2_internal_grid()
    broadened = blr._fe2_broadened_on_grid(fwhm)
    window_flux = jnp.sum(window_weights * broadened)
    on_caller = jnp.interp(wave, grid, broadened, left=0.0, right=0.0) / window_flux
    return np.asarray(on_caller), float(window_flux), float(jnp.trapezoid(broadened, grid))


def _fwhm_values():
    rng = np.random.default_rng(2768)
    draws = np.exp(rng.uniform(np.log(_FWHM_MIN), np.log(_FWHM_MAX), _N_FWHM_DRAWS))
    return np.concatenate([[_FWHM_MIN, _FWHM_MAX], draws])


class TestAccuracy:
    """The band-limited path reproduces the full-lattice one."""

    def test_spectrum_over_fwhm_range_and_strengths(self):
        rng = np.random.default_rng(1)
        probes = np.concatenate(
            [
                np.geomspace(1100.0, 8800.0, _PROBES_PER_DRAW),
                rng.uniform(1100.0, 8800.0, _PROBES_PER_DRAW),
                [900.0, 3500.0, 4434.0, 4684.0, 9000.0],
            ]
        )
        wave = jnp.asarray(probes)
        worst_relative, worst_peak = 0.0, 0.0
        for fwhm in _fwhm_values():
            ref, _, _ = _reference(wave, float(fwhm))
            for strength in _STRENGTHS:
                got = np.asarray(blr._fe2_pseudo_continuum(wave, float(fwhm), strength))
                want = strength * ref
                worst_peak = max(worst_peak, np.abs(got - want).max() / want.max())
                live = want > 1.0e-3 * want.max()
                worst_relative = max(worst_relative, (np.abs(got - want)[live] / want[live]).max())
        assert worst_relative <= _RELATIVE_ACCURACY, f"{worst_relative:.3e}"
        assert worst_peak <= _PEAK_ACCURACY, f"{worst_peak:.3e}"

    def test_rfe_window_flux_and_total_power(self):
        for fwhm in _fwhm_values():
            _, window_ref, total_ref = _reference(jnp.asarray([5000.0]), float(fwhm))
            window, total = blr._fe2_window_flux_and_total(float(fwhm))
            assert float(window) == pytest.approx(window_ref, rel=_NORMALIZATION_ACCURACY)
            assert float(total) == pytest.approx(total_ref, rel=_NORMALIZATION_ACCURACY)

    @pytest.mark.parametrize("strength", _STRENGTHS)
    def test_spectrum_integrates_to_rfe_over_the_window(self, strength):
        """sum_i w_i L_lambda(node_i) = R_Fe on the lattice the normalization is defined on."""
        grid, _, window_weights, _ = blr._fe2_internal_grid()
        for fwhm in (_FWHM_MIN, 2000.0, 12000.0):
            lam = blr._fe2_pseudo_continuum(grid, fwhm, strength)
            flux = float(jnp.sum(window_weights * lam))
            assert flux == pytest.approx(strength, rel=_NORMALIZATION_ACCURACY)

    def test_total_power_matches_the_spectrum_integral(self):
        grid, _, _, _ = blr._fe2_internal_grid()
        fwhm = 3000.0
        lam = blr._fe2_pseudo_continuum(grid, fwhm, 1.0)
        assert float(blr._fe2_total_power(fwhm, 1.0)) == pytest.approx(
            float(jnp.trapezoid(lam, grid)), rel=_NORMALIZATION_ACCURACY
        )

    def test_the_two_paths_agree_at_the_switch(self):
        """Just below the minimum width the full-lattice path runs; the two meet there."""
        wave = jnp.asarray(np.geomspace(1100.0, 8800.0, 800))
        below = np.asarray(blr._fe2_pseudo_continuum(wave, _FWHM_MIN - 1.0e-6, 1.0))
        above = np.asarray(blr._fe2_pseudo_continuum(wave, _FWHM_MIN, 1.0))
        np.testing.assert_allclose(above, below, rtol=0.0, atol=_PEAK_ACCURACY * below.max())

    def test_narrow_concrete_width_is_the_full_lattice_spectrum(self):
        wave = jnp.asarray(np.geomspace(1100.0, 8800.0, 800))
        ref, _, _ = _reference(wave, 100.0)
        got = np.asarray(blr._fe2_pseudo_continuum(wave, 100.0, 1.0))
        np.testing.assert_allclose(got, ref, rtol=1.0e-12, atol=0.0)


class TestTransforms:
    """The band-limited path stays jit/grad/vmap safe and differentiable in the width."""

    def test_jit_matches_eager_and_is_finite(self):
        wave = jnp.asarray(_CALLER_GRID)
        eager = blr._fe2_pseudo_continuum(wave, 4000.0, 1.0)
        jitted = jax.jit(lambda f: blr._fe2_pseudo_continuum(wave, f, 1.0))(4000.0)
        assert bool(jnp.all(jnp.isfinite(jitted)))
        np.testing.assert_allclose(
            np.asarray(jitted), np.asarray(eager), rtol=0.0, atol=1.0e-12 * float(eager.max())
        )

    def test_vmap_over_fwhm_matches_a_loop(self):
        wave = jnp.asarray(_CALLER_GRID)
        fwhms = jnp.array([600.0, 2500.0, 9000.0])
        batched = jax.vmap(lambda f: blr._fe2_pseudo_continuum(wave, f, 1.0))(fwhms)
        looped = jnp.stack([blr._fe2_pseudo_continuum(wave, float(f), 1.0) for f in fwhms])
        np.testing.assert_allclose(
            np.asarray(batched), np.asarray(looped), rtol=0.0, atol=1.0e-12 * float(looped.max())
        )

    @pytest.mark.parametrize("fwhm", [_FWHM_MIN, 3000.0])
    def test_fwhm_gradient_matches_the_full_lattice_gradient(self, fwhm):
        wave = jnp.asarray(_CALLER_GRID)
        weight = jnp.sin(wave / 300.0)

        def new(f):
            return jnp.sum(blr._fe2_pseudo_continuum(wave, f, 1.0) * weight)

        def old(f):
            grid, _, window_weights, _ = blr._fe2_internal_grid()
            b = blr._fe2_broadened_on_grid(f)
            on_caller = jnp.interp(wave, grid, b, left=0.0, right=0.0)
            return jnp.sum(on_caller / jnp.sum(window_weights * b) * weight)

        g_new, g_old = float(jax.grad(new)(fwhm)), float(jax.grad(old)(fwhm))
        assert np.isfinite(g_old), "a non-finite full-lattice FWHM gradient"
        assert g_old != 0.0, "an identically zero full-lattice FWHM gradient"
        assert np.isfinite(g_new), "a non-finite FWHM gradient"
        assert g_new != 0.0, "an identically zero FWHM gradient"
        assert g_new == pytest.approx(g_old, rel=1.0e-6)


def _flops(fn, x0) -> float:
    return jax.jit(jax.grad(fn)).lower(x0).compile().cost_analysis()["flops"]


class TestGradientCost:
    """Compiled gradient FLOPs, as a ratio to the full-lattice path measured in the same process.

    ``cost_analysis`` accounting (FFT flops, fusion) moves across JAX/XLA releases, so the two
    FeII graphs are asserted as a ratio against the retained full-lattice computation compiled
    in the same test: measured 0.06 (``_fe2_pseudo_continuum``) and 0.001 (``_fe2_total_power``),
    asserted at 0.15 and 0.01. A reference that shifts with the compiler shifts both sides. The
    composition has no in-process reference, so it keeps an absolute ceiling at 2 x the
    measured value (main costs 12 x that).
    """

    def test_pseudo_continuum(self):
        wave = jnp.asarray(_CALLER_GRID)
        weight = jnp.sin(wave / 300.0)

        def new(x):
            return jnp.sum(blr._fe2_pseudo_continuum(wave, x[0], x[1]) * weight)

        def full_lattice(x):
            grid, _, window_weights, _ = blr._fe2_internal_grid()
            b = blr._fe2_broadened_on_grid(x[0])
            on_caller = jnp.interp(wave, grid, b, left=0.0, right=0.0)
            return jnp.sum(x[1] * on_caller / jnp.sum(window_weights * b) * weight)

        x0 = jnp.array([5000.0, 1.0])
        flops, reference = _flops(new, x0), _flops(full_lattice, x0)
        assert flops <= _PSEUDO_CONTINUUM_RATIO * reference, (
            f"{flops:,.0f} vs full-lattice {reference:,.0f} ({flops / reference:.3f})"
        )

    def test_total_power(self):
        def full_lattice(x):
            grid, _, window_weights, _ = blr._fe2_internal_grid()
            b = blr._fe2_broadened_on_grid(x[0])
            return x[1] * jnp.trapezoid(b, grid) / jnp.sum(window_weights * b)

        x0 = jnp.array([5000.0, 1.0])
        flops = _flops(lambda x: blr._fe2_total_power(x[0], x[1]), x0)
        reference = _flops(full_lattice, x0)
        assert flops <= _TOTAL_POWER_RATIO * reference, (
            f"{flops:,.0f} vs full-lattice {reference:,.0f} ({flops / reference:.4f})"
        )

    def test_composition_with_blr_and_feii(self):
        """Analytic BLR + FeII blocks under a conserving ledger, strength and width traced."""
        wave = jnp.asarray(_COMPOSITION_GRID)

        def fn(x):
            return jnp.sum(
                compose_l_nu(
                    wave,
                    x[0],
                    agn_cos_inc=0.5,
                    agn_log_mbh=x[1],
                    agn_disc_block="multicolor",
                    agn_torus_block="none",
                    agn_attenuation_block="none",
                    agn_norm="conserving",
                    agn_nlr_block="none",
                    agn_blr_block="analytic",
                    agn_feii_block="boroson_green",
                    agn_fe2_strength=x[2],
                    agn_blr_fwhm_kms=x[3],
                )
            )

        flops = _flops(fn, jnp.array([11.5, 8.0, 1.0, 5000.0]))
        assert flops <= _COMPOSITION_FLOP_BUDGET, f"{flops:,.0f} > {_COMPOSITION_FLOP_BUDGET:,}"


class TestWidthContract:
    """The band-limited path is only reachable with a width it was built for."""

    def test_traced_narrow_width_is_nan_not_a_lowpassed_spectrum(self):
        wave = jnp.asarray(_CALLER_GRID)
        narrow = jax.jit(lambda f: blr._fe2_pseudo_continuum(wave, f, 1.0))(_FWHM_MIN - 1.0)
        power = jax.jit(lambda f: blr._fe2_total_power(f, 1.0))(_FWHM_MIN - 1.0)
        assert bool(jnp.all(jnp.isnan(narrow)))
        assert bool(jnp.isnan(power))

    def test_traced_width_at_the_minimum_is_finite(self):
        wave = jnp.asarray(_CALLER_GRID)
        at_min = jax.jit(lambda f: blr._fe2_pseudo_continuum(wave, f, 1.0))(_FWHM_MIN)
        assert bool(jnp.all(jnp.isfinite(at_min)))
        assert float(at_min.max()) > 0.0

    def test_concrete_narrow_width_is_finite_on_the_full_lattice_path(self):
        wave = jnp.asarray(_CALLER_GRID)
        narrow = blr._fe2_pseudo_continuum(wave, 100.0, 1.0)
        assert bool(jnp.all(jnp.isfinite(narrow)))
        assert float(narrow.max()) > 0.0

    def test_no_declared_fe2_or_blr_width_prior_reaches_below_the_minimum(self):
        """Any declared BLR/FeII width parameter must have a prior inside the fast path.

        None is declared today: the width is a block keyword, not a parameter.
        """
        from tengri.components.agn._params import PARAMS

        widths = [d for d in PARAMS if "fwhm" in d.name and "nlr" not in d.name]
        for decl in widths:
            prior = decl.free_prior if decl.free_prior is not None else decl.prior
            assert float(prior.lo) >= _FWHM_MIN, f"{decl.name} prior reaches {prior.lo}"

    def test_the_grammar_refuses_a_blr_width_so_a_model_never_traces_one(self):
        """``agn_blr_fwhm_kms`` cannot be freed through ``SEDModel.build``.

        The model path therefore always evaluates the FeII block at its concrete default width.
        """
        import glob
        import os

        from tengri import DEFAULT, Fixed, SEDModel, Uniform
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.observation import Observation, Photometry

        paths = sorted(glob.glob(os.path.join("data", "ssp_*.h5")))
        if not paths:
            pytest.skip("no SSP grid under data/")
        ssp = load_ssp_data(os.path.abspath(paths[0]))
        obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
        agn = {
            "type": "composable",
            "disc": {"type": "multicolor"},
            "blr": {"type": "analytic", "agn_blr_fwhm_kms": Uniform(100.0, 3000.0)},
            "all_params": Fixed(DEFAULT),
        }
        with pytest.raises(ValueError, match="Unknown key 'agn_blr_fwhm_kms'"):
            SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "const", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                agn=agn,
                redshift=Fixed(0.1),
            )

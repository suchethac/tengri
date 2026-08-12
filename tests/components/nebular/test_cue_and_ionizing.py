# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Cue emulator and ionizing spectrum fitting."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Ionizing spectrum fitting ─────────────────────────────────────


pytestmark = pytest.mark.bounds


class TestIonizingSpectrumFit:
    """Test ionizing spectrum parameter extraction from SSP spectra."""

    @pytest.fixture(scope="class")
    def ssp(self, ssp_data_fsps):
        return ssp_data_fsps

    def test_fit_returns_7_params(self, ssp):
        from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])  # solar Z, young
        result = fit_ionizing_spectrum(wave, flux)
        assert "ionspec_index1" in result
        assert "ionspec_index4" in result
        assert "ionspec_logLratio3" in result
        assert "gas_logqion" in result
        assert result["powerlaw_params"].shape == (4, 2)

    def test_fit_within_cue_ranges(self, ssp):
        from tengri.components.nebular.ionizing_spectrum import _CLIP_RANGES, fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        for key, (lo, hi) in _CLIP_RANGES.items():
            assert lo <= result[key] <= hi, f"{key}={result[key]} outside [{lo}, {hi}]"

    def test_qion_reasonable(self, ssp):
        """Q_H should be ~10^46-10^48 for young SSPs."""
        from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        assert 45 < result["gas_logqion"] < 49, (
            f"logQion={result['gas_logqion']:.1f}, expected 45-49"
        )

    def test_index1_steep(self, ssp):
        """The extreme UV slope (index1) should be steep (>5)."""
        from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        assert result["ionspec_index1"] > 5, (
            f"index1={result['ionspec_index1']:.1f}, expected >5 (steep UV)"
        )

    def test_index4_shallow(self, ssp):
        """The optical slope (index4, near Lyman limit) should be shallower."""
        from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        assert result["ionspec_index4"] < result["ionspec_index1"], (
            "index4 should be shallower than index1"
        )


# ── Precomputed ionizing params table ─────────────────────────────


class TestIonizingParamsTable:
    """Test precomputation and interpolation of ionizing params."""

    @pytest.fixture(scope="class")
    def ssp(self, ssp_data_fsps):
        return ssp_data_fsps

    def test_interpolation_within_bounds(self, ssp):
        from tengri.components.nebular.ionizing_spectrum import (
            interpolate_ionizing_params,
            precompute_ionizing_params_table,
        )

        result = precompute_ionizing_params_table(
            np.array(ssp.ssp_wave),
            np.array(ssp.ssp_flux[:3, :10, :]),  # small subset for speed
            np.array(ssp.ssp_lgmet[:3]),
        )
        ionspec, _logqion = interpolate_ionizing_params(
            jnp.array(result["ionspec_table"]),
            jnp.array(result["logqion_table"]),
            jnp.array(ssp.ssp_lgmet[:3]),
            jnp.array(ssp.ssp_lg_age_gyr[:10]) + 9.0,
            -1.85,  # solar Z
            6.5,  # 3 Myr
        )
        chex.assert_shape(ionspec, (7,))
        chex.assert_tree_all_finite(ionspec)

    @pytest.mark.regression_bug
    def test_float32_ssp_does_not_overflow(self, ssp, tmp_path, monkeypatch):
        """Float32 SSP flux must not overflow the Q_H integration (issue #458).

        SSPs may ship in float32 (e.g. BC03-from-CIGALE) to save disk. The
        Q_H integration ``trapezoid(flux * L_SUN / (h*nu), nu)`` produces
        intermediates ~ 1e30 and integrates over ~ 1e16 Hz bandwidth (~ 1e46),
        well above float32's 3.4e38 max. Without an explicit float64 cast,
        ``logqion_table`` collapses to ``inf`` for every (Z, age) bin and
        downstream Cue emits ~zero nebular SED silently.

        The test guards against regressions by re-running the fit fresh —
        bypassing both the in-memory ``_IONSPEC_TABLE_CACHE`` and the disk
        cache (#448). Without these isolations, a pre-fix cache hit would
        return finite values from a prior good run and the test would
        vacuously pass even with the bug reintroduced.
        """
        from tengri.components.nebular import ionizing_spectrum as ions_mod
        from tengri.components.nebular.ionizing_spectrum import (
            precompute_ionizing_params_table,
        )

        # Isolate from prior runs: clear in-memory cache; redirect disk
        # cache dir to a tmp path so no pre-existing .npz can be loaded.
        ions_mod._IONSPEC_TABLE_CACHE.clear()
        monkeypatch.setattr(ions_mod, "_ionspec_disk_cache_dir", lambda: tmp_path)

        # Use slice indices known to contain ionizing-bright young bins.
        # FSPS prsc-miles fixture: lgmet[0] ≈ −4, ages[0:10] cover 0.3–1 Myr,
        # well within the regime where (flux × L_sun / (h × nu)) is ~ 1e30+
        # and the trapezoid integration overflows float32 if not cast.
        wave_f32 = np.asarray(ssp.ssp_wave, dtype=np.float32)
        flux_f32 = np.asarray(ssp.ssp_flux[:2, :10, :], dtype=np.float32)
        lgmet_f32 = np.asarray(ssp.ssp_lgmet[:2], dtype=np.float32)
        assert flux_f32.dtype == np.float32, "test setup: flux must be float32"

        result = precompute_ionizing_params_table(wave_f32, flux_f32, lgmet_f32)
        logq = np.asarray(result["logqion_table"])
        assert not np.any(np.isinf(logq)), (
            "Q_H integration overflowed float32 — logqion_table contains inf"
        )
        # Young (< 10 Myr) bins of a bare-stellar SSP should yield realistic Q_H.
        assert logq.max() > 40.0, (
            f"max logqion = {logq.max():.2f}; expected > 40 for a young bare "
            f"stellar SSP. Bug is likely back."
        )

    def test_precompute_is_memoized(self, ssp_data_wne):
        """Repeat calls with the same SSP grid must hit the cache (issue #416).

        ``precompute_ionizing_params_table`` runs scipy curve-fits per (met, age)
        SSP cell; on a full grid this costs ~6 s and was re-entered on every
        ``SEDModel.build(neb={'type':'cue', ...})``. Memoization keyed on the
        SSP wavelength + metallicity grids guarantees the second call returns
        without re-fitting.
        """
        import time

        from tengri.components.nebular.ionizing_spectrum import (
            _IONSPEC_TABLE_CACHE,
            precompute_ionizing_params_table,
        )

        ssp = ssp_data_wne
        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[:3, :10, :])
        lgmet = np.array(ssp.ssp_lgmet[:3])

        # Clear any cached entry from prior tests so the first call is cold.
        _IONSPEC_TABLE_CACHE.clear()

        t0 = time.time()
        first = precompute_ionizing_params_table(wave, flux, lgmet)
        cold = time.time() - t0

        t0 = time.time()
        second = precompute_ionizing_params_table(wave, flux, lgmet)
        warm = time.time() - t0

        # Same Python object → memoization hit (not a re-fit returning
        # an equal-but-distinct dict).
        assert second is first

        # And the warm call must be at least 100x faster than the cold one;
        # in practice it's microseconds vs seconds. A generous bound keeps
        # the test stable on slow CI machines.
        assert warm < max(cold / 100.0, 0.05), (
            f"second call should be near-instant (cold={cold:.2f}s warm={warm:.4f}s)"
        )


# ── Cue backend ───────────────────────────────────────────────────


class TestCueBackend:
    """Test the Cue neural net emulator backend."""

    @pytest.fixture(scope="class")
    def backend(self):
        import os

        from tengri.components.nebular.cue import CueBackend

        weights_path = "data/cue_weights.npz"
        if not os.path.exists(weights_path):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        return CueBackend(weights_path)

    def test_load_weights(self, backend):
        assert backend.name == "cue"
        assert backend.has_free_params is True
        assert backend.weights is not None

    def test_predict_lines(self, backend):
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        assert len(wave) > 0
        assert len(lum) == len(wave)
        chex.assert_tree_all_finite(lum)

    def test_predict_continuum(self, backend):
        result = backend.predict_nebular_continuum(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        # Result is (wavelength, luminosity) tuple or just array
        if isinstance(result, tuple):
            _wave_cont, lum_cont = result
            assert len(lum_cont) > 0
            chex.assert_tree_all_finite(lum_cont)
        else:
            chex.assert_tree_all_finite(result)

    def test_halpha_positive(self, backend):
        """H-alpha should be one of the brightest lines."""
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        ha_idx = jnp.argmin(jnp.abs(wave - 6562.8))
        ha_lum = float(lum[ha_idx])
        assert ha_lum > 0, "H-alpha luminosity should be positive"

    def test_logU_affects_lines(self, backend):
        """Higher logU should produce brighter lines."""
        _, lum_low = backend.predict_nebular_line_luminosities(
            gas_logu=-3.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        _, lum_high = backend.predict_nebular_line_luminosities(
            gas_logu=-1.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        assert float(jnp.sum(lum_high)) != float(jnp.sum(lum_low)), (
            "Different logU should give different total line luminosity"
        )

    def test_gradient_through_cue(self, backend):
        """Cue predictions should be differentiable w.r.t. gas params.

        This was ``xfail(strict=True)`` — "the Cue NN architecture includes
        operations that break JAX's gradient tape ... FD gives a finite value
        (~6e36)". That diagnosis was wrong, and the ``6e36`` was the clue: it is
        an **erg/s-magnitude** number. Nothing in the network is
        non-differentiable; the backward pass was simply running at a magnitude
        where it lost precision, because the backend multiplied its [Lsun]
        output by ``L_sun`` before returning.

        #1559 moved that conversion to ``NebularSEDComponent``, so this method
        again returns [Lsun] (~1e3 here) and the gradient matches finite
        differences. The identical thing happened when the multiply was first
        removed in 2026-05-17 and the xfail was deleted then too — it came back
        when the multiply did. If it returns a third time, look at the units
        before auditing the network.
        """

        def loss(logu):
            _, lum = backend.predict_nebular_line_luminosities(
                gas_logu=logu,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
            )
            return jnp.sum(lum)

        grad_jax = float(jax.grad(loss)(-2.5))
        grad_fd = fd_grad(loss, -2.5)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )


# ── Cue with precomputed ionizing params ──────────────────────────


class TestCueWithSSP:
    """Test Cue backend with SSP-derived ionizing parameters."""

    @pytest.fixture(scope="class")
    def backend_with_ssp(self, ssp_data_fsps):
        import os

        from tengri.components.nebular.cue import CueBackend

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found")
        return CueBackend("data/cue_weights.npz", ssp_data=ssp_data_fsps)

    def test_ionspec_table_computed(self, backend_with_ssp):
        assert backend_with_ssp._ionspec_table is not None
        assert backend_with_ssp._ionspec_table.shape[2] == 7

    def test_get_ionizing_params(self, backend_with_ssp):
        ionspec, logqion = backend_with_ssp.get_ionizing_params_at(-1.85, 6.5)
        chex.assert_shape(ionspec, (7,))
        assert jnp.isfinite(logqion)
        assert float(logqion) > 40  # should have meaningful Q_H at 3 Myr

    def test_predict_with_ssp_params(self, backend_with_ssp):
        """Predict lines using SSP-derived ionizing params."""
        ionspec, logqion = backend_with_ssp.get_ionizing_params_at(-1.85, 6.5)
        _wave, lum = backend_with_ssp.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=float(logqion),
            ionspec_index1=float(ionspec[0]),
            ionspec_index2=float(ionspec[1]),
            ionspec_index3=float(ionspec[2]),
            ionspec_index4=float(ionspec[3]),
            ionspec_logLratio1=float(ionspec[4]),
            ionspec_logLratio2=float(ionspec[5]),
            ionspec_logLratio3=float(ionspec[6]),
        )
        assert float(jnp.sum(lum)) > 0, "Should produce positive line emission"


# ── Multi-build warm budget (issue #423 / #416 follow-up) ────────


class TestCueMultiBuildBudget:
    """Lock in the per-build cost when many ``SEDModel.build(neb={'type':'cue'})``
    calls share one process — the pattern the CIGALE-reproduction notebook
    (``reproduction/cigale/01_cigale.py``) exercises.

    Issue #416 reported ~6 s per build; #418 memoized
    :func:`precompute_ionizing_params_table` so the second-and-later builds
    re-use the cached scipy fit. Issue #423 alleged the fix only worked
    for the scipy step (with Cue still re-tracing per build). We could not
    reproduce that on origin/main: same-physics builds reuse the structural
    kernel cache, so the warm cost is dominated by Python-side construction.

    This test guards both regressions at once: if either the ionspec-table
    cache or the structural-kernel cache stops hitting, the warm budget
    blows up by 50–100×.
    """

    def test_six_builds_warm_budget(self, ssp_data_fsps):
        import os
        import time

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found")

        from tengri import FIXED, Fixed, SEDModel

        ssp = ssp_data_fsps

        def _build_one():
            return SEDModel.build(
                ssp_data=ssp,
                sfh={"type": "delayed", "*": FIXED},
                neb={"type": "cue", "*": FIXED},
                dust={
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "*": FIXED,
                },
                redshift=Fixed(0.0),
            )

        # Cold build: pays the ionspec-table scipy fit + first JAX trace.
        # We measure it but only use it as a sanity reference — the bound
        # is on warm builds, which is what the regression is about.
        t0 = time.time()
        _build_one()
        cold = time.time() - t0

        # Five warm builds back-to-back. With the #418 memoization and the
        # structural-kernel cache both hitting, each should be << 1 s.
        warm_times = []
        for _ in range(5):
            t0 = time.time()
            _build_one()
            warm_times.append(time.time() - t0)

        warm_total = sum(warm_times)
        # Generous bound: locally each warm build runs in ~40 ms; CI tends
        # to be 5-10× slower. Anything over 1 s/build means the cache broke.
        assert warm_total < 5.0, (
            f"Five warm Cue builds should complete in << 5 s (cold={cold:.2f}s "
            f"warm_total={warm_total:.2f}s individual={warm_times}). "
            f"Likely cause: either precompute_ionizing_params_table memoization "
            f"(issue #416 / PR #418) regressed, or the structural-kernel cache "
            f"stopped hitting on same-physics builds."
        )


# ── Kennicutt 1998 Hα calibration ─────────────────────────────────


class TestKennicutt1998Halpha:
    """Hα luminosity calibration against Kennicutt 1998 SFR relation.

    Kennicutt 1998, ARA&A, 36, 189, Eq. 2:
      SFR [Msun/yr] = L(Hα) / 1.26e41 [erg/s]
    i.e. at SFR = 1 Msun/yr, L(Hα) = 1.26e41 erg/s.

    Tests the nebular Cue backend's Hα luminosity against the canonical SFR
    calibration at an ionizing photon rate consistent with SFR=1 Msun/yr.
    The Kennicutt relation assumes Case B recombination at T=10^4 K and
    Salpeter IMF.  Here we use gas_logqion = 52.8 (Hα=1.26e41 erg/s
    from Kennicutt 1998 recombination factor α_B = 2.6e-13 cm^3/s).
    """

    @pytest.fixture(scope="class")
    def backend(self):
        import os

        from tengri.components.nebular.cue import CueBackend

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        return CueBackend("data/cue_weights.npz")

    def test_halpha_kennicutt_calibration(self, backend):
        """L(Hα) ≈ 1.26e41 erg/s at SFR=1 Msun/yr.

        Kennicutt 1998, ARA&A 36, 189, Eq. 2.
        logQ_H ≈ 52.8 s^-1 corresponds to SFR~1 Msun/yr for a Salpeter IMF
        (Kennicutt & Evans 2012, ARA&A 50, 531, Table 1).
        We allow ±50% (rtol=0.50) because the Cue emulator is trained on
        BPASS/fsps ionizing fields at fixed age/Z, not a galaxy-averaged SFR.
        The calibration provides a physical plausibility check, not a
        precision requirement.
        """
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=52.8,
        )
        from tengri.utils.physics_constants import L_SUN_CUE

        ha_idx = int(jnp.argmin(jnp.abs(wave - 6562.8)))
        # The backend returns [Lsun] (#1559); Kennicutt's calibration is in
        # erg/s, so the conversion belongs here. ``L_SUN_CUE``, not the IAU
        # value: this is Cue's own catalog, and the two differ by 0.287%.
        ha_lum = float(lum[ha_idx]) * L_SUN_CUE

        assert ha_lum > 0, "Hα luminosity must be positive"
        assert 6.3e40 < ha_lum < 3.78e41, (
            f"Hα luminosity {ha_lum:.3e} erg/s outside Kennicutt+1998 ±50% range "
            f"[6.3e40, 3.78e41] erg/s at SFR~1 Msun/yr"
        )


# ── Ionspec defaults warning ──────────────────────────────────────


class TestCueIonspecDefaultsWarning:
    """Test the UserWarning when silent young-starburst defaults are used."""

    @pytest.fixture(scope="class")
    def backend_no_ssp(self):
        import os

        from tengri.components.nebular.cue import CueBackend

        weights_path = "data/cue_weights.npz"
        if not os.path.exists(weights_path):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        # Create a fresh backend with no ssp_data — this is the dangerous case
        return CueBackend(weights_path, ssp_data=None)

    def test_warning_fires_when_no_ssp_no_overrides(self, backend_no_ssp):
        """Warning should fire when ssp_data=None AND no ionspec_* overrides."""
        import tengri.components.nebular.cue as cue_module

        # Reset the warning flag to allow re-testing
        cue_module._IONSPEC_DEFAULT_WARNED = False

        with pytest.warns(UserWarning, match="young-starburst ionizing spectrum defaults"):
            backend_no_ssp.predict_nebular_line_luminosities(
                gas_logu=-2.5,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
                # No ionspec_* overrides — triggers warning
            )

    def test_warning_fires_only_once(self, backend_no_ssp):
        """Warning should fire only once per process (module-level flag)."""
        import warnings

        import tengri.components.nebular.cue as cue_module

        # Reset the warning flag
        cue_module._IONSPEC_DEFAULT_WARNED = False

        with pytest.warns(UserWarning, match="young-starburst ionizing spectrum defaults"):
            backend_no_ssp.predict_nebular_line_luminosities(
                gas_logu=-2.5,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
            )

        # Second call should NOT warn (flag is set)
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            backend_no_ssp.predict_nebular_line_luminosities(
                gas_logu=-2.5,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
            )
        # Check that no UserWarning about defaults was raised
        assert not any(
            issubclass(w.category, UserWarning) and "young-starburst" in str(w.message)
            for w in warning_list
        )

    def test_no_warning_when_ionspec_override_provided(self, backend_no_ssp):
        """Warning should NOT fire when any ionspec_* override is provided."""
        import warnings

        import tengri.components.nebular.cue as cue_module

        # Reset the warning flag
        cue_module._IONSPEC_DEFAULT_WARNED = False

        # Providing even one ionspec_* override should suppress the warning
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            backend_no_ssp.predict_nebular_line_luminosities(
                gas_logu=-2.5,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
                ionspec_index1=10.0,  # Override one param
            )
        # Check that no UserWarning about defaults was raised
        assert not any(
            issubclass(w.category, UserWarning) and "young-starburst" in str(w.message)
            for w in warning_list
        )

    def test_no_warning_when_ssp_data_provided(self, ssp_data_fsps):
        """Warning should NOT fire when ssp_data was provided at init."""
        import os
        import warnings

        import tengri.components.nebular.cue as cue_module
        from tengri.components.nebular.cue import CueBackend

        weights_path = "data/cue_weights.npz"
        if not os.path.exists(weights_path):
            pytest.skip("Cue weights not found")

        # Reset the warning flag
        cue_module._IONSPEC_DEFAULT_WARNED = False

        # Create backend WITH ssp_data — should never warn
        backend_with_ssp = CueBackend(weights_path, ssp_data=ssp_data_fsps)

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            backend_with_ssp.predict_nebular_line_luminosities(
                gas_logu=-2.5,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
                # No ionspec_* overrides, but ssp_data was provided
            )
        # Check that no UserWarning about defaults was raised
        user_warnings_with_defaults = [
            w
            for w in warning_list
            if issubclass(w.category, UserWarning) and "young-starburst" in str(w.message)
        ]
        assert not user_warnings_with_defaults, (
            f"Unexpected warning(s): {user_warnings_with_defaults}"
        )

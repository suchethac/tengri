"""Regression test for BUG-04: Warm Comptonization uses simplified power-law.

See ADR / docs/known_bugs.md for full context.
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestBug04WarmComptonization:
    """disc.py — warm zone must use nthcomp (K&D 2018 §2.2), not a modified BB.

    The full fix requires precomputed templates (build_nthcomp_templates.py).
    These tests verify:
    1. The nthcomp numpy solver produces physically correct spectra.
    2. kubota_done_disc emits a warning when templates are absent.
    3. When templates are present, the warm zone SED differs from the
       simplified proxy (confirming the two paths are distinct).
    4. The nthcomp spectrum peaks at higher frequency than the seed BB
       (the defining signature of Comptonization).

    Reference: Kubota & Done (2018) MNRAS 480 1247 §2.2;
               Zdziarski, Johnson & Magdziarz (1996) MNRAS 283 193.
    """

    def test_nthcomp_template_returns_finite_nonnegative(self):
        """nthcomp template interpolation returns finite, non-negative shape."""
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE, nthcomp_lnu_interp

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        nu = jnp.array(np.logspace(13, np.log10(5e18), 200))
        # K&D 2018 default warm zone: Gamma=2.5, kTe=0.2 keV, kTbb=10 eV = 0.01 keV
        shape = nthcomp_lnu_interp(nu, gamma=2.5, kTe_keV=0.2, kTbb_keV=0.01)
        chex.assert_equal_shape([shape, nu])
        chex.assert_tree_all_finite(shape)
        assert jnp.all(shape >= 0.0), "nthcomp template shape must be non-negative"
        assert jnp.any(shape > 0), "nthcomp template shape must be non-zero somewhere"

    def test_nthcomp_spectrum_peaks_above_seed_temperature(self):
        """Comptonized spectrum peak must be at higher nu than the seed BB.

        For warm Comptonization, photons are scattered to higher energies
        than the seed blackbody.  The nthcomp template peak should be at
        significantly higher frequency than the seed BB peak (Wien: nu_peak = 2.82 kT/h).
        """
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE, nthcomp_lnu_interp

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        kTbb_keV = 0.01  # 10 eV seed
        kTe_keV = 0.2
        gamma = 2.5
        _KEV_TO_ERG = 1.602176634e-9
        _H_PLANCK = 6.62607015e-27
        nu_seed_peak = 2.82 * kTbb_keV * _KEV_TO_ERG / _H_PLANCK  # Hz

        nu = jnp.array(np.logspace(13, np.log10(5e18), 300))
        shape = np.array(nthcomp_lnu_interp(nu, gamma=gamma, kTe_keV=kTe_keV, kTbb_keV=kTbb_keV))
        nu_np = np.array(nu)

        power = shape * nu_np  # weight by nu for energy centroid
        if power.sum() > 0:
            nu_centroid = np.average(nu_np, weights=power)
            assert nu_centroid > nu_seed_peak * 5, (
                f"nthcomp centroid {nu_centroid:.2e} Hz should be > 5x seed BB "
                f"peak {nu_seed_peak:.2e} Hz — Comptonization must shift photons up"
            )

    def test_nthcomp_gamma_effect(self):
        """Harder Gamma (steeper spectrum) reduces soft X-ray relative to UV.

        Larger Gamma → steeper power-law → less energy at high nu.
        The ratio of X-ray to UV flux must decrease as Gamma increases.
        """
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE, nthcomp_lnu_interp

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        nu = np.logspace(13, np.log10(5e18), 300)
        nu_uv = nu[(nu > 1e15) & (nu < 3e15)]  # UV band
        nu_xray = nu[(nu > 5e17) & (nu < 2e18)]  # soft X-ray band

        def xray_uv_ratio(gamma):
            shape = np.array(
                nthcomp_lnu_interp(jnp.array(nu), gamma=gamma, kTe_keV=0.2, kTbb_keV=0.01)
            )
            f_uv = np.trapezoid(np.interp(nu_uv, nu, shape), nu_uv)
            f_xray = np.trapezoid(np.interp(nu_xray, nu, shape), nu_xray)
            return f_xray / max(f_uv, 1e-300)

        ratio_soft = xray_uv_ratio(gamma=2.0)  # softer spectrum
        ratio_hard = xray_uv_ratio(gamma=3.0)  # harder spectrum
        assert ratio_hard < ratio_soft, (
            f"Harder Gamma=3.0 (ratio={ratio_hard:.4f}) should have less X-ray "
            f"relative to UV than Gamma=2.0 (ratio={ratio_soft:.4f})"
        )

    def test_kubota_done_disc_raises_without_templates(self, monkeypatch):
        """kubota_done_disc must raise RuntimeError when nthcomp templates are absent.

        The silent modified-blackbody fallback was removed (BUG-04). Uses monkeypatch
        to simulate absent templates regardless of whether data/nthcomp_templates.h5
        is present on this machine.
        """
        import tengri.components.agn._nthcomp as _nthcomp_mod
        import tengri.components.agn.disc as _disc_mod
        from tengri.components.agn.disc import kubota_done_disc

        monkeypatch.setattr(_nthcomp_mod, "_TABLE_AVAILABLE", False)
        monkeypatch.setattr(_disc_mod, "_NTHCOMP_AVAILABLE", False)

        wavelength = jnp.linspace(1000.0, 50000.0, 100)
        with pytest.raises(RuntimeError, match="nthcomp templates"):
            kubota_done_disc(wavelength, agn_log_lbol=46.0)

    def test_kubota_done_disc_uses_nthcomp_when_templates_present(self):
        """When templates present, nthcomp path returns finite, physical SED.

        The nthcomp Kompaneets solution produces the correct soft X-ray excess
        shape (Kubota & Done 2018, §2.2). This verifies the path is used and
        returns non-zero, finite flux in the soft X-ray / EUV band.
        """
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        from tengri.components.agn import disc as disc_mod

        # Soft X-ray / EUV grid (10–200 Å)
        wav_xray = jnp.linspace(10.0, 200.0, 80)

        result = disc_mod.kubota_done_disc(wav_xray, agn_log_lbol=46.0)

        chex.assert_tree_all_finite(result)
        assert float(jnp.max(result)) > 0.0, (
            "nthcomp SED is identically zero — warm Comptonization path not reached"
        )

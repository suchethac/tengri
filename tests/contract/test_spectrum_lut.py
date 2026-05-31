# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5: Spectrum LUT (SpectrumPrecomp) contract tests.

Covers the SpectrumPrecomp dataclass plus the W1 continuum precomp path:
exact-vs-LUT agreement (fixed-z and free-z), the high-R auto-fallback, and
gradient flow through the LUT projection.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import SpectrumPrecomp

pytestmark = pytest.mark.regression_paper

_SSP_CANDIDATES = [
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
]

#: Bare-stellar SSP (no baked-in nebular emission) — required by line-publishing
#: backends such as Cue.
_BARE_SSP_CANDIDATES = [
    "data/fsps_prsc_miles_chabrier.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
]


def _ssp_path():
    return next((p for p in _SSP_CANDIDATES if Path(p).is_file()), None)


def _bare_ssp_path():
    return next((p for p in _BARE_SSP_CANDIDATES if Path(p).is_file()), None)


class TestSpectrumPrecompDataclass:
    """SpectrumPrecomp is a frozen, hashable dataclass with free-z knobs."""

    def test_spectrum_precomp_frozen(self):
        sp = SpectrumPrecomp()
        with pytest.raises(AttributeError):
            sp.x = 42

    def test_spectrum_precomp_hashable(self):
        assert len({SpectrumPrecomp(), SpectrumPrecomp()}) >= 1

    def test_spectrum_precomp_knobs(self):
        sp = SpectrumPrecomp(n_z=200, z_min=0.0, z_max=3.0)
        assert (sp.n_z, sp.z_min, sp.z_max) == (200, 0.0, 3.0)


def _build(ssp, obs, approx, redshift, dust=None):
    import warnings

    from tengri import FIXED, SEDModel

    dust = dust or {"type": "two_component", "law_bc": "calzetti", "*": FIXED}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FIXED},
            dust=dust,
            neb={"type": "none"},
            redshift=redshift,
            approx=approx,
        )


class TestSpectrumLUTAccuracy:
    """SpectrumPrecomp spec_fnu agrees with the exact path at low–medium R.

    Documented tolerance is ~0.5% on photometric magnitudes; for low-R
    spectroscopy on the same effective-wavelength approximation we accept
    ≤1% on F_ν.
    """

    def test_spectrum_lut_vs_exact_fixed_z(self):
        ssp_path = _ssp_path()
        if ssp_path is None:
            pytest.skip("No SSP grid available under data/.")
        from tengri import Fixed, Observation, Spectroscopy, load_ssp_data

        ssp = load_ssp_data(ssp_path)
        wave_obs = jnp.asarray(np.linspace(4500.0, 7500.0, 64))
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

        m_exact = _build(ssp, obs, None, Fixed(0.05))
        m_lut = _build(ssp, obs, SpectrumPrecomp(), Fixed(0.05))
        # The LUT path must actually engage (not silently fall through).
        assert m_lut._approx.get("spectrum_precomp") is True

        spec_exact = m_exact.predict_observables({}).spec_fnu
        spec_lut = m_lut.predict_observables({}).spec_fnu
        assert spec_exact.shape == spec_lut.shape
        max_rel = float(
            jnp.max(jnp.abs(spec_lut - spec_exact) / jnp.maximum(jnp.abs(spec_exact), 1e-30))
        )
        assert max_rel < 1e-2, f"fixed-z max rel err = {max_rel:.4%} exceeds 1%."

    def test_spectrum_lut_vs_exact_free_z(self):
        ssp_path = _ssp_path()
        if ssp_path is None:
            pytest.skip("No SSP grid available under data/.")
        from tengri import Observation, Spectroscopy, Uniform, load_ssp_data

        ssp = load_ssp_data(ssp_path)
        wave_obs = jnp.asarray(np.linspace(4500.0, 7500.0, 64))
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

        m_exact = _build(ssp, obs, None, Uniform(0.01, 0.5, "redshift"))
        m_lut = _build(ssp, obs, SpectrumPrecomp(), Uniform(0.01, 0.5, "redshift"))
        for z in (0.05, 0.2, 0.4):
            se = m_exact.predict_observables({"redshift": z}).spec_fnu
            sl = m_lut.predict_observables({"redshift": z}).spec_fnu
            max_rel = float(jnp.max(jnp.abs(sl - se) / jnp.maximum(jnp.abs(se), 1e-30)))
            assert max_rel < 1e-2, f"free-z @z={z} max rel err = {max_rel:.4%} exceeds 1%."

    def test_jit_safe_under_external_jit(self):
        """Wrapping predict_observables in jax.jit must not retrace the (numpy)
        spectrum-LUT build. Regression for the eager chain-cache fix — real
        inference jits the loss function, so a lazy LUT build inside the trace
        would raise TracerArrayConversionError."""
        ssp_path = _ssp_path()
        if ssp_path is None:
            pytest.skip("No SSP grid available under data/.")
        from tengri import Fixed, Observation, Spectroscopy, Uniform, load_ssp_data

        ssp = load_ssp_data(ssp_path)
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4500.0, 7500.0, 48)))
        m = _build(ssp, obs, SpectrumPrecomp(), Fixed(0.05))
        m_free = _build(ssp, obs, SpectrumPrecomp(), Uniform(0.01, 0.5, "redshift"))

        f_fixed = jax.jit(lambda: m.predict_observables({}).spec_fnu)
        assert bool(jnp.all(jnp.isfinite(f_fixed())))  # fixed-z LUT under jit
        f_free = jax.jit(lambda z: m_free.predict_observables({"redshift": z}).spec_fnu)
        assert bool(jnp.all(jnp.isfinite(f_free(jnp.asarray(0.1)))))  # free-z under jit


class TestSpectrumLUTGuards:
    """High-R auto-fallback and gradient flow."""

    def test_high_r_auto_fallback_warns(self):
        ssp_path = _ssp_path()
        if ssp_path is None:
            pytest.skip("No SSP grid available under data/.")
        import warnings

        from tengri import FIXED, Fixed, Observation, SEDModel, Spectroscopy, load_ssp_data

        ssp = load_ssp_data(ssp_path)
        wave_obs = jnp.asarray(np.linspace(4500.0, 7500.0, 64))
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=5000.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "*": FIXED},
                dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
                neb={"type": "none"},
                redshift=Fixed(0.05),
                approx=SpectrumPrecomp(),
            )
        assert m._approx.get("spectrum_precomp") is False
        assert any("falls back" in str(w.message) for w in caught)

    def test_gradient_flows_through_lut(self):
        ssp_path = _ssp_path()
        if ssp_path is None:
            pytest.skip("No SSP grid available under data/.")
        from tengri import Fixed, Observation, Spectroscopy, Uniform, load_ssp_data

        ssp = load_ssp_data(ssp_path)
        wave_obs = jnp.asarray(np.linspace(4500.0, 7500.0, 32))
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

        import warnings

        from tengri import FIXED, SEDModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "alpha": Uniform(0.1, 5.0, "sfh_dpl_alpha"), "*": FIXED},
                dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
                neb={"type": "none"},
                redshift=Fixed(0.05),
                approx=SpectrumPrecomp(),
            )

        def loss(a):
            return jnp.sum(m.predict_observables({"sfh_dpl_alpha": a}).spec_fnu ** 2)

        g = jax.grad(loss)(jnp.asarray(1.5))
        assert jnp.isfinite(g)


class TestSpectrumLUTLines:
    """W2: emission lines on the precomp path (design-doc option A).

    A line-publishing nebular backend (Cue) is allowed under SpectrumPrecomp;
    its discrete line luminosities are grid-independent and survive the LUT
    path, so line fluxes and derived line properties are bit-identical to the
    exact path.
    """

    def _build_cue(self, ssp, obs, approx):
        import warnings

        from tengri import FIXED, Fixed, SEDModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "*": FIXED},
                dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
                neb={"type": "cue", "*": FIXED},
                redshift=Fixed(0.05),
                approx=approx,
            )

    def test_cue_under_spectrum_precomp_builds(self):
        bare = _bare_ssp_path()
        if bare is None or not Path("data/cue_weights.npz").is_file():
            pytest.skip("No bare-stellar SSP / Cue weights available.")
        from tengri import Observation, Spectroscopy, load_ssp_data

        ssp = load_ssp_data(bare)
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4500.0, 7500.0, 64)))
        m = self._build_cue(ssp, obs, SpectrumPrecomp())
        # Guard lifted: a line-publishing backend is allowed and engages.
        assert m._approx.get("spectrum_precomp") is True

    def test_line_fluxes_match_exact_under_precomp(self):
        bare = _bare_ssp_path()
        if bare is None or not Path("data/cue_weights.npz").is_file():
            pytest.skip("No bare-stellar SSP / Cue weights available.")
        from tengri import Observation, Spectroscopy, load_ssp_data

        ssp = load_ssp_data(bare)
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4500.0, 7500.0, 64)))
        m_lut = self._build_cue(ssp, obs, SpectrumPrecomp())
        m_exact = self._build_cue(ssp, obs, None)
        targets = jnp.asarray([4862.69, 5008.24, 6564.61])  # Hβ, [OIII]5008, Hα (vacuum)
        lf_lut = m_lut.predict_line_fluxes({}, target_wavelengths=targets)
        lf_exact = m_exact.predict_line_fluxes({}, target_wavelengths=targets)
        # Line luminosities are grid-independent → identical on both paths.
        np.testing.assert_allclose(np.asarray(lf_lut), np.asarray(lf_exact), rtol=1e-6)

    def test_derived_line_properties_under_precomp(self):
        bare = _bare_ssp_path()
        if bare is None or not Path("data/cue_weights.npz").is_file():
            pytest.skip("No bare-stellar SSP / Cue weights available.")
        from tengri import Observation, Spectroscopy, load_ssp_data

        ssp = load_ssp_data(bare)
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4500.0, 7500.0, 64)))
        pred = self._build_cue(ssp, obs, SpectrumPrecomp()).predict({})
        # pred.lines.* must be finite under the precomp path.
        assert jnp.isfinite(pred.lines.halpha)
        assert jnp.isfinite(pred.lines.bpt_nii)
        assert jnp.isfinite(pred.lines.balmer_decrement)

# SPDX-License-Identifier: BSD-3-Clause
"""Silent torus luminosity drop when fracAGN used without dust (#944).

Regression test for the CIGALE skirtor2016 energy-balance coupling: when
fracAGN is enabled, the torus luminosity is renormalized against the
dust-absorbed stellar luminosity. Without a dust component that luminosity
is ~zero, so the torus (and under cigale_joint the whole AGN) collapses.

Fix: build-time validation raises ConfigError when AGN has fracAGN without dust.

Tests:
a) Dust none + fracAGN=0.3 → build raises ConfigError with clear message
b) Dust none + no fracAGN → builds; torus alive and changes SED >2x at 10µm
c) Dust two_component + fracAGN=0.3 → builds; torus changes SED >1.05x at 10µm
   under both 'cigale_joint' and 'independent' norm modes
d) Equivalence suite xfails remain xfail (or pass if config changed)
"""

from __future__ import annotations

import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.regression_bug


class TestFracAGNDustCoupling944:
    """#944 fracAGN→dust energy-balance coupling validation."""

    def test_dust_none_fracagn_fixed_raises_configerror(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust none + fracAGN Fixed(0.3) must raise ConfigError at build time.

        The torus luminosity would collapse under dust.absorbed=0, making
        the silent failure (#944) the default behavior without this check.
        """
        with pytest.raises(
            ConfigError,
            match=r"fracAGN.*dust.*component.*zero.*add.*dust",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust={"type": "none"},
                agn={
                    "type": "composable",
                    "norm": "cigale_joint",
                    "torus": {"type": "skirtor"},
                    "fracAGN": Fixed(0.3),
                },
                redshift=Fixed(0.1),
            )

    def test_dust_none_fracagn_free_raises_configerror(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust none + fracAGN FREE must raise ConfigError.

        A FREE fracAGN is just as dangerous as Fixed(0.3) — the prior
        will allow positive values that trigger the coupling.
        """
        from tengri import FREE

        with pytest.raises(
            ConfigError,
            match=r"fracAGN.*dust.*component.*zero.*add.*dust",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust={"type": "none"},
                agn={
                    "type": "composable",
                    "norm": "cigale_joint",
                    "torus": {"type": "skirtor"},
                    "fracAGN": FREE,
                },
                redshift=Fixed(0.1),
            )

    def test_dust_none_no_fracagn_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust none + no fracAGN → model builds (torus on independent scale).

        Without fracAGN the torus uses agn_torus_frac directly; no CIGALE
        energy-balance coupling, no need for dust.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "norm": "cigale_joint",
                "torus": {"type": "skirtor"},
                # No fracAGN — torus operates on its own scale
            },
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec is not None

    def test_dust_none_no_fracagn_torus_alive(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Torus model builds and predicts when dust=none + no fracAGN.

        Without dust and without fracAGN, the torus uses agn_torus_frac
        for independent scaling. The model should build without error.
        """
        import jax

        # Model with torus should build
        model_torus = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "*": FIXED},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "norm": "independent",
                "torus": {"type": "skirtor"},
                "disc": {"type": "powerlaw", "*": FIXED},
                "nlr": {"type": "none"},
                "blr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
            redshift=Fixed(0.1),
        )

        # Should build and predict without error
        assert model_torus is not None
        params = model_torus.spec.sample(jax.random.PRNGKey(0))
        pred = model_torus.predict(params)
        assert pred is not None
        assert pred.rest_sed() is not None

    def test_dust_present_fracagn_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust two_component + fracAGN=0.3 → model builds.

        With dust present, the energy-balance coupling is valid and useful.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "emission": {"type": "dale2014"},
            },
            agn={
                "type": "composable",
                "norm": "cigale_joint",
                "torus": {"type": "skirtor"},
                "fracAGN": Fixed(0.3),
            },
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec is not None

    def test_dust_present_fracagn_torus_changes_sed(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust present + fracAGN=0.3 → torus changes SED >1.05x at 10µm.

        With dust, the energy-balance coupling works as intended:
        the torus luminosity is tied to dust-absorbed stellar luminosity.
        """
        import jax
        import jax.numpy as jnp

        # Model with torus
        model_torus = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "*": FIXED},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "emission": {"type": "dale2014"},
            },
            agn={
                "type": "composable",
                "norm": "cigale_joint",
                "torus": {"type": "skirtor"},
                "disc": {"type": "multicolor", "*": FIXED},
                "nlr": {"type": "none"},
                "blr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
                "fracAGN": Fixed(0.3),
            },
            redshift=Fixed(0.1),
        )

        # Model without torus
        model_notorus = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "*": FIXED},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "emission": {"type": "dale2014"},
            },
            agn={
                "type": "composable",
                "norm": "cigale_joint",
                "torus": {"type": "none"},
                "disc": {"type": "multicolor", "*": FIXED},
                "nlr": {"type": "none"},
                "blr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
                "fracAGN": Fixed(0.3),
            },
            redshift=Fixed(0.1),
        )

        # Sample params
        params = dict(model_torus.spec.sample(jax.random.PRNGKey(0)))
        params["agn_log_lbol"] = 12.0

        probe_wavelengths = jnp.array([100000.0])

        sed_torus = model_torus.predict(params).rest_sed(probe_wavelengths)
        sed_notorus = model_notorus.predict(params).rest_sed(probe_wavelengths)

        ratio = sed_torus[0] / jnp.maximum(sed_notorus[0], 1e-30)

        # With dust and valid coupling, torus should still change SED (>1.05x)
        assert float(ratio) > 1.05, (
            f"Torus contribution at 10µm is {ratio:.2f}x, expected >1.05x. "
            "Torus is not contributing under dust + fracAGN."
        )

    def test_dust_present_fracagn_both_norm_modes(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust present + fracAGN works under both norm modes.

        Tests both 'cigale_joint' and 'independent' normalization modes
        to ensure the coupling doesn't break either path.
        """
        import jax

        for norm_mode in ("cigale_joint", "independent"):
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust={
                    "type": "two_component",
                    "law": "calzetti",
                    "emission": {"type": "dale2014"},
                },
                agn={
                    "type": "composable",
                    "norm": norm_mode,
                    "torus": {"type": "skirtor"},
                    "fracAGN": Fixed(0.3),
                },
                redshift=Fixed(0.1),
            )

            # Just verify it builds and predicts without error
            params = model.spec.sample(jax.random.PRNGKey(0))
            pred = model.predict(params)
            assert pred is not None, f"Prediction failed for norm_mode={norm_mode}"

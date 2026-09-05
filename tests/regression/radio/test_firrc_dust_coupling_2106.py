# SPDX-License-Identifier: BSD-3-Clause
"""Silent zero radio SED when FIRRC blocks used without dust (#2106).

Regression test for the FIR-radio correlation (FIRRC) normalization: when
any of the three FIRRC models (bell2003, delvecchio2021, mccheyne2022) is
enabled, the synchrotron luminosity is normalized against L_ir, the
dust-absorbed stellar luminosity. Without a dust component, that luminosity
is ~zero, so the radio SED silently becomes all zeros with no signal to
the user that the configuration is invalid.

Fix: build-time validation raises ConfigError when FIRRC is used without dust.

Tests:
a) Dust none + bell2003 → build raises ConfigError with clear message
b) Dust none + delvecchio2021 → build raises ConfigError
c) Dust none + mccheyne2022 → build raises ConfigError
d) Dust none + powerlaw (AGN-only, sfr_mode='none') → builds successfully
e) Dust none + dpl (AGN-only, sfr_mode='none') → builds successfully
f) Dust two_component + bell2003 → builds and predicts nonzero radio SED
g) Dust two_component + delvecchio2021 → builds and predicts nonzero radio SED
h) Dust two_component + mccheyne2022 → builds and predicts nonzero radio SED
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.regression_bug


class TestFIRRCDustCoupling2106:
    """#2106 FIRRC→dust coupling validation."""

    # Test a) Dust none + bell2003 raises ConfigError
    def test_dust_none_bell2003_raises_configerror(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust none + bell2003 FIRRC must raise ConfigError at build time.

        The synchrotron luminosity would be all-zero under L_ir=0, making
        the silent failure (#2106) the default behavior without this check.
        """
        with pytest.raises(
            ConfigError,
            match=r"FIRRC.*radio_sfr_mode.*L_ir.*dust.*component.*add.*dust",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust_attenuation={"type": "none"},
                radio={"sf": {"type": "bell2003"}},
                redshift=Fixed(0.1),
            )

    # Test b) Dust none + delvecchio2021 raises ConfigError
    def test_dust_none_delvecchio2021_raises_configerror(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust none + delvecchio2021 FIRRC must raise ConfigError at build time."""
        with pytest.raises(
            ConfigError,
            match=r"FIRRC.*radio_sfr_mode.*L_ir.*dust.*component.*add.*dust",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust_attenuation={"type": "none"},
                radio={"sf": {"type": "delvecchio2021"}},
                redshift=Fixed(0.1),
            )

    # Test c) Dust none + mccheyne2022 raises ConfigError
    def test_dust_none_mccheyne2022_raises_configerror(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust none + mccheyne2022 FIRRC must raise ConfigError at build time."""
        with pytest.raises(
            ConfigError,
            match=r"FIRRC.*radio_sfr_mode.*L_ir.*dust.*component.*add.*dust",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust_attenuation={"type": "none"},
                radio={"sf": {"type": "mccheyne2022"}},
                redshift=Fixed(0.1),
            )

    # Test d) Dust none + powerlaw (AGN-only, sfr_mode='none') still builds
    def test_dust_none_agn_only_powerlaw_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust none + agn_radio_model='powerlaw' + sfr_mode='none' builds.

        Powerlaw AGN radio does not depend on L_ir, so it is safe without dust.
        This is negative coverage: ensure the fix does not over-trigger.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust_attenuation={"type": "none"},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "powerlaw"},
            },
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec.radio_sfr_mode == "none"

    # Test e) Dust none + dpl (AGN-only, sfr_mode='none') still builds
    def test_dust_none_agn_only_dpl_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust none + agn_radio_model='dpl' + sfr_mode='none' builds.

        Double power-law AGN radio does not depend on L_ir, so it is safe without dust.
        This is negative coverage: ensure the fix does not over-trigger.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust_attenuation={"type": "none"},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "dpl"},
            },
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec.radio_sfr_mode == "none"

    # Test f) Dust two_component + bell2003 builds and predicts
    def test_dust_present_bell2003_builds_and_predicts(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust two_component + bell2003 FIRRC builds and predicts.

        With dust present, L_ir is nonzero and the FIRRC is valid.
        """
        import jax

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014_cigale"},
            radio={"sf": {"type": "bell2003"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

        # Predict and check for finite/nonzero radio
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        pred = model.predict(params)
        sed = pred.rest_sed()
        assert jnp.all(jnp.isfinite(sed)), "SED contains non-finite values"
        # Radio should be nonzero somewhere in the radio bands
        assert jnp.any(sed > 0.0), "Radio SED is identically zero (silent failure)"

    # Test g) Dust two_component + delvecchio2021 builds and predicts
    def test_dust_present_delvecchio2021_builds_and_predicts(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust two_component + delvecchio2021 FIRRC builds and predicts."""
        import jax

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014_cigale"},
            radio={"sf": {"type": "delvecchio2021"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

        # Predict and check for finite/nonzero radio
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        pred = model.predict(params)
        sed = pred.rest_sed()
        assert jnp.all(jnp.isfinite(sed)), "SED contains non-finite values"
        # Radio should be nonzero somewhere
        assert jnp.any(sed > 0.0), "Radio SED is identically zero (silent failure)"

    # Test h) Dust two_component + mccheyne2022 builds and predicts
    def test_dust_present_mccheyne2022_builds_and_predicts(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Dust two_component + mccheyne2022 FIRRC builds and predicts."""
        import jax

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014_cigale"},
            radio={"sf": {"type": "mccheyne2022"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

        # Predict and check for finite/nonzero radio
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        pred = model.predict(params)
        sed = pred.rest_sed()
        assert jnp.all(jnp.isfinite(sed)), "SED contains non-finite values"
        # Radio should be nonzero somewhere
        assert jnp.any(sed > 0.0), "Radio SED is identically zero (silent failure)"

    # Additional test: free FIRRC params with dust should work
    def test_dust_present_free_firrc_params_predict(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Free FIRRC params with dust present should build and predict."""
        import jax

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014_cigale"},
            radio={"sf": {"type": "delvecchio2021", "delv_q0": Uniform(2.4, 3.1)}},
            redshift=Fixed(0.1),
        )
        assert "radio_delv_q0" in model.spec.free_params

        # Predict with the free parameter supplied
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        pred = model.predict(params)
        sed = pred.rest_sed()
        assert jnp.all(jnp.isfinite(sed)), "SED with free FIRRC params contains non-finite"
        assert jnp.any(sed > 0.0), "Radio SED with free FIRRC params is identically zero"

    # Negative coverage: no radio group at all + dust off must build.
    # radio_sfr_mode defaults to "bell2003" even when radio is disabled, so
    # the validator must first check the radio flag (the dale2014 sibling
    # validator guards the same way).
    def test_radio_disabled_dust_off_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Dust none + no radio group builds: FIRRC validation must not fire."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust_attenuation={"type": "none"},
            redshift=Fixed(0.1),
        )
        assert model is not None

# SPDX-License-Identifier: BSD-3-Clause
"""Dale2014 template embeds SF radio; combining with SF radio block double-counts (~#1970).

The Dale+2014 dust emission template (dale2014) contains an embedded
star-forming radio synchrotron continuum rising out to 2.2459e9 Å = 1.335 GHz.
The Dale2014Cigale variant (dale2014_cigale) strips the radio tail beyond
7.727e7 Å per CIGALE convention.

When dale2014 is combined with an active SF radio block (radio_sfr_mode != 'none'),
the synchrotron is double-counted in rest_sed between ~1.34 and ~10 GHz (3–22 cm), measured as
a 2x composition attenuation at the template edge and a −4.93 slope vs. +0.77 expected.

Fix: build-time validation raises ConfigError when BOTH:
  (a) dust.emission == 'dale2014' (the radio-bearing variant), AND
  (b) radio is enabled with SF synchrotron active (radio=True and radio_sfr_mode != 'none')

The error message names the remedy: use dust.emission='dale2014_cigale' instead.

Tests:
a) dale2014 + SF radio (bell2003) raises ConfigError with the remedy in the message
b) dale2014 + SF radio (delvecchio2021) raises too (any SF block, not just bell2003)
c) dale2014 + AGN-only radio (sf='none') builds fine
d) dale2014_cigale + SF radio (bell2003) builds fine AND composed SED is smooth
e) Data-contract pin: dale2014_cigale templates are zero beyond 1e8 Å while the
   plain dale2014 file still carries its embedded radio tail (the guard's premise)

Documented non-guard: the radio component's free-free term (active only when a
nebular component publishes log_nion; there is no grammar knob for it) overlaps
the template's embedded thermal radio at the <~10% level near 1.4 GHz. That
combination stays legal — refusing it would block dale2014 + AGN radio + nebular
with no grammar-reachable remedy — and is documented on both Dale components.
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.regression_bug


class TestDale2014RadioDoubleCount1970:
    """#1970 Dale2014 embedded SF radio double-count guard."""

    def test_dale2014_sf_radio_bell2003_raises_configerror(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """dale2014 + SF radio (bell2003) must raise ConfigError at build time.

        The embedded radio tail in dale2014 and the SF synchrotron block
        would double-count, causing 2x attenuation in the radio band.
        """
        with pytest.raises(
            ConfigError,
            match=r"Dale.*2014.*embeds.*radio.*double.*dale2014_cigale",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                },
                dust_emission={"type": "dale2014"},
                radio={"sf": {"type": "bell2003"}},
                redshift=Fixed(0.1),
            )

    def test_dale2014_sf_radio_delvecchio_raises_configerror(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """dale2014 + SF radio (delvecchio2021) must raise ConfigError.

        Any SF radio model combined with dale2014 will double-count.
        """
        with pytest.raises(
            ConfigError,
            match=r"Dale.*2014.*embeds.*radio.*double.*dale2014_cigale",
        ):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                },
                dust_emission={"type": "dale2014"},
                radio={"sf": {"type": "delvecchio2021"}},
                redshift=Fixed(0.1),
            )

    def test_dale2014_agn_radio_only_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """dale2014 + AGN-only radio (sf='none') builds fine.

        With SF synchrotron disabled (the default), no double-count occurs:
        the template's embedded SF radio is inert, and AGN radio is independent.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014"},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "powerlaw"},
            },
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec is not None

    def test_dale2014_no_radio_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """dale2014 without radio (radio=False) builds fine.

        No radio at all means no conflict.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014"},
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec is not None

    def test_dale2014_cigale_sf_radio_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """dale2014_cigale + SF radio (bell2003) builds fine.

        The CIGALE variant has the radio tail stripped, so combining with
        an SF radio block is safe and intended.
        """
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "const"},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
            },
            dust_emission={"type": "dale2014_cigale"},
            radio={"sf": {"type": "bell2003"}},
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model.spec is not None

    def test_dale2014_cigale_sf_radio_composed_sed_is_smooth(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """dale2014_cigale + SF radio produces smooth composed SED.

        The smoothness test: between 0.5 and 30 GHz (1e8–6e9 Å in wavelength),
        every node-to-node log-log slope should be within 0.15 of the median.
        This would have caught #1970: the unguarded dale2014 combo shows
        a −4.93 slope at the template edge vs. +0.77 expected.
        """
        import jax
        import jax.numpy as jnp

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

        params = model.spec.sample(jax.random.PRNGKey(0))
        pred = model.predict(params)
        sed = pred.rest_sed()

        # Get wavelengths in the radio range: 0.5 to 30 GHz → 1e8 to 6e9 Å
        wave_rest = pred.wave_rest
        radio_mask = (wave_rest >= 1e8) & (wave_rest <= 6e9)

        if jnp.sum(radio_mask) < 2:
            pytest.skip("Insufficient wavelength nodes in the radio band for smoothness test")

        radio_wave = wave_rest[radio_mask]
        radio_sed = sed[radio_mask]

        # Compute log-log slopes between consecutive nodes
        log_wave = jnp.log10(radio_wave)
        log_sed = jnp.log10(jnp.maximum(radio_sed, 1e-30))

        slopes = jnp.diff(log_sed) / jnp.diff(log_wave)

        # Smoothness criterion: |slope - median| < 0.15
        median_slope = jnp.median(slopes)
        max_deviation = jnp.max(jnp.abs(slopes - median_slope))

        assert float(max_deviation) < 0.15, (
            f"Composed SED is not smooth in radio band: max deviation {float(max_deviation):.3f} "
            f"from median slope {float(median_slope):.3f}. This suggests double-counting "
            f"or other incompatibility at the template edge."
        )

    def test_dale2014_cigale_templates_zero_beyond_1e8_angstrom(self):
        """dale2014_cigale radio tail is zero beyond 1e8 Å (CIGALE convention).

        Data contract: the templates must be strictly zero in the radio band
        per CIGALE's convention of not extending the radio synchrotron. The
        plain dale2014 file, by contrast, must still CARRY its embedded radio
        tail — the guard exists precisely because that tail is real.
        """
        from pathlib import Path

        import h5py
        import numpy as np

        data_dir = Path(__file__).resolve().parents[3] / "data"

        cigale_path = data_dir / "dale2014_templates_cigale.h5"
        assert cigale_path.exists(), f"tracked template file missing: {cigale_path}"
        with h5py.File(cigale_path, "r") as f:
            wavelength = np.asarray(f["wavelength_aa"][:])
            templates = np.asarray(f["templates_sf"][:])
        radio_region = wavelength > 1e8
        assert radio_region.any(), "template grid unexpectedly ends below 1e8 Å"
        radio_templates = templates[:, radio_region]
        assert np.allclose(radio_templates, 0.0), (
            f"dale2014_cigale templates are not zero beyond 1e8 Å; "
            f"max |value| = {np.max(np.abs(radio_templates)):.2e}"
        )

        plain_path = data_dir / "dale2014_templates.h5"
        assert plain_path.exists(), f"tracked template file missing: {plain_path}"
        with h5py.File(plain_path, "r") as f:
            wavelength = np.asarray(f["wavelength_aa"][:])
            templates = np.asarray(f["templates_sf"][:])
        radio_region = wavelength > 1e8
        assert np.abs(templates[:, radio_region]).max() > 0.0, (
            "plain dale2014 templates carry no radio tail beyond 1e8 Å — "
            "the #1970 guard's premise no longer holds; re-verify and retire it"
        )

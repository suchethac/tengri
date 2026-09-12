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
        """dale2014_cigale + SF radio produces a smoothly curving composed SED.

        Smoothness criterion: the log-log slope between adjacent radio-band
        nodes (0.5-30 GHz, 1e8-6e9 Angstrom) changes by less than 0.05 from
        one interval to the next. Deviation-from-median was the wrong
        quantity: it also flags the physically required, gradual free-free
        flattening toward high frequency as a defect, whereas #1970's real
        double-count defect is a node-to-node slope JUMP (-4.93 next to
        +0.77, a change of order 5).

        Physical pin: the free-free thermal fraction rises from ~10% at
        1.4 GHz to ~50% at 30 GHz (Condon 1992, ARA&A, 30, 575), so the
        lowest-frequency node sits close to the pure-synchrotron slope the
        sampled ``radio_alpha_sf`` implies, while the highest-frequency node
        is measurably flatter, bounded below by the pure free-free slope the
        sampled ``radio_alpha_ff`` implies.
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

        # Smoothness criterion: adjacent-interval slope CHANGE stays small.
        # A genuine double-count kink jumps the slope by ~5.7 at the
        # template edge; physical synchrotron+free-free curvature changes
        # by ~0.015 per node here.
        slope_change = jnp.diff(slopes)
        max_slope_change = jnp.max(jnp.abs(slope_change))

        assert float(max_slope_change) < 0.05, (
            f"Composed SED curvature changes abruptly between adjacent radio-band "
            f"nodes: max |Δslope| {float(max_slope_change):.3f}. This suggests "
            f"double-counting or other incompatibility at the template edge."
        )

        # Physical pin: radio_alpha_sf/radio_alpha_ff are read from the sampled
        # params (never hardcoded) since only the sampled value is meaningful
        # per draw. L_nu ~ nu**-alpha_sf and L_nu ~ nu**alpha_ff give
        # d log L_nu / d log wave = +alpha_sf (pure synchrotron) and
        # -alpha_ff (pure free-free) respectively.
        alpha_sf = params["radio_alpha_sf"]
        alpha_ff = params["radio_alpha_ff"]
        lowest_freq_slope = slopes[-1]  # nearest 6e9 Å / 0.5 GHz
        highest_freq_slope = slopes[0]  # nearest 1e8 Å / 30 GHz

        assert float(jnp.abs(lowest_freq_slope - alpha_sf)) < 0.10, (
            f"Lowest-frequency slope {float(lowest_freq_slope):.3f} strays from the "
            f"pure-synchrotron value radio_alpha_sf={float(alpha_sf):.3f} by more "
            f"than 0.10; the ~10% Condon (1992) thermal fraction at 1.4 GHz should "
            f"leave this node close to the synchrotron-only spectral index."
        )
        assert float(highest_freq_slope) < float(lowest_freq_slope) - 0.15, (
            f"Highest-frequency slope {float(highest_freq_slope):.3f} is not at "
            f"least 0.15 flatter than the lowest-frequency slope "
            f"{float(lowest_freq_slope):.3f}; the free-free thermal fraction rising "
            f"toward ~50% by 30 GHz (Condon 1992) should flatten the spectrum "
            f"measurably at the high-frequency end."
        )
        assert float(highest_freq_slope) > -float(alpha_ff) - 1e-6, (
            f"Highest-frequency slope {float(highest_freq_slope):.3f} has overshot "
            f"below the pure free-free slope -radio_alpha_ff="
            f"{-float(alpha_ff):.3f}; a synchrotron+free-free mixture cannot be "
            f"flatter than its flattest (free-free) component."
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

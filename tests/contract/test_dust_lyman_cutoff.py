# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end wiring for the dust Lyman-limit attenuation cutoff.

Guards that ``dust={'lyman_cutoff': True}`` actually reaches the forward model
— the builder -> spec -> SEDModel -> factory -> DustSEDComponent chain is not a
silent no-op — and reproduces the intended physics: with the clip enabled the
attenuation curve is zeroed below 912 Å, so far-UV starlight is transmitted
unattenuated (CIGALE ``dustatt_modified_starburst`` behavior) instead of being
suppressed by tengri's default FUV polynomial extrapolation.

Also covers the unit-level helper, the JIT/grad safety of the clip, the static
``compile_signature`` distinctness (no kernel-cache color-leak), the
single/wg00 guard, and the grammar round-trip.

Uses the synthetic SSP (spans ~100 Å – 1 mm, so it has FUV grid points below
912 Å) + top-hat fixtures so it runs in default CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.dust.attenuation import (
    apply_lyman_cutoff,
    calzetti,
    two_component_dust,
)

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

LYMAN_LIMIT_AA = 912.0


def _build(ssp, obs, **dust_extra):
    dust = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": tengri.FIXED,
        "tau_bc": 0.0,
        "tau_diff": 0.5,
    }
    dust.update(dust_extra)
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "tsnorm", "all_params": tengri.FIXED},
        dust_attenuation=dust,
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=tengri.Fixed(0.0),
    )


def _rest_sed(model):
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    return np.asarray(out.wavelength), np.asarray(out.sed)


class TestHelper:
    def test_off_is_noop(self):
        wave = jnp.array([300.0, 911.0, 913.0, 5500.0])
        k = calzetti(wave)
        np.testing.assert_array_equal(np.asarray(apply_lyman_cutoff(k, wave, 0.0)), np.asarray(k))

    def test_zeros_below_cutoff_only(self):
        wave = jnp.array([300.0, 500.0, 911.0, 913.0, 2000.0, 5500.0])
        k = calzetti(wave)
        clipped = np.asarray(apply_lyman_cutoff(k, wave, LYMAN_LIMIT_AA))
        assert np.all(clipped[:3] == 0.0), "λ < 912 Å must be zeroed"
        np.testing.assert_array_equal(clipped[3:], np.asarray(k)[3:])

    def test_jit_and_grad_safe(self):
        wave = jnp.linspace(200.0, 20000.0, 64)

        def loss(tau):
            k = calzetti(wave)
            k = apply_lyman_cutoff(k, wave, LYMAN_LIMIT_AA)
            return jnp.sum(jnp.exp(-tau * k))

        g = jax.jit(jax.grad(loss))(jnp.asarray(0.5))
        assert jnp.isfinite(g), "grad through the clip must be finite under jit"


class TestTransmission:
    def test_fuv_transmitted_with_cutoff(self):
        wave = jnp.array([300.0, 700.0, 911.0, 913.0, 2000.0, 5500.0])
        ages = jnp.array([1e6, 1e8, 1e10])
        kw = dict(law_bc="calzetti", law_diff="calzetti")
        t_off = two_component_dust(wave, ages, 0.5, 0.3, **kw)
        t_on = two_component_dust(wave, ages, 0.5, 0.3, lyman_cutoff_aa=LYMAN_LIMIT_AA, **kw)
        # Below 912 Å: cutoff -> full transmission (T == 1); default -> attenuated.
        assert np.allclose(np.asarray(t_on[:, :3]), 1.0)
        assert not np.allclose(np.asarray(t_off[:, :3]), 1.0)
        # At and above 912 Å: identical to the un-clipped curve.
        np.testing.assert_allclose(np.asarray(t_on[:, 3:]), np.asarray(t_off[:, 3:]), rtol=1e-12)


class TestForwardWiring:
    def test_cutoff_changes_fuv_sed(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """The clip must change the emergent rest SED below 912 Å (no-op guard)."""
        wave, sed_off = _rest_sed(_build(synthetic_ssp_wide, synthetic_tophat_obs))
        _, sed_on = _rest_sed(_build(synthetic_ssp_wide, synthetic_tophat_obs, lyman_cutoff=True))
        fuv = wave < LYMAN_LIMIT_AA
        assert fuv.any(), "fixture must include FUV grid points below 912 Å"
        assert not np.allclose(sed_on[fuv], sed_off[fuv]), (
            "lyman_cutoff=True produced no FUV change — wiring is a silent no-op"
        )
        # With the curve zeroed, the clipped FUV flux is >= the attenuated one.
        assert np.all(sed_on[fuv] >= sed_off[fuv] - 1e-30)

    def test_optical_unchanged(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """The clip touches only λ < 912 Å; the optical SED is bit-identical."""
        wave, sed_off = _rest_sed(_build(synthetic_ssp_wide, synthetic_tophat_obs))
        _, sed_on = _rest_sed(_build(synthetic_ssp_wide, synthetic_tophat_obs, lyman_cutoff=True))
        opt = (wave >= 3000.0) & (wave <= 10000.0)
        np.testing.assert_allclose(sed_on[opt], sed_off[opt], rtol=1e-12)

    def test_compile_signature_distinct(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Two models differing only by the clip MUST get distinct signatures."""
        sig_off = _build(synthetic_ssp_wide, synthetic_tophat_obs).compile_signature()
        sig_on = _build(
            synthetic_ssp_wide, synthetic_tophat_obs, lyman_cutoff=True
        ).compile_signature()
        assert sig_off != sig_on, "clip must enter compile_signature (color-leak guard)"


class TestGrammar:
    def test_single_component_guard(self):
        from tengri.parameters.groups import _translate_dust_attenuation

        with pytest.raises(ValueError, match="two_component"):
            _translate_dust_attenuation({"type": "single_component", "lyman_cutoff": True}, {})

    def test_wg00_guard(self):
        from tengri.parameters.groups import _translate_dust_attenuation

        with pytest.raises(ValueError, match="two_component"):
            _translate_dust_attenuation({"type": "wg00", "lyman_cutoff": True}, {})

    def test_translate_sets_cutoff(self):
        from tengri.parameters.groups import _translate_dust_attenuation

        result: dict = {}
        _translate_dust_attenuation(
            {
                "type": "two_component",
                "law": "calzetti",
                "lyman_cutoff": True,
            },
            result,
        )
        assert result["dust_lyman_cutoff_aa"] == LYMAN_LIMIT_AA

    def test_round_trip(self, synthetic_ssp_wide, synthetic_tophat_obs):
        model = _build(synthetic_ssp_wide, synthetic_tophat_obs, lyman_cutoff=True)
        groups = model.spec.to_groups()
        assert groups["dust_attenuation"].get("lyman_cutoff") is True

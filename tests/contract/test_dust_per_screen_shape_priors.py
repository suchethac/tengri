# SPDX-License-Identifier: BSD-3-Clause
"""Per-screen dust attenuation shape priors: grammar, prediction, and gates."""

import re

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Parameters, SEDModel, Uniform, WavePrecomp
from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract


class TestPerScreenDustShapePriors:
    """Test per-screen dust attenuation shape parameters as declared priors."""

    def test_grammar_accepts_per_screen_prior(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Grammar: per-screen shape keys with priors are declared and free."""
        spec = parse_groups(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "slope_bc": Uniform(-1.5, -0.3),
                "delta_diff": Uniform(-0.5, 0.5),
            },
            redshift=Fixed(0.1),
        )

        assert "dust_slope_bc" in spec.free_params
        assert "dust_delta_diff" in spec.free_params
        assert "dust_slope" not in spec.free_params
        assert "dust_delta" not in spec.free_params

    def test_prediction_per_screen_moves_differently(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Prediction: sweeping per-screen params moves predictions PHYSICALLY.

        A young, dusty birth cloud (small ``age_gyr``/``tau_gyr``, large
        ``tau_bc``) is where the birth-cloud screen actually matters: the
        default DPL SFH used before leaves almost no stellar mass under
        10 Myr, so a "some nonzero difference exists" assertion there passed
        on a vacuous ~5e-4 effect regardless of whether the per-screen value
        was really threaded through. This configuration measures ~11%.
        """

        def build(slope_bc, slope_diff):
            return SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={
                    "type": "dpl",
                    "age_gyr": Fixed(0.3),
                    "tau_gyr": Fixed(0.1),
                    "other_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "power_law",
                    "law_diff": "power_law",
                    "slope_bc": slope_bc,
                    "slope_diff": slope_diff,
                    "tau_bc": Fixed(4.0),
                    "other_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )

        model_bc = build(Uniform(-1.5, -0.3), Fixed(-0.7))
        sed_bc_low = np.asarray(model_bc.predict_photometry({"dust_slope_bc": -1.5}))
        sed_bc_high = np.asarray(model_bc.predict_photometry({"dust_slope_bc": -0.3}))
        rel_bc = (sed_bc_high - sed_bc_low) / sed_bc_low

        model_diff = build(Fixed(-0.7), Uniform(-1.5, -0.3))
        sed_diff_low = np.asarray(model_diff.predict_photometry({"dust_slope_diff": -1.5}))
        sed_diff_high = np.asarray(model_diff.predict_photometry({"dust_slope_diff": -0.3}))
        rel_diff = (sed_diff_high - sed_diff_low) / sed_diff_low

        # Floor: a physical effect, not sampling noise off a fixture whose
        # default SFH leaves almost no mass in the screen being swept.
        floor = np.max(np.abs(rel_bc))
        assert floor > 1e-2, f"slope_bc sweep effect {floor:.2%} is too small to be physical"

        # Contrast: the birth-cloud sweep must move the SED differently from
        # the diffuse-screen sweep -- proving the two screens are genuinely
        # independent, not merely that "something" moves either way.
        contrast = np.max(np.abs(rel_bc - rel_diff)) / np.max(np.abs(rel_diff))
        assert contrast > 0.10, (
            f"slope_bc and slope_diff sweeps agree to within {1 - contrast:.2%}; "
            f"the two screens are not behaving independently"
        )

    def test_fixed_per_screen_matches_scalar_static_override(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Fixed per-screen param is bit-identical to scalar static override
        (#2428) -- not only in PREDICTION, but in what the declared
        parameter itself carries: ``get_fixed_values()`` and ``summary()``
        must show -1.0, the value actually driving the screen's law, not
        the untouched registry default (-0.7) the bug left behind."""

        def build(slope_bc):
            return SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "power_law",
                    "law_diff": "power_law",
                    "slope_bc": slope_bc,
                    "slope_diff": -0.7,
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )

        model_per_screen = build(Fixed(-1.0))
        model_scalar = build(-1.0)

        # Both models are fully Fixed: predict_photometry needs no seed dict
        # (params dicts stay free-only throughout this module).
        sed_per_screen = model_per_screen.predict_photometry({})
        sed_scalar = model_scalar.predict_photometry({})

        np.testing.assert_array_equal(sed_per_screen, sed_scalar)

        # The bare-scalar spelling's declared dust_slope_bc must ALSO carry
        # -1.0 -- not the registry default (-0.7) -- with "user_fixed"
        # provenance, exactly like the Fixed(-1.0) spelling.
        for model, label in ((model_scalar, "scalar"), (model_per_screen, "Fixed(v)")):
            fixed_values = model.spec.get_fixed_values()
            assert float(fixed_values["dust_slope_bc"]) == pytest.approx(-1.0), (
                f"{label} spelling: get_fixed_values()['dust_slope_bc'] should be -1.0, "
                f"got {fixed_values['dust_slope_bc']}"
            )
            provenance = model.spec._group_provenance
            assert provenance["dust_slope_bc"] == "user_fixed", (
                f"{label} spelling: dust_slope_bc provenance should be 'user_fixed', "
                f"got {provenance['dust_slope_bc']!r}"
            )

        summary = model_scalar.spec.summary_str()
        assert re.search(r"dust_slope_bc\s+Fixed\s+-1\b", summary), (
            f"summary() should show dust_slope_bc Fixed at -1, not the registry "
            f"default:\n{summary}"
        )

    def test_two_component_law_defaults_gate(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Gate #1833: per-screen free names not freed by wildcard."""
        spec = parse_groups(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "all_params": FREE,
            },
            redshift=Fixed(0.1),
        )

        free_params = spec.free_params
        per_screen_names = [
            f"dust_{s}_{sc}"
            for s in ["slope", "delta", "bump_strength", "Rv"]
            for sc in ["bc", "diff", "neb"]
        ]
        for name in per_screen_names:
            assert name not in free_params, f"{name} should not be freed by wildcard"

    def test_flat_surface_diagnostic_rejects_per_screen_prior(self):
        """Flat surface: dust_law_overrides with prior raises ParameterError."""
        with pytest.raises(ParameterError, match=r"per-screen.*override"):
            Parameters(dust_law_overrides={"bc": {"dust_slope": Uniform(-1.5, -0.3)}})

    def test_neb_screen_prior(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Neb screen: per-screen shape with prior works like bc/diff."""
        spec = parse_groups(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
                "law_neb": "cardelli",
                "Rv_neb": Uniform(2.0, 6.0),
            },
            redshift=Fixed(0.1),
        )

        assert "dust_Rv_neb" in spec.free_params

    def test_energy_balance_lut_disabled_by_free_per_screen(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Energy balance LUT: a free per-screen shape disables it; a Fixed one
        keeps it and agrees with the exact path within a stated margin."""

        def build(approx, slope_bc):
            return SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                approx=approx,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "power_law",
                    "law_diff": "calzetti",
                    "slope_bc": slope_bc,
                    "other_params": Fixed(DEFAULT),
                },
                dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.1),
            )

        model_free = build(WavePrecomp(), Uniform(-1.5, -0.3))
        assert model_free._energy_balance_lut_cache is None, (
            "a free per-screen dust_slope_bc must disable the WavePrecomp "
            "energy-balance LUT (same unsafe_free gate as a free shared dust_slope)"
        )

        model_fixed = build(WavePrecomp(), Fixed(-0.9))
        assert model_fixed._energy_balance_lut_cache is not None, (
            "a Fixed per-screen dust_slope_bc must keep the energy-balance LUT"
        )

        model_exact = build(None, Fixed(-0.9))
        lut_phot = np.asarray(model_fixed.predict_photometry({}))
        exact_phot = np.asarray(model_exact.predict_photometry({}))
        rel = np.abs(lut_phot - exact_phot) / np.abs(exact_phot)
        # Measured 1.82e-4 max relative on this fixture; margin to 5e-4 keeps
        # the assertion meaningful without pinning the WavePrecomp
        # approximation's exact digit.
        assert rel.max() < 5e-4, f"LUT vs exact drifted {rel.max():.2e} (> 5e-4)"

    def test_round_trip_to_groups(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Round-trip: to_groups() preserves per-screen prior."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "slope_bc": Uniform(-1.5, -0.3),
            },
            redshift=Fixed(0.1),
        )

        groups = model.spec.to_groups()
        dust_atten = groups["dust_attenuation"]

        assert "slope_bc" in dust_atten


class TestPerScreenFreeRoundTrip:
    """``to_groups()`` -> reparse preserves a per-screen ``FREE`` (#2428).

    ``_get_explicit_overrides`` and ``parameters_to_groups``'s no-wildcard
    branch re-emit a per-screen name only when its own provenance says a
    caller asked for it; before this fix that check was a bare
    ``("user_prior", "user_fixed")`` tuple, omitting ``"user_free"``. A
    ``slope_bc: FREE`` build resolved and predicted correctly at build time,
    but round-tripping through ``to_groups()`` silently dropped ``slope_bc``
    from the emitted dict: the reparsed model had no free dust parameter at
    all, and ``predict_photometry({"dust_slope_bc": ...})`` on it silently
    accepted and ignored the value (measured 7.04e-2 max relative difference
    from the original model's own prediction at the same value).
    """

    @staticmethod
    def _roundtrip(model, synthetic_ssp_wide, synthetic_tophat_obs):
        groups = model.spec.to_groups()
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide, observation=synthetic_tophat_obs, **groups
        )

    def test_free_slope_bc_survives_round_trip(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """A lone ``slope_bc: FREE`` keeps its free-ness across a round-trip."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": FREE,
                "slope_diff": Fixed(-0.7),
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        model2 = self._roundtrip(model, synthetic_ssp_wide, synthetic_tophat_obs)

        assert set(model2.spec.free_params) == set(model.spec.free_params)
        assert "dust_slope_bc" in model2.spec.free_params

        params = {"dust_slope_bc": -1.4}
        np.testing.assert_array_equal(
            model.predict_photometry(params), model2.predict_photometry(params)
        )

    def test_free_bc_diff_pair_survives_round_trip(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """A ``slope_bc``/``slope_diff`` FREE pair both keep their free-ness."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": FREE,
                "slope_diff": FREE,
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        model2 = self._roundtrip(model, synthetic_ssp_wide, synthetic_tophat_obs)

        assert set(model2.spec.free_params) == set(model.spec.free_params)
        assert {"dust_slope_bc", "dust_slope_diff"} <= set(model2.spec.free_params)

        params = {"dust_slope_bc": -1.4, "dust_slope_diff": -0.5}
        np.testing.assert_array_equal(
            model.predict_photometry(params), model2.predict_photometry(params)
        )

    def test_free_rv_neb_survives_round_trip(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """A nebular-screen ``Rv_neb: FREE`` keeps its free-ness too."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
                "law_neb": "cardelli",
                "Rv_neb": FREE,
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
        model2 = self._roundtrip(model, synthetic_ssp_wide, synthetic_tophat_obs)

        assert set(model2.spec.free_params) == set(model.spec.free_params)
        assert "dust_Rv_neb" in model2.spec.free_params

        params = {"dust_Rv_neb": 5.0}
        np.testing.assert_array_equal(
            model.predict_photometry(params), model2.predict_photometry(params)
        )


class TestPerScreenFreeAndFixedDefault:
    """``FREE`` and ``Fixed(DEFAULT)`` on a per-screen key (#2428).

    Before this fix, ``slope_bc: FREE`` raised the issue's own ``float()``
    ``TypeError`` (the per-screen key's value was handed unconditionally to
    ``float()`` unless it was already a resolved ``Distribution``, and
    ``FREE`` is a bare sentinel, not one) and ``slope_bc: Fixed(DEFAULT)``
    was stored unresolved, reaching ``priors.py``'s "Fixed(DEFAULT) is a
    build-grammar token" guard with a message naming the wrong context.
    """

    def test_free_frees_on_declared_free_prior(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """slope_bc: FREE frees dust_slope_bc on its declared free_prior."""
        # Young, dusty birth cloud (see test_prediction_per_screen_moves_
        # differently above): the default DPL SFH leaves almost no mass
        # under 10 Myr, so slope_bc has no physical leverage to move under it.
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={
                "type": "dpl",
                "age_gyr": Fixed(0.3),
                "tau_gyr": Fixed(0.1),
                "other_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": FREE,
                "slope_diff": Fixed(-0.7),
                "tau_bc": Fixed(4.0),
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )

        assert "dust_slope_bc" in model.spec.free_params
        # Matches the declared free_prior mirrored from the shared dust_slope
        # stem (Uniform(-1.5, -0.3)), not some other ad hoc range.
        lo, hi = model.spec.get_distribution("dust_slope_bc").bounds
        assert (float(lo), float(hi)) == (-1.5, -0.3)

        sed_low = np.asarray(model.predict_photometry({"dust_slope_bc": -1.5}))
        sed_high = np.asarray(model.predict_photometry({"dust_slope_bc": -0.3}))
        assert not np.allclose(sed_low, sed_high, atol=0)

    def test_fixed_default_pins_at_registry_default(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """slope_bc: Fixed(DEFAULT) pins at the registry default, exactly as
        the scalar spelling of that same default value does."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": Fixed(DEFAULT),
                "slope_diff": -0.7,
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )

        assert "dust_slope_bc" not in model.spec.free_params
        fixed_value = model.spec.get_fixed_values()["dust_slope_bc"]
        assert float(fixed_value) == pytest.approx(-0.7)  # dust_slope's registry default

        model_scalar = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": -0.7,
                "slope_diff": -0.7,
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )

        np.testing.assert_array_equal(
            model.predict_photometry({}), model_scalar.predict_photometry({})
        )

    def test_junk_value_raises_parameter_error_not_type_error(self):
        """A non-numeric, non-Distribution, non-FREE per-screen value raises
        the grammar's own ParameterError, never a bare float() TypeError."""
        with pytest.raises(ParameterError, match=r"dust_attenuation 'slope_bc'"):
            parse_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "power_law",
                    "law_diff": "power_law",
                    "slope_bc": "bogus",
                    "slope_diff": Fixed(-0.7),
                },
                redshift=Fixed(0.1),
            )


class TestNebularScreenLiveOverrides:
    """The nebular (``_neb``) per-screen live-override merge (#2428).

    Both nebular call sites in ``DustSEDComponent``
    (``attenuate_line_catalog`` for the discrete line catalog, ``apply`` for
    the nebular continuum) used to derive the attenuation-law keyword as
    ``stem.replace("dust_", "")`` (e.g. ``"Rv"`` for ``dust_Rv``), which no
    law's signature declares (``law_kwarg_names('cardelli') == ('dust_Rv',)``),
    so ``select_law_kwargs`` silently discarded every live ``*_neb`` value: a
    ``Rv_neb`` sweep left the nebular channel bit-identical, and a Fixed
    ``Rv_neb`` silently equaled the law's own default. Both sites now go
    through ``merge_neb_screen_live_overrides``, tested directly here via a
    hand-built ``DustSEDComponent`` -- no SSP/model machinery needed, since
    ``attenuate_line_catalog``/``apply`` are pure functions of
    ``(params, ...)``, and direct construction is the documented low-level
    testing pattern (``DustSEDComponentConfig``'s own docstring).
    """

    LINE_WAVE = jnp.asarray([1300.0, 2200.0, 4862.71, 6564.72])
    LOG_LINE_LUMS = jnp.asarray([40.4, 41.2, 40.9, 41.4])

    @staticmethod
    def _line_component(*, live: bool):
        config = DustSEDComponentConfig(
            law_bc="cardelli",
            law_diff="cardelli",
            live_shape_params=frozenset({"dust_Rv_neb"}) if live else None,
        )
        return DustSEDComponent(config=config)

    def test_line_rv_neb_sweep_moves_the_lines(self):
        """Rv_neb 2 -> 6 must move the attenuated line luminosities."""
        comp = self._line_component(live=True)
        params = dict(dust_tau_bc=1.0, dust_tau_diff=0.3, dust_Rv=3.1, redshift=0.1)

        out_low = comp.attenuate_line_catalog(
            {**params, "dust_Rv_neb": 2.0}, self.LINE_WAVE, self.LOG_LINE_LUMS
        )
        out_high = comp.attenuate_line_catalog(
            {**params, "dust_Rv_neb": 6.0}, self.LINE_WAVE, self.LOG_LINE_LUMS
        )
        assert not np.allclose(np.asarray(out_low), np.asarray(out_high))

    def test_line_rv_neb_fixed_matches_static_override(self):
        """A live Fixed(v) Rv_neb equals the static neb_law_overrides route,
        bit-for-bit -- the two mechanisms for the same value must agree."""
        params = dict(dust_tau_bc=1.0, dust_tau_diff=0.3, dust_Rv=3.1, redshift=0.1)

        comp_live = self._line_component(live=True)
        out_live = comp_live.attenuate_line_catalog(
            {**params, "dust_Rv_neb": 5.0}, self.LINE_WAVE, self.LOG_LINE_LUMS
        )

        config_static = DustSEDComponentConfig(
            law_bc="cardelli", law_diff="cardelli", neb_law_overrides=(("dust_Rv", 5.0),)
        )
        comp_static = DustSEDComponent(config=config_static)
        out_static = comp_static.attenuate_line_catalog(params, self.LINE_WAVE, self.LOG_LINE_LUMS)

        np.testing.assert_array_equal(np.asarray(out_live), np.asarray(out_static))

    @staticmethod
    def _continuum_state():
        n_wave, n_age = 40, 6
        wave = jnp.logspace(2.5, 4.5, n_wave)
        ages_yr = jnp.logspace(6.0, 10.0, n_age)
        lnu_age = jnp.ones((n_age, n_wave)) * 1e30
        sed_nebular = jnp.ones(n_wave) * 5e29
        return ForwardState(
            wave=wave,
            derived={"lnu_age": lnu_age, "ssp_ages_yr": ages_yr, "sed_nebular": sed_nebular},
        )

    def test_continuum_rv_neb_sweep_moves_the_sed(self):
        """Rv_neb 2 -> 6 must move the attenuated nebular continuum."""
        state = self._continuum_state()
        config = DustSEDComponentConfig(
            law_bc="cardelli", law_diff="cardelli", live_shape_params=frozenset({"dust_Rv_neb"})
        )
        comp = DustSEDComponent(config=config)
        params = dict(dust_tau_bc=1.0, dust_tau_diff=0.3, dust_Rv=3.1, redshift=0.1)

        out_low = comp.apply(state, {**params, "dust_Rv_neb": 2.0})
        out_high = comp.apply(state, {**params, "dust_Rv_neb": 6.0})
        assert not np.allclose(
            np.asarray(out_low.sed_intrinsic), np.asarray(out_high.sed_intrinsic), atol=0
        )

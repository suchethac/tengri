# SPDX-License-Identifier: BSD-3-Clause
"""Issue #721: AGN double-count guard (composable AGN + Dale2014 dust_frac_agn).

The composable AGN's ``agn_ir_frac`` (CIGALE-joint tie) and Dale2014's embedded
quasar template ``dust_frac_agn`` are two distinct AGN surfaces, both keyed off
the same stellar ``L_absorbed`` (ADR-0018 §5). With both > 0 the AGN mid/far-IR
is double-counted. ``SEDModel.build`` emits a filterable
:class:`AGNDustDoubleCountWarning` — value-aware, so a ``dust_frac_agn`` pinned
to 0 (the flagship recipes) never warns.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform, recipes
from tengri.config.exceptions import AGNDustDoubleCountWarning
from tengri.forward.sed_model import _warn_agn_dust_double_count
from tengri.parameters import parse_groups

pytestmark = pytest.mark.contract


def _spec(dust_param_value, agn_fracagn, *, emission="dale2014", with_agn=True):
    """Build a spec with a chosen dust emission + (optional) composable AGN.

    For dale2014/dale2014_cigale, dust_param_value is used as frac_agn.
    For energy_balance_split, dust_param_value is used as L_agn_ir.
    """
    groups = dict(
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
    )
    if emission in ("dale2014", "dale2014_cigale"):
        groups["dust_emission"] = {"type": emission, "frac_agn": dust_param_value}
    elif emission == "energy_balance_split":
        groups["dust_emission"] = {"type": emission, "L_agn_ir": dust_param_value}
    else:
        groups["dust_emission"] = {"type": emission}

    if with_agn:
        groups["agn"] = {
            "type": "composable",
            "disc": {"type": "multicolor"},
            "torus": {"type": "none"},
            "agn_ir_frac": agn_fracagn,
            "all_params": Fixed(DEFAULT),
        }
    return parse_groups(**groups)


def _fires(spec) -> bool:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_agn_dust_double_count(spec)
    return any(issubclass(x.category, AGNDustDoubleCountWarning) for x in w)


@pytest.mark.unit
class TestDoubleCountGuardFires:
    """The guard fires exactly when both AGN surfaces are positive-active on cigale."""

    def test_cigale_both_fixed_positive_warns(self):
        # Only dale2014_cigale carries the QSO template where frac_agn is live.
        assert _fires(_spec(Fixed(0.3), Fixed(0.5), emission="dale2014_cigale"))

    def test_cigale_both_free_warns(self):
        # A free param can take positive values → counts as active.
        assert _fires(_spec(Uniform(0.0, 0.9), Uniform(0.01, 0.99), emission="dale2014_cigale"))

    def test_plain_dale2014_both_positive_no_warn(self):
        # Plain dale2014 lacks the QSO template; frac_agn is inert there.
        # No warning even with both parameters nominally positive. Note: plain
        # dale2014 doesn't declare frac_agn, so we can't pass it explicitly.
        # Create the spec manually to test that the gate doesn't fire on plain dale2014.
        spec = parse_groups(
            redshift=Fixed(0.1),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014"},
            agn={
                "type": "composable",
                "disc": {"type": "multicolor"},
                "torus": {"type": "none"},
                "agn_ir_frac": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
        )
        assert not _fires(spec)

    def test_dust_frac_agn_zero_no_warn(self):
        # Zero on either surface prevents double-count on cigale.
        assert not _fires(_spec(Fixed(0.0), Fixed(0.5), emission="dale2014_cigale"))

    def test_agn_fracagn_zero_no_warn(self):
        assert not _fires(_spec(Fixed(0.3), Fixed(0.0), emission="dale2014_cigale"))

    def test_no_agn_no_warn(self):
        # Dale2014 frac_agn alone (the embedded-proxy use) is legitimate.
        assert not _fires(
            _spec(Fixed(0.3), Fixed(0.0), emission="dale2014_cigale", with_agn=False)
        )

    def test_non_dale_emission_no_warn(self):
        # Only dale2014_cigale carries the embedded quasar template.
        # Plain dale2014 lacks it; other engines don't have the parameter at all.
        assert not _fires(_spec(Fixed(0.0), Fixed(0.5), emission="casey2012"))

    def test_ebs_both_fixed_positive_warns(self):
        # energy_balance_split uses dust_L_agn_ir (not frac_agn).
        assert _fires(_spec(Fixed(1.0e43), Fixed(0.5), emission="energy_balance_split"))

    def test_ebs_both_free_warns(self):
        # A free dust_L_agn_ir counts as active even if the prior is large.
        assert _fires(
            _spec(Uniform(0.0, 1e45), Uniform(0.01, 0.99), emission="energy_balance_split")
        )

    def test_ebs_dust_l_agn_ir_zero_no_warn(self):
        # Zero on the dust_L_agn_ir side prevents double-count.
        assert not _fires(_spec(Fixed(0.0), Fixed(0.5), emission="energy_balance_split"))

    def test_ebs_agn_fracagn_zero_no_warn(self):
        # Zero on the AGN side prevents double-count.
        assert not _fires(_spec(Fixed(1.0e43), Fixed(0.0), emission="energy_balance_split"))

    def test_ebs_no_agn_no_warn(self):
        # energy_balance_split with dust_L_agn_ir alone (no composable AGN) is legitimate.
        assert not _fires(
            _spec(Fixed(1.0e43), Fixed(0.0), emission="energy_balance_split", with_agn=False)
        )


@pytest.mark.unit
class TestFlagshipRecipesDoNotWarn:
    """The curated AGN recipes pin dust_frac_agn=0 — they must stay silent."""

    @pytest.mark.parametrize("name", ["composable_agn", "agn_panchromatic"])
    def test_recipe_does_not_double_count(self, name):
        recipe = dict(getattr(recipes, name)())
        recipe.pop("approx", None)  # SEDModel-only kwarg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = parse_groups(**recipe)
        assert not _fires(spec)


@pytest.mark.unit
class TestGuardEndToEndAndFilterable:
    def test_build_warns_on_both_active(self, synthetic_ssp_wide):
        # Warning fires on dale2014_cigale (where frac_agn is live), not plain dale2014.
        with pytest.warns(AGNDustDoubleCountWarning, match="DOUBLE-COUNTED"):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                dust_emission={"type": "dale2014_cigale", "frac_agn": Fixed(0.3)},
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor"},
                    "torus": {"type": "none"},
                    "agn_ir_frac": Fixed(0.5),
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.05),
            )

    def test_build_silent_when_dust_frac_agn_zero(self, synthetic_ssp_wide):
        # Even on cigale (where frac_agn is live), zero value means no double-count warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", AGNDustDoubleCountWarning)
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                dust_emission={"type": "dale2014_cigale", "frac_agn": Fixed(0.0)},
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor"},
                    "torus": {"type": "none"},
                    "agn_ir_frac": Fixed(0.5),
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.05),
            )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_warning_is_filterable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AGNDustDoubleCountWarning)
            # Must not raise even though both surfaces are active on cigale.
            _warn_agn_dust_double_count(_spec(Fixed(0.3), Fixed(0.5), emission="dale2014_cigale"))

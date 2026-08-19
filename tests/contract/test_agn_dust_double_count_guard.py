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

from tengri import FIXED, Fixed, SEDModel, Uniform, recipes
from tengri.config.exceptions import AGNDustDoubleCountWarning
from tengri.forward.sed_model import _warn_agn_dust_double_count
from tengri.parameters import parse_groups

pytestmark = pytest.mark.contract


def _spec(dust_frac_agn, agn_fracagn, *, emission="dale2014", with_agn=True):
    """Build a spec with a chosen dust emission + (optional) composable AGN."""
    groups = dict(
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "law": "power_law",
            "type": "two_component",
            "emission": {"type": emission, "frac_agn": dust_frac_agn}
            if emission == "dale2014"
            else {"type": emission},
            "*": FIXED,
        },
    )
    if with_agn:
        groups["agn"] = {
            "type": "composable",
            "disc": {"type": "multicolor"},
            "torus": {"type": "none"},
            "agn_ir_frac": agn_fracagn,
            "*": FIXED,
        }
    return parse_groups(**groups)


def _fires(spec) -> bool:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_agn_dust_double_count(spec)
    return any(issubclass(x.category, AGNDustDoubleCountWarning) for x in w)


@pytest.mark.unit
class TestDoubleCountGuardFires:
    """The guard fires exactly when both AGN surfaces are positive-active."""

    def test_both_fixed_positive_warns(self):
        assert _fires(_spec(Fixed(0.3), Fixed(0.5)))

    def test_both_free_warns(self):
        # A free param can take positive values → counts as active.
        assert _fires(_spec(Uniform(0.0, 0.9), Uniform(0.01, 0.99)))

    def test_dust_frac_agn_zero_no_warn(self):
        assert not _fires(_spec(Fixed(0.0), Fixed(0.5)))

    def test_agn_fracagn_zero_no_warn(self):
        assert not _fires(_spec(Fixed(0.3), Fixed(0.0)))

    def test_no_agn_no_warn(self):
        # Dale2014 fracAGN alone (the embedded-proxy use) is legitimate.
        assert not _fires(_spec(Fixed(0.3), Fixed(0.0), with_agn=False))

    def test_non_dale_emission_no_warn(self):
        # Only Dale2014 carries the embedded quasar template.
        assert not _fires(_spec(Fixed(0.0), Fixed(0.5), emission="casey2012"))


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
        with pytest.warns(AGNDustDoubleCountWarning, match="DOUBLE-COUNTED"):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "*": FIXED},
                dust={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "emission": {"type": "dale2014", "frac_agn": Fixed(0.3)},
                    "*": FIXED,
                },
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor"},
                    "torus": {"type": "none"},
                    "agn_ir_frac": Fixed(0.5),
                    "*": FIXED,
                },
                redshift=Fixed(0.05),
            )

    def test_build_silent_when_dust_frac_agn_zero(self, synthetic_ssp_wide):
        with warnings.catch_warnings():
            warnings.simplefilter("error", AGNDustDoubleCountWarning)
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "*": FIXED},
                dust={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "emission": {"type": "dale2014"},  # frac_agn defaults to 0
                    "*": FIXED,
                },
                agn={
                    "type": "composable",
                    "disc": {"type": "multicolor"},
                    "torus": {"type": "none"},
                    "agn_ir_frac": Fixed(0.5),
                    "*": FIXED,
                },
                redshift=Fixed(0.05),
            )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_warning_is_filterable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AGNDustDoubleCountWarning)
            # Must not raise even though both surfaces are active.
            _warn_agn_dust_double_count(_spec(Fixed(0.3), Fixed(0.5)))

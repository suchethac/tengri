# SPDX-License-Identifier: BSD-3-Clause
"""Grammar, refusals and ratio-coordinate bit identity for the young-fraction continuity.

``sfh={'type': 'continuity', 'coordinates': 'young_fraction'}`` selects the stick-breaking
fractions. The default ``coordinates='ratios'`` must stay exactly the shipped continuity:
its predicted SED and stellar mass are pinned here as SHA-256 and hex digests taken on the
pre-change tree.

Every path that cannot evaluate the Student-t ratio prior refuses the young-fraction spec
with :class:`~tengri.config.exceptions.ConfigError`, naming the path. None drops the term.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.stellar.sfh.young_fraction import (
    young_fraction_prior_for_spec,
)
from tengri.config.exceptions import ConfigError
from tengri.inference.loss_functions import build_logprior_fn, standardized_neg_log_prior
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


def _yf_spec(**extra):
    """A young-fraction continuity spec with all six fractions free."""
    sfh = {"type": "continuity", "coordinates": "young_fraction", "all_params": FREE, **extra}
    return parse_groups(sfh=sfh, redshift=Fixed(0.1))


def test_young_fraction_type_and_free_parameters():
    spec = _yf_spec()
    assert list(spec.mean_sfh_type) == ["continuity_young_fraction"]
    assert sorted(p for p in spec.free_params if p.startswith("sfh_cont_young_frac_")) == [
        f"sfh_cont_young_frac_{k}" for k in range(6)
    ]
    assert "sfh_cont_log_total_mass" in spec.free_params


def test_ratios_coordinates_keep_the_shipped_continuity():
    spec = parse_groups(
        sfh={"type": "continuity", "coordinates": "ratios", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    assert list(spec.mean_sfh_type) == ["continuity"]


def test_to_groups_round_trips_the_coordinates_key():
    groups = _yf_spec().to_groups()
    assert groups["sfh"]["type"] == "continuity"
    assert groups["sfh"]["coordinates"] == "young_fraction"
    again = parse_groups(sfh=groups["sfh"], redshift=Fixed(0.1))
    assert list(again.mean_sfh_type) == ["continuity_young_fraction"]


def test_unknown_coordinates_are_refused():
    with pytest.raises(ValueError, match="coordinates"):
        parse_groups(sfh={"type": "continuity", "coordinates": "log"}, redshift=Fixed(0.1))


def test_coordinates_on_another_sfh_type_are_refused():
    with pytest.raises(ValueError, match="applies only to sfh type 'continuity'"):
        parse_groups(
            sfh={"type": "dpl", "coordinates": "young_fraction", "all_params": FREE},
            redshift=Fixed(0.1),
        )


def test_partly_fixed_fractions_have_no_exact_pullback():
    spec = _yf_spec(young_frac_0=Fixed(0.3))
    with pytest.raises(ConfigError, match="all six young fractions free"):
        young_fraction_prior_for_spec(spec)


def test_physical_space_logprior_refuses_the_young_fraction_spec():
    spec = _yf_spec()
    fitter = SimpleNamespace(
        spec=spec,
        _free_names=list(spec.free_params),
        _extra_log_prior=None,
    )
    with pytest.raises(ConfigError, match="build_logprior_fn"):
        build_logprior_fn(fitter)


def test_standardized_prior_refuses_latents_without_their_term():
    names = ("sfh_cont_log_total_mass", *(f"sfh_cont_young_frac_{k}" for k in range(6)))
    latents = {name: jnp.zeros(()) for name in names}
    with pytest.raises(ConfigError, match="young-fraction prior"):
        standardized_neg_log_prior(latents, names, stochastic=False)


def test_hierarchical_flat_seam_refuses_the_young_fraction_spec():
    from tengri.inference._hierarchical_flat import build_flat_problem

    spec = _yf_spec()
    fitter = SimpleNamespace(n_galaxies=2, _spec=spec, _free_names=list(spec.free_params))
    with pytest.raises(ConfigError, match="hierarchical flat seam"):
        build_flat_problem(fitter, key=None)


# SHA-256 of the rest-frame SED (float64 bytes), stellar mass and SED sum, as hex, of a
# continuity model in ratio coordinates on the synthetic narrow SSP, recorded on the tree
# before the young-fraction change. Three parameter points, each with a different ladder.
_RATIO_GOLDEN = (
    {
        "params": {
            "sfh_cont_log_total_mass": 10.0,
            **{f"sfh_cont_ratio_{k}": 0.0 for k in range(6)},
        },
        "sha256": "e7b79580b6099d1bf2811eda10d5932e8f32c9ec9554768c2b77291de217a73f",
        "stellar_mass_hex": "0x1.2a05f20000000p+33",
        "sed_sum_hex": "0x1.27ef9d4f44926p+141",
    },
    {
        "params": {
            "sfh_cont_log_total_mass": 9.4,
            "sfh_cont_ratio_0": 0.6,
            "sfh_cont_ratio_1": -0.4,
            "sfh_cont_ratio_2": 0.8,
            "sfh_cont_ratio_3": -0.5,
            "sfh_cont_ratio_4": 0.3,
            "sfh_cont_ratio_5": -0.7,
        },
        "sha256": "0f8cda714fe0e0e03ef598f583d3f98a53254034a6b1b9975d769524cf548250",
        "stellar_mass_hex": "0x1.2b70b0bf04e7fp+31",
        "sed_sum_hex": "0x1.27a268adc5279p+139",
    },
    {
        "params": {
            "sfh_cont_log_total_mass": 11.1,
            "sfh_cont_ratio_0": -1.2,
            "sfh_cont_ratio_1": 0.25,
            "sfh_cont_ratio_2": 0.1,
            "sfh_cont_ratio_3": 1.5,
            "sfh_cont_ratio_4": -0.05,
            "sfh_cont_ratio_5": 0.9,
        },
        "sha256": "21587c6d21de64b2806f2f0a109db8b016c3a072ddf5dc62ebfcb5b5c350d7dc",
        "stellar_mass_hex": "0x1.d4fc7bafb6aa7p+36",
        "sed_sum_hex": "0x1.c96a83a869ae3p+144",
    },
)


@pytest.fixture
def ratio_model(synthetic_ssp):
    return SEDModel.build(
        ssp_data=synthetic_ssp,
        sfh={"type": "continuity", "coordinates": "ratios", "all_params": FREE},
        met={"logzsol": Fixed(-0.5), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.mark.parametrize("point", _RATIO_GOLDEN, ids=["flat", "tilted", "young-heavy"])
def test_ratio_coordinates_are_bit_identical_to_the_shipped_continuity(ratio_model, point):
    pred = ratio_model.predict(point["params"])
    sed = np.asarray(pred.rest_sed())
    assert hashlib.sha256(sed.tobytes()).hexdigest() == point["sha256"]
    assert float(np.asarray(pred.properties["stellar_mass"])).hex() == point["stellar_mass_hex"]
    assert float(sed.sum()).hex() == point["sed_sum_hex"]


def test_young_fraction_sed_equals_ratio_sed_at_the_same_history(synthetic_ssp):
    """The two coordinate systems predict the same SED for the same SFH, bit for bit.

    The dense integrand puts exact knots at the bin edges of a continuity-type SFH. A
    young-fraction SFH that missed those knots would differ by up to a percent of the SED
    while its stellar mass matched, so the SED is the check.
    """
    from tengri.components.stellar.sfh.young_fraction import (
        log_width_steps,
        young_fractions_from_ratios,
    )

    kw = {
        "ssp_data": synthetic_ssp,
        "met": {"logzsol": Fixed(-0.5), "all_params": Fixed(DEFAULT)},
        "neb": {"type": "none"},
        "redshift": Fixed(0.1),
    }
    model_r = SEDModel.build(
        sfh={"type": "continuity", "coordinates": "ratios", "all_params": FREE}, **kw
    )
    model_y = SEDModel.build(
        sfh={"type": "continuity", "coordinates": "young_fraction", "all_params": FREE}, **kw
    )
    ratios = np.array([0.4, -0.2, 0.3, -0.1, 0.2, -0.3])
    fractions = np.asarray(
        young_fractions_from_ratios(
            jnp.asarray(ratios), log_width_steps(np.asarray(model_r.spec.bin_edges_gyr))
        )
    )
    params_r = {"sfh_cont_log_total_mass": 9.8}
    params_r.update({f"sfh_cont_ratio_{k}": float(ratios[k]) for k in range(6)})
    params_y = {"sfh_cont_log_total_mass": 9.8}
    params_y.update({f"sfh_cont_young_frac_{k}": float(fractions[k]) for k in range(6)})
    sed_r = np.asarray(model_r.predict(params_r).rest_sed())
    sed_y = np.asarray(model_y.predict(params_y).rest_sed())
    np.testing.assert_allclose(sed_y, sed_r, rtol=1e-12, atol=0)

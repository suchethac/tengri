"""Tests for PopulationSED (hierarchical-population SubModel wrapper)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward.population_sed import PopulationSED

pytestmark = pytest.mark.contract


class _StubSED:
    """A minimal SED-like object for shape testing."""

    name = "stub"
    spec = None  # only fit() needs spec; structural tests don't


def _gal(n_filters: int = 5):
    return {"flux_obs": jnp.zeros(n_filters), "noise": jnp.ones(n_filters)}


def test_population_sed_construction_with_defaults() -> None:
    pop = PopulationSED(sed=_StubSED(), galaxies=[_gal(), _gal(), _gal()])
    assert pop.n_galaxies == 3
    assert pop.name == "population_sed"
    # Defaults to the two PSD hyperparameters
    assert "sfh_field_psd_sigma" in pop.shared
    assert "sfh_field_psd_tau_myr" in pop.shared


def test_population_sed_rejects_zero_galaxies() -> None:
    with pytest.raises(ValueError, match="at least one galaxy"):
        PopulationSED(sed=_StubSED(), galaxies=[])


def test_population_sed_priors_must_cover_shared_names() -> None:
    """Every shared parameter needs a prior."""
    with pytest.raises(ValueError, match="missing entries"):
        PopulationSED(
            sed=_StubSED(),
            galaxies=[_gal()],
            shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
            priors={"sfh_field_psd_sigma": (0.1, 4.0)},  # missing tau
        )


def test_population_sed_custom_shared_and_priors() -> None:
    pop = PopulationSED(
        sed=_StubSED(),
        galaxies=[_gal()],
        shared=("met_logzsol",),
        priors={"met_logzsol": (-2.0, 0.5)},
    )
    assert pop.shared == ("met_logzsol",)
    assert pop.priors["met_logzsol"] == (-2.0, 0.5)


def test_population_sed_n_galaxies_property() -> None:
    pop = PopulationSED(sed=_StubSED(), galaxies=[_gal()] * 7)
    assert pop.n_galaxies == 7


def test_population_sed_galaxies_is_frozen_tuple() -> None:
    """galaxies field is held as a tuple — list mutation outside doesn't leak in."""
    galaxies = [_gal(), _gal()]
    pop = PopulationSED(sed=_StubSED(), galaxies=galaxies)
    galaxies.append(_gal())  # mutate the original list
    assert pop.n_galaxies == 2  # PopulationSED was snapshotted


def test_population_sed_top_level_import() -> None:
    """``from tengri import PopulationSED`` works."""
    import tengri

    assert tengri.PopulationSED is PopulationSED

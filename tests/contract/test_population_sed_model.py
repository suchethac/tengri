"""Tests for PopulationSEDModel — SubModel for hierarchical galaxy populations."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel

pytestmark = pytest.mark.contract


class _StubSED:
    """Smallest SED-like object that satisfies the population template role."""

    name = "stub_sed"


def _gal(n_filters: int = 5):
    return {"flux_obs": jnp.zeros(n_filters), "noise": jnp.ones(n_filters)}


# ── Construction ────────────────────────────────────────────────────


def test_construction_with_defaults() -> None:
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal(), _gal(), _gal()])
    assert pop.n_galaxies == 3
    assert pop.name == "population_sed_model"
    assert "sfh_field_psd_sigma" in pop.shared
    assert "sfh_field_psd_tau_myr" in pop.shared


def test_rejects_zero_galaxies() -> None:
    with pytest.raises(ValueError, match="at least one galaxy"):
        PopulationSEDModel(sed=_StubSED(), galaxies=[])


def test_priors_must_cover_shared_names() -> None:
    with pytest.raises(ValueError, match="missing entries"):
        PopulationSEDModel(
            sed=_StubSED(),
            galaxies=[_gal()],
            shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
            priors={"sfh_field_psd_sigma": (0.1, 4.0)},  # missing tau
        )


def test_custom_shared_and_priors() -> None:
    pop = PopulationSEDModel(
        sed=_StubSED(),
        galaxies=[_gal()],
        shared=("met_logzsol",),
        priors={"met_logzsol": (-2.0, 0.5)},
    )
    assert pop.shared == ("met_logzsol",)
    assert pop.priors["met_logzsol"] == (-2.0, 0.5)


def test_galaxies_snapshot() -> None:
    """galaxies field is a tuple — outside mutation doesn't leak in."""
    galaxies = [_gal(), _gal()]
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=galaxies)
    galaxies.append(_gal())
    assert pop.n_galaxies == 2


# ── SubModel contract ───────────────────────────────────────────────


def test_declared_parameters_returns_shared_names() -> None:
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal()])
    declared = pop.declared_parameters()
    assert {d.name for d in declared} == {
        "sfh_field_psd_sigma",
        "sfh_field_psd_tau_myr",
    }


def test_run_raises_pending_forward_path() -> None:
    """Forward-time batched run is not yet wired; the inference route is."""
    from tengri.protocols.component import ForwardState

    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal()])
    state = ForwardState(wave=jnp.zeros(1))
    with pytest.raises(NotImplementedError, match="not yet wired"):
        pop.run(state, {})


# ── ForwardModel.build(population=...) ──────────────────────────────


def test_forward_model_build_with_population_kwarg() -> None:
    """``ForwardModel.build(population=pop, observation=obs)`` wraps the population."""
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal(), _gal()])
    forward = ForwardModel.build(population=pop, observation=object())
    assert len(forward.populations) == 1
    assert forward.populations[0].sed is pop
    assert forward.populations[0].name == "default"


def test_forward_model_build_rejects_mixing_sed_and_population() -> None:
    """``sed=`` and ``population=`` are mutually exclusive."""
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal()])
    with pytest.raises(ValueError, match="exactly one of"):
        ForwardModel.build(sed=_StubSED(), population=pop, observation=object())


def test_forward_model_build_rejects_mixing_population_and_populations() -> None:
    """``population=`` and ``populations=`` are mutually exclusive."""
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal()])
    with pytest.raises(ValueError, match="exactly one of"):
        ForwardModel.build(
            population=pop,
            populations=[],  # any value triggers the conflict
            observation=object(),
        )


def test_top_level_import() -> None:
    import tengri

    assert tengri.PopulationSEDModel is PopulationSEDModel

# SPDX-License-Identifier: BSD-3-Clause
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


def test_parameter_axes_partitions_shared_and_per_galaxy() -> None:
    """``parameter_axes`` returns 0 for per-galaxy and None for shared."""
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal()])
    axes = pop.parameter_axes(
        {
            "sfh_field_psd_sigma": 1.0,
            "sfh_field_psd_tau_myr": 50.0,
            "sfh_dpl_alpha": jnp.zeros(3),
            "sfh_dpl_log_total_mass": jnp.zeros(3),
        }
    )
    assert axes["sfh_field_psd_sigma"] is None
    assert axes["sfh_field_psd_tau_myr"] is None
    assert axes["sfh_dpl_alpha"] == 0
    assert axes["sfh_dpl_log_total_mass"] == 0


# ── Forward-time batched ``.run`` path (issue #211 final item) ──────


def _real_template(synthetic_ssp, simple_observation):
    """Build a minimal real SEDModel template for batched-vmap tests."""
    from tengri import FIXED, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        # Keep the SFH simple but with one free per-galaxy parameter so
        # we can verify the vmap actually fans out across galaxies.
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(-1.0, 3.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=0.05,
    )


def test_run_vmaps_over_population(synthetic_ssp, simple_observation) -> None:
    """``run`` builds a batched ForwardState across the population via vmap.

    Verifies that:
    - The vmap path executes cleanly on a real SED template.
    - Per-galaxy free params (shape ``(N,)``) produce per-galaxy
      derived quantities with a leading ``N`` axis.
    """
    from tengri.protocols.component import ForwardState

    template = _real_template(synthetic_ssp, simple_observation)

    N = 3
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{"flux_obs": jnp.zeros(5), "noise": jnp.ones(5)} for _ in range(N)],
    )
    # Per-galaxy params: each free param is an array of length N.
    params: dict[str, jnp.ndarray] = {
        name: jnp.array([0.5] * N) for name in template.spec.free_params
    }
    state = ForwardState(wave=jnp.zeros(1))
    out = pop.run(state, params)
    # The output state should have a leading N axis on per-galaxy derived
    # quantities. Find at least one such quantity and check the shape.
    leading_axes = []
    for _key, val in dict(out.derived).items():
        if hasattr(val, "shape") and len(val.shape) > 0:
            leading_axes.append(val.shape[0])
    assert leading_axes, "expected at least one derived array to vmap"
    # Every vmapped quantity should have the same N along axis 0.
    assert all(axis == N for axis in leading_axes), (
        f"expected leading axis of size {N} on vmapped derived quantities; got {leading_axes}"
    )


def test_batched_axes_publishes_galaxy_axis() -> None:
    """PopulationSEDModel.batched_axes publishes the named galaxy axis at position 0."""
    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal()])
    assert pop.batched_axes == {"galaxy": 0}


def test_predict_one_is_un_batched_primitive(synthetic_ssp, simple_observation) -> None:
    """predict_one runs the SED template once, no vmap, no leading axis.

    Composable with outer jax.vmap / pmap / shard_map.
    """
    import jax.numpy as _jnp

    from tengri.protocols.component import ForwardState

    template = _real_template(synthetic_ssp, simple_observation)
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{"flux_obs": _jnp.zeros(5), "noise": _jnp.ones(5)}],
    )
    params_one: dict[str, _jnp.ndarray] = {
        name: _jnp.float64(0.5) for name in template.spec.free_params
    }
    state = ForwardState(wave=_jnp.zeros(1))
    direct = template.run(state, params_one)
    via_predict_one = pop.predict_one(state, params_one)
    # predict_one should be identical to the bare template (it's a thin pass-through).
    for key, val_direct in dict(direct.derived).items():
        if not hasattr(val_direct, "shape"):
            continue
        val_via = dict(via_predict_one.derived).get(key)
        if val_via is not None and hasattr(val_via, "shape"):
            assert val_via.shape == val_direct.shape
            assert _jnp.allclose(val_via, val_direct, rtol=1e-10, atol=0.0, equal_nan=True)


def test_predict_one_composes_with_outer_vmap(synthetic_ssp, simple_observation) -> None:
    """Outer vmap over predict_one matches run — composability sanity check."""
    import jax
    import jax.numpy as _jnp

    from tengri.protocols.component import ForwardState

    template = _real_template(synthetic_ssp, simple_observation)
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{"flux_obs": _jnp.zeros(5), "noise": _jnp.ones(5)} for _ in range(3)],
    )
    state = ForwardState(wave=_jnp.zeros(1))
    params = {name: _jnp.array([0.5] * 3) for name in template.spec.free_params}

    # Default-batched path (uses internal vmap of predict_one)
    via_run = pop.run(state, params)
    # External vmap of predict_one (the un-batched primitive)
    axes = pop.parameter_axes(params)
    via_outer_vmap = jax.vmap(pop.predict_one, in_axes=(None, axes))(state, params)

    # The two paths should be numerically identical.
    for key, val_run in dict(via_run.derived).items():
        if not hasattr(val_run, "shape"):
            continue
        val_outer = dict(via_outer_vmap.derived).get(key)
        if val_outer is not None and hasattr(val_outer, "shape"):
            assert val_outer.shape == val_run.shape
            assert _jnp.allclose(val_outer, val_run, rtol=1e-10, atol=0.0, equal_nan=True)


def test_run_single_galaxy_matches_template_directly(synthetic_ssp, simple_observation) -> None:
    """N=1 vmapped run should match a direct template.run (modulo broadcasting).

    The per-galaxy axis is just a length-1 leading dim; after a squeeze,
    the numbers should match the bare-template forward pass.
    """
    import jax.numpy as _jnp

    from tengri.protocols.component import ForwardState

    template = _real_template(synthetic_ssp, simple_observation)

    # Single-galaxy direct path.
    params_single: dict[str, _jnp.ndarray] = {
        name: _jnp.float64(0.5) for name in template.spec.free_params
    }
    state = ForwardState(wave=_jnp.zeros(1))
    direct = template.run(state, params_single)

    # Vmap path with N=1.
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{"flux_obs": _jnp.zeros(5), "noise": _jnp.ones(5)}],
    )
    params_batched = {name: _jnp.array([0.5]) for name in template.spec.free_params}
    batched = pop.run(state, params_batched)

    # Compare one derived quantity, squeezing the leading axis.
    found_match = False
    for key, val_direct in dict(direct.derived).items():
        if not hasattr(val_direct, "shape"):
            continue
        val_batched = dict(batched.derived).get(key)
        if val_batched is None or not hasattr(val_batched, "shape"):
            continue
        # Vmap added one leading axis of size 1
        if val_batched.shape[0] == 1 and val_batched.shape[1:] == val_direct.shape:
            assert _jnp.allclose(val_batched[0], val_direct, rtol=1e-10, atol=0.0, equal_nan=True)
            found_match = True
    assert found_match, "could not find a comparable derived quantity"


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


# ── PopulationSpecView wiring (single-Hamiltonian-Fitter plan, Task 3) ──


def test_spec_returns_population_spec_view() -> None:
    """``PopulationSEDModel.spec`` returns the batched view, not the bare template."""
    from typing import ClassVar

    from tengri.parameters._population_view import PopulationSpecView

    class _StubSpec:
        free_params: ClassVar[list[str]] = ["sfh_dpl_alpha"]
        all_params: ClassVar[list[str]] = ["sfh_dpl_alpha"]
        fixed_params: ClassVar[list[str]] = []
        n_free: int = 1
        stochastic: bool = False
        _distributions: ClassVar[dict] = {}

        def get_fixed_values(self):
            return {}

        def get_distribution(self, name):
            raise KeyError(name)

        def sample(self, key):
            return {"sfh_dpl_alpha": 0.5}

    class _StubSED:
        name = "stub"
        spec = _StubSpec()

    pop = PopulationSEDModel(sed=_StubSED(), galaxies=[_gal(), _gal()])
    view = pop.spec
    assert isinstance(view, PopulationSpecView)
    # The view wraps the template spec
    assert view.free_params == _StubSpec.free_params
    # Population size flows through
    assert view._n_galaxies == 2


# ── batched_data helper (single-Hamiltonian-Fitter plan, Task 4) ──────


def test_batched_data_stacks_flux_and_noise() -> None:
    """``batched_data()`` returns (N, n_filters) arrays for Fitter consumption."""
    pop = PopulationSEDModel(
        sed=_StubSED(),
        galaxies=[
            {"flux_obs": jnp.array([1.0, 2.0, 3.0]), "noise": jnp.array([0.1, 0.2, 0.3])},
            {"flux_obs": jnp.array([4.0, 5.0, 6.0]), "noise": jnp.array([0.4, 0.5, 0.6])},
            {"flux_obs": jnp.array([7.0, 8.0, 9.0]), "noise": jnp.array([0.7, 0.8, 0.9])},
        ],
    )
    flux, noise = pop.batched_data()
    assert flux.shape == (3, 3)
    assert noise.shape == (3, 3)
    assert jnp.allclose(flux[0], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(noise[2], jnp.array([0.7, 0.8, 0.9]))


def test_construction_raises_on_missing_keys() -> None:
    """Galaxy dict missing flux_obs/noise → fail fast at construction."""
    with pytest.raises(ValueError, match="missing required 'noise'"):
        PopulationSEDModel(
            sed=_StubSED(),
            galaxies=[{"flux_obs": jnp.zeros(3)}],  # missing noise
        )


def test_construction_raises_on_heterogeneous_grids() -> None:
    """Galaxies with mismatched data shapes violate the shared-grid contract."""
    with pytest.raises(ValueError, match="must share one measurement grid"):
        PopulationSEDModel(
            sed=_StubSED(),
            galaxies=[
                {"flux_obs": jnp.zeros(3), "noise": jnp.ones(3)},
                {"flux_obs": jnp.zeros(5), "noise": jnp.ones(5)},  # different n
            ],
        )


# ── Fitter standard path (single-Hamiltonian milestone) ────────────────
# The legacy ``_maybe_population_delegate`` was removed once the
# standard inference path (Fitter(forward).run(...)) was proven to
# work end-to-end on hierarchical fits (PRs #241-#245). Tests of
# that delegation are deleted with the code. The standard path's
# behavior is pinned in test_single_hamiltonian_path_probe.py.

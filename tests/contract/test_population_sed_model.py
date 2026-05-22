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
            "sfh_dpl_log_peak_sfr": jnp.zeros(3),
        }
    )
    assert axes["sfh_field_psd_sigma"] is None
    assert axes["sfh_field_psd_tau_myr"] is None
    assert axes["sfh_dpl_alpha"] == 0
    assert axes["sfh_dpl_log_peak_sfr"] == 0


# ── Forward-time batched ``.run`` path (issue #211 final item) ──────


def _real_template(synthetic_ssp, simple_observation):
    """Build a minimal real SEDModel template for batched-vmap tests."""
    from tengri import FIXED, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        # Keep the SFH simple but with one free per-galaxy parameter so
        # we can verify the vmap actually fans out across galaxies.
        sfh={"type": "dpl", "*": FIXED, "log_peak_sfr": Uniform(-1.0, 3.0)},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
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
            assert _jnp.allclose(val_batched[0], val_direct, rtol=1e-10, atol=0.0)
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


# ── Fitter routing (issue #211 — inference side) ────────────────────


def test_fitter_routes_to_population_fitter_for_population_forward() -> None:
    """`Fitter(forward)` where forward holds PopulationSEDModel
    constructs a PopulationFitter under the hood, no data/noise needed."""
    from tengri.inference.fitter import _maybe_population_delegate
    from tengri.inference.hierarchical import PopulationFitter

    class _StubSpec:
        free_params: tuple = ()

    class _StubSEDWithSpec:
        name = "stub_with_spec"
        spec = _StubSpec()

        def with_fixed(self, **kw):
            return self

    pop = PopulationSEDModel(
        sed=_StubSEDWithSpec(),
        galaxies=[_gal(), _gal()],
    )
    forward = ForwardModel.build(population=pop, observation=object())
    delegate = _maybe_population_delegate(forward)
    assert isinstance(delegate, PopulationFitter)
    assert delegate.n_galaxies == 2


def test_fitter_rejects_unsupported_shared_for_routing() -> None:
    """If shared= isn't the canonical PSD pair, the routing surfaces a clear error."""
    from tengri.inference.fitter import _maybe_population_delegate

    class _StubSEDWithSpec:
        name = "stub_with_spec"
        spec = None

        def with_fixed(self, **kw):
            return self

    pop = PopulationSEDModel(
        sed=_StubSEDWithSpec(),
        galaxies=[_gal()],
        shared=("met_logzsol",),
        priors={"met_logzsol": (-2.0, 0.5)},
    )
    forward = ForwardModel.build(population=pop, observation=object())
    with pytest.raises(NotImplementedError, match="issue #211"):
        _maybe_population_delegate(forward)


def test_fitter_returns_none_for_non_population_models() -> None:
    """Non-population models pass through with no delegate."""
    from tengri.inference.fitter import _maybe_population_delegate

    assert _maybe_population_delegate("not even a forward") is None
    assert _maybe_population_delegate(object()) is None

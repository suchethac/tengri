# SPDX-License-Identifier: BSD-3-Clause
"""Conformance tests for :class:`PopulationSpecView`.

The view duck-types :class:`Parameters` so the standard
:class:`Fitter` consumes it without knowing it's hierarchical. These
tests pin every attribute / method on the implicit Protocol surface
that :mod:`tengri.inference` reaches for, so a duck-typing leak
fails loudly here rather than silently producing wrong-shape outputs
inside a JIT'd loss function.

Task 2.5 from the plan
``docs/internal/plans/2026-05-22-single-hamiltonian-fitter.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri.parameters._population_view import PopulationSpecView
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)


def _template() -> Parameters:
    """A minimal template Parameters spec with both free and fixed params."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Fixed(0.3),
        redshift=Fixed(0.1),
    )


def _view(n: int = 3) -> PopulationSpecView:
    return PopulationSpecView(
        template=_template(),
        n_galaxies=n,
        shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
    )


# ── Pass-through attributes ─────────────────────────────────────────


def test_free_params_matches_template() -> None:
    view = _view()
    assert view.free_params == _template().free_params


def test_all_params_matches_template() -> None:
    view = _view()
    assert view.all_params == _template().all_params


def test_fixed_params_matches_template() -> None:
    view = _view()
    assert view.fixed_params == _template().fixed_params


def test_n_free_matches_template() -> None:
    view = _view()
    assert view.n_free == _template().n_free


def test_stochastic_flag_matches_template() -> None:
    view = _view()
    assert view.stochastic == _template().stochastic


def test_get_fixed_values_matches_template() -> None:
    view = _view()
    assert view.get_fixed_values() == _template().get_fixed_values()


def test_distributions_dict_matches_template() -> None:
    view = _view()
    assert set(view._distributions) == set(_template()._distributions)


def test_get_distribution_matches_template() -> None:
    """View routes ``get_distribution`` to the wrapped template."""
    template = _template()
    view = PopulationSpecView(
        template=template,
        n_galaxies=3,
        shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
    )
    for name in template.free_params:
        # Same backing template ⇒ identity holds.
        assert view.get_distribution(name) is template.get_distribution(name)


# ── Construction guards ────────────────────────────────────────────


def test_rejects_zero_galaxies() -> None:
    with pytest.raises(ValueError, match="n_galaxies"):
        PopulationSpecView(template=_template(), n_galaxies=0, shared=())


# ── Batched sampling ───────────────────────────────────────────────


def test_sample_per_galaxy_params_have_leading_n_axis() -> None:
    """Per-galaxy free params get a (N,) leading axis in sampled draws."""
    N = 4
    view = _view(n=N)
    key = jax.random.PRNGKey(0)
    sample = view.sample(key)
    for name in view.free_params:
        if name in view._shared:
            continue
        val = sample[name]
        assert hasattr(val, "shape"), f"{name} should be a JAX array"
        assert val.shape == (N,), (
            f"per-galaxy free param {name} expected shape ({N},), got {val.shape}"
        )


def test_sample_shared_params_are_scalar() -> None:
    """Shared parameters get a single (scalar) draw."""
    N = 5
    view = _view(n=N)
    sample = view.sample(jax.random.PRNGKey(1))
    for name in view._shared:
        if name not in sample:
            continue
        val = sample[name]
        shape = getattr(val, "shape", ())
        assert shape == (), f"shared param {name} expected scalar, got shape {shape}"


def test_sample_values_in_bounds() -> None:
    """Every sampled per-galaxy value lies inside its Uniform prior bounds."""
    view = _view(n=10)
    sample = view.sample(jax.random.PRNGKey(2))
    template = _template()
    for name in view.free_params:
        dist = template.get_distribution(name)
        if not hasattr(dist, "low"):
            continue
        val = sample[name]
        assert bool(jnp.all(val >= dist.low - 1e-9))
        assert bool(jnp.all(val <= dist.high + 1e-9))


def test_sample_is_jit_compilable() -> None:
    """sample must be wrappable in jax.jit (closes over the view)."""
    view = _view(n=3)

    @jax.jit
    def call_sample(key):
        return view.sample(key)

    out = call_sample(jax.random.PRNGKey(7))
    # Just verify we got something out
    assert len(out) > 0

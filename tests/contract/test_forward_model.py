# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ForwardModel (forward-model architecture §5–§6).

Covers both the single-population convenience path and the
multi-population namespace (ADR-0012).
"""

from __future__ import annotations

import pytest

from tengri import Fixed
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population import Population

pytestmark = pytest.mark.contract


@pytest.fixture
def sed_model_minimal(synthetic_ssp, simple_observation):
    from tengri import FIXED, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def test_build_single_population_from_sed_kwarg(sed_model_minimal, simple_observation) -> None:
    forward = ForwardModel.build(
        sed=sed_model_minimal,
        observation=simple_observation,
    )
    assert isinstance(forward, ForwardModel)
    assert len(forward.populations) == 1
    assert forward.populations[0].name == "default"


def test_build_rejects_no_sed_no_populations(simple_observation) -> None:
    with pytest.raises(ValueError, match=r"sed=.*populations="):
        ForwardModel.build(observation=simple_observation)


def test_build_accepts_explicit_populations(sed_model_minimal, simple_observation) -> None:
    pop = Population(name="only", sed=sed_model_minimal)
    forward = ForwardModel.build(populations=[pop], observation=simple_observation)
    assert forward.populations[0].name == "only"


def test_build_accepts_multi_population(sed_model_minimal, simple_observation) -> None:
    """ADR-0012: build accepts >1 distinct populations."""
    pops = [
        Population(name="a", sed=sed_model_minimal),
        Population(name="b", sed=sed_model_minimal),
    ]
    forward = ForwardModel.build(populations=pops, observation=simple_observation)
    assert {p.name for p in forward.populations} == {"a", "b"}


def test_build_rejects_duplicate_population_names(sed_model_minimal, simple_observation) -> None:
    """Distinct names required (ADR-0012)."""
    pops = [
        Population(name="agn", sed=sed_model_minimal),
        Population(name="agn", sed=sed_model_minimal),
    ]
    with pytest.raises(ValueError, match="distinct"):
        ForwardModel.build(populations=pops, observation=simple_observation)


def test_predict_returns_mapping_with_expected_keys(sed_model_minimal, simple_observation) -> None:
    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    params = {name: 0.5 for name in sed_model_minimal.spec.free_params}
    pred = forward.predict_observables(params)
    assert isinstance(pred, dict)
    # Must publish at least one photometric-channel key.
    assert any(k in pred for k in ("phot_fnu", "fnu_obs"))


def test_build_accepts_spatial_kwarg(sed_model_minimal, simple_observation) -> None:
    """ForwardModel.build(spatial=...) wraps SED + Spatial into one Population."""
    from tengri.components.spatial.sersic import Sersic
    from tengri.forward.spatial_model import SpatialModel

    spatial = SpatialModel(components=[Sersic()])
    forward = ForwardModel.build(
        sed=sed_model_minimal,
        spatial=spatial,
        observation=simple_observation,
    )
    assert forward.populations[0].spatial is spatial


def test_build_rejects_spatial_without_sed(simple_observation) -> None:
    """spatial=... requires sed=... too."""
    from tengri.components.spatial.sersic import Sersic
    from tengri.forward.spatial_model import SpatialModel

    spatial = SpatialModel(components=[Sersic()])
    with pytest.raises(ValueError, match="requires sed"):
        ForwardModel.build(spatial=spatial, observation=simple_observation)


def test_predict_with_spatial_threads_state(sed_model_minimal, simple_observation) -> None:
    """When spatial is present, the spatial sub-model runs without breaking photometry."""
    import jax.numpy as jnp

    from tengri.components.spatial.sersic import Sersic
    from tengri.forward.spatial_model import SpatialModel

    forward = ForwardModel.build(
        sed=sed_model_minimal,
        spatial=SpatialModel(components=[Sersic()]),
        observation=simple_observation,
    )
    # SED params (all fixed in the minimal model) + Sersic free params
    params: dict = dict.fromkeys(sed_model_minimal.spec.free_params, 0.5)
    params.update(
        {
            "spatial_re_kpc": jnp.float64(1.0),
            "spatial_n": jnp.float64(1.0),
            "spatial_axis_ratio": jnp.float64(1.0),
            "spatial_pa_deg": jnp.float64(0.0),
        }
    )
    pred = forward.predict_observables(params)
    # Photometry still works (no observation adapter consumes the spatial
    # profile yet — that's the next slice of item #6).
    new_phot = pred.get("phot_fnu", pred.get("fnu_obs"))
    assert new_phot is not None
    assert jnp.all(jnp.isfinite(new_phot))


def test_predict_matches_legacy_sedmodel(sed_model_minimal, simple_observation) -> None:
    """The shell must not change the numerical result vs the existing path."""
    import jax.numpy as jnp

    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    params = {name: 0.5 for name in sed_model_minimal.spec.free_params}
    pred_new = forward.predict_observables(params)
    pred_old = sed_model_minimal.predict_photometry(params)

    new_phot = pred_new.get("phot_fnu", pred_new.get("fnu_obs"))
    assert new_phot is not None, f"Prediction dict missing photometric key: {list(pred_new)}"
    assert jnp.allclose(new_phot, pred_old, rtol=1e-10, atol=0.0)


# ── Multi-population (ADR-0012) ─────────────────────────────────────


def test_multi_population_predict_sums_in_linear_flux(
    sed_model_minimal, simple_observation
) -> None:
    """Two populations with identical SEDs ⇒ summed flux = 2 × single-pop flux."""
    import jax.numpy as jnp

    single = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    twin = ForwardModel.build(
        populations=[
            Population(name="a", sed=sed_model_minimal),
            Population(name="b", sed=sed_model_minimal),
        ],
        observation=simple_observation,
    )
    pred_single = single.predict_observables({})
    pred_twin = twin.predict_observables({})

    single_phot = pred_single.get("phot_fnu", pred_single.get("fnu_obs"))
    twin_phot = pred_twin.get("phot_fnu", pred_twin.get("fnu_obs"))
    assert jnp.allclose(twin_phot, 2.0 * single_phot, rtol=1e-10)


def test_multi_population_params_slice_by_namespace(sed_model_minimal, simple_observation) -> None:
    """Namespaced params reach the right population; bare names flow everywhere."""
    import jax.numpy as jnp

    forward = ForwardModel.build(
        populations=[
            Population(name="a", sed=sed_model_minimal),
            Population(name="b", sed=sed_model_minimal),
        ],
        observation=simple_observation,
    )
    # No free params in this minimal model, but the namespace path should
    # still produce finite output (this exercises _params_for_population).
    pred = forward.predict_observables({"redshift": 0.05})
    phot = pred.get("phot_fnu", pred.get("fnu_obs"))
    assert phot is not None
    assert jnp.all(jnp.isfinite(phot))


def test_multi_population_cross_pop_namespaced_extras(sed_model_minimal) -> None:
    """Pass 2 of `ForwardModel.predict` injects every other population's derived
    bundle under namespaced keys (``"<pop>.<key>"``) into each population's
    ``state.derived._extras``, so downstream components / observations can read
    cross-population products (e.g. AGN ``L_bolometric``). This is the plumbing
    that backs ADR-0012 §6.2 and architecture spec §9.1.
    """

    captured: dict[str, dict] = {}

    class _CapturingObservation:
        def predict_summed(self, per_pop_states, per_pop_params):
            for name, state in per_pop_states.items():
                captured[name] = dict(state.derived._extras)
            # Trivial return — test inspects ``captured`` not the result.
            return {"phot_fnu": 0.0}

    forward = ForwardModel.build(
        populations=[
            Population(name="a", sed=sed_model_minimal),
            Population(name="b", sed=sed_model_minimal),
        ],
        observation=_CapturingObservation(),
    )
    forward.predict_observables({"redshift": 0.05})

    assert set(captured) == {"a", "b"}
    # Every typed-derived key published by population "a" must surface in
    # "b"'s _extras under the "a." prefix, and vice versa.
    a_namespaced = {k for k in captured["a"] if k.startswith("b.")}
    b_namespaced = {k for k in captured["b"] if k.startswith("a.")}
    assert a_namespaced, "Pop 'a' should see 'b.*' namespaced keys after Pass 2"
    assert b_namespaced, "Pop 'b' should see 'a.*' namespaced keys after Pass 2"
    # Symmetric: identical SubModels publish identical key sets.
    assert {k[2:] for k in a_namespaced} == {k[2:] for k in b_namespaced}


# ── Migration-2 public-property delegation contract ─────────────────


@pytest.mark.parametrize(
    "attr",
    [
        "wave_obs",
        "has_fixedz_photometry_precompute",
        "hybrid",
        "z_fixed",
        "dl_cm_fixed",
        "n_grid",
        "uses_stochastic_sfh",
        "wavelengths",
    ],
)
def test_forward_model_public_property_matches_inner_sed(
    sed_model_minimal, simple_observation, attr
) -> None:
    """Migration 2 step 2/4 contract: each public property promoted off
    the legacy ``__getattr__`` fall-through must read identically to the
    inner SED's same-named attribute.

    Pins the property surface so future refactors don't silently drop a
    delegation (or, worse, redirect it through stale state).
    """
    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    inner_value = getattr(sed_model_minimal, attr)
    forward_value = getattr(forward, attr)
    assert forward_value is inner_value or forward_value == inner_value


def test_forward_model_compile_signature_matches_inner_sed(
    sed_model_minimal, simple_observation
) -> None:
    """``compile_signature`` is the JIT cache key — it MUST be stable across
    the SEDModel / ForwardModel boundary so Fitter cache lookups don't fork.
    """
    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    assert forward.compile_signature() == sed_model_minimal.compile_signature()


def test_forward_model_predict_delegates_match_inner_sed(
    sed_model_minimal, simple_observation
) -> None:
    """Spot-check that explicit method delegates produce the same output as
    calling the inner SED directly. Guards against accidental wrapper drift.
    """
    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    params = {"redshift": 0.05}
    import jax.numpy as jnp

    direct_phot = sed_model_minimal.predict_photometry(params)
    forward_phot = forward.predict_photometry(params)
    assert jnp.allclose(direct_phot, forward_phot, rtol=1e-12)


def test_forward_model_feature_delegates_forward_state_kwarg(
    sed_model_minimal, simple_observation
) -> None:
    """The line-flux / line-ratio / spectral-index delegates must forward
    keyword extras (``tolerance_aa`` and the shared-forward ``state=``).

    Regression: the joint-loss fast path (``_build_prediction`` computes
    ``predict_state`` once and threads it into each feature channel via
    ``state=``). ForwardModel's explicit delegates dropped that kwarg, so a
    joint phot+lines fit on the canonical ForwardModel path raised
    ``TypeError: unexpected keyword argument 'state'``. ``predict_line_ratios``
    had no delegate at all. Signature-based so it needs no SSP data.
    """
    import inspect

    forward = ForwardModel.build(sed=sed_model_minimal, observation=simple_observation)
    for name in ("predict_line_fluxes", "predict_line_ratios", "predict_spectral_indices"):
        assert hasattr(forward, name), f"ForwardModel missing {name} delegate"
        params = inspect.signature(getattr(forward, name)).parameters
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
            f"ForwardModel.{name} must accept **kwargs to forward state=/tolerance_aa "
            f"to the inner SEDModel; got {list(params)}"
        )


def test_forward_model_delegation_walks_into_population_sed_model(
    sed_model_minimal, simple_observation
) -> None:
    """ForwardModel wrapping a :class:`PopulationSEDModel` must resolve the
    promoted properties through the *two-level* chain
    ``forward.populations[0].sed`` → ``pop.sed`` (the template SEDModel).

    Migration 2's ``_inner_sed_for_delegation`` does ``getattr(sub, 'sed', sub)``
    — this test pins that walk so a future refactor doesn't accidentally
    flatten back to the single-level lookup and break hierarchical fits.
    """
    import jax.numpy as jnp

    from tengri.forward.population_sed_model import PopulationSEDModel

    pop = PopulationSEDModel(
        sed=sed_model_minimal,
        galaxies=[
            {"flux_obs": jnp.ones(3) * 1e-18, "noise": jnp.ones(3) * 1e-19} for _ in range(2)
        ],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    # Each promoted property must reach the *template* (sed_model_minimal),
    # not stop at the PopulationSEDModel wrapper.
    template = sed_model_minimal
    assert forward.wave_obs is getattr(template, "wave_obs", None) or forward.wave_obs == getattr(
        template, "wave_obs", None
    )
    assert forward.has_fixedz_photometry_precompute == template.has_fixedz_photometry_precompute
    assert forward.z_fixed == template.z_fixed
    assert forward.n_grid == template.n_grid
    assert forward.uses_stochastic_sfh == template.uses_stochastic_sfh
    assert forward.wavelengths is template.wavelengths
    # compile_signature is structural — must match the template, not the wrapper.
    assert forward.compile_signature() == template.compile_signature()

"""Tests for ForwardModel (forward-model architecture §5–§6).

Covers both the single-population convenience path and the
multi-population namespace (ADR-0012).
"""

from __future__ import annotations

import pytest

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
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
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


def test_build_rejects_duplicate_population_names(
    sed_model_minimal, simple_observation
) -> None:
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
    pred = forward.predict(params)
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
    pred = forward.predict(params)
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
    pred_new = forward.predict(params)
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
    pred_single = single.predict({})
    pred_twin = twin.predict({})

    single_phot = pred_single.get("phot_fnu", pred_single.get("fnu_obs"))
    twin_phot = pred_twin.get("phot_fnu", pred_twin.get("fnu_obs"))
    assert jnp.allclose(twin_phot, 2.0 * single_phot, rtol=1e-10)


def test_multi_population_params_slice_by_namespace(
    sed_model_minimal, simple_observation
) -> None:
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
    pred = forward.predict({"redshift": 0.05})
    phot = pred.get("phot_fnu", pred.get("fnu_obs"))
    assert phot is not None
    assert jnp.all(jnp.isfinite(phot))

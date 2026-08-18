# SPDX-License-Identifier: BSD-3-Clause
"""Fresh-user experience scenarios beyond ``test_user_scenarios.py``.

Each test walks a path a new user actually takes -- picking a recipe, reading a
menu, calling an accessor, mistyping a name -- and asserts the *documented*
contract, not an implementation detail. Where a surface is allowed to refuse,
the test asserts the refusal is actionable: it must name the problem and a fix.

The contracts asserted here are stated in ``CLAUDE.md`` (Prediction API section)
and ``docs/dev/NAMING_CONTRACT.md`` section 4b.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tengri
from tengri import FREE, Fitter, Observation, Photometry, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_WNE_SSP = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

pytestmark = pytest.mark.skipif(not _WNE_SSP.is_file(), reason="shipped wNE SSP grid required")

# Recipes split by whether they accept the one SSP grid the repo tracks.
_RECIPES_THAT_BUILD = ("dust_demo", "high_z", "mock_recovery_minimal", "photoz")
_RECIPES_THAT_REFUSE = (
    "star_forming_photometry",
    "quiescent_z0",
    "stochastic_sfh_jwst",
    "agn_panchromatic",
    "composable_agn",
)


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_WNE_SSP))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_filter_set(
            load_filter_set(["hst_f606w", "hst_f160w", "irac_36"])
        )
    )


@pytest.fixture(scope="module")
def model(ssp, obs):
    return SEDModel.build(ssp_data=ssp, observation=obs, **tengri.recipes.mock_recovery_minimal())


@pytest.fixture(scope="module")
def params(model):
    return model.spec.sample(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def pred(model, params):
    return model.predict(params)


# ── A. Recipes and construction ───────────────────────────────────


@pytest.mark.contract
def test_uc01_every_listed_recipe_is_callable():
    """The advertised menu and the importable functions agree."""
    listed = tengri.list_recipes()
    assert len(listed) > 0, "list_recipes() advertises nothing"
    public = [r for r in dir(tengri.recipes) if not r.startswith("_")]
    for name in public:
        fn = getattr(tengri.recipes, name)
        if callable(fn):
            assert isinstance(fn(), dict), f"{name}() must return build kwargs"


@pytest.mark.contract
@pytest.mark.parametrize("recipe", _RECIPES_THAT_BUILD)
def test_uc02_recipe_builds_with_the_shipped_grid(ssp, obs, recipe):
    """A recipe compatible with the tracked SSP builds without extra downloads.

    ``dust_demo`` is documented as forward-only with *every* parameter FIXED,
    so zero free parameters is its contract, not a defect. Every other recipe
    here is meant to be fitted and must expose something to fit.
    """
    kwargs = getattr(tengri.recipes, recipe)()
    built = SEDModel.build(ssp_data=ssp, observation=obs, **kwargs)
    free = built.spec.free_params
    if recipe == "dust_demo":
        assert not free, (
            "dust_demo documents every parameter as FIXED for sweep_parameter; "
            f"it now exposes free parameters: {free}"
        )
    else:
        assert free, f"{recipe} is a fitting recipe but built with nothing free"


@pytest.mark.contract
@pytest.mark.parametrize("recipe", _RECIPES_THAT_REFUSE)
def test_uc03_incompatible_recipe_refuses_actionably(ssp, obs, recipe):
    """A recipe that cannot use this grid must say so *and* offer a way out.

    Refusing is correct -- stacking Cue on a wNE grid double-counts nebular
    emission. Refusing without naming a fix is a dead end for a new user.
    """
    kwargs = getattr(tengri.recipes, recipe)()
    with pytest.raises(Exception) as excinfo:
        SEDModel.build(ssp_data=ssp, observation=obs, **kwargs)
    msg = str(excinfo.value).lower()
    assert "ssp" in msg or "grid" in msg, f"{recipe}: refusal never mentions the SSP"
    assert any(w in msg for w in ("fix", "use", "drop", "instead")), (
        f"{recipe}: refusal offers no remedy -- message was: {excinfo.value}"
    )


@pytest.mark.contract
def test_uc04_nested_dict_grammar_builds(ssp, obs):
    """The documented hand-rolled grammar works as written in CLAUDE.md."""
    built = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1.0, 3.0)},
        redshift=tengri.Fixed(0.5),
    )
    assert any("sfh_dpl" in p for p in built.spec.free_params)


@pytest.mark.contract
def test_uc05_builder_factory_matches_dict_grammar(ssp, obs):
    """``builders.sfh.dpl`` is sugar for the dict form, not a second dialect."""
    from tengri import builders

    via_builder = SEDModel.build(ssp_data=ssp, observation=obs, sfh=builders.sfh.dpl(_=FREE))
    via_dict = SEDModel.build(
        ssp_data=ssp, observation=obs, sfh={"type": "dpl", "all_params": FREE}
    )
    assert set(via_builder.spec.free_params) == set(via_dict.spec.free_params)


@pytest.mark.contract
def test_uc06_summary_reports_provenance(model, capsys):
    """``spec.summary()`` prints provenance tags so a user can audit defaults."""
    model.spec.summary()
    out = capsys.readouterr().out
    assert out.strip(), "summary() printed nothing"
    assert any(tag in out for tag in ("[user]", "all_params", "[default]")), (
        "summary() prints no provenance tags"
    )


@pytest.mark.contract
def test_uc07_to_groups_round_trips(ssp, obs, model):
    """``to_groups()`` output can rebuild an equivalent model (#1589)."""
    groups = model.spec.to_groups()
    assert isinstance(groups, dict) and groups
    rebuilt = SEDModel.build(ssp_data=ssp, observation=obs, **groups)
    assert set(rebuilt.spec.free_params) == set(model.spec.free_params), (
        "round-tripping to_groups() changed the free-parameter set"
    )


@pytest.mark.contract
def test_uc08_free_params_use_full_prefixes(model):
    """NAMING_CONTRACT 3.2: free parameters carry their full prefix."""
    for name in model.spec.free_params:
        assert "_" in name, f"{name!r} looks like un-prefixed shorthand"
        assert name == name.lower(), f"{name!r} is not snake_case"


@pytest.mark.contract
def test_uc09_removed_stellar_group_is_refused_with_a_translation(ssp, obs):
    """The retired ``stellar={'met_mode':...}`` spelling must not fail silently."""
    with pytest.raises(Exception) as excinfo:
        SEDModel.build(ssp_data=ssp, observation=obs, stellar={"met_mode": "table"})
    msg = str(excinfo.value)
    assert "met" in msg.lower(), f"refusal does not point at the replacement group: {msg}"


@pytest.mark.contract
def test_uc10_unknown_type_suggests_valid_names(ssp, obs):
    """A typo gets a domain-scoped suggestion, not a bare KeyError (#1917)."""
    with pytest.raises(Exception) as excinfo:
        SEDModel.build(ssp_data=ssp, observation=obs, sfh={"type": "dlp"})
    msg = str(excinfo.value).lower()
    assert "dpl" in msg or "did you mean" in msg or "valid" in msg, (
        f"no suggestion offered for a near-miss type name: {excinfo.value}"
    )


# ── B. Prediction API contracts (CLAUDE.md section 4b) ────────────


@pytest.mark.contract
def test_uc11_predict_takes_params_and_nothing_else(model, params, pred):
    """Rule 1: resampling lives on the accessor, never on ``predict``."""
    with pytest.raises(TypeError):
        model.predict(params, wave=pred.wave_rest)


@pytest.mark.contract
def test_uc12_bare_accessor_fails_loudly(pred):
    """Rule 5: ``pred.rest_sed`` without ``()`` must raise, not coerce."""
    with pytest.raises(TypeError) as excinfo:
        jnp.asarray(pred.rest_sed)
    assert "parenthes" in str(excinfo.value).lower(), (
        "the error does not tell the user what they did wrong"
    )


@pytest.mark.contract
def test_uc13_rest_and_obs_sed_are_both_luminosities(pred):
    """Rule: both are L_nu; they differ by axis and IGM, not by unit."""
    rest, obs_ = pred.rest_sed(), pred.obs_sed()
    assert rest.shape == obs_.shape
    assert jnp.all(jnp.isfinite(rest)) and jnp.all(jnp.isfinite(obs_))
    ratio = jnp.median(obs_[obs_ > 0]) / jnp.median(rest[rest > 0])
    assert 1e-3 < ratio < 1e3, (
        f"obs_sed/rest_sed median ratio {ratio:.3e} suggests a unit change, "
        "but both must be L_nu [erg/s/Hz]"
    )


@pytest.mark.contract
def test_uc14_obs_sed_is_not_a_flux(pred):
    """CLAUDE.md 4b.3b: treating obs_sed as a flux is wrong by ~57 dex."""
    lum = jnp.median(pred.obs_sed()[pred.obs_sed() > 0])
    flux = jnp.median(pred.photometry())
    assert lum / flux > 1e20, (
        "obs_sed and photometry are within 20 dex -- one of them is not in the "
        "unit its contract claims"
    )


@pytest.mark.contract
def test_uc15_observed_axis_is_the_rest_axis_redshifted(model, params, pred):
    """Rule 4: never hand-roll ``wave*(1+z)`` -- the object carries its axis."""
    z = float(model._get_redshift(params))
    np.testing.assert_allclose(
        np.asarray(pred.wave_obs), np.asarray(pred.wave_rest) * (1.0 + z), rtol=1e-6
    )


@pytest.mark.contract
def test_uc16_photometry_is_finite_and_positive(pred, model):
    """A user's first plot must not contain NaN or negative flux."""
    phot = pred.photometry()
    assert phot.shape[0] == 3
    assert jnp.all(jnp.isfinite(phot)), "photometry contains non-finite entries"
    assert jnp.all(phot > 0), "photometry contains non-positive flux"


@pytest.mark.gradient
def test_uc17_predict_photometry_is_jit_and_vmap_safe(model, params):
    """The inference hot path must survive jit, vmap and grad."""
    jitted = jax.jit(model.predict_photometry)
    np.testing.assert_allclose(
        np.asarray(jitted(params)),
        np.asarray(model.predict_photometry(params)),
        rtol=1e-6,
    )
    batch = jax.tree_util.tree_map(lambda v: jnp.stack([v, v]), params)
    out = jax.vmap(model.predict_photometry)(batch)
    assert out.shape[0] == 2
    free = model.spec.free_params[0]
    g = jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))(params)
    assert jnp.isfinite(g[free]), f"gradient wrt {free} is not finite"


@pytest.mark.contract
def test_uc18_predict_properties_returns_what_was_asked(model, params):
    """The single jit/vmap surface for derived quantities honours ``names``."""
    got = model.predict_properties(params, names=("stellar_mass",))
    assert set(got) == {"stellar_mass"}
    assert jnp.isfinite(jnp.asarray(got["stellar_mass"]))


@pytest.mark.contract
def test_uc19_fixed_redshift_must_not_be_read_with_dict_get(model, params):
    """Rule 2: a Fixed redshift is legitimately absent from ``params``.

    ``params.get("redshift", 0.0)`` would silently place the galaxy at 10 pc.
    """
    z = float(model._get_redshift(params))
    assert z > 0.0, "test model has no redshift to speak of"
    if "redshift" not in params:
        assert params.get("redshift", 0.0) == 0.0
        assert z != 0.0, (
            "redshift is absent from params and _get_redshift agrees with the "
            "dangerous default -- the guard would be untestable"
        )


# ── C. Discovery and error quality ────────────────────────────────


@pytest.mark.contract
def test_uc20_menus_are_non_empty():
    """A user exploring by menu must not meet an empty list."""
    everything = tengri.list_all()
    assert isinstance(everything, dict) and everything
    empty = [k for k, v in everything.items() if not v]
    assert not empty, f"list_all() advertises empty domains: {empty}"


@pytest.mark.contract
def test_uc21_describe_resolves_advertised_names():
    """``describe()`` must resolve the names the menus advertise (#1560)."""
    names, unresolved = [], []
    for value in tengri.list_all().values():
        # Registry tables expose their advertised names directly; fall back to
        # the row dicts so a table shape change surfaces as a failure, not a
        # silently empty sweep.
        table_names = getattr(value, "names", None)
        if callable(table_names):
            table_names = table_names()
        if table_names:
            names.extend(str(n) for n in table_names)
        else:
            for row in value:
                if isinstance(row, dict) and "name" in row:
                    names.append(str(row["name"]))
    assert names, "could not enumerate any advertised name"
    # Every advertised name, not a sample: a capped sweep reports "clean" while
    # leaving the untested tail free to break.
    for name in names:
        try:
            tengri.describe(name)
        except Exception as exc:
            unresolved.append(f"{name} -> {type(exc).__name__}")
    assert not unresolved, (
        f"describe() failed on {len(unresolved)}/{len(names)} advertised names: {unresolved[:15]}"
    )


@pytest.mark.contract
def test_uc22_unknown_inference_method_is_a_parameter_error(model):
    """``"vi_native"`` does not exist; the refusal must list real methods."""
    rng = np.random.default_rng(0)
    flux = jnp.asarray(rng.uniform(1e-29, 1e-27, size=3))
    fitter = Fitter(model, data=flux, noise=flux / 10.0)
    with pytest.raises(ValueError) as excinfo:
        fitter.run("vi_native", key=jax.random.PRNGKey(0))
    assert not isinstance(excinfo.value, KeyError)
    assert "vi_native" in str(excinfo.value) or "method" in str(excinfo.value).lower()

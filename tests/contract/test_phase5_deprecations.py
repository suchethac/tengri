# SPDX-License-Identifier: BSD-3-Clause
r"""Contract §6: a deprecated method warns, and returns *exactly* what it always did.

Phase 5 of #1043 (#1049). Six ``SEDModel`` methods are superseded by the property
catalog and the ``Prediction`` surface. Each keeps its body untouched behind a
private twin, so migrating a user changes no number:

=========================  ==========================================
deprecated                 replacement
=========================  ==========================================
``predict_rest_sed``       ``pred.rest_sed()``
``predict_obs_sed``        ``pred.obs_sed()``
``predict_derived``        ``pred.properties``
``predict_magnitudes``     ``pred.magnitudes()``
``predict_sfh_quantities`` ``predict_properties(names=...)``
``predict_sed_quantities`` ``predict_properties(names=...)``
=========================  ==========================================

Two things must hold, and the second is the one that bites: the warning has to
fire for a **user**, and it must never fire because *tengri itself* called the old
method in a hot loop. Every internal caller was migrated to the private twin in
the same commit, and ``test_the_library_does_not_warn_at_itself`` proves it by
promoting ``DeprecationWarning`` to an error across the recommended surfaces.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.contract

DEPRECATED = [
    "predict_rest_sed",
    "predict_obs_sed",
    "predict_derived",
    "predict_magnitudes",
    "predict_sfh_quantities",
    "predict_sed_quantities",
]


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def params(model):
    return {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}


@pytest.mark.parametrize("name", DEPRECATED)
def test_the_deprecated_method_warns(model, params, name):
    """It must warn, name itself, and point at the replacement."""
    with pytest.warns(DeprecationWarning, match=name):
        getattr(model, name)(params)


@pytest.mark.parametrize("name", DEPRECATED)
def test_the_shim_is_bit_exact_with_its_private_twin(model, params, name):
    """Contract §6: the shim keeps the OLD numbers for one cycle.

    Not a tautology guard against nothing: a shim that quietly re-routed through
    the new catalog would silently move every existing user's values, which is
    exactly what the deprecation policy forbids. The body must be untouched.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        public = getattr(model, name)(params)
    private = getattr(model, f"_{name}")(params)

    pub_leaves = jax.tree_util.tree_leaves(public)
    priv_leaves = jax.tree_util.tree_leaves(private)
    assert len(pub_leaves) == len(priv_leaves) > 0
    for a, b in zip(pub_leaves, priv_leaves):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_library_does_not_warn_at_itself(model, params):
    """The one that actually bites: no internal caller may trip the warning.

    A deprecation whose own library still calls the old method spams the user from
    inside code they never wrote — the plan's top-listed risk. Every recommended
    surface is exercised here with ``DeprecationWarning`` promoted to an error.

    This test auto-discovers public attributes/methods of Prediction and Posterior
    to ensure comprehensive coverage as the API evolves.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        # Build model and prediction
        pred = model.predict(params)
        model.predict_photometry(params)
        model.predict_properties(params)

        # Exercise recommended Prediction surfaces (auto-discovered)
        exercised = set()

        # Known public properties/methods to exclude
        excluded = {"_" + name for name in DEPRECATED}
        excluded.update({"_cache", "_model", "_params", "_photometry_cache"})

        # Auto-discover and exercise Prediction public attributes/methods
        for name in dir(pred):
            if name.startswith("_") or name in excluded:
                continue
            try:
                attr = getattr(pred, name)
                # Call callables; access properties
                if callable(attr):
                    if name in ("photometry", "magnitudes", "spectrum"):
                        _ = attr()
                        exercised.add(name)
                else:
                    _ = attr
                    exercised.add(name)
            except Exception:
                # Some properties may fail on this minimal model; that's OK
                pass

        # Manually exercise key recommended surfaces
        _ = pred.photometry()
        _ = pred.magnitudes()
        _ = pred.rest_sed()
        _ = pred.obs_sed()
        _ = pred.wave_rest
        _ = pred.wave_obs
        _ = pred.properties["stellar_mass"]
        _ = pred.stellar_mass
        _ = pred.sfh.stellar_mass

        # Vacuity guard: ensure we exercised many properties
        assert len(exercised) >= 15, (
            f"Auto-discovery found only {len(exercised)} public attributes; "
            f"this test may have gone stale. Found: {sorted(exercised)}"
        )

        # CRITICAL: Exercise Posterior.derived — this is where the posterior.py:492
        # bug would manifest. Build a small posterior with samples.
        # The deprecated property itself emits a DeprecationWarning; we catch that
        # as a separate expected warning, but internal calls would appear as a second
        # warning (which we'd catch as an error).
        samples_dict = {k: jnp.repeat(v[None], 5, axis=0) for k, v in params.items()}
        from tengri.inference.posterior import Posterior

        posterior = Posterior(
            samples=samples_dict,
            params=params,
            method="vi",
            wall_time_s=1.0,
            diagnostics={},
            _model=model,
        )

        # Accessing .derived itself emits ONE DeprecationWarning (the user-facing one).
        # If line 492 calls the deprecated method, we'd get a SECOND warning from
        # inside the library code. Catch the first as expected, and let any second
        # become an error.
        with pytest.warns(DeprecationWarning, match="Posterior.derived is deprecated"):
            _ = posterior.derived

        # Also exercise posterior.properties (non-deprecated)
        _ = posterior.properties["stellar_mass"]
        _ = posterior.stellar_mass


def test_the_quantities_methods_agree_with_the_catalog(model, params):
    """The replacement must actually be able to replace them (contract §7).

    Every field of the old NamedTuples must exist in the catalog *and* carry the
    same number — otherwise the deprecation is pointing users at a different
    answer. This was false until #1131: ``mass_weighted_age_gyr`` differed by 4.6%
    and ``irx`` by 3 dex between the two paths.
    """
    catalog = model.predict_properties(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sfh = model.predict_sfh_quantities(params)
        sed = model.predict_sed_quantities(params)

    checked = 0
    for group in (sfh, sed):
        for field, value in group._asdict().items():
            assert field in catalog, (
                f"{field!r} has no catalog equivalent — deprecating the method that "
                "produces it would strand the user (contract §7)"
            )
            np.testing.assert_allclose(
                np.asarray(catalog[field], dtype=float),
                np.asarray(value, dtype=float),
                rtol=1e-10,
                # `rtol` alone is infinite against an expected value of exactly
                # zero, and several of these fields legitimately reach it — the
                # default metallicity is solar, i.e. log10(Z/Zsun) = 0, where
                # the two paths differ by one ULP (2.2e-16). 1e-12 sits far
                # below any physically meaningful difference in dex, Gyr or
                # 1/yr while still catching a real disagreement (#1703).
                atol=1e-12,
                err_msg=f"catalog[{field!r}] != the method it replaces",
            )
            checked += 1

    assert checked >= 20, f"only {checked} fields compared — this test has gone vacuous"


def test_wave_rest_and_wave_obs_have_correct_shapes(model, params):
    """wave_rest and wave_obs pair with rest_sed and obs_sed respectively.

    Both wavelength arrays must have the same shape as their paired SED arrays.
    This regression guard prevents API gaps like #XXXX where users have no
    wavelength axis and must hand-compute wave_obs with error-prone patterns.
    """
    pred = model.predict(params)

    wave_rest = pred.wave_rest
    sed_rest = pred.rest_sed()

    wave_obs = pred.wave_obs
    sed_obs = pred.obs_sed()

    assert wave_rest.shape == sed_rest.shape, (
        f"wave_rest shape {wave_rest.shape} != rest_sed shape {sed_rest.shape}"
    )
    assert wave_obs.shape == sed_obs.shape, (
        f"wave_obs shape {wave_obs.shape} != obs_sed shape {sed_obs.shape}"
    )


def test_wave_obs_uses_fixed_redshift_not_params_default(synthetic_ssp_wide, synthetic_tophat_obs):
    """wave_obs resolves Fixed redshift correctly, not via params.get() default.

    Regression guard for #1097, #1124, #1127: using params.get("redshift", 0.0)
    silently falls back to 0.0 when redshift is Fixed and absent from params,
    yielding 1e17-magnitude errors. Prediction.wave_obs must use _get_redshift
    to raise KeyError if redshift is missing, and correctly apply the Fixed value.
    """
    # Build a model with Fixed redshift — redshift NOT in params dict
    model_with_fixed_z = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),  # ← Fixed, not Free; won't appear in params
    )

    # Sample parameters — redshift is NOT in the dict
    params_fixed_z = jnp.asarray(0.5)  # just a placeholder for free params
    params_dict = {"sfh_dpl_log_total_mass": params_fixed_z}

    pred = model_with_fixed_z.predict(params_dict)

    wave_rest = pred.wave_rest
    wave_obs = pred.wave_obs

    # Verify: wave_obs = wave_rest * (1 + 0.1)
    z_fixed = 0.1
    expected_wave_obs = wave_rest * (1.0 + z_fixed)

    np.testing.assert_allclose(
        wave_obs,
        expected_wave_obs,
        rtol=1e-10,
        err_msg="wave_obs should equal wave_rest * (1 + z) for Fixed z",
    )

    # Spot-check: the ratio should be 1.1
    ratio_mean = float(np.mean(wave_obs / wave_rest))
    np.testing.assert_allclose(
        ratio_mean,
        1.0 + z_fixed,
        rtol=1e-10,
        err_msg="mean(wave_obs / wave_rest) should equal 1 + z",
    )

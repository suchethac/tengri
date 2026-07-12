# SPDX-License-Identifier: BSD-3-Clause
r"""Contract §6: a deprecated method warns, and returns *exactly* what it always did.

Phase 5 of #1043 (#1049). Six ``SEDModel`` methods are superseded by the property
catalog and the ``Prediction`` surface. Each keeps its body untouched behind a
private twin, so migrating a user changes no number:

=========================  ==========================================
deprecated                 replacement
=========================  ==========================================
``predict_rest_sed``       ``pred.rest_sed``
``predict_obs_sed``        ``pred.obs_sed``
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

from tengri import FIXED, Fixed, SEDModel, Uniform

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
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
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
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        pred = model.predict(params)
        model.predict_photometry(params)
        model.predict_properties(params)

        _ = pred.photometry()
        _ = pred.magnitudes()
        _ = pred.rest_sed
        _ = pred.obs_sed
        _ = pred.properties["stellar_mass"]
        _ = pred.stellar_mass
        _ = pred.sfh.stellar_mass


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
                err_msg=f"catalog[{field!r}] != the method it replaces",
            )
            checked += 1

    assert checked >= 20, f"only {checked} fields compared — this test has gone vacuous"

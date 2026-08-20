# SPDX-License-Identifier: BSD-3-Clause
"""#1313: catalog summaries must carry WHICH percentiles they hold.

``Catalog.fit(store="summary", percentiles=...)` accepts arbitrary
percentiles — the API spec's own example is ``(2.5, 16, 50, 84, 97.5)`` —
but ``CatalogPosterior`` stored only the arrays and threw the requested
levels away. Three silent consequences followed:

1. ``post[name]``, documented as the per-galaxy **median**, returned
   ``arr[:, min(1, n_pct - 1)]``. With the five-percentile example that is
   the 16th percentile; with ``(16, 84)`` it is the 84th. Measured on a
   three-galaxy NUTS fit, ``post["sfh_dpl_alpha"]`` gave ``[0.76, 0.46,
   2.93]`` where the true median was ``[2.35, 3.41, 4.11]``.
2. ``to_table()`` re-derived column *labels* from the array width, so the
   five-percentile block exported as ``_p0/_p25/_p50/_p75/_p100`` while
   holding 2.5/16/50/84/97.5 data — mislabeled numbers in a file meant to
   leave the process.
3. The summary block held **parameters only**, so ``stellar_mass`` — the
   quantity the spec's worked example asks for, and the reason the
   memory-bounded path exists at N ~ 1e5 — was absent.

Plus one no-op: ``store="summary"`` on a sample-free method (MAP) left
``.percentiles`` and ``.summary`` at ``None`` while ``.store`` still read
``"summary"``, with nothing warned.
"""

import warnings

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

SPEC_PERCENTILES = (2.5, 16, 50, 84, 97.5)


def _posterior(levels, columns_equal_levels=True):
    """A CatalogPosterior whose column *values* are the percentile levels.

    Making value == level is what turns a labeling bug into a visible
    numeric one: the median column must read 50.0.
    """
    from tengri.inference.catalog_fitter import CatalogPosterior

    n_gal = 4
    block = np.tile(np.asarray(levels, dtype=float), (n_gal, 1))
    return CatalogPosterior(
        posteriors=[],
        method="mcmc_nuts",
        n_galaxies=n_gal,
        percentiles={"stellar_mass": block},
        percentile_levels=tuple(levels),
        store="summary",
    )


def test_percentile_levels_are_retained():
    """The requested levels survive on the result object."""
    post = _posterior(SPEC_PERCENTILES)
    assert post.percentile_levels == SPEC_PERCENTILES


def test_median_is_looked_up_by_value_not_position():
    """``post[name]`` is the 50th percentile wherever 50 sits in the tuple."""
    post = _posterior(SPEC_PERCENTILES)
    np.testing.assert_allclose(np.asarray(post["stellar_mass"]), 50.0)


@pytest.mark.parametrize(
    "levels", [(16, 50, 84), (50,), (2.5, 16, 50, 84, 97.5), (5, 25, 50, 75, 95)]
)
def test_median_correct_for_every_level_layout(levels):
    """Whatever the layout, the median column is the one labeled 50."""
    np.testing.assert_allclose(np.asarray(_posterior(levels)["stellar_mass"]), 50.0)


def test_median_without_50_raises_and_says_why():
    """No 50 requested and no samples kept: refuse, do not hand back a neighbor."""
    post = _posterior((16, 84))
    with pytest.raises(KeyError, match="50"):
        post["stellar_mass"]


def test_to_table_labels_columns_from_the_requested_levels():
    """Exported column names must match the percentiles actually computed."""
    table = _posterior(SPEC_PERCENTILES).to_table()
    for level in SPEC_PERCENTILES:
        label = f"stellar_mass_p{level:g}".replace(".", "p")
        assert label in table, f"missing column {label}; got {sorted(table)}"
        np.testing.assert_allclose(np.asarray(table[label]), level)
    np.testing.assert_allclose(np.asarray(table["stellar_mass"]), 50.0)


def test_to_table_omits_the_bare_column_without_a_median():
    """A bare ``stellar_mass`` column would be an unlabeled guess — omit it."""
    table = _posterior((16, 84)).to_table()
    assert "stellar_mass" not in table
    assert "stellar_mass_p16" in table and "stellar_mass_p84" in table


# ── end-to-end: properties in the block, and the MAP no-op ────────────


@pytest.fixture
def fwd_catalog(synthetic_ssp_wide, simple_observation):
    from tengri import FIXED, FREE, ForwardModel, SEDModel, WavePrecomp

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"law": "power_law", "type": "two_component", "all_params": FIXED, "tau_bc": 0.5},
        neb={"type": "none"},
        redshift=FIXED,
        approx=WavePrecomp(catalog_z_range=(0.05, 1.5), n_z=60),
    )
    return ForwardModel.build(sed=sed, observation=simple_observation)


@pytest.fixture
def table_2gal(fwd_catalog):
    truth = {}
    for name in fwd_catalog.spec.free_params:
        dist = fwd_catalog.spec.get_distribution(name)
        truth[name] = float((dist.lo + dist.hi) / 2.0) if hasattr(dist, "lo") else 0.5
    flux = np.asarray(fwd_catalog.predict_photometry(truth))
    err = np.abs(flux) * 0.05 + 1e-30
    table = {f"band_{i}": np.full(2, flux[i]) for i in range(3)}
    table.update({f"band_{i}_err": np.full(2, err[i]) for i in range(3)})
    table["z"] = np.array([0.1, 0.6])
    return table


def test_summary_block_includes_derived_properties(fwd_catalog, table_2gal):
    """``stellar_mass`` must be summarizable — it is the point of §9.2."""
    from tengri import Catalog

    post = Catalog(fwd_catalog, table_2gal, flux_unit="cgs_fnu", redshift_col="z").fit(
        method="mcmc_nuts",
        key=jax.random.PRNGKey(0),
        n_warmup=5,
        n_samples=8,
        store="summary",
        percentiles=SPEC_PERCENTILES,
        properties=("stellar_mass",),
    )
    assert post.percentiles is not None
    assert "stellar_mass" in post.percentiles, (
        f"derived properties missing from the summary block; got {sorted(post.percentiles)}"
    )
    assert np.asarray(post.percentiles["stellar_mass"]).shape == (2, 5)
    # And the median accessor agrees with the labeled column.
    block = np.asarray(post.percentiles["stellar_mass"])
    np.testing.assert_allclose(np.asarray(post["stellar_mass"]), block[:, 2])


def test_summary_on_a_sample_free_method_warns_instead_of_silently_doing_nothing(
    fwd_catalog, table_2gal
):
    """MAP has no samples. Say so; never report ``store='summary'`` with None."""
    from tengri import Catalog

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        post = Catalog(fwd_catalog, table_2gal, flux_unit="cgs_fnu", redshift_col="z").fit(
            method="map",
            key=jax.random.PRNGKey(0),
            store="summary",
            percentiles=SPEC_PERCENTILES,
        )
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("summary" in m and "map" in m.lower() for m in messages), (
        f"no warning explained the skipped summary; saw {messages}"
    )
    assert post.store == "full", (
        "the result must not claim store='summary' when no summary block was built"
    )

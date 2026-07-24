# SPDX-License-Identifier: BSD-3-Clause
"""#1317: one noun, action verbs. Wraps the existing engine; ingestion
and validation happen at construction (fail fast, before any compile)."""

import jax
import numpy as np
import pytest


@pytest.fixture
def fwd_3band(synthetic_ssp_wide, simple_observation):
    """Synthetic 3-band ForwardModel for testing."""
    from tengri import FIXED, FREE, ForwardModel, SEDModel, WavePrecomp

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "two_component", "all_params": FIXED, "tau_bc": 0.5},
        neb={"type": "none"},
        redshift=FIXED,
        approx=WavePrecomp(catalog_z_range=(0.01, 2.0)),
    )
    fwd = ForwardModel.build(sed=sed, observation=simple_observation)
    return fwd


@pytest.fixture
def fwd_3band_zrange(synthetic_ssp_wide, simple_observation):
    """3-band ForwardModel with WavePrecomp catalog_z_range."""
    from tengri import FIXED, FREE, ForwardModel, SEDModel, WavePrecomp

    sed = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "two_component", "all_params": FIXED, "tau_bc": 0.5},
        neb={"type": "none"},
        redshift=FIXED,
        approx=WavePrecomp(catalog_z_range=(0.05, 1.5)),
    )
    fwd = ForwardModel.build(sed=sed, observation=simple_observation)
    return fwd


@pytest.fixture
def table_3band():
    """3-row table with 3 bands and redshifts."""
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        "band_1_err": np.array([0.2, 0.2, 0.2]),
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.1, 0.5, 0.3]),
    }


@pytest.fixture
def table_3band_bad_missing_col():
    """3-row table missing an error column."""
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        # Missing band_1_err!
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.1, 0.5, 0.3]),
    }


@pytest.fixture
def table_z_outside():
    """3-row table with redshifts outside the catalog_z_range (0.05, 1.5)."""
    return {
        "band_0": np.array([1.0, 2.0, 1.5]),
        "band_0_err": np.array([0.1, 0.1, 0.1]),
        "band_1": np.array([3.0, 4.0, 3.5]),
        "band_1_err": np.array([0.2, 0.2, 0.2]),
        "band_2": np.array([5.0, 6.0, 5.5]),
        "band_2_err": np.array([0.3, 0.3, 0.3]),
        "z": np.array([0.01, 2.0, 0.3]),  # 0.01 < 0.05 and 2.0 > 1.5
    }


@pytest.fixture
def param_table_3rows(fwd_3band):
    """Parameter table with 3 rows and all free params at prior midpoints."""
    from tengri import Uniform

    free_params = fwd_3band.spec.free_params
    param_table = {}
    for name in free_params:
        # Get the prior from the spec
        prior = fwd_3band.spec.get_distribution(name)
        # Use prior midpoint for initialization
        if isinstance(prior, Uniform):
            midpoint = (prior.lo + prior.hi) / 2.0
        else:
            # For other distributions, use a reasonable default
            midpoint = 0.5
        param_table[name] = np.full(3, midpoint, dtype=np.float64)
    return param_table


def test_construction_validates_eagerly(fwd_3band, table_3band_bad_missing_col):
    from tengri import Catalog

    with pytest.raises(ValueError):  # missing err column found at __init__
        Catalog(fwd_3band, table_3band_bad_missing_col, flux_unit="cgs_fnu")


def test_fit_default_is_map_and_returns_catalog_posterior(fwd_3band, table_3band):
    from tengri import Catalog

    cat = Catalog(fwd_3band, table_3band, flux_unit="cgs_fnu", redshift_col="z")
    post = cat.fit(key=jax.random.PRNGKey(0))  # no method= -> "map"
    assert post.n_galaxies == 3
    assert np.asarray(post.properties["stellar_mass"]).shape == (3,)


def test_redshift_span_validated_against_catalog_z_range(fwd_3band_zrange, table_z_outside):
    from tengri import Catalog

    with pytest.raises(ValueError, match="catalog_z_range"):
        Catalog(fwd_3band_zrange, table_z_outside, flux_unit="cgs_fnu", redshift_col="z")


def test_predict_mock_shapes(fwd_3band, param_table_3rows):
    from tengri import Catalog

    cat = Catalog(fwd_3band, None, flux_unit="cgs_fnu")  # prediction-only: no data table
    mock = cat.predict(param_table_3rows)
    assert mock.shape == (3, fwd_3band.observation.photometry.n_filters)


def test_no_vi_default_anywhere(fwd_3band, table_3band):
    import inspect

    from tengri import Catalog

    sig = inspect.signature(Catalog.fit)
    assert sig.parameters["method"].default == "map"

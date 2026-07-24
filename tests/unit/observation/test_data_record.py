# SPDX-License-Identifier: BSD-3-Clause
"""#1321: Data is the measurement record validated against the
Observation schema. One seam for shape checks, censor alignment,
and line-name subsetting (spec 2026-07-23, sections 3.2-3.3)."""

import jax.numpy as jnp
import numpy as np
import pytest


def _obs(n=3):
    from tengri.observation import Observation, Photometry

    names = ["sdss_g", "sdss_r", "sdss_i"][:n]
    return Observation(photometry=Photometry.from_names(names))


def test_photometry_shapes_validate():
    from tengri import Data

    d = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)))
    v = d.validate_against(_obs(3))
    assert v.flux.shape == (3,) and v.noise.shape == (3,)


def test_wrong_band_count_raises_naming_both():
    from tengri import Data

    d = Data(photometry=(jnp.ones(4), jnp.full(4, 0.1)))
    with pytest.raises(ValueError, match=r"4.*3|3.*4"):
        d.validate_against(_obs(3))


def test_censor_must_align_and_be_flags_not_bool():
    from tengri import Data

    ok = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)), censor=jnp.array([0, 1, -1]))
    ok.validate_against(_obs(3))  # 0/1/-1 fine
    boolean = Data(
        photometry=(jnp.ones(3), jnp.full(3, 0.1)), censor=jnp.array([True, False, True])
    )
    with pytest.raises(ValueError, match="censor"):  # bool rejected (spec 3.3)
        boolean.validate_against(_obs(3))


def test_lines_must_be_subset_of_schema_linelist():
    from tengri import Data

    d = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)), lines={"Halpha": (3.2e-17, 0.4e-17)})
    with pytest.raises(ValueError, match="Halpha"):  # obs declares no LineList
        d.validate_against(_obs(3))


def test_nan_in_single_galaxy_data_raises():
    from tengri import Data

    d = Data(photometry=(jnp.array([1.0, np.nan, 1.0]), jnp.full(3, 0.1)))
    with pytest.raises(ValueError, match=r"NaN.*sdss_r|sdss_r.*NaN"):
        d.validate_against(_obs(3))  # single-galaxy Data is complete (spec 3.3)


def test_empty_data_rejected():
    from tengri import Data

    with pytest.raises(ValueError):
        Data().validate_against(_obs(3))

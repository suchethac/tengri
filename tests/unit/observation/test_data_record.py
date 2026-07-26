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


# ── #1365: every channel Data carries must be validated, not just photometry flux ──
#
# The module docstring promises "shape errors, boolean-censor traps, NaNs, and unknown
# line names all fail loudly with the offending channel named". Before this, only the
# photometry FLUX was checked: a NaN noise value, a sign-flipped error bar, or a
# spectrum noise array of the wrong length all passed validation and produced a NaN or
# silently wrong logdensity with no error anywhere. A wrong-but-finite likelihood raises
# nothing and simply reports wrong science, so each channel is pinned explicitly.


def _obs_spec(npix=5):
    from tengri.observation import Observation, Spectroscopy

    return Observation(spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 5000.0, npix)))


def _obs_lines():
    from tengri.observation import LineList, Observation, Photometry

    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        lines=LineList.from_names(["Halpha"]),
    )


def test_nan_in_photometry_noise_raises_naming_the_band():
    from tengri import Data

    d = Data(photometry=(jnp.ones(3), jnp.array([0.1, np.nan, 0.1])))
    with pytest.raises(ValueError, match=r"sdss_r"):
        d.validate_against(_obs(3))


def test_nonpositive_photometry_sigma_raises_naming_the_band():
    from tengri import Data

    # A negative sigma is the dangerous one: chi2 = ((d-m)/sigma)**2 squares the sign
    # away, so the fit runs to completion and reports a confidently wrong answer.
    neg = Data(photometry=(jnp.ones(3), jnp.array([0.1, -0.1, 0.1])))
    with pytest.raises(ValueError, match=r"sdss_r"):
        neg.validate_against(_obs(3))
    zero = Data(photometry=(jnp.ones(3), jnp.array([0.1, 0.0, 0.1])))
    with pytest.raises(ValueError, match=r"sdss_i|sdss_r|positive"):
        zero.validate_against(_obs(3))


def test_spectrum_noise_length_must_match_the_wave_grid():
    from tengri import Data

    d = Data(spectrum=(jnp.ones(5), jnp.full(4, 0.1)))
    with pytest.raises(ValueError, match=r"noise|4|5"):
        d.validate_against(_obs_spec(5))


def test_nan_in_spectrum_flux_or_noise_raises():
    from tengri import Data

    bad_flux = Data(spectrum=(jnp.array([1.0, np.nan, 1.0, 1.0, 1.0]), jnp.full(5, 0.1)))
    with pytest.raises(ValueError, match=r"NaN|inf|finite"):
        bad_flux.validate_against(_obs_spec(5))
    bad_noise = Data(spectrum=(jnp.ones(5), jnp.array([0.1, 0.1, np.nan, 0.1, 0.1])))
    with pytest.raises(ValueError, match=r"NaN|inf|finite"):
        bad_noise.validate_against(_obs_spec(5))


def test_nonpositive_spectrum_sigma_raises():
    from tengri import Data

    d = Data(spectrum=(jnp.ones(5), jnp.array([0.1, 0.1, -0.1, 0.1, 0.1])))
    with pytest.raises(ValueError, match=r"positive|negative|sigma|noise"):
        d.validate_against(_obs_spec(5))


def test_line_value_nan_or_nonpositive_error_raises_naming_the_line():
    from tengri import Data

    nan_val = Data(lines={"Halpha": (np.nan, 0.4e-17)})
    with pytest.raises(ValueError, match=r"Halpha"):
        nan_val.validate_against(_obs_lines())
    neg_err = Data(lines={"Halpha": (3.2e-17, -0.4e-17)})
    with pytest.raises(ValueError, match=r"Halpha"):
        neg_err.validate_against(_obs_lines())


def test_valid_data_on_every_channel_still_passes():
    """Anti-vacuity: the new checks must not reject well-formed records."""
    from tengri import Data

    v = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1))).validate_against(_obs(3))
    assert v.flux.shape == (3,) and v.noise.shape == (3,)
    vs = Data(spectrum=(jnp.ones(5), jnp.full(5, 0.1))).validate_against(_obs_spec(5))
    assert vs.spec_flux.shape == (5,) and vs.spec_noise.shape == (5,)
    vl = Data(lines={"Halpha": (3.2e-17, 0.4e-17)}).validate_against(_obs_lines())
    assert vl.line_values["Halpha"][0] == pytest.approx(3.2e-17)

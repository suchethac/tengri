# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 forward has no erg/s-scale linear intermediate that overflows (#3149).

The float32 ceiling is 3.4e38. A 1e10 Msun galaxy carries a peak SED of ~1e31
erg/s/Hz, a bolometric luminosity of ~1e46 erg/s and a mass scale
``total_mass x L_sun`` of ~3.8e43 erg/s. Each of those, formed in float32 as a
linear value, is ``inf`` or swamps a cumulative sum. The tests below pin each
path the issue names:

(a) the conserving resample, on the bright synthetic SSP at log10 M = 6;
(b) float64 bit-identity of the conserving resample against the unfactored integral;
(c) spectral indices from a real SSP at log10 M = 10 through the window LUT;
(d) the shock path, both the model and the ``emission_helpers`` helper;
(e) the published stellar mass scale, which is log10 only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    FeaturePrecomp,
    Fixed,
    Observation,
    SEDModel,
    Spectroscopy,
    load_ssp_data,
)
from tengri._data_setup import find_data

pytestmark = pytest.mark.regression_bug

_SSP_NAME = "fsps_prsc_miles_chabrier.h5"
_LOG_M_FLOAT32 = 6.0
_LOG_M_REAL = 10.0


def _real_ssp_path():
    """Path to the real SSP grid, or skip when it is absent from the data directories."""
    path = find_data(_SSP_NAME)
    if path is None:
        pytest.skip(f"real SSP grid {_SSP_NAME} not on this machine")
    return path


def _spectroscopic_model(ssp, resample="conserving", resolution=2000):
    """Pure DPL SFH with only the total mass free, at z = 0.1, no nebular emission.

    ``resolution=None`` takes the bin-integral path with no line-spread convolution,
    which is the path the resample fix covers on its own.
    """
    wave_obs = jnp.linspace(4000.0, 7500.0, 160)
    observation = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=resolution, resample=resample)
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": FREE},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def _max_relative_error(values, reference):
    """Largest absolute deviation relative to the largest reference magnitude."""
    scale = np.max(np.abs(reference))
    return float(np.max(np.abs(values - reference)) / scale)


# ── (a) float32 conserving resample is finite and agrees with float64 ─────


@pytest.mark.parametrize("resolution", [None, 2000], ids=["no_lsf", "lsf_r2000"])
def test_float32_conserving_spectrum_is_finite_and_matches_float64(synthetic_ssp, resolution):
    """Bright synthetic SSP at log10 M = 6: the resample overflowed before the fix.

    The cumulative integral of a 1e6 Msun SED runs past 3.4e38 in float32 when
    formed on the raw erg/s/Hz values; the peak-factored integral stays in range.
    """
    params = {"sfh_dpl_log_total_mass": _LOG_M_FLOAT32}
    reference = np.asarray(
        _spectroscopic_model(synthetic_ssp, resolution=resolution).predict_spectrum(params)
    )
    assert np.all(np.isfinite(reference))
    assert np.any(reference != 0.0), "reference is identically zero"

    with jax.enable_x64(False):
        model32 = _spectroscopic_model(synthetic_ssp, resolution=resolution)
        spectrum32 = np.asarray(model32.predict_spectrum(params))
        assert spectrum32.dtype == np.float32

    assert np.all(np.isfinite(spectrum32)), "float32 conserving spectrum is non-finite (#3149)"
    assert np.any(spectrum32 != 0.0), "spectrum32 is identically zero"
    assert _max_relative_error(spectrum32.astype(np.float64), reference) < 1e-5


# ── (b) float64 bit-identity against the unfactored integral ───────────────


def _unfactored_resample(wave_rest, sed_rest, wave_query, edge_aa):
    """The conserving resample exactly as it was before the peak factoring (#3149).

    Reproduced here as the reference: the cumulative integral is formed on the
    raw ``sed_rest``, with no power-of-two scaling.
    """
    from tengri.observation.spectrum import _step_minus_ramp

    mid = 0.5 * (wave_query[1:] + wave_query[:-1])
    lo = wave_query[:1] - 0.5 * (wave_query[1:2] - wave_query[:1])
    hi = wave_query[-1:] + 0.5 * (wave_query[-1:] - wave_query[-2:-1])
    edges = jnp.concatenate([lo, mid, hi])
    h = jnp.diff(wave_rest)
    node_cum = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (sed_rest[1:] + sed_rest[:-1]) * h)]
    )
    e = jnp.clip(edges, wave_rest[0], wave_rest[-1])
    i = jnp.clip(jnp.searchsorted(wave_rest, e, side="right") - 1, 0, wave_rest.shape[0] - 2)
    t = e - wave_rest[i]
    f0 = sed_rest[i]
    flux_at_edges = node_cum[i] + f0 * t + 0.5 * (sed_rest[i + 1] - f0) / h[i] * t * t
    if edge_aa is not None:
        flux_at_edges = flux_at_edges + _step_minus_ramp(wave_rest, sed_rest, e, edge_aa)
    return (flux_at_edges[1:] - flux_at_edges[:-1]) / (edges[1:] - edges[:-1])


@pytest.mark.parametrize("edge_aa", [None, 912.0])
def test_float64_conserving_resample_is_bit_identical_to_unfactored_integral(
    synthetic_ssp, edge_aa
):
    """Power-of-two factoring is exact, so float64 output does not move by one bit."""
    from tengri.observation.spectrum import _flux_conserving_resample

    assert jax.config.jax_enable_x64
    wave_rest = jnp.asarray(synthetic_ssp.ssp_wave)
    sed_rest = jnp.asarray(synthetic_ssp.ssp_flux[1, 5]) * 1e25
    wave_query = jnp.linspace(3200.0, 9500.0, 60)

    fixed = np.asarray(_flux_conserving_resample(wave_rest, sed_rest, wave_query, edge_aa))
    unfixed = np.asarray(_unfactored_resample(wave_rest, sed_rest, wave_query, edge_aa))
    assert np.array_equal(fixed, unfixed)


# ── (c) spectral indices from a real SSP at log10 M = 10 ───────────────────


def _index_model(ssp):
    """Two-component dust, FeaturePrecomp window LUT, the fast index path."""
    return SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(_LOG_M_REAL)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=FeaturePrecomp(lines=[6564.61]),
    )


def test_float32_spectral_indices_on_real_ssp_match_float64():
    """Break and EW indices at log10 M = 10 agree with float64 to 1e-4.

    The window LUT used to scale its point terms by total_mass x L_sun before
    the F_lambda conversion, which overflowed float32. Ratios cancel the scale,
    so the fix drops it.
    """
    from tengri.observation.spectral_indices import STANDARD_INDICES

    path = _real_ssp_path()
    defs = [STANDARD_INDICES[k] for k in ("D4000", "Hbeta", "Mgb")]
    reference = np.asarray(
        _index_model(load_ssp_data(path)).predict_spectral_indices({}, defs, approx=True)
    )
    assert np.all(np.isfinite(reference))
    assert np.any(reference != 0.0), "reference is identically zero"

    with jax.enable_x64(False):
        model32 = _index_model(load_ssp_data(path))
        values32 = np.asarray(model32.predict_spectral_indices({}, defs, approx=True))

    assert np.all(np.isfinite(values32)), f"float32 indices non-finite (#3149): {values32}"
    assert np.any(values32 != 0.0), "values32 is identically zero"
    np.testing.assert_allclose(values32, reference, rtol=1e-4)


# ── (d) the shock path ──────────────────────────────────────────────────────


def _shock_model(ssp):
    """MAPPINGS V shock on the galaxy H-alpha, the default ``norm="frac"``."""
    observation = Observation(
        spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 7500.0, 160), resample="point")
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(_LOG_M_REAL)},
        neb={"type": "none"},
        shock={"type": "mappings", "frac": Fixed(0.1)},
        redshift=Fixed(0.1),
    )


def test_float32_shock_spectrum_on_real_ssp_matches_float64():
    """The shock component at log10 M = 10 on a real SSP is finite and matches float64."""
    path = _real_ssp_path()
    reference = np.asarray(_shock_model(load_ssp_data(path)).predict_spectrum({}))
    assert np.all(np.isfinite(reference))
    assert np.any(reference != 0.0), "reference is identically zero"

    with jax.enable_x64(False):
        spectrum32 = np.asarray(_shock_model(load_ssp_data(path)).predict_spectrum({}))

    assert np.all(np.isfinite(spectrum32)), "float32 shock spectrum non-finite (#3149)"
    assert np.any(spectrum32 != 0.0), "spectrum32 is identically zero"
    assert _max_relative_error(spectrum32.astype(np.float64), reference) < 1e-4


def test_float32_emission_helper_shock_matches_float64():
    """``emission_helpers.shock_emission`` at a 1e10 Msun peak: finite and matches float64.

    The SED peaks at ~1e31 erg/s/Hz, so its bolometric integral (~1e46 erg/s) and
    the shock anchor (~1e41 erg/s) are both past float32's ceiling in linear form.
    """
    from tengri.forward.emission_helpers import shock_emission

    wave = np.linspace(1000.0, 10000.0, 1500)
    sed = 1e31 * np.exp(-0.5 * ((wave - 4000.0) / 800.0) ** 2) + 1e29
    reference = np.asarray(shock_emission(jnp.asarray(wave), jnp.asarray(sed), shock_frac=0.5))
    assert np.all(np.isfinite(reference))
    assert np.any(reference != 0.0), "reference is identically zero"

    with jax.enable_x64(False):
        out32 = np.asarray(
            shock_emission(
                jnp.asarray(wave, dtype=jnp.float32),
                jnp.asarray(sed, dtype=jnp.float32),
                shock_frac=0.5,
            )
        )

    assert np.all(np.isfinite(out32)), "float32 shock_emission non-finite (#3149)"
    assert np.any(out32 != 0.0), "out32 is identically zero"
    assert _max_relative_error(out32.astype(np.float64), reference) < 1e-4


# ── (e) the published stellar mass scale is log10 only ─────────────────────


def test_linear_stellar_mass_scale_is_not_published_and_log_is_finite_in_float32():
    """``stellar_mass_scale`` (~1e43) is gone; ``log_stellar_mass_scale`` is finite in float32."""
    path = _real_ssp_path()
    params = {"sfh_dpl_log_total_mass": _LOG_M_REAL}
    derived64 = _spectroscopic_model(load_ssp_data(path)).predict_state(params).derived
    assert "stellar_mass_scale" not in derived64

    with jax.enable_x64(False):
        model32 = _spectroscopic_model(load_ssp_data(path))
        derived32 = model32.predict_state(params).derived
        log_scale = np.asarray(derived32["log_stellar_mass_scale"])

    assert "stellar_mass_scale" not in derived32
    assert np.isfinite(log_scale)
    assert float(log_scale) != 0.0
    assert float(log_scale) == pytest.approx(_LOG_M_REAL + np.log10(3.828e33), abs=1e-3)

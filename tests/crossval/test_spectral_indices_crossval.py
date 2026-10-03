# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: spectral index measurement against Bagpipes.

Measures Dn4000 and HdA on synthetic spectra using both Tengri's
soft-sigmoid JAX measurement and Bagpipes' hard-mask numpy measurement.
Agreement is expected to ~1% on dense grids (soft sigmoid edges converge
to hard masks as pixel spacing → 0).

Bagpipes' ``single_index`` is not the Lick definition: it takes a constant
continuum, the mean of the two sideband means, on the array it is given.
Tengri's default is the Lick definition (Trager et al. 1998, ApJS 116, 1,
Eqs. 1-3: F_λ, straight line through the sideband means); the Bagpipes
comparison therefore uses ``pseudo_continuum="mean"``, and the equivalent
widths of the default are checked on sloped SSP spectra against a numpy
implementation of the definition (#2690).

Invoke with:  pytest -m crossval tests/crossval/test_spectral_indices_crossval.py
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import load_ssp_data

pytestmark = pytest.mark.crossval

bp_indices = pytest.importorskip("bagpipes.input.spectral_indices")

from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexData,
    measure_index_jax,
)


def _mean_def(name: str):
    """The registry index with Bagpipes' constant-continuum arithmetic."""
    return dataclasses.replace(STANDARD_INDICES[name], pseudo_continuum="mean")


def _bp_index_dict(name: str) -> dict:
    """Build a Bagpipes-style index definition dict from Tengri's catalog."""
    idx = STANDARD_INDICES[name]
    d: dict = {
        "type": idx.index_type,
        "continuum": list(idx.continuum),
    }
    if idx.feature is not None:
        d["feature"] = list(idx.feature)
    if idx.units != "AA":
        d["units"] = idx.units
    return d


class TestDn4000CrossVal:
    """Dn4000 break index: Tengri vs Bagpipes on synthetic spectra."""

    def _make_spectrum(self, slope: float, n: int = 5000):
        wave = np.linspace(3600.0, 4300.0, n)
        flux = 1.0 + slope * (wave - 3600.0) / 700.0
        return wave, flux

    def test_flat_spectrum(self):
        wave, flux = self._make_spectrum(slope=0.0)
        bp_spec = np.column_stack([wave, flux])
        bp_val = bp_indices.single_index(_bp_index_dict("Dn4000"), bp_spec, 0.0)

        dn4000 = STANDARD_INDICES["Dn4000"]
        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flux), dn4000)

        assert_allclose(float(tengri_val), float(bp_val), rtol=0.01)
        assert_allclose(float(tengri_val), 1.0, atol=0.01)

    @pytest.mark.parametrize("slope", [0.3, 0.8, 1.5, -0.3])
    def test_tilted_spectrum(self, slope):
        wave, flux = self._make_spectrum(slope=slope)
        bp_spec = np.column_stack([wave, flux])
        bp_val = bp_indices.single_index(_bp_index_dict("Dn4000"), bp_spec, 0.0)

        dn4000 = STANDARD_INDICES["Dn4000"]
        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flux), dn4000)

        assert_allclose(float(tengri_val), float(bp_val), rtol=0.01)

    def test_old_stellar_population(self):
        """Simulate an old stellar population with strong 4000A break."""
        wave, _ = self._make_spectrum(slope=0.0)
        flux = np.where(wave < 3975.0, 0.5, 1.2)
        flux += 0.05 * np.sin(wave / 30.0)

        bp_spec = np.column_stack([wave, flux])
        bp_val = bp_indices.single_index(_bp_index_dict("Dn4000"), bp_spec, 0.0)

        dn4000 = STANDARD_INDICES["Dn4000"]
        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flux), dn4000)

        assert_allclose(float(tengri_val), float(bp_val), rtol=0.02)
        assert float(tengri_val) > 2.0


class TestHdACrossVal:
    """HdA equivalent width: Tengri (``"mean"`` option) vs Bagpipes on synthetic F_λ spectra."""

    def _make_spectrum(self, n: int = 5000):
        wave = np.linspace(3900.0, 4300.0, n)
        return wave

    def test_flat_spectrum_zero_ew(self):
        wave = self._make_spectrum()
        flux = np.ones_like(wave)

        bp_spec = np.column_stack([wave, flux])
        bp_val = bp_indices.single_index(_bp_index_dict("HdA"), bp_spec, 0.0)

        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flux), _mean_def("HdA"))

        assert_allclose(float(tengri_val), float(bp_val), atol=0.15)
        assert_allclose(float(tengri_val), 0.0, atol=0.15)

    def test_absorption_line(self):
        """Gaussian absorption dip in the HdA feature window → positive EW."""
        wave = self._make_spectrum()
        hda = STANDARD_INDICES["HdA"]
        center = (hda.feature[0] + hda.feature[1]) / 2.0
        flux = 1.0 - 0.4 * np.exp(-0.5 * ((wave - center) / 5.0) ** 2)

        bp_spec = np.column_stack([wave, flux])
        bp_val = bp_indices.single_index(_bp_index_dict("HdA"), bp_spec, 0.0)

        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flux), _mean_def("HdA"))

        assert_allclose(float(tengri_val), float(bp_val), rtol=0.05)
        assert float(tengri_val) > 0.0

    def test_emission_line(self):
        """Gaussian emission bump → negative EW."""
        wave = self._make_spectrum()
        hda = STANDARD_INDICES["HdA"]
        center = (hda.feature[0] + hda.feature[1]) / 2.0
        flux = 1.0 + 0.3 * np.exp(-0.5 * ((wave - center) / 5.0) ** 2)

        bp_spec = np.column_stack([wave, flux])
        bp_val = bp_indices.single_index(_bp_index_dict("HdA"), bp_spec, 0.0)

        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flux), _mean_def("HdA"))

        assert_allclose(float(tengri_val), float(bp_val), rtol=0.05)
        assert float(tengri_val) < 0.0


class TestD4000CrossVal:
    """D4000 (wide break): verify wider windows produce different value from Dn4000."""

    def test_wide_vs_narrow(self):
        wave = np.linspace(3500.0, 4500.0, 8000)
        flux = 1.0 + 0.5 * (wave - 3500.0) / 1000.0

        dn_val = measure_index_jax(jnp.array(wave), jnp.array(flux), STANDARD_INDICES["Dn4000"])
        d_val = measure_index_jax(jnp.array(wave), jnp.array(flux), STANDARD_INDICES["D4000"])

        assert float(dn_val) != pytest.approx(float(d_val), abs=0.001)
        assert float(dn_val) > 1.0
        assert float(d_val) > 1.0


class TestSpectralIndexDataConvenience:
    """Verify chi2 and log_likelihood methods match manual calculations."""

    def test_chi2_matches_manual(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000", "HdA"],
            values=[1.8, -1.2],
            errors=[0.05, 0.3],
        )
        model = jnp.array([1.75, -0.9])

        manual = jnp.sum(((sid.values - model) / sid.errors) ** 2)
        assert_allclose(float(sid.chi2(model)), float(manual), rtol=1e-10)

    def test_log_likelihood_matches_manual(self):
        sid = SpectralIndexData.from_names(
            names=["Dn4000"],
            values=[1.8],
            errors=[0.05],
        )
        model = jnp.array([1.75])

        residual = (sid.values - model) / sid.errors
        manual = float(
            jnp.sum(-0.5 * residual**2 - jnp.log(sid.errors) - 0.5 * jnp.log(2.0 * jnp.pi))
        )
        assert_allclose(float(sid.log_likelihood(model)), manual, rtol=1e-10)


# ── Lick definition vs Bagpipes on sloped SSP spectra (#2690) ───────

C_AA = 2.99792458e18  # speed of light [Å/s]
_SSP_CASES = [("HgA", 1.0), ("HgA", 10.0), ("Fe4383", 1.0), ("Fe4383", 10.0), ("Hbeta", 3.0)]
#: |Tengri "mean" - Bagpipes| <= 0.157 Å on these cases (1 Å soft edges and trapezoid
#: means against hard masks and per-pixel means); tolerance 1.25 x, rounded up.
_BAGPIPES_MEAN_TOL_AA = 0.2
#: Largest soft-window - hard-window difference of the default over three ages [Å],
#: 1.25 x, rounded up: HgA 0.050, Fe4383 0.050, Hbeta 0.130.
_EDGE_TOL_AA = {"HgA": 0.050, "Fe4383": 0.050, "Hbeta": 0.130}


def _ssp_flam(age_gyr: float):
    """Optical solar-metallicity SSP: wave, L_ν, and F_λ (up to a constant)."""
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    wave = np.asarray(ssp.ssp_wave, dtype=float)
    sel = (wave > 3600.0) & (wave < 5600.0)
    i_met = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) + 1.848)))
    i_age = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(age_gyr))))
    lnu = np.asarray(ssp.ssp_flux, dtype=float)[i_met, i_age][sel]
    return wave[sel], lnu, lnu * C_AA / wave[sel] ** 2


def _lick_hard(wave, flam, idx) -> float:
    """Trager et al. 1998 Eqs. 1-2 with hard windows, flux interpolated at the bounds."""

    def seg(lo, hi):
        x = np.concatenate(([lo], wave[(wave > lo) & (wave < hi)], [hi]))
        return x, np.interp(x, wave, flam)

    def mean(lo, hi):
        x, v = seg(lo, hi)
        return np.trapezoid(v, x) / (hi - lo)

    (b0, b1), (r0, r1) = idx.continuum
    f_b, f_r = mean(b0, b1), mean(r0, r1)
    x, v = seg(*idx.feature)
    line = f_b + (f_r - f_b) * (x - 0.5 * (b0 + b1)) / (0.5 * (r0 + r1) - 0.5 * (b0 + b1))
    return float(np.trapezoid(1.0 - v / line, x))


class TestLickDefinitionVsBagpipes:
    """The split: ``"mean"`` is Bagpipes; the default is the published definition."""

    @pytest.mark.parametrize("name,age", _SSP_CASES)
    def test_mean_option_reproduces_bagpipes_on_f_lambda(self, name, age):
        wave, _, flam = _ssp_flam(age)
        bp_val = bp_indices.single_index(_bp_index_dict(name), np.column_stack([wave, flam]), 0.0)
        tengri_val = measure_index_jax(jnp.array(wave), jnp.array(flam), _mean_def(name))
        assert_allclose(float(tengri_val), float(bp_val), atol=_BAGPIPES_MEAN_TOL_AA)

    @pytest.mark.parametrize("name,age", _SSP_CASES)
    def test_default_is_the_published_definition_not_bagpipes(self, name, age):
        """Default (from L_ν) agrees with a numpy Lick definition, to the soft-edge effect."""
        wave, lnu, flam = _ssp_flam(age)
        idx = STANDARD_INDICES[name]
        tengri_val = float(measure_index_jax(jnp.array(wave), jnp.array(lnu), idx))
        assert abs(tengri_val - _lick_hard(wave, flam, idx)) < _EDGE_TOL_AA[name]

    @pytest.mark.parametrize("name,min_gap", [("HgA", 0.5), ("Fe4383", 0.1)])
    def test_bagpipes_misses_the_published_definition_on_an_old_population(self, name, min_gap):
        """Constant continuum vs sideband line: 0.69 Å (HgA) and 0.17 Å (Fe4383) at 10 Gyr."""
        wave, _, flam = _ssp_flam(10.0)
        bp_val = bp_indices.single_index(_bp_index_dict(name), np.column_stack([wave, flam]), 0.0)
        assert abs(float(bp_val) - _lick_hard(wave, flam, STANDARD_INDICES[name])) > min_gap

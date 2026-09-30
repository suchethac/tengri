# SPDX-License-Identifier: BSD-3-Clause
"""#2105: the LUT-bias advisory spans the free-redshift prior and names approx=None.

With redshift free, the z-interpolation error along the LUT's z axis is a
second error term invisible to a single-point forward probe. The advisory
probes up to three redshifts — the prior median plus the nearest z-table
inter-node midpoint on each side (or the prior bounds when no midpoint lies
inside the prior) — takes the worst case, and names the peak redshift.

Stub design: only the LUT stub carries the z-dependent bias term. An earlier
draft applied the same triangle bias to both stubs, so the relative bias
``|lut - exact| / |exact|`` was a constant at every z and the z dependence
the file claims to test cancelled out.
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np
import pytest

from tengri.config.exceptions import PrecompBiasWarning
from tengri.inference.fitter import _warn_if_lut_bias_amplified

pytestmark = pytest.mark.regression_bug


class _StubDistribution:
    def __init__(self, z_min, z_max):
        self.bounds = (z_min, z_max)

    def unstandardize(self, std_val):
        return (self.bounds[0] + self.bounds[1]) / 2.0


class _StubSpec:
    def __init__(self, free_params=None, distributions=None):
        self.free_params = free_params or ()
        self.stochastic = False
        self._distributions = distributions or {}

    def get_distribution(self, name):
        return self._distributions[name]


class _ExactStub:
    """Exact model: constant flux, NO z-dependent term."""

    def __init__(self, flux, spec=None):
        self.spec = spec or _StubSpec()
        self._flux = np.asarray(flux, dtype=float)
        self.n_calls = 0

    def predict_photometry(self, params):
        self.n_calls += 1
        return self._flux

    def predict_spectrum(self, params):
        return self.predict_photometry(params)


class _LutStub(_ExactStub):
    """LUT model: carries the z-dependent interpolation error.

    Triangle per z-table step — zero at nodes, maximal at the inter-node
    midpoint — with amplitude proportional to the step WIDTH (wider steps
    interpolate worse), so an uneven table has its largest error in its
    widest step.
    """

    def __init__(self, flux, spec=None, z_grid=None):
        super().__init__(flux, spec)
        self._z_grid = None if z_grid is None else np.asarray(z_grid, dtype=float)

    def predict_photometry(self, params):
        self.n_calls += 1
        flux = self._flux
        if self._z_grid is not None:
            z = float(params.get("redshift", 0.5))
            idx = int(np.searchsorted(self._z_grid, z))
            if 0 < idx < self._z_grid.size:
                lo, hi = float(self._z_grid[idx - 1]), float(self._z_grid[idx])
                rel = (z - lo) / (hi - lo)
                amp = 0.003 * (hi - lo) / 0.5
                flux = flux * (1.0 + 4.0 * rel * (1.0 - rel) * amp)
        return flux

    def _ztable_data_for_jit(self):
        if self._z_grid is None:
            return None
        return namedtuple("ZTable", ["z_grid"])(z_grid=self._z_grid)


def _capture(exact, lut, data, noise):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_if_lut_bias_amplified(exact, lut, data, noise, "photometry", surface="Test")
    return [x for x in w if issubclass(x.category, PrecompBiasWarning)]


def _free_z_spec(z_min, z_max):
    return _StubSpec(
        free_params=("redshift",),
        distributions={"redshift": _StubDistribution(z_min, z_max)},
    )


_FLUX = np.array([1.0, 2.0, 3.0, 4.0])
_NOISE = np.abs(_FLUX) / 100.0  # SNR 100


def test_fixed_redshift_single_evaluation():
    """Fixed redshift: exactly one LUT forward."""
    exact = _ExactStub(_FLUX)
    lut = _LutStub(_FLUX * 1.002, exact.spec)
    _capture(exact, lut, _FLUX, _NOISE)
    assert lut.n_calls == 1


def test_free_redshift_probes_median_and_both_midpoints():
    """Free z with a table: three forwards, and the peak is AT a midpoint.

    The median (1.0) sits on a table node where the LUT stub's error is zero,
    so the 0.2% constant offset is all a median-only probe can see; the
    midpoints (0.75 and 1.25) carry the triangle maximum. The warning must
    fire (not pass vacuously) and name a midpoint, proving the probe left
    the median.
    """
    spec = _free_z_spec(0.1, 1.9)  # median 1.0, a node of the grid below
    exact = _ExactStub(_FLUX, spec)
    lut = _LutStub(_FLUX * 1.002, spec, z_grid=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    rec = _capture(exact, lut, _FLUX, _NOISE)
    assert lut.n_calls == 3, f"median + one midpoint per side = 3 forwards, got {lut.n_calls}"
    assert rec, "the amplified-bias warning must fire for this SNR"
    msg = str(rec[0].message)
    assert "peaking at z=0.750" in msg or "peaking at z=1.250" in msg, msg
    assert "redshift is free" in msg
    assert "approx=None" in msg


def test_same_sided_nearest_midpoints_still_probe_one_each_side():
    """Two-nearest-by-distance would pick two midpoints on the SAME side.

    Table nodes [0.5, 0.7, 0.9, 2.9] give midpoints [0.6, 0.8, 1.9]; with
    the prior median at 1.0, the two nearest by distance are 0.8 and 0.6 —
    both below — and the wide last step's midpoint 1.9, which carries the
    largest interpolation error (amplitude scales with step width), would
    never be probed. The intent is one below and one above: the peak the
    message names must be 1.9.
    """
    spec = _free_z_spec(0.1, 1.9)  # median 1.0
    exact = _ExactStub(_FLUX, spec)
    lut = _LutStub(_FLUX * 1.002, spec, z_grid=np.array([0.5, 0.7, 0.9, 2.9]))
    rec = _capture(exact, lut, _FLUX, _NOISE)
    assert lut.n_calls == 3
    assert rec, "the amplified-bias warning must fire for this SNR"
    assert "peaking at z=1.900" in str(rec[0].message), str(rec[0].message)


def test_no_table_probes_prior_bounds():
    """Free z with no z-table: the probe spans the prior itself (3 forwards)."""
    spec = _free_z_spec(0.0, 2.0)
    exact = _ExactStub(_FLUX, spec)
    lut = _LutStub(_FLUX * 1.002, spec, z_grid=None)
    rec = _capture(exact, lut, _FLUX, _NOISE)
    assert lut.n_calls == 3, f"median + lo + hi = 3 forwards, got {lut.n_calls}"
    assert rec, "the amplified-bias warning must fire for this SNR"


def test_nodes_outside_narrow_prior_probes_bounds():
    """A REAL table whose midpoints all miss a narrow prior still spans it.

    The old gate was ``z_grid is None or len(z_grid) <= 1``, so this case —
    a 5-node table with every inter-node midpoint outside the prior
    (0.62, 0.68) — probed the median only. The gate must be "no inter-node
    midpoint inside [lo, hi]": median + both prior bounds, 3 forwards.
    """
    spec = _free_z_spec(0.62, 0.68)  # midpoints 0.25/0.75/1.25/1.75 all outside
    exact = _ExactStub(_FLUX, spec)
    lut = _LutStub(_FLUX * 1.002, spec, z_grid=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    rec = _capture(exact, lut, _FLUX, _NOISE)
    assert lut.n_calls == 3, f"median + prior lo + prior hi = 3 forwards, got {lut.n_calls}"
    assert rec, "the amplified-bias warning must fire for this SNR"


def test_free_redshift_message_names_z_peak():
    """Free z + material bias: the message carries the z clause. Must FIRE."""
    spec = _free_z_spec(0.1, 1.9)
    exact = _ExactStub(_FLUX, spec)
    lut = _LutStub(_FLUX * 1.01, spec, z_grid=np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    rec = _capture(exact, lut, _FLUX, _NOISE)
    assert rec, "the amplified-bias warning must fire for this SNR"
    msg = str(rec[0].message)
    assert "peaking at z=" in msg
    assert "redshift is free" in msg
    assert "approx=None" in msg


def test_fixed_redshift_message_no_z_peak():
    """Fixed z: the message must FIRE and must NOT carry the z clause."""
    exact = _ExactStub(_FLUX)
    lut = _LutStub(_FLUX * 1.01, exact.spec)
    rec = _capture(exact, lut, _FLUX, _NOISE)
    assert rec, "the amplified-bias warning must fire for this SNR"
    msg = str(rec[0].message)
    assert "peaking at z=" not in msg
    assert "redshift is free" not in msg
    assert "approx=None" in msg


def test_bias_low_snr_no_warn():
    """Low SNR keeps the advisory silent (bias x SNR below the threshold)."""
    exact = _ExactStub(_FLUX)
    lut = _LutStub(_FLUX * 1.002, exact.spec)
    rec = _capture(exact, lut, _FLUX, np.abs(_FLUX) / 10.0)
    assert rec == []


def test_real_wave_precomp_free_redshift_names_z_peak(
    synthetic_ssp_wide, synthetic_tophat_obs, monkeypatch
):
    """Measured on a real WavePrecomp LUT with redshift=Uniform spanning table nodes.

    The advisory fires at Fitter construction (where the LUT clone is
    resolved). The warn threshold is monkeypatched to zero so the measured
    numbers are always reported; the assertion is that the free-z probe ran
    against the real ``_ztable_data_for_jit()`` axis and the message names a
    peak redshift.
    """
    import jax

    from tengri import DEFAULT, Fixed, ForwardModel, SEDModel, Uniform, WavePrecomp
    from tengri.inference import fitter as fitter_mod

    sed = SEDModel.build(
        synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 1.5),
        },
        dust_emission=None,
        neb={"type": "none"},
        redshift=Uniform(0.02, 0.30),
    )
    truth = sed.spec.sample(jax.random.PRNGKey(0))
    mock = sed.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
    fwd = ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)

    monkeypatch.setattr(fitter_mod, "_LUT_BIAS_GRAD_WARN", 0.0)
    f = fitter_mod.Fitter(fwd, mock.flux_obs, mock.noise, approx=WavePrecomp(n_z=17))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f.run("map", n_steps=2)
    rec = [x for x in w if issubclass(x.category, PrecompBiasWarning)]
    assert rec, "with a zero threshold the advisory must fire on the real model"
    msg = str(rec[0].message)
    assert "peaking at z=" in msg, msg
    assert "redshift is free" in msg

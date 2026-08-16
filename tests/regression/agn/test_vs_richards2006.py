# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the Richards+2006 empirical BBB composite.

Verifies that ``richards2006_disc`` reproduces the Richards+2006 SDSS
composite as tabulated upstream by AGNfitter
(`models/BBB/R06.pickle`), correctly normalizes to the user's L_bol
anchor, and registers in ``AGN_MODELS``.

References
----------
Richards, G. T. et al. 2006, ApJ, 166, 470.
https://doi.org/10.1086/506525
"""

import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager


@pytest.mark.regression_paper
def test_richards2006_data_file_sha256():
    """The shipped Richards+2006 data file must match the expected SHA256."""
    from importlib.resources import files

    path = files("tengri.data.agn_bbb") / "richards2006.dat"
    with path.open("rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    expected = "17a4e0b655a967744a36341281cf3f28d05632ce526a9575f34e53f1c5ed8c97"
    assert digest == expected, (
        f"Richards+2006 data file SHA256 changed: {digest} != {expected}. "
        f"If you re-extracted the file, update PROVENANCE.md and this test."
    )


@pytest.mark.regression_paper
def test_richards2006_template_wavelength_range():
    """Template covers ~30 Å (soft X-ray) through ~3×10⁸ Å (radio), monotonic."""
    from tengri.components.agn.richards2006_disc import RICHARDS2006_WAVE_AA

    assert RICHARDS2006_WAVE_AA[0] < 50.0
    assert RICHARDS2006_WAVE_AA[-1] > 1e8
    assert np.all(np.diff(RICHARDS2006_WAVE_AA) > 0), "wavelength grid not ascending"
    assert RICHARDS2006_WAVE_AA.shape == (438,), "expected 438 grid points"


@pytest.mark.regression_paper
def test_richards2006_bolometric_normalization():
    """L_nu integrated over the template's ν range recovers 10^log_lbol · L_sun."""
    from tengri.components.agn.richards2006_disc import richards2006_disc

    log_lbol = 12.0  # log10(L_sun)
    # Wave grid spans the whole template so the integral picks up all energy
    wave = jnp.logspace(np.log10(31.0), np.log10(2.9e8), 5000)
    sed = richards2006_disc(wave, log_lbol=log_lbol)
    nu = 2.99792458e18 / wave
    order = jnp.argsort(nu)
    lbol = float(jnp.trapezoid(sed[order], nu[order]))
    target = 10.0**log_lbol * 3.828e33  # L_sun in erg/s
    np.testing.assert_allclose(lbol, target, rtol=0.02)


@pytest.mark.regression_paper
def test_richards2006_registered_in_agn_models():
    """``resolve_agn_model("richards2006")`` returns a callable."""
    import warnings

    from tengri.components.agn.unified import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("richards2006")
    assert callable(fn)
    wave = jnp.logspace(2, 5, 10)
    out = fn(wave, agn_log_lbol=12.0)
    assert out.shape == wave.shape


@pytest.mark.regression_paper
def test_richards2006_agn_frac_scaling():
    """``agn_lum_ratio`` linearly scales the SED."""
    from tengri.components.agn import richards2006

    wave = jnp.logspace(2, 5, 100)
    full = richards2006(wave, agn_log_lbol=12.0, agn_lum_ratio=1.0)
    half = richards2006(wave, agn_log_lbol=12.0, agn_lum_ratio=0.5)
    np.testing.assert_allclose(np.asarray(half), 0.5 * np.asarray(full), rtol=1e-6)


@pytest.mark.regression_paper
def test_richards2006_jit_compatible():
    """Function is JIT-compatible (pure jnp.interp)."""

    from tengri.components.agn.richards2006_disc import richards2006_disc

    wave = jnp.logspace(2, 5, 50)
    out = assert_jit_matches_eager(richards2006_disc, wave, 12.0)
    assert out.shape == wave.shape
    assert bool(jnp.all(jnp.isfinite(out)))

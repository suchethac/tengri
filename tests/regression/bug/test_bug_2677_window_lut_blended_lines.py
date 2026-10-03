# SPDX-License-Identifier: BSD-3-Clause
r"""Regression #2677: the window LUT measures blended lines the way the exact path does.

The exact path multiplies the SED by the dust transmission :math:`T(\lambda)` and
then takes each window mean; the LUT used to take :math:`T` at the window center.
The difference is a fraction of a percent of a window mean, but a faint line beside
a strong one ([N II] 6584 next to H-alpha) is a small difference of two large
means, so it came out 13 % off. The LUT now keeps the SSP integrand per grid point
and applies :math:`T` where the exact path does.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.line_measurement import DESI_LINES, LineDef, default_line_defs
from tengri.observation.spectral_indices import STANDARD_INDICES

pytestmark = pytest.mark.regression_bug

_SSP_PATH = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

#: LUT and exact path now evaluate the same sum; only float rounding separates them.
_TOL = 1e-9

#: Faint-beside-strong census (rest vacuum centers [Å]).
_LINE_SETS = {
    "NII_Halpha": (("NII_6549", 6549.86), ("Halpha", 6564.61), ("NII_6584", 6585.28)),
    "SII_doublet": (("SII_6717", 6718.29), ("SII_6731", 6732.67)),
    "OIII_doublet": (("OIII_4960", 4960.30), ("OIII_5007", 5008.24)),
    "Hbeta_OIII": (("Hbeta", 4862.68), ("OIII_4960", 4960.30), ("OIII_5007", 5008.24)),
}


@pytest.fixture(scope="module")
def model():
    if not _SSP_PATH.is_file():
        pytest.skip("wNE SSP grid not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=load_ssp_data(str(_SSP_PATH)),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(10.0),
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Fixed(0.75),
                "tau_bc": Fixed(0.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )


def _defs(pairs):
    return default_line_defs(np.array([w for _, w in pairs]), tuple(n for n, _ in pairs))


@pytest.mark.parametrize("key", list(_LINE_SETS))
def test_line_lut_matches_exact_for_faint_and_strong_lines(model, key):
    defs = _defs(_LINE_SETS[key])
    approx = np.asarray(model.measure_line_fluxes({}, defs, approx=True))
    exact = np.asarray(model.measure_line_fluxes({}, defs, approx=False))
    rel = np.abs(approx / exact - 1.0)
    assert np.all(rel < _TOL), f"{key}: {dict(zip([d.name for d in defs], rel))}"


def test_desi_lines_lut_matches_exact(model):
    approx = np.asarray(model.measure_line_fluxes({}, DESI_LINES, approx=True))
    exact = np.asarray(model.measure_line_fluxes({}, DESI_LINES, approx=False))
    assert np.all(np.abs(approx / exact - 1.0) < _TOL)


def test_indices_lut_matches_exact(model):
    idx = tuple(d for d in STANDARD_INDICES.values() if d.index_type in ("break", "EW"))
    approx = np.asarray(model.predict_spectral_indices({}, idx, approx=True))
    exact = np.asarray(model.predict_spectral_indices({}, idx, approx=False))
    rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-12)
    assert np.all(rel < _TOL), dict(zip([d.name for d in idx], rel))


def test_faint_line_is_really_blended(model):
    """Vacuity: the exact [N II] 6584 value is set by the H-alpha wing in its windows.

    Moving only the red side-band by 100 A (off the H-alpha wing structure) changes
    the exact faint-line flux by far more than the parity tolerance, so the parity
    test above is not comparing two insensitive numbers.
    """
    (nii,) = _defs((("NII_6584", 6585.28),))
    (b, (rlo, rhi)) = nii.continuum
    moved = LineDef(nii.name, nii.wavelength, (b, (rlo + 100.0, rhi + 100.0)), nii.feature)
    base = float(np.asarray(model.measure_line_fluxes({}, (nii,), approx=False))[0])
    shifted = float(np.asarray(model.measure_line_fluxes({}, (moved,), approx=False))[0])
    assert abs(shifted / base - 1.0) > 0.05


def test_line_lut_gradient_is_finite_and_nonzero_in_float32():
    """The per-point contraction must not overflow the float32 backward pass.

    ``window_means * 2**112`` restores ``L_sun`` after the ~1e10 mass scale. When the
    mass scale multiplied the returned means, XLA reassociated the two adjacent
    scalar multiplies in the backward pass to ``ct * (scale * 2**112)``, ~1e44 and
    ``inf`` in float32: every cotangent into the window means was ``inf`` and the
    parameter gradient ``nan``, with every forward value finite. The mass scale now
    enters through the SFH weights.
    """
    import jax
    import jax.numpy as jnp

    if not _SSP_PATH.is_file():
        pytest.skip("wNE SSP grid not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with jax.enable_x64(False):
            m32 = SEDModel.build(
                ssp_data=load_ssp_data(str(_SSP_PATH)),
                sfh={
                    "type": "delayed",
                    "all_params": Fixed(DEFAULT),
                    "log_total_mass": Uniform(9.0, 11.0),
                    "tau_gyr": Fixed(1.0),
                    "age_gyr": Fixed(5.0),
                },
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                    "tau_diff": Uniform(0.1, 1.5),
                    "tau_bc": Fixed(0.0),
                },
                neb={"type": "none"},
                redshift=Fixed(0.1),
            )
            params = {"log_total_mass": jnp.float32(10.0), "dust_tau_diff": jnp.float32(0.5)}
            f0 = m32.measure_line_fluxes(params, DESI_LINES, approx=True)

            @jax.jit
            def grads(p, obs, sig):
                def chi2(q):
                    f = m32.measure_line_fluxes(q, DESI_LINES, approx=True)
                    return 0.5 * jnp.sum(((f - obs) / sig) ** 2)

                return jax.grad(chi2)(p)

            g = {k: np.asarray(v) for k, v in grads(params, 0.9 * f0, 0.1 * f0).items()}
    assert f0.dtype == jnp.float32 and np.all(np.isfinite(np.asarray(f0)))
    assert all(v.dtype == np.float32 and np.isfinite(v) for v in g.values()), g
    assert g["dust_tau_diff"] != 0.0, g

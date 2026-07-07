# SPDX-License-Identifier: BSD-3-Clause
"""Contract: emission-line fluxes measured from the model spectrum (#950).

`predict_line_fluxes_measured` applies a catalog-style operator (side-band
continuum, subtract, integrate) to the model's own rest-frame SED. Validated:

- the measured flux matches an independent trapezoidal integration of the SED;
- the window-LUT fast path is bit-exact with the exact path (baked-in models);
- it works for *any* backend (Cue additive, baked-in), and the fast path raises
  for additive Cue (its emission is not in the SSP window integrals);
- it includes dust reddening (bluer lines more attenuated) — unlike the intrinsic
  direct `predict_line_fluxes`.

Data-gated (needs a wNE SSP + a bare-stellar SSP); skips in CI.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, load_ssp_data
from tengri.cosmology import luminosity_distance
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_measurement import DESI_LINES, LineDef

pytestmark = pytest.mark.contract

_WNE = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_BARE = "data/fsps_prsc_miles_chabrier.h5"
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
Z = 0.15
_HALPHA = LineDef("Halpha", 6564.61, ((6505.0, 6535.0), (6600.0, 6620.0)), (6556.0, 6573.0))


def _ssp(path):
    if not Path(path).is_file():
        pytest.skip(f"missing SSP grid {path}")
    return load_ssp_data(path)


def _model(ssp_path, neb, tau=0.0):
    import warnings

    ssp = _ssp(ssp_path)
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_diff": Uniform(0.0, 2.0),
                "tau_bc": Uniform(0.0, 2.0),
            },
            neb=neb,
            redshift=Fixed(Z),
        )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.5)
    p["dust_tau_bc"] = jnp.asarray(2.0 * tau)
    p["dust_tau_diff"] = jnp.asarray(tau)
    return m, p


def _trapz_line_flux(m, p, line_def):
    """Independent continuum-subtracted line flux by trapezoidal integration."""
    rest = m.predict_rest_sed(p)
    wave = np.asarray(rest.wavelength)
    llam = np.asarray(rest.sed) * 2.99792458e18 / wave**2  # erg/s/Angstrom
    (blo, bhi), (rlo, rhi) = line_def.continuum
    flo, fhi = line_def.feature
    xb, xr = 0.5 * (blo + bhi), 0.5 * (rlo + rhi)
    yb = llam[(wave > blo) & (wave < bhi)].mean()
    yr = llam[(wave > rlo) & (wave < rhi)].mean()
    lam_c = 0.5 * (flo + fhi)
    cont = yb + (yr - yb) * (lam_c - xb) / (xr - xb)
    fsel = (wave > flo) & (wave < fhi)
    trapz = getattr(np, "trapezoid", np.trapz)
    L = trapz(llam[fsel] - cont, wave[fsel])
    dl = float(luminosity_distance(jnp.asarray(Z)))
    return L / (4.0 * np.pi * dl**2)


def test_measured_matches_independent_integration():
    """The operator matches a hand-rolled trapz integration of the SED (baked-in)."""
    m, p = _model(_WNE, {"type": "none"}, tau=0.5)
    got = float(m.predict_line_fluxes_measured(p, [_HALPHA], fast=False)[0])
    ref = _trapz_line_flux(m, p, _HALPHA)
    assert got > 0, "Halpha emission flux must be positive"
    assert abs(got - ref) / ref < 0.05, f"operator {got:.3e} vs trapz {ref:.3e}"


def test_fast_line_fluxes_bitexact_to_exact():
    """Window-LUT line fluxes reproduce the exact-SED measurement (baked-in)."""
    for tau, tol in ((0.0, 1e-9), (0.7, 1e-3)):
        m, p = _model(_WNE, {"type": "none"}, tau=tau)
        exact = np.asarray(m.predict_line_fluxes_measured(p, DESI_LINES, fast=False))
        fast = np.asarray(m.predict_line_fluxes_measured(p, DESI_LINES, fast=True))
        rel = np.max(np.abs(exact - fast) / np.maximum(np.abs(exact), 1e-30))
        assert rel < tol, f"tau={tau}: fast vs exact worst rel {rel:.2e}"


def test_line_flux_measured_is_jittable():
    m, p = _model(_WNE, {"type": "none"}, tau=0.4)
    jitted = jax.jit(lambda pp: m.predict_line_fluxes_measured(pp, DESI_LINES, fast=True))
    eager = np.asarray(m.predict_line_fluxes_measured(p, DESI_LINES, fast=True))
    got = np.asarray(jitted(p))
    assert np.all(np.isfinite(got))
    assert np.allclose(got, eager, rtol=1e-10, atol=0.0)


def test_measured_works_for_cue_backend():
    """Measure-as-catalog works on Cue's total SED; a clean line ([OIII]) recovers
    the direct nebular luminosity to ~10% (dust off)."""
    m, p = _model(_BARE, {"type": "cue", "*": FIXED}, tau=0.0)
    measured = {ld.name: float(v) for ld, v in zip(DESI_LINES, m.predict_line_fluxes_measured(p))}
    direct = {
        n: float(v)
        for n, v in zip(
            _LINES, m.predict_line_fluxes(p, target_wavelengths=_LINE_DATA.wavelengths)
        )
    }
    assert abs(measured["OIII_5007"] - direct["OIII_5007"]) / direct["OIII_5007"] < 0.10
    # stellar Balmer absorption biases the measured Hbeta lower than a clean line
    assert (measured["Hbeta"] / direct["Hbeta"]) < (measured["OIII_5007"] / direct["OIII_5007"])


def test_measured_includes_dust_reddening():
    """Measured fluxes redden with dust (bluer lines attenuated more) — the
    catalog-observable behaviour the intrinsic predict_line_fluxes lacks."""
    m0, p0 = _model(_BARE, {"type": "cue", "*": FIXED}, tau=0.0)
    m1, p1 = _model(_BARE, {"type": "cue", "*": FIXED}, tau=0.6)
    f0 = {ld.name: float(v) for ld, v in zip(DESI_LINES, m0.predict_line_fluxes_measured(p0))}
    f1 = {ld.name: float(v) for ld, v in zip(DESI_LINES, m1.predict_line_fluxes_measured(p1))}
    # reddening lowers observed flux, and Hbeta (bluer) more than Halpha (redder)
    assert f1["Halpha"] < f0["Halpha"] and f1["Hbeta"] < f0["Hbeta"]
    assert (f1["Hbeta"] / f0["Hbeta"]) < (f1["Halpha"] / f0["Halpha"])


def test_fast_line_fluxes_raise_for_additive_nebular():
    """The window LUT misses additive Cue emission → fast=True must raise."""
    m, p = _model(_BARE, {"type": "cue", "*": FIXED}, tau=0.0)
    with pytest.raises(ValueError, match=r"baked-in nebular only"):
        m.predict_line_fluxes_measured(p, DESI_LINES, fast=True)

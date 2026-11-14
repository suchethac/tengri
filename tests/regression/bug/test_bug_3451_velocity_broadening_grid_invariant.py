# SPDX-License-Identifier: BSD-3-Clause
"""#3451: velocity broadening is grid-invariant (ln-lambda lattice correction).

The velocity-dispersion correction is computed on a uniform ln-lambda lattice, so the
pixel output depends on the SED's interpolant, not on how the model grid is partitioned.
Adding nodes to a grid whose interpolant is unchanged must not move the output; the
flux of a line inside the window is conserved exactly; and halving the lattice spacing
moves the output by only a small amount.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation import spectrum as spectrum_mod
from tengri.observation.spectrum import project_spectrum
from tengri.utils.cosmology import luminosity_distance

pytestmark = pytest.mark.regression_bug

_Z = 0.1
_DL = float(luminosity_distance(jnp.asarray(_Z)))
_SIGMA = 150.0
_OBS = np.linspace(3800.0, 9200.0, 200)


def _coarse_and_fine_grids():
    """A coarse grid and a superset grid, with the coarse SED's linear interpolant on both."""
    coarse = np.geomspace(1000.0, 2.0e4, 1500)
    fine = np.unique(np.concatenate([coarse, 0.5 * (coarse[1:] + coarse[:-1])]))

    def sed(w):
        cont = 1e28 * (w / 5000.0) ** -1.0
        lines = 3e29 * np.exp(-0.5 * ((w - 5007.0) / 2.0) ** 2)
        return cont + lines

    f_coarse = sed(coarse)
    f_fine = np.interp(fine, coarse, f_coarse)
    return (coarse, f_coarse), (fine, f_fine)


def _project(wave, flux, sigma=_SIGMA, conserving=True):
    return np.asarray(
        project_spectrum(
            jnp.asarray(flux),
            jnp.asarray(wave),
            jnp.asarray(_OBS),
            _Z,
            _DL,
            sigma_v_kms=sigma,
            conserving=conserving,
        )
    )


@pytest.mark.parametrize("conserving", [True, False], ids=["pixel-means", "point"])
def test_output_is_invariant_under_grid_refinement(conserving):
    (wc, fc), (wf, ff) = _coarse_and_fine_grids()
    coarse = _project(wc, fc, conserving=conserving)
    fine = _project(wf, ff, conserving=conserving)
    rel = np.max(np.abs(coarse - fine)) / np.max(np.abs(coarse))
    assert rel <= 1e-5, f"grid refinement moved the output by {rel:.3e} of peak"


def test_line_flux_inside_window_is_conserved_exactly():
    """A narrow line well inside the window carries the same flux after broadening."""
    wave = np.geomspace(1000.0, 2.0e4, 6000)
    lam0 = 6000.0 / (1.0 + _Z)
    line = 1e-27 * np.exp(-0.5 * ((wave - lam0) / 3.0) ** 2)
    centres = 0.5 * (_OBS[1:] + _OBS[:-1])
    widths = np.diff(_OBS)

    def total(sigma):
        flux = np.asarray(
            project_spectrum(
                jnp.asarray(line),
                jnp.asarray(wave),
                jnp.asarray(centres),
                _Z,
                _DL,
                sigma_v_kms=sigma,
                conserving=True,
            )
        )
        return np.sum(flux * widths)

    ref = total(0.0)
    assert ref > 0.0
    assert abs(total(_SIGMA) / ref - 1.0) <= 1e-12


def test_sigma_zero_limit_matches_exact_projection():
    """The lattice correction vanishes as sigma_v goes to zero (the D4 check, by construction)."""
    (wc, fc), _ = _coarse_and_fine_grids()
    exact = _project(wc, fc, sigma=0.0)
    tiny = _project(wc, fc, sigma=1e-9)
    rel = np.max(np.abs(tiny - exact)) / np.max(np.abs(exact))
    assert rel <= 1e-6


def test_halving_the_lattice_spacing_converges(monkeypatch):
    """Halving the lattice spacing from its default moves the output by at most 1e-5 of peak."""
    (wc, fc), _ = _coarse_and_fine_grids()
    base_h = spectrum_mod._LATTICE_DLNLAMBDA
    monkeypatch.setattr(spectrum_mod, "_LATTICE_DLNLAMBDA", base_h / 2.0)
    fine_h = _project(wc, fc)
    monkeypatch.setattr(spectrum_mod, "_LATTICE_DLNLAMBDA", base_h)
    default_h = _project(wc, fc)
    rel = np.max(np.abs(fine_h - default_h)) / np.max(np.abs(default_h))
    assert rel <= 1e-5, f"halving the spacing moved the output by {rel:.3e} of peak"

# SPDX-License-Identifier: BSD-3-Clause
"""The line-emitter precompute builders project delta lines with the observation's convention.

This is a builder-level check: it calls each ``line_weight_matrix`` builder directly, not the
runtime model. ``mappings_photo_precompute`` is not registered in
``forward/precompute/registry.py`` yet (#3074), so its entry tests the builder alone.

A delta line's band flux carries the weight at its observed wavelength and the filter
normalization of the convention (#3505); a builder that takes only the transmission at the line
wavelength is wrong by a different factor for every convention. For each adapter that builds a
``line_weight_matrix`` this test checks, under both conventions and for every line the bands
reach:

* the matrix applied to a unit line luminosity [erg/s] reproduces the band flux of the same line
  rendered as a narrow Gaussian and projected with the exact integral, to 6e-3;
* the energy and the photon-counting matrices differ by more than 1e-3, so the convention is
  in the number.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.observation.photometry import FilterConvention, lnu_filter_integral
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.contract

Z = 0.3
RTOL = 6e-3
SIGMA_AA = 1.0
CONVENTIONS = (FilterConvention.BESSELL, FilterConvention.ENERGY)


def _bands():
    """Three broad sine-shaped bands covering the optical to the near infrared."""
    waves, trans = [], []
    for center in (4500.0, 6200.0, 8800.0):
        w = np.linspace(center * 0.84, center * 1.16, 120)
        waves.append(w)
        trans.append(np.sin(np.linspace(0.0, np.pi, w.size)) * 0.6)
    return waves, trans


def _nlr(fw, ft, convention):
    from tengri.components.agn import nlr_gaussian_precompute as m

    out = m.precompute(fw, ft, Z, None, convention=convention)
    return np.asarray(out["line_weight_matrix"]), np.asarray(out["line_wavelengths_obs"]) / (1 + Z)


def _blr(fw, ft, convention):
    from tengri.components.agn import blr_precompute as m

    out = m.precompute(fw, ft, Z, None, convention=convention)
    rest = np.concatenate(
        [np.asarray(out["line_wavelengths_obs"]), np.asarray(out["feii_centers"])]
    ) / (1 + Z)
    return np.asarray(out["line_weight_matrix"]), rest


def _grid_builder(module_path, **kwargs):
    def build(fw, ft, convention):
        import importlib

        m = importlib.import_module(module_path)
        try:
            out = m.precompute(fw, ft, Z, None, convention=convention, **kwargs)
        except FileNotFoundError as exc:
            pytest.skip(f"the grid {module_path} reads is absent: {exc}")
        return np.asarray(out["line_weight_matrix"]), np.asarray(out["line_wavelengths"])

    return build


BUILDERS = {
    "nlr_gaussian": _nlr,
    "blr": _blr,
    "feltre": _grid_builder("tengri.components.nebular.feltre_precompute"),
    "cb19": _grid_builder("tengri.components.nebular.cb19_precompute"),
    "mappings_photo": _grid_builder("tengri.components.nebular.mappings_photo_precompute"),
    "mappings_shock": _grid_builder("tengri.components.nebular.mappings_shock_precompute"),
}


def _reference(rest_wave, fw, ft, convention):
    """Band flux of a unit-luminosity narrow Gaussian line, by the exact projector."""
    row = []
    for w, t in zip(fw, ft, strict=True):
        lo, hi = w[0] / (1 + Z) - 50.0, w[-1] / (1 + Z) + 50.0
        wave = np.arange(max(lo, 1.0), hi, 0.1)
        phi = np.exp(-0.5 * ((wave - rest_wave) / SIGMA_AA) ** 2) / (SIGMA_AA * np.sqrt(2 * np.pi))
        lnu = phi * wave**2 / C_AA
        row.append(float(lnu_filter_integral(lnu, wave, w, t, Z, convention=convention)))
    return np.asarray(row)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_builder_line_weights_match_the_exact_projection_and_follow_the_convention(name):
    fw, ft = _bands()
    matrices = {}
    for convention in CONVENTIONS:
        matrix, rest = BUILDERS[name](fw, ft, convention)
        matrices[convention] = matrix
        # Lines whose Gaussian lies wholly inside a band's interior, away from its edges.
        checked = 0
        for k, lam in enumerate(rest):
            obs = lam * (1 + Z)
            for b, (w, _) in enumerate(zip(fw, ft, strict=True)):
                if w[0] * 1.1 < obs < w[-1] * 0.9 and 0.0 < lam < 1.0e5:
                    ref = _reference(lam, [w], [ft[b]], convention)[0]
                    got = float(matrix[k, b])
                    assert ref > 0.0
                    assert abs(got / ref - 1.0) < RTOL, (
                        f"{name}/{convention.value}: line {lam:.1f} A band {b}: "
                        f"matrix {got:.4e} vs exact {ref:.4e}"
                    )
                    checked += 1
        assert checked > 0, f"{name}: no line falls inside a band; the canary checks nothing"
    live = (matrices[FilterConvention.BESSELL] > 0.0) & (matrices[FilterConvention.ENERGY] > 0.0)
    ratio = matrices[FilterConvention.ENERGY][live] / matrices[FilterConvention.BESSELL][live]
    assert np.max(np.abs(ratio - 1.0)) > 1e-3, f"{name}: the matrices ignore the convention"

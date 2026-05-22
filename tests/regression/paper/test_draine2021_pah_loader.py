"""Tests for the Draine+2021 PAHspec template loader.

Loader returns a frozen ``Draine2021PAHTemplates`` dataclass with:
  - wavelength_um (n_wave,) strictly increasing
  - lgU (15,) values 0..7 step 0.5
  - 4 spectrum cubes shaped (n_starlight, n_slab, 15, 3, 3, n_wave)
  - axis name tuples (starlight_names, ion_names, size_names)

Smoke fixture under ``tests/fixtures/pahspec_smoke.h5`` ships
mMMP non-slab only (135 cells = 15 lgU x 3 ion x 3 size).
"""

from __future__ import annotations

import chex
import pytest

pytestmark = pytest.mark.regression_paper
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pahspec_smoke.h5"


@pytest.fixture(scope="module")
def templates():
    pytest.importorskip("h5py")
    from tengri.components.dust.emission_templates import (
        load_draine2021_pahspec_templates,
    )

    return load_draine2021_pahspec_templates(str(FIXTURE))


def test_grid_shape_and_axes(templates):
    n_sl = len(templates.starlight_names)
    n_slab = templates.slab.size
    n_lgU = templates.lgU.size
    n_ion = len(templates.ion_names)
    n_size = len(templates.size_names)
    n_wave = templates.wavelength_um.size
    assert n_sl == 1  # mMMP only
    assert n_slab == 1  # non-slab only
    assert n_lgU == 15
    assert n_ion == 3
    assert n_size == 3
    assert n_wave > 1000
    chex.assert_shape(templates.nu_pnu_total, (n_sl, n_slab, n_lgU, n_ion, n_size, n_wave))
    np.testing.assert_allclose(
        np.asarray(templates.lgU),
        np.arange(15) * 0.5,
        atol=1e-6,
    )
    assert templates.starlight_names == ("mMMP",)
    assert templates.ion_names == ("lo", "st", "hi")
    assert templates.size_names == ("sma", "std", "lrg")


def test_wavelength_monotonic_and_positive(templates):
    wave = np.asarray(templates.wavelength_um)
    assert wave[0] > 0.5 and wave[-1] < 1e5
    assert np.all(np.diff(wave) > 0)
    # Common wavelength grid coverage required for SED fitting.
    assert wave[0] < 2.0 and wave[-1] > 100.0


def test_spectra_finite_and_nonnegative(templates):
    for arr in (
        templates.nu_pnu_total,
        templates.nu_pnu_astrodust,
        templates.nu_pnu_pah_plus,
        templates.nu_pnu_pah_neutral,
    ):
        a = np.asarray(arr)
        assert np.isfinite(a).all()
        assert (a >= 0).all()


def test_total_equals_sum_of_components(templates):
    total = np.asarray(templates.nu_pnu_total)
    summed = (
        np.asarray(templates.nu_pnu_astrodust)
        + np.asarray(templates.nu_pnu_pah_plus)
        + np.asarray(templates.nu_pnu_pah_neutral)
    )
    np.testing.assert_allclose(total, summed, rtol=1e-3, atol=0)


def test_u_scaling_linear(templates):
    """At fixed (ion, size), TIR should scale linearly with U.
    Because lgU=0 means U=1 (M_MMP) and lgU=1 means U=10, integral of
    nu*P_nu over the full grid should be 10x larger at lgU=1 than at
    lgU=0 (within a few percent due to wavelength gridding).
    """
    from tengri.analysis.feature_strengths import total_ir_power

    wave = np.asarray(templates.wavelength_um)
    spec_u0 = np.asarray(templates.nu_pnu_total[0, 0, 0, 1, 1, :])  # lgU=0, st, std
    spec_u1 = np.asarray(templates.nu_pnu_total[0, 0, 2, 1, 1, :])  # lgU=1, st, std
    f0 = total_ir_power(wave, spec_u0)
    f1 = total_ir_power(wave, spec_u1)
    np.testing.assert_allclose(f1 / f0, 10.0, rtol=0.02)


def test_loader_unit_metadata(templates):
    """Verify the loader exposes expected attributes for traceability."""
    assert hasattr(templates, "paper")
    assert "Draine" in templates.paper or "draine" in templates.paper.lower()
    assert hasattr(templates, "arxiv")
    assert templates.arxiv == "2011.07046"

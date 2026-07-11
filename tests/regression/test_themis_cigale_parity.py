# SPDX-License-Identifier: BSD-3-Clause
"""External parity: tengri's THEMIS matches CIGALE's THEMIS grain-for-grain.

CIGALE's ``themis`` module is an independent implementation of the same
Jones et al. (2017) DustEM grid, so it is a genuine external reference — unlike
the ``themis.npy`` golden, which is captured from tengri itself and will happily
record a wrong grain model.

That gap is exactly how the qhac unit-convention bug survived: the shipped grid
tabulates qhac in FSPS scaling (CIGALE value x 100/2.2) while the user-facing
parameter follows CIGALE ([0.02, 0.40], default 0.17), so every physical qhac
clipped to the grid minimum and selected the wrong grain composition. Nothing in
CI compared the resulting *shape* against CIGALE, so nothing failed.

This test closes that: at matched knobs (qhac=0.17, umin=1.0, gamma=0.1) the
normalized FIR shape must track CIGALE across the alpha sweep, and the FIR peak
must land at the same wavelength. Pre-fix these diverged by tens of percent with
different peak positions.

Reference data: ``data/cigale_themis_reference.npz``, regenerated with
``scripts/regenerate_themis_from_cigale.py`` (needs pcigale; the test itself does
not import it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

_REF = Path(__file__).resolve().parents[2] / "data" / "cigale_themis_reference.npz"
_C_AA_S = 2.998e18  # speed of light [Angstrom/s]

# Compare on the FIR/MIR range where the dust template dominates.
_FIR_MIN_AA = 3.0e4
_FIR_MAX_AA = 1.0e7

# The aromatic (PAH) features that ``qhac`` actually controls, 5-20 um.
_MIR_MIN_AA = 5.0e4
_MIR_MAX_AA = 2.0e5


def _normalized(l_nu: np.ndarray, wave: np.ndarray) -> np.ndarray:
    """Unit-bolometric shape, so the comparison is scale-free."""
    nu = _C_AA_S / wave
    integral = -np.trapezoid(l_nu, nu)
    return l_nu / integral


@pytest.fixture(scope="module")
def reference() -> dict:
    if not _REF.is_file():
        pytest.skip(f"CIGALE THEMIS reference not on disk: {_REF}")
    return dict(np.load(_REF))


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_themis_shape_matches_cigale(reference: dict, alpha: float) -> None:
    """Normalized THEMIS FIR shape tracks CIGALE at matched qhac/umin/gamma."""
    from tengri.components.dust.emission import themis as themis_emission

    wave = np.asarray(reference["wave_aa"], dtype=np.float64)
    ref = np.asarray(reference[f"alpha_{alpha:.1f}"], dtype=np.float64)
    qhac = float(reference["qhac"])
    umin = float(reference["umin"])
    gamma = float(reference["gamma"])

    band = (wave >= _FIR_MIN_AA) & (wave <= _FIR_MAX_AA) & (ref > 0.0)
    w = wave[band]

    tng = np.asarray(
        themis_emission(
            w, 1.0, dust_umin=umin, dust_gamma_dl=gamma, dust_qhac=qhac, dust_alpha=alpha
        ),
        dtype=np.float64,
    )

    a = _normalized(ref[band], w)
    b = _normalized(tng, w)
    rel = np.abs(b - a) / np.max(a)
    median_rel = float(np.median(rel))

    assert median_rel < 0.05, (
        f"THEMIS shape disagrees with CIGALE at alpha={alpha}, qhac={qhac}: "
        f"median |rel diff| = {median_rel:.3f} (>5%). A qhac convention/clipping "
        f"regression selects the wrong grain model."
    )


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_themis_mir_pah_shape_matches_cigale(reference: dict, alpha: float) -> None:
    """The 5-20 um aromatic-feature band matches CIGALE — the sharpest qhac gate.

    ``qhac`` *is* the a-C(:H) aromatic mass fraction, so its signature lives in
    the mid-IR PAH features, not the FIR peak (which is set mostly by ``umin``
    and ``alpha``). The FIR check above only separates the wrong grain at
    alpha=1; this band separates it at every alpha, so it is the test that would
    actually have caught the clipping bug: the wrong grain sits at 8-15% here
    versus <3% for the correct one.
    """
    from tengri.components.dust.emission import themis as themis_emission

    wave = np.asarray(reference["wave_aa"], dtype=np.float64)
    ref = np.asarray(reference[f"alpha_{alpha:.1f}"], dtype=np.float64)
    qhac = float(reference["qhac"])
    umin = float(reference["umin"])
    gamma = float(reference["gamma"])

    band = (wave >= _MIR_MIN_AA) & (wave <= _MIR_MAX_AA) & (ref > 0.0)
    w = wave[band]
    tng = np.asarray(
        themis_emission(
            w, 1.0, dust_umin=umin, dust_gamma_dl=gamma, dust_qhac=qhac, dust_alpha=alpha
        ),
        dtype=np.float64,
    )

    a = _normalized(ref[band], w)
    b = _normalized(tng, w)
    median_rel = float(np.median(np.abs(b - a) / np.max(a)))

    assert median_rel < 0.05, (
        f"THEMIS mid-IR (PAH) shape disagrees with CIGALE at alpha={alpha}, "
        f"qhac={qhac}: median |rel diff| = {median_rel:.3f} (>5%). The aromatic "
        f"fraction is wrong — a qhac convention/clipping regression."
    )


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_themis_absolute_normalization_matches_cigale(reference: dict, alpha: float) -> None:
    """Absolute scale, not just shape: fed CIGALE's L_dust, tengri emits the same total.

    The shape tests above normalize both sides to unit bolometric, so they are
    blind to the *normalization* — a template emitting twice the absorbed energy
    would sail through them. This pins the absolute scale.

    Regression: the CMB contrast factor was applied to the emitted SED after the
    unit-integral renormalization, so tengri re-emitted only 98.4% of the
    absorbed energy at alpha=1 while CIGALE conserves to 0.01%.
    """
    from tengri.components.dust.emission import themis as themis_emission

    wave = np.asarray(reference["wave_aa"], dtype=np.float64)
    ref = np.asarray(reference[f"alpha_{alpha:.1f}"], dtype=np.float64)
    qhac = float(reference["qhac"])
    umin = float(reference["umin"])
    gamma = float(reference["gamma"])

    # CIGALE's own emitted bolometric == the energy its dust absorbed.
    nu = _C_AA_S / wave
    order = np.argsort(nu)
    l_dust = float(np.trapezoid(ref[order], nu[order]))

    # Feed tengri the same absorbed luminosity; it must re-emit the same total.
    tng = np.asarray(
        themis_emission(
            wave, l_dust, dust_umin=umin, dust_gamma_dl=gamma, dust_qhac=qhac, dust_alpha=alpha
        ),
        dtype=np.float64,
    )
    emitted = float(np.trapezoid(tng[order], nu[order]))

    ratio = emitted / l_dust
    assert abs(ratio - 1.0) < 0.01, (
        f"THEMIS absolute normalization at alpha={alpha}: tengri emits {ratio:.4f} of "
        f"the absorbed energy that CIGALE re-emits (expected 1.0 +/- 1%). Something "
        f"scales the SED after the energy-balance renormalization."
    )


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_themis_fir_peak_matches_cigale(reference: dict, alpha: float) -> None:
    """The FIR peak lands at the same wavelength as CIGALE.

    The peak position is the sharpest signature of the grain composition: the
    clipped (low-aromatic) grain peaked in a visibly different place.
    """
    from tengri.components.dust.emission import themis as themis_emission

    wave = np.asarray(reference["wave_aa"], dtype=np.float64)
    ref = np.asarray(reference[f"alpha_{alpha:.1f}"], dtype=np.float64)
    qhac = float(reference["qhac"])
    umin = float(reference["umin"])
    gamma = float(reference["gamma"])

    band = (wave >= _FIR_MIN_AA) & (wave <= _FIR_MAX_AA) & (ref > 0.0)
    w = wave[band]
    tng = np.asarray(
        themis_emission(
            w, 1.0, dust_umin=umin, dust_gamma_dl=gamma, dust_qhac=qhac, dust_alpha=alpha
        ),
        dtype=np.float64,
    )

    peak_ref = w[np.argmax(ref[band])]
    peak_tng = w[np.argmax(tng)]
    assert abs(peak_tng - peak_ref) / peak_ref < 0.10, (
        f"THEMIS FIR peak at alpha={alpha}: tengri {peak_tng / 1e4:.1f} um vs "
        f"CIGALE {peak_ref / 1e4:.1f} um (>10% apart) — wrong grain composition."
    )

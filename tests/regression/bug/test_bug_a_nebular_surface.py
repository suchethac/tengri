# SPDX-License-Identifier: BSD-3-Clause
"""Regression: nebular output surface cleanup (#301, #303, #313).

Three related fixes on the predict_emission_lines + nebular pipeline:

- **#313**: predict_emission_lines did not apply dust attenuation —
  Balmer decrement stayed at intrinsic ~2.85 across an A_V sweep.
  Fix: attenuate line luminosities through the active dust model
  (Charlot & Fall 2000 BC + diffuse, by default).
- **#303**: EmissionLines NamedTuple exposed only 11 hardcoded
  species. The Cue backend trains ~138 lines and CloudyGrid more.
  Fix: add ``all_waves`` / ``all_lums`` arrays + ``.get(wavelength)``
  helper exposing the full catalog.
- **#301**: neb_fesc scaled the nebular continuum and lines but left
  the stellar LyC continuum untouched, breaking the "stellar LyC ×
  fesc + nebular ∝ (1 − fesc)" energy balance.
  Fix: attenuate stellar continuum below 912 Å by ``fesc`` in
  NebularSEDComponent.apply.

The Cue-needing assertions are guarded with skips when the bare-stellar
SSP file isn't present (CI / minimal install). The BakedIn assertions
verify the predict_emission_lines NotImplementedError path is intact.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug

import tengri
from tengri import FIXED, Fixed
from tengri.forward.prediction import EmissionLines

_BARE_STELLAR_SSP = pathlib.Path("data/ssp_prsc_miles_chabrier_noNE.h5").resolve()


@pytest.fixture(scope="module")
def ssp_bare():
    if not _BARE_STELLAR_SSP.exists():
        pytest.skip(f"Bare-stellar SSP not present at {_BARE_STELLAR_SSP}")
    return tengri.load_ssp_data(str(_BARE_STELLAR_SSP))


def test_emission_lines_namedtuple_has_full_catalog_fields():
    """#303: EmissionLines exposes all_waves / all_lums + .get(wavelength)."""
    assert "all_waves" in EmissionLines._fields
    assert "all_lums" in EmissionLines._fields
    assert hasattr(EmissionLines, "get")


def test_emission_lines_get_handles_empty_catalog():
    """#303: .get() returns NaN when the active backend has no catalog."""
    empty = jnp.asarray([], dtype=jnp.float64)
    nan_scalar = jnp.asarray(jnp.nan)
    lines = EmissionLines(
        lya=nan_scalar,
        civ_1549=nan_scalar,
        oii=nan_scalar,
        hbeta=nan_scalar,
        oiii_4959=nan_scalar,
        oiii_5007=nan_scalar,
        nii_6548=nan_scalar,
        halpha=nan_scalar,
        nii_6584=nan_scalar,
        sii_6717=nan_scalar,
        sii_6731=nan_scalar,
        all_waves=empty,
        all_lums=empty,
    )
    assert bool(jnp.isnan(lines.get(1640.0)))


def test_emission_lines_get_returns_nearest_within_tolerance():
    """#303: .get() returns the nearest line within ``tol_aa``, else NaN."""
    waves = jnp.asarray([1216.0, 1640.0, 5007.0])
    lums = jnp.asarray([10.0, 5.0, 100.0])
    lines = EmissionLines(
        lya=jnp.asarray(jnp.nan),
        civ_1549=jnp.asarray(jnp.nan),
        oii=jnp.asarray(jnp.nan),
        hbeta=jnp.asarray(jnp.nan),
        oiii_4959=jnp.asarray(jnp.nan),
        oiii_5007=jnp.asarray(jnp.nan),
        nii_6548=jnp.asarray(jnp.nan),
        halpha=jnp.asarray(jnp.nan),
        nii_6584=jnp.asarray(jnp.nan),
        sii_6717=jnp.asarray(jnp.nan),
        sii_6731=jnp.asarray(jnp.nan),
        all_waves=waves,
        all_lums=lums,
    )
    assert float(lines.get(1640.0)) == pytest.approx(5.0)
    assert float(lines.get(5006.5, tol_aa=2.0)) == pytest.approx(100.0)
    # Outside tolerance → NaN
    assert bool(jnp.isnan(lines.get(2000.0, tol_aa=5.0)))


def test_balmer_decrement_rises_under_dust_sweep(ssp_bare):
    """#313: Hα/Hβ should rise from ~2.85 (no dust) to ~6+ (A_V=2 Calzetti)."""
    ratios = []
    for tau_diff in [0.0, 1.0, 2.0]:
        m = tengri.SEDModel.build(
            ssp_bare,
            sfh={"type": "const", "*": FIXED, "log_sfr": 0.0},
            dust={
                "type": "two_component",
                "*": FIXED,
                "law_bc": "calzetti",
                "tau_diff": tau_diff,
                "tau_bc": 0.0,
                "slope": -0.7,
            },
            neb={"type": "cue", "*": FIXED},
            redshift=Fixed(0.05),
        )
        p = dict(m.spec.sample(jax.random.PRNGKey(0)))
        lines = m.predict_emission_lines(p)
        ratios.append(float(lines.halpha) / float(lines.hbeta))

    # Intrinsic ratio is ~2.85 (case-B). Calzetti pushes higher with A_V.
    # Pre-fix: all three were ~2.85 (the bug). Post-fix: strictly increasing.
    assert ratios[0] < ratios[1] < ratios[2], (
        f"Balmer decrement should rise under dust sweep, got {ratios}"
    )
    assert ratios[0] == pytest.approx(2.86, abs=0.5)


def test_cue_exposes_more_than_thirteen_species(ssp_bare):
    """#303: Cue's all_waves should expose ≫13 lines (full catalog)."""
    m = tengri.SEDModel.build(
        ssp_bare,
        sfh={"type": "const", "*": FIXED, "log_sfr": 0.0},
        dust={"type": "two_component", "*": FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        neb={"type": "cue", "*": FIXED},
        redshift=Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    lines = m.predict_emission_lines(p)
    assert lines.all_waves.size > 50, f"Expected >50 species, got {lines.all_waves.size}"
    # HeII 1640 — the issue's canonical example
    heii = float(lines.get(1640.4, tol_aa=5.0))
    assert jnp.isfinite(heii) and heii > 0


def test_stellar_lyc_attenuated_by_fesc(ssp_bare):
    """#301: stellar LyC continuum should scale with fesc (0 absorption ⇒ 0 LyC observed)."""
    import numpy as np

    lyc_means = {}
    for fesc in [0.0, 0.5, 1.0]:
        m = tengri.SEDModel.build(
            ssp_bare,
            sfh={"type": "const", "*": FIXED, "log_sfr": 0.0},
            neb={"type": "cue", "*": FIXED, "fesc": fesc},
            dust={"type": "two_component", "*": FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
            redshift=Fixed(0.05),
        )
        p = dict(m.spec.sample(jax.random.PRNGKey(0)))
        out = m.predict_rest_sed(p)
        w = np.asarray(out.wavelength)
        L = np.asarray(out.sed)
        # Stellar-only band (just below 912 Å): nebular cont is small there
        # vs the stellar LyC. After fix, fesc=0 should knock out the stellar
        # continuum here, fesc=1 leaves it alone.
        lyc_mean = float(np.mean(L[(w > 700) & (w < 912)]))
        lyc_means[fesc] = lyc_mean

    # Pre-fix: all three lyc_means were identical (bug).
    # Post-fix: lyc_means[0.0] < lyc_means[0.5] < lyc_means[1.0]
    assert lyc_means[0.0] < lyc_means[1.0], (
        f"LyC at fesc=0 should be ≪ LyC at fesc=1, got {lyc_means}"
    )


def test_bakedin_predict_lines_still_raises():
    """Pre-existing behavior: BakedIn nebular raises NotImplementedError."""
    if not _BARE_STELLAR_SSP.exists():
        pytest.skip("Need any SSP for this test")
    ssp = tengri.load_ssp_data(str(_BARE_STELLAR_SSP))
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "const", "*": FIXED, "log_sfr": 0.0},
        dust={"type": "two_component", "*": FIXED},
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with pytest.raises(NotImplementedError, match="BakedIn"):
        m.predict_emission_lines(p)

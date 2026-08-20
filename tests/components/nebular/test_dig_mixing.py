# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #259: DIG parameters propagate into Cue line predictions.

Before the fix, the ``neb_dig_frac`` and ``neb_dig_delta_logU`` knobs were
silently dropped — the Cue backend was called once with the HII ``neb_logU``
and the result was returned unchanged. Sweeping ``dig_delta_logU`` across
its full prior range with ``dig_frac=0.9`` left every line ratio
bit-identical.

The fix wires a second backend call (DIG branch with shifted ``neb_logU``)
and linearly mixes the result by ``dig_frac``. This test pins both:

1. ``dig_frac=0`` reproduces the HII-only result (no-DIG case unchanged).
2. ``dig_frac > 0`` makes ``dig_delta_logU`` measurably move the line
   ratios. The exact magnitudes are model-dependent, but the direction
   of the canonical DIG signature (Belfiore+2022, Vale Asari+2019) holds:
   low Δlog U → high [N II]/Hα and [S II]/Hα, low [O III]/Hβ.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platforms", "cpu")

import tengri
from tengri.parameters.priors import Fixed, Uniform

_SSP_PATH = Path(__file__).resolve().parents[3] / "data" / "fsps_prsc_miles_chabrier.h5"


def _has_cue_weights() -> bool:
    return (Path(__file__).resolve().parents[3] / "data" / "cue_weights.npz").is_file()


@pytest.fixture(scope="module")
def base_model():
    if not _SSP_PATH.is_file() or not _has_cue_weights():
        pytest.skip("Cue weights or SSP fixture missing")
    return tengri.SEDModel.build(
        tengri.load_ssp_data(str(_SSP_PATH)),
        sfh={
            "type": "dpl",
            "*": tengri.FIXED,
            "tau_gyr": 0.3,
            "log_total_mass": 1.5,
            "alpha": 1.0,
            "beta": 2.5,
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        neb={
            "type": "cue",
            "*": tengri.FIXED,
            "dig_frac": Uniform(0.0, 1.0),
            "dig_delta_logU": Uniform(-3.0, -0.1),
        },
        redshift=Fixed(0.05),
    )


def _line_ratios(model, dig_frac: float, dig_delta_logU: float) -> dict:
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    p["neb_dig_frac"] = jnp.float64(dig_frac)
    p["neb_dig_delta_logU"] = jnp.float64(dig_delta_logU)
    lines = model.predict_emission_lines(p)
    return {
        "nii_ha": float(lines.nii_6584 / lines.halpha),
        "sii_ha": float((lines.sii_6717 + lines.sii_6731) / lines.halpha),
        "oiii_hb": float(lines.oiii_5007 / lines.hbeta),
    }


@pytest.mark.regression_bug
def test_dig_frac_zero_matches_hii_only(base_model):
    """dig_frac=0 must reproduce the pre-fix (HII-only) ratios bit-exactly."""
    # Two calls with dig_frac=0 but different dig_delta_logU must be
    # bit-identical: the DIG branch is short-circuited away.
    r1 = _line_ratios(base_model, dig_frac=0.0, dig_delta_logU=-3.0)
    r2 = _line_ratios(base_model, dig_frac=0.0, dig_delta_logU=-0.3)
    for key in r1:
        assert r1[key] == pytest.approx(r2[key], rel=0, abs=0), (
            f"{key} changed when dig_frac=0: {r1[key]} vs {r2[key]}"
        )


@pytest.mark.regression_bug
def test_dig_delta_logU_moves_line_ratios(base_model):
    """At dig_frac=0.9, sweeping dig_delta_logU must move line ratios.

    Tracks issue #259: before the fix, the four readings were bit-identical.
    """
    ratios = {
        dlu: _line_ratios(base_model, dig_frac=0.9, dig_delta_logU=dlu)
        for dlu in (-3.0, -2.0, -1.0, -0.3)
    }
    # At least one ratio must vary meaningfully across the sweep.
    for key in ("nii_ha", "sii_ha", "oiii_hb"):
        values = np.array([ratios[dlu][key] for dlu in ratios])
        spread = float(values.max() - values.min())
        rel_spread = spread / float(np.median(values))
        assert rel_spread > 0.05, (
            f"{key} barely moves under dig_delta_logU sweep at dig_frac=0.9: "
            f"values={values}, rel_spread={rel_spread:.4f}"
        )


@pytest.mark.regression_bug
def test_oiii_hb_rises_as_dig_approaches_hii(base_model):
    """Δlog U → 0 should make the DIG component look more like an HII
    region (more [O III]). Δlog U = -3 keeps it deep in low-ionization
    territory."""
    deep_dig = _line_ratios(base_model, dig_frac=0.9, dig_delta_logU=-3.0)
    near_hii = _line_ratios(base_model, dig_frac=0.9, dig_delta_logU=-0.3)
    # [O III]/Hβ should be higher when the DIG ionization parameter
    # is closer to the HII value.
    assert near_hii["oiii_hb"] > deep_dig["oiii_hb"], (
        f"[O III]/Hβ should rise as Δlog U → 0: "
        f"deep={deep_dig['oiii_hb']:.4f}, near={near_hii['oiii_hb']:.4f}"
    )

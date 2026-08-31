# SPDX-License-Identifier: BSD-3-Clause
"""Contract: WavePrecomp warns when its first-order dust projection (#617) biases
rest-UV bands.

The photometry LUT bakes the SSP×filter integral at zero dust and re-applies
attenuation as a first-order Taylor expansion across each filter. That is
accurate in the optical/IR but silently wrong in the rest-UV (steep, extrapolated
attenuation) — catastrophically so at moderate/high redshift. The model must emit
a build-time ``UserWarning`` for such configurations so no fit is biased
unknowingly. Runs on the synthetic wide SSP + hand-made top-hat filters (no
``data/`` needed).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

_MATCH = "first-order Taylor projection"

#: The warning describes the Taylor form, which since #1122 lives only behind an
#: explicit opt-out. Every "does not warn" case below must be built HERE, not on
#: the default: on the default the quadrature short-circuits the check, and the
#: tests would pass without exercising the dust-off / optical-only reason they
#: claim to test.
_TAYLOR = WavePrecomp(n_subbands=0, taylor_correction=True)

# Observed-frame centers: at z=1, 1500 Å -> rest 750 Å (deep UV, flagged);
# 6000 Å -> rest 3000 Å and 8000 Å -> rest 4000 Å (optical/NIR, not flagged).
_UV = (1500.0, "uv")
_OPT = (6000.0, "opt")
_NIR = (8000.0, "nir")


def _tophat(center: float, name: str, frac: float = 0.12, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


def _obs(bands: list[tuple[float, str]]) -> Observation:
    return Observation(photometry=Photometry(filters=tuple(_tophat(c, n) for c, n in bands)))


def _build(ssp, obs, approx, tau, z: float = 1.0) -> SEDModel:
    dust: dict = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
    }
    if tau == "free":
        dust["tau_diff"] = Uniform(0.0, 1.0)
    else:  # explicit numeric optical depth on both components
        dust["tau_bc"] = Fixed(float(tau))
        dust["tau_diff"] = Fixed(float(tau))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=approx,
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation=dust,
        neb={"type": "none"},
        redshift=Fixed(z),
    )


def _flagged(recwarn) -> list:
    return [w for w in recwarn if _MATCH in str(w.message)]


def test_warns_for_uv_band_with_dust_under_wave_precomp(synthetic_ssp_wide):
    """A rest-UV band + dust + the Taylor projection must raise the bias warning."""
    obs = _obs([_UV, _OPT])
    with pytest.warns(UserWarning, match=_MATCH):
        _build(synthetic_ssp_wide, obs, _TAYLOR, tau="free")


def test_no_warning_on_exact_path(synthetic_ssp_wide, recwarn):
    """The exact path (approx=None) is unbiased — no warning even with a UV band."""
    _build(synthetic_ssp_wide, _obs([_UV, _OPT]), None, tau="free")
    assert not _flagged(recwarn)


def test_no_warning_when_dust_off(synthetic_ssp_wide, recwarn):
    """Zero dust → the LUT is exact → no warning."""
    _build(synthetic_ssp_wide, _obs([_UV, _OPT]), _TAYLOR, tau=0.0)
    assert not _flagged(recwarn)


def test_no_warning_for_optical_only(synthetic_ssp_wide, recwarn):
    """No rest-UV band → the Taylor projection is accurate → no warning."""
    _build(synthetic_ssp_wide, _obs([_OPT, _NIR]), _TAYLOR, tau="free")
    assert not _flagged(recwarn)


def test_the_default_quadrature_does_not_warn(synthetic_ssp_wide, recwarn):
    """The default WavePrecomp() EVALUATES the screen at n_subbands=5 nodes per band
    (#1122) and folds the IGM into the same nodes at build time (#1135), so the
    Taylor-era bias is gone and the warning must not fire.

    This is the whole point of the sub-band quadrature. A warning here would send
    astronomers to ``approx=None`` — 80x slower — to escape a bias that measurement
    against the exact path puts at <=0.5% in the worst rest-UV band. Guards against
    the warning silently outliving the defect it describes, which is exactly what
    happened between #1122 landing and #1135.
    """
    _build(synthetic_ssp_wide, _obs([_UV, _OPT]), WavePrecomp(), tau="free")
    assert not _flagged(recwarn)

# SPDX-License-Identifier: BSD-3-Clause
"""A joint model's photometry depended on *which* precompute object you passed.

On a joint photometry+spectroscopy observation, any precompute opt-in promotes
to BOTH LUT families (``wave_precomp`` and ``spectrum_precomp``). But the dust
screen's sub-band quadrature order (#1122) was read off whichever config object
happened to be passed::

    self._approx["n_subbands"] = int(getattr(self._approx_config, "n_subbands", 0))

``SpectrumPrecomp`` has no ``n_subbands`` field — a spectrum pixel is a point,
not a bandpass, so there is no effective-wavelength residual to correct. The
``getattr`` default therefore fired, and ``0`` is not ``WavePrecomp``'s default
(``5``): it is the sentinel that *disables* the quadrature. Its ``taylor_correction
= True`` (declared "for API symmetry", documented as inert for spectroscopy) was
picked up the same way.

So a joint model built with ``approx=SpectrumPrecomp()`` promoted photometry onto
the LUT and then ran it through the **pre-#1122 effective-wavelength + Taylor**
path, whose own docstring records the failure mode it was replaced for:

======================================  ==============  ==============
joint model, photometry err vs exact    ``WavePrecomp`` ``SpectrumPrecomp``
======================================  ==============  ==============
NUV, z = 0.05, tau = 2                  -0.11 %         -2.6 %
FUV, z = 0.5,  tau = 2                  -0.82 %         -5.6 %
NUV, z = 1.0,  tau = 0.5                -1.07 %         **+8.4 %**
======================================  ==============  ==============

A *speed* knob silently changed the physics — the same class as the LUT dropping
AGN and nebular emission (#737/#740) and the flux calibration (#1031).

The headline assertion is **parity across the three promotion forms**: they build
identical LUT families, so any difference between them is a knob leak, whatever
its size. Asserting only "close to the exact path" would have to admit the LUT's
own documented residual (~0.6 % at K=5) and would have waved the 0.16 % optical
case straight through.

These use the synthetic SSP deliberately. The equivalent contract test
(``tests/contract/test_spectrum_lut.py::TestJointPrecomp``) needs a bare-stellar
grid under ``data/``, which CI does not have, so it *skips* there — which is how
this reached main with the Tests workflow green.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    SpectrumPrecomp,
    WavePrecomp,
)
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

# Rest-UV through optical: the sub-band quadrature only matters where the
# attenuation curve is steep across a bandpass. An optical-only setup registers
# the leak as a benign 0.16 % and hides how bad it gets.
_BANDS = {"FUV": (1400.0, 1800.0), "NUV": (2000.0, 2800.0), "g": (4000.0, 5500.0)}

# Every precompute opt-in that promotes a joint observation to both LUT families.
_JOINT_FORMS = {
    "WavePrecomp": lambda: WavePrecomp(),
    "SpectrumPrecomp": lambda: SpectrumPrecomp(),
    "composite": lambda: (WavePrecomp(), SpectrumPrecomp()),
}


def _joint_obs():
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 60), trans=jnp.ones(60) * 0.5, name=name)
        for name, (lo, hi) in _BANDS.items()
    )
    return Observation(
        photometry=Photometry(filters=curves),
        spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4500.0, 7500.0, 64)),
    )


def _build(ssp, obs, approx, *, tau_diff=1.5, z=0.05):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FIXED},
            # Diffuse-only: isolates the screen quadrature from the birth-cloud
            # LUT residual (#617).
            dust={"law_diff": 'calzetti', 
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_bc": 0.0,
                "tau_diff": tau_diff,
            },
            neb={"type": "none"},
            redshift=Fixed(z),
            approx=approx,
        )


@pytest.mark.parametrize("form", sorted(_JOINT_FORMS))
def test_joint_promotion_keeps_the_dust_quadrature(synthetic_ssp_wide, form):
    """The root cause, pinned directly.

    Every opt-in that promotes photometry onto the LUT must give that LUT the
    quadrature it needs. Compared against ``WavePrecomp``'s own defaults rather
    than the literal ``5`` so raising K in the dataclass does not silently make
    this test assert the stale order.
    """
    model = _build(synthetic_ssp_wide, _joint_obs(), _JOINT_FORMS[form]())

    assert model._approx["wave_precomp"] is True
    assert model._approx["spectrum_precomp"] is True
    assert model._approx["n_subbands"] == WavePrecomp().n_subbands
    assert model._approx["taylor_correction"] == WavePrecomp().taylor_correction


def test_joint_photometry_is_identical_across_promotion_forms(synthetic_ssp_wide):
    """The contract: all three forms build the same two LUT families.

    They therefore have to produce the *same* photometry. A difference of any
    size means a knob reached one family and not the other.
    """
    obs = _joint_obs()
    phot = {
        form: _build(synthetic_ssp_wide, obs, factory()).predict_photometry({})
        for form, factory in _JOINT_FORMS.items()
    }

    reference = phot["composite"]
    for form, values in phot.items():
        assert jnp.allclose(values, reference, rtol=1e-12, atol=0.0), (
            f"joint approx={form} photometry differs from the composite form: "
            f"{jnp.max(jnp.abs(values - reference) / reference):.4%} — a "
            f"precompute knob leaked into one LUT family but not the other"
        )


# Redshifts that keep the bluest band (1400 Å observed) above the Lyman limit in
# the rest frame. Push past z ~ 0.53 and it samples rest-frame < 912 Å, where the
# flux is ~0 and a *relative* error explodes for reasons unrelated to the
# quadrature — the LUT reads 12 % out there even when it is behaving correctly.
_UV_SAFE_Z = [0.05, 0.5]

# One bound, shared by the guard and its teeth check, so they cannot drift apart.
# It has to sit above the LUT's own residual (≤ 0.42 % measured, ~0.6 % documented
# at K=5) and below the pre-fix error (1.93 % at z=0.05, worse at z=0.5). The
# window is real but not generous — hence the teeth check.
_MAX_LUT_ERR = 0.01


@pytest.mark.parametrize("z", _UV_SAFE_Z)
def test_joint_spectrum_precomp_tracks_the_exact_path_in_the_uv(synthetic_ssp_wide, z):
    """Severity guard, measured against the exact path — never another precompute.

    ``approx=None`` is the reference. The LUT keeps a documented residual of its
    own (~0.6 % at K=5; ≤ 0.42 % measured here), so the bound is loose by design.
    Before the fix these bands ran several percent out, well past it.
    """
    obs = _joint_obs()
    exact = _build(synthetic_ssp_wide, obs, None, tau_diff=2.0, z=z).predict_photometry({})
    lut = _build(synthetic_ssp_wide, obs, SpectrumPrecomp(), tau_diff=2.0, z=z).predict_photometry(
        {}
    )

    rel = float(jnp.max(jnp.abs(lut - exact) / jnp.abs(exact)))
    assert rel < _MAX_LUT_ERR, f"joint SpectrumPrecomp photometry is {rel:.2%} off exact"


@pytest.mark.parametrize("z", _UV_SAFE_Z)
def test_the_pre_fix_path_really_was_this_wrong(synthetic_ssp_wide, z):
    """Teeth check: the bound above is not vacuous.

    ``WavePrecomp(n_subbands=0, taylor_correction=True)`` reconstructs exactly what
    a joint ``SpectrumPrecomp()`` used to resolve to. Asserting it *violates* the
    guard proves the guard would have caught the bug, without reverting the fix.
    """
    obs = _joint_obs()
    exact = _build(synthetic_ssp_wide, obs, None, tau_diff=2.0, z=z).predict_photometry({})
    pre_fix = _build(
        synthetic_ssp_wide,
        obs,
        WavePrecomp(n_subbands=0, taylor_correction=True),
        tau_diff=2.0,
        z=z,
    ).predict_photometry({})

    rel = float(jnp.max(jnp.abs(pre_fix - exact) / jnp.abs(exact)))
    assert rel > _MAX_LUT_ERR, (
        f"the pre-#1122 effective-wavelength path is only {rel:.2%} off exact at "
        f"z={z}, inside the {_MAX_LUT_ERR:.0%} guard above — that guard would NOT "
        f"have caught the leak, so it needs a harsher band or a tighter bound"
    )

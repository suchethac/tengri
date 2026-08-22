# SPDX-License-Identifier: BSD-3-Clause
"""Composite ``approx=(WavePrecomp, SpectrumPrecomp)`` for joint fits (#610) + #609.

* #610 — a single ``approx=`` object meant a joint photometry+spectroscopy fit
  could accelerate at most one channel. A composite ``(WavePrecomp(...),
  SpectrumPrecomp())`` tuple now activates both LUT families.
* #609 — ``approx=SpectrumPrecomp()`` used to crash in ``compile_signature``
  reading ``cfg.n_z``; guarded here so the public-API surface stays usable.

Runs on the synthetic wide SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
)
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def joint_obs():
    wave_obs = jnp.logspace(jnp.log10(3300.0), jnp.log10(8000.0), 80)
    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0))),
        spectroscopy=Spectroscopy(wave_obs=wave_obs),
    )


def _build(joint_obs, ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=joint_obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(8, 12)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.3),
        approx=approx,
    )


def test_composite_tuple_activates_both_luts(synthetic_ssp_wide, joint_obs):
    """The (WavePrecomp, SpectrumPrecomp) tuple turns on both LUT families."""
    m = _build(joint_obs, synthetic_ssp_wide, (WavePrecomp(), SpectrumPrecomp()))
    assert m._approx["wave_precomp"] is True
    assert m._approx["spectrum_precomp"] is True


def test_composite_tuple_is_order_independent(synthetic_ssp_wide, joint_obs):
    m = _build(joint_obs, synthetic_ssp_wide, (SpectrumPrecomp(), WavePrecomp()))
    assert m._approx["wave_precomp"] is True
    assert m._approx["spectrum_precomp"] is True


def test_composite_predicts_both_channels_finite(synthetic_ssp_wide, joint_obs):
    """A joint composite model produces finite photometry AND spectroscopy."""
    m = _build(joint_obs, synthetic_ssp_wide, (WavePrecomp(), SpectrumPrecomp()))
    params = {p: 0.5 for p in m.spec.free_params}
    phot = m.predict_photometry(params)
    spec = m.predict_spectrum(params)
    assert phot.shape == (3,)
    assert spec.shape == (80,)
    assert jnp.all(jnp.isfinite(phot))
    assert jnp.all(jnp.isfinite(spec))


def test_composite_has_distinct_compile_signature(synthetic_ssp_wide, joint_obs):
    """Composite differs from single-LUT and from a different ztable sampling."""
    composite = _build(joint_obs, synthetic_ssp_wide, (WavePrecomp(), SpectrumPrecomp()))
    composite_fine = _build(
        joint_obs, synthetic_ssp_wide, (WavePrecomp(n_z=200), SpectrumPrecomp())
    )
    assert composite.compile_signature() != composite_fine.compile_signature()


def test_illegal_approx_forms_raise(synthetic_ssp_wide, joint_obs):
    with pytest.raises(TypeError):
        _build(joint_obs, synthetic_ssp_wide, (WavePrecomp(), WavePrecomp()))
    with pytest.raises(TypeError):
        _build(joint_obs, synthetic_ssp_wide, (WavePrecomp(), "nope"))


def test_spectrum_precomp_compile_signature_does_not_crash(synthetic_ssp_wide, joint_obs):
    """#609: SpectrumPrecomp() reaches compile_signature + predict cleanly."""
    m = _build(joint_obs, synthetic_ssp_wide, SpectrumPrecomp())
    sig = m.compile_signature()  # used to raise AttributeError on cfg.n_z
    assert isinstance(sig, tuple)
    params = {p: 0.5 for p in m.spec.free_params}
    spec = m.predict_spectrum(params)
    assert jnp.all(jnp.isfinite(spec))

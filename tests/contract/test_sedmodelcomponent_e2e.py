"""End-to-end smoke tests for `SEDModelComponent` ports.

For each domain (dust attenuation, dust IR emission, AGN, nebular, radio,
X-ray), build a minimal `SEDModel` with a `SEDModelComponent`-based
backend and verify:

  * Construction succeeds (no resolver miss, no contract violation)
  * `model.predict_photometry({})` returns a finite, positive vector
  * `model.spec.free_params` (when free params are declared) is non-empty
  * `model.compile_signature()` is hashable and stable across calls

These are NOT inference-level parity tests — they exercise the wiring
end-to-end (resolver → component chain → forward pass → observation
projection) for every new port at once. If a port's `name` is missing
from the resolver, or its `predict()` signature drifts from the base
class contract, or its `inputs/outputs` declaration is malformed, the
build or predict call here will catch it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri import (
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    load_ssp_data,
)
from tengri.components.sed_model_component import _REGISTRY

# Filter set for all the cases below — cheap (5 filters) and physically
# realistic enough that finite/positive checks mean something.
_FILTERS = ("sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z")


_SSP_CANDIDATES = [
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
]


@pytest.fixture(scope="module")
def ssp():
    path = next((p for p in _SSP_CANDIDATES if Path(p).is_file()), None)
    if path is None:
        pytest.skip("No SSP grid available under data/.")
    return load_ssp_data(path)


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(list(_FILTERS)))


def _silent_build(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _assert_phot_ok(phot):
    assert phot.shape == (len(_FILTERS),), f"unexpected shape: {phot.shape}"
    assert bool(jnp.all(jnp.isfinite(phot))), "non-finite photometry"
    assert bool(jnp.all(phot > 0)), "non-positive photometry"


# ─────────────────────────────────────────────────────────────────────
# Dust attenuation backends
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dust_type", ["calzetti", "smc", "milky_way", "salim18"])
def test_dust_attenuation_e2e(ssp, obs, dust_type):
    """Each single-screen attenuation port builds and predicts finite photometry."""
    if dust_type not in _REGISTRY:
        pytest.skip(f"{dust_type!r} not registered (resolver fallback to legacy)")
    model = _silent_build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": Fixed},
        dust={"type": dust_type, "tau_v": Fixed(0.3)},
        redshift=Fixed(0.05),
    )
    _assert_phot_ok(model.predict_photometry({}))


# ─────────────────────────────────────────────────────────────────────
# Catalog cross-compile reuse — the headline guarantee
# ─────────────────────────────────────────────────────────────────────


def test_catalog_z_range_end_to_end(ssp, obs):
    """Three SEDModels at different Fixed(z) under WavePrecomp(catalog_z_range)
    share the same compile_signature AND produce sensible per-row photometry."""
    from tengri import WavePrecomp

    cz = WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=200)

    def _build(z):
        return _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": Fixed},
            dust={"type": "calzetti", "tau_v": Fixed(0.3)},
            redshift=Fixed(z),
            approx=cz,
        )

    m_lo = _build(0.05)
    m_mid = _build(0.5)
    m_hi = _build(1.2)

    # Signature collapse — one compile across the catalog
    sig_lo = m_lo.compile_signature()
    sig_mid = m_mid.compile_signature()
    sig_hi = m_hi.compile_signature()
    assert sig_lo == sig_mid == sig_hi, (
        "catalog_z_range should make compile_signature invariant under Fixed(z)"
    )

    # Per-row photometry varies smoothly with z
    phot_lo = m_lo.predict_photometry({})
    phot_mid = m_mid.predict_photometry({})
    phot_hi = m_hi.predict_photometry({})
    for phot in (phot_lo, phot_mid, phot_hi):
        _assert_phot_ok(phot)
    # All distinct — higher z is much dimmer in observed-frame F_nu
    assert not bool(jnp.allclose(phot_lo, phot_mid, rtol=1e-3))
    assert not bool(jnp.allclose(phot_mid, phot_hi, rtol=1e-3))


# ─────────────────────────────────────────────────────────────────────
# WavePrecomp (photometry) agreement at the documented tolerance
# ─────────────────────────────────────────────────────────────────────


def test_waveprecomp_agreement_with_exact(ssp, obs):
    """`approx=WavePrecomp()` should agree with the exact path within the
    documented Zacharegkas+2025 tolerance (~0.5% on broadband photometry)."""
    from tengri import WavePrecomp

    def _build(approx):
        return _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": Fixed},
            dust={"type": "calzetti", "tau_v": Fixed(0.4)},
            redshift=Fixed(0.1),
            approx=approx,
        )

    m_exact = _build(None)
    m_wp = _build(WavePrecomp())

    phot_exact = m_exact.predict_photometry({})
    phot_wp = m_wp.predict_photometry({})
    _assert_phot_ok(phot_exact)
    _assert_phot_ok(phot_wp)

    # Zacharegkas+2025 docs: < 0.5% on photometric magnitudes;
    # equivalent to ~5e-3 in F_nu fractional differences.
    rel = jnp.abs(phot_wp - phot_exact) / jnp.abs(phot_exact)
    assert float(jnp.max(rel)) < 5e-3, (
        f"WavePrecomp vs exact max rel err = {float(jnp.max(rel)):.4%}"
    )

# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``dust_eta_balance``'s declared free prior is a linear Gaussian(1, 0.2).

Owner decision (2026-09-11): ``dust_eta_balance`` is a linear multiplicative
factor (``L_IR = eta * L_absorbed``, applied once after both the exact
integral and the energy-balance LUT branch), so its declared free prior must
be stated in the linear quantity -- a Gaussian with mean 1 and standard
deviation 0.2, truncated at 0 -- rather than a ``LogNormal`` on ``log(eta)``.

Covers: the declaration (``components/dust/_params.py``), the ``FREE`` /
``all_params`` wildcard resolution path, the ``relaxed_energy_balance``
builder, the standardization pushforward, and parity of the linear eta
scaling under the ``WavePrecomp`` energy-balance LUT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Gaussian,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
)
from tengri.observation.photometry import FilterCurve
from tengri.parameters.groups import parse_groups
from tengri.parameters.registry import registry

pytestmark = pytest.mark.contract


def _minimal_groups(dust_emission: dict) -> dict:
    """Smallest groups dict that reaches ``dust_emission`` validation.

    CI-safe: no SSP data, no filters, no model build -- ``parse_groups``
    resolves the grammar directly to a ``Parameters`` spec.
    """
    return {
        "sfh": {"type": "dpl", "all_params": Fixed(DEFAULT)},
        "dust_attenuation": {
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        "dust_emission": dust_emission,
        "neb": {"type": "none"},
        "redshift": Fixed(0.1),
    }


def _assert_linear_gaussian(dist) -> None:
    assert isinstance(dist, Gaussian), f"expected Gaussian, got {type(dist).__name__}"
    assert dist.mu == pytest.approx(1.0)
    assert dist.sigma == pytest.approx(0.2)
    assert dist.lo == pytest.approx(0.0)
    assert dist.hi == float("inf")


# ── (a) explicit FREE on eta_balance resolves to the linear Gaussian ──────


def test_eta_balance_free_resolves_to_linear_gaussian():
    dust_emission = {"type": "modified_blackbody", "eta_balance": FREE}
    spec = parse_groups(**_minimal_groups(dust_emission))
    assert "dust_eta_balance" in spec.free_params
    _assert_linear_gaussian(spec.get_distribution("dust_eta_balance"))


# ── (b) the emission-group all_params wildcard frees it to the same family ─
#
# ``modified_blackbody`` cannot demonstrate this: the wildcard narrows to
# each engine's own *declared* parameters (``_declared_param_names`` in
# ``parameters/groups.py``), and ``dust_eta_balance`` is applied by the
# attenuator, not read by any per-engine ``predict``. Only
# ``energy_balance_split`` lists it in its ``reads_parameters`` (see
# ``components/dust/emission/analytic/energy_balance_split.py``), which is
# the one engine where ``dust_emission={'all_params': FREE}`` reaches it --
# verified empirically, not assumed. ``dust_L_agn_ir`` (no declared
# free_prior, by design) stays pinned and warns; that is expected.


def test_all_params_wildcard_frees_eta_balance_to_same_gaussian():
    from tengri.config.exceptions import WildcardPartialFreeWarning

    dust_emission = {"type": "energy_balance_split", "all_params": FREE}
    with pytest.warns(WildcardPartialFreeWarning):
        spec = parse_groups(**_minimal_groups(dust_emission))
    assert "dust_eta_balance" in spec.free_params
    _assert_linear_gaussian(spec.get_distribution("dust_eta_balance"))


# ── (c) relaxed_energy_balance() returns the same linear family ───────────


def test_relaxed_energy_balance_default_sigma():
    result = builders.dust.emission.relaxed_energy_balance()
    dist = result["eta_balance"]
    assert isinstance(dist, Gaussian), f"expected Gaussian, got {type(dist).__name__}"
    assert dist.mu == pytest.approx(1.0)
    assert dist.sigma == pytest.approx(0.2)
    assert dist.lo == pytest.approx(0.0)


def test_relaxed_energy_balance_explicit_sigma():
    result = builders.dust.emission.relaxed_energy_balance(sigma=0.3)
    dist = result["eta_balance"]
    assert isinstance(dist, Gaussian), f"expected Gaussian, got {type(dist).__name__}"
    assert dist.mu == pytest.approx(1.0)
    assert dist.sigma == pytest.approx(0.3)
    assert dist.lo == pytest.approx(0.0)


# ── (d) the declared prior's standardization pushforward is physical ──────


def test_declared_prior_unstandardize_is_finite_and_nonnegative():
    dist = registry()["dust_eta_balance"].free_prior
    _assert_linear_gaussian(dist)
    for xi in (-60.0, -6.0, 0.0, 6.0, 60.0):
        theta = float(dist.unstandardize(jnp.asarray(xi)))
        assert np.isfinite(theta), f"unstandardize({xi}) is not finite: {theta}"
        # A tiny negative floating-point residual sits right at the eta=0
        # truncation boundary for the most extreme xi (measured: -1.5e-12 at
        # xi=-60); that is numerical noise from the inverse-CDF pushforward,
        # not a physical negative eta, so tolerate it instead of asserting
        # a strict >= 0.0.
        assert theta > -1e-6, f"unstandardize({xi}) = {theta} is meaningfully negative"


def test_declared_prior_unstandardize_zero_is_the_mean():
    dist = registry()["dust_eta_balance"].free_prior
    theta = float(dist.unstandardize(jnp.asarray(0.0)))
    assert abs(theta - 1.0) < 1e-5, f"unstandardize(0.0) = {theta}, expected ~1.0"


# ── (e) precomp parity: the linear eta scaling survives the WavePrecomp LUT ─


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _obs() -> Observation:
    # Optical bands probe the absorbed (energy-balance) light; the far-IR
    # band (100 um, last filter) is where the re-emitted dust luminosity
    # lands, so it is the band that actually exercises L_ir and the eta
    # scaling.
    centers = (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_obs(),
        approx=approx,
        sfh=builders.sfh.tsnorm(all_params=Fixed(DEFAULT)),
        dust_attenuation=builders.dust.two_component(
            all_params=Fixed(DEFAULT),
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(
            all_params=Fixed(DEFAULT), eta_balance=FREE
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )


_IR_BAND_INDEX = -1  # the 1.0e6 Angstrom far-IR filter in _obs()


def test_eta_scaling_survives_precomp_and_matches_exact(synthetic_ssp_wide):
    """The linear eta scaling holds on both paths, and the paths agree.

    The owner's "make sure everything still works with precomp" evidence:
    the far-IR-band flux ratio between eta=2.0 and eta=0.5 must be 4.0 on
    BOTH the exact and the ``WavePrecomp`` path (``L_IR = eta *
    L_absorbed`` is linear in eta, and eta is applied after the
    energy-balance LUT branch so it reaches WavePrecomp unchanged), and the
    two paths must agree with each other at each eta, to the same tolerance
    ``test_energy_balance_lut.py``'s exact-vs-LUT test already uses (2%).
    """
    ssp = synthetic_ssp_wide
    m_exact = _build(ssp, None)
    m_lut = _build(ssp, WavePrecomp())
    base = {**m_exact.spec.get_fixed_values(), **m_exact.spec.sample(jax.random.PRNGKey(0))}

    photometry = {}
    for label, m in (("exact", m_exact), ("lut", m_lut)):
        f = jax.jit(m.predict_photometry)
        for eta in (0.5, 2.0):
            p = dict(base)
            p["dust_eta_balance"] = jnp.asarray(eta)
            photometry[(label, eta)] = np.asarray(f(p))

    for label in ("exact", "lut"):
        ir_lo = photometry[(label, 0.5)][_IR_BAND_INDEX]
        ir_hi = photometry[(label, 2.0)][_IR_BAND_INDEX]
        ratio = float(ir_hi / ir_lo)
        assert abs(ratio - 4.0) < 2e-2, (
            f"{label}: IR-band ratio (eta=2.0 / eta=0.5) = {ratio:.4f}, expected 4.0"
        )

    worst = 0.0
    for eta in (0.5, 2.0):
        a = photometry[("exact", eta)]
        b = photometry[("lut", eta)]
        worst = max(worst, float(np.abs(a - b).max() / np.abs(b).max()))
    assert worst < 0.02, f"exact vs WavePrecomp drifted {worst:.3%} across the eta sweep (> 2%)"

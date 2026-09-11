# SPDX-License-Identifier: BSD-3-Clause
"""``narayanan_prior``/``narayanan_tau_prior`` returned priors their own
parameters cannot carry (#2226).

https://github.com/suchethac/tengri/issues/2226

``dust_bump_strength`` and ``dust_tau_diff`` both declare a ``bound_check``
requiring ``lo >= 0`` (``ATTENUATION_PARAMS`` in
``components/dust/_params.py``). ``narayanan_prior(z)`` and
``narayanan_tau_prior(z, log_mstar)`` centered unbounded ``Gaussian(mu, sigma)``
distributions on them, so ``Parameters(..., **narayanan_prior(z=2.0))`` --
literally the docstring's own Example -- raised::

    ValueError: Parameter 'dust_bump_strength' (...): bounds (-inf, inf)
    violate physical constraint: must be >= 0

``dust_delta`` declares no ``bound_check`` (its MUFASA-fitted means run
-0.556 to +0.138, genuinely straddling zero) and must stay unbounded.

A second, independent defect made the obvious fix (``Gaussian(mu, sigma,
lo=0.0)``) inert on its own: :class:`~tengri.parameters.priors.Gaussian`
decided whether it was truncated from ``self._cdf_lo > 0.0``, and for these
means (2-3.6) and sigma=0.3 the ``lo=0`` bound sits 6.6-12 sigma away, where
``erf`` underflows to exactly 0.0 in float64. ``_truncated`` then read
``False`` and ``unstandardize``/``sample`` fell back to the untruncated affine
map, which can and does return negative values. The fix reads the bound
directly (``self._lo > -inf``), a structural test immune to the underflow.

Finally, ``dust_bump_strength``'s declared ``free_prior`` was
``Uniform(0.0, 2.0)``, which cannot reach the fitted table's own values (up to
3.634 at z=4) -- so even a *bounded* ``narayanan_prior`` disagreed with the
range a caller freeing the same parameter via ``FREE`` would get. The ceiling
is now ``Uniform(0.0, 4.0)``, ~10% above the highest node.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
import scipy.integrate

from tengri import DEFAULT, Fixed, Parameters, parse_groups
from tengri.components.dust.attenuation import _NARAYANAN_BUMP_STRENGTH
from tengri.components.dust.priors import narayanan_prior, narayanan_tau_prior
from tengri.parameters.groups import _expand_free

pytestmark = pytest.mark.regression_bug

_REDSHIFTS = (0.0, 2.0, 6.0)


# ── Flat form: Parameters(..., **narayanan_prior(z)) must build ──────────


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_narayanan_prior_flat_form_builds(z):
    """The docstring's own Example: this raised ValueError before the fix."""
    spec = Parameters(
        mean_sfh_type="dpl",
        dust_model="single_component",
        dust_law_bc="kriek_conroy",
        redshift=z,
        **narayanan_prior(z),
    )
    assert not spec.get_distribution("dust_delta").is_fixed
    assert not spec.get_distribution("dust_bump_strength").is_fixed


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_narayanan_prior_bump_strength_bounds_are_zero_to_inf(z):
    spec = Parameters(
        mean_sfh_type="dpl",
        dust_model="single_component",
        dust_law_bc="kriek_conroy",
        redshift=z,
        **narayanan_prior(z),
    )
    assert spec.get_distribution("dust_bump_strength").bounds == (0.0, math.inf)


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_narayanan_prior_dust_delta_stays_unbounded(z):
    """``dust_delta`` has no ``bound_check`` and its fitted means straddle
    zero (-0.556 to +0.138) -- it must NOT gain a lower bound."""
    spec = Parameters(
        mean_sfh_type="dpl",
        dust_model="single_component",
        dust_law_bc="kriek_conroy",
        redshift=z,
        **narayanan_prior(z),
    )
    assert spec.get_distribution("dust_delta").bounds == (-math.inf, math.inf)


# ── Group form: parse_groups must build and free both ────────────────────


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_narayanan_prior_group_form_builds_and_frees_both(z):
    spec = parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "kriek_conroy",
            "tau_v": 1.0,
            **narayanan_prior(z),
        },
        redshift=Fixed(z),
    )
    assert "dust_delta" in spec.free_params
    assert "dust_bump_strength" in spec.free_params
    assert spec.get_distribution("dust_bump_strength").bounds == (0.0, math.inf)


# ── narayanan_tau_prior: dust_tau_diff must build, free, and bound ───────


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_narayanan_tau_prior_flat_form_builds(z):
    spec = Parameters(mean_sfh_type="dpl", redshift=z, **narayanan_tau_prior(z))
    dist = spec.get_distribution("dust_tau_diff")
    assert not dist.is_fixed
    assert dist.bounds == (0.0, math.inf)


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_narayanan_tau_prior_group_form_builds(z):
    """Two-component group: ``dust_tau_bc`` stays Fixed at its declared
    default, only ``dust_tau_diff`` is freed by the helper.

    Trap: the completeness guard for the two-component ``tau_bc``/``tau_diff``
    pair recognizes only the short stems (``tau_bc``, ``tau_diff``), so this
    must pass the helper's own full-name key (``dust_tau_diff=``), not mix it
    with a short-stem ``tau_bc=``.
    """
    spec = parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            **narayanan_tau_prior(z),
        },
        redshift=Fixed(z),
    )
    assert "dust_tau_diff" in spec.free_params
    assert spec.get_distribution("dust_tau_diff").bounds == (0.0, math.inf)
    tau_bc = spec.get_distribution("dust_tau_bc")
    assert tau_bc.is_fixed
    assert float(tau_bc.bounds[0]) == pytest.approx(1.0)  # declared default


# ── log_prob(-eps) == -inf: mutation kill for dropping lo=0.0 ────────────


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_bump_log_prob_is_neg_inf_just_below_zero(z):
    dist = narayanan_prior(z)["dust_bump_strength"]
    assert dist.log_prob(jnp.asarray(-1e-9)) == -jnp.inf


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_tau_diff_log_prob_is_neg_inf_just_below_zero(z):
    dist = narayanan_tau_prior(z)["dust_tau_diff"]
    assert dist.log_prob(jnp.asarray(-1e-9)) == -jnp.inf


# ── unstandardize deep tail: mutation kill for the _truncated defect ─────


def test_bump_unstandardize_deep_negative_tail_stays_nonnegative():
    """xi=-15 is a ~10^-51 tail event under N(0,1): before the ``_truncated``
    fix, ``_cdf_lo`` underflowed to exactly 0.0, the bump Gaussian's
    ``_truncated`` flag read False, and ``unstandardize`` fell back to the
    plain affine map ``mu + sigma*xi`` -- for z=2 (mu~1.98, sigma=0.3) that is
    a negative "bump strength", -2.5 sigma below the parameter's own zero
    bound."""
    dist = narayanan_prior(z=2.0)["dust_bump_strength"]
    assert float(dist.unstandardize(jnp.asarray(-15.0))) >= 0.0


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_bump_log_prob_grad_at_mu_is_finite(z):
    dist = narayanan_prior(z)["dust_bump_strength"]
    grad = jax.grad(lambda x: dist.log_prob(x))(jnp.asarray(dist.mu))
    assert jnp.isfinite(grad)


# ── 10k-sample negativity check: dust_tau_diff only ───────────────────────
#
# The bump Gaussian sits 6.6-12 sigma from its zero bound (mutation-blind: a
# 10k-sample draw will not visit that tail). ``dust_tau_diff``'s mean is
# frequently within ~1-2 sigma of zero (low-mass, low-z galaxies especially),
# so a truncation defect shows up directly in a finite sample -- this is the
# sample-based check that is actually sensitive to the fix.
@pytest.mark.parametrize("z", (0.0, 1.0, 3.0))
@pytest.mark.parametrize("log_mstar", (8.0, 9.0, 10.0, 11.0))
def test_tau_diff_10k_samples_never_negative(z, log_mstar):
    dist = narayanan_tau_prior(z, log_mstar)["dust_tau_diff"]
    keys = jax.random.split(jax.random.PRNGKey(2226), 10_000)
    samples = jax.vmap(dist.sample)(keys)
    assert float(jnp.min(samples)) >= 0.0


# ── Normalization: truncated density integrates to 1 ──────────────────────


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_bump_gaussian_normalized(z):
    dist = narayanan_prior(z)["dust_bump_strength"]
    mass, _ = scipy.integrate.quad(lambda x: float(jnp.exp(dist.log_prob(x))), 0.0, 60.0)
    assert mass == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("z", _REDSHIFTS)
def test_tau_diff_gaussian_normalized(z):
    dist = narayanan_tau_prior(z)["dust_tau_diff"]
    mass, _ = scipy.integrate.quad(lambda x: float(jnp.exp(dist.log_prob(x))), 0.0, 60.0)
    assert mass == pytest.approx(1.0, rel=1e-6)


# ── Ceiling: FREE must reach every fitted node ────────────────────────────


def test_free_bump_strength_ceiling_covers_every_fitted_node():
    free_prior = _expand_free("dust_bump_strength", Fixed(0.0))
    assert free_prior.bounds[1] >= float(max(_NARAYANAN_BUMP_STRENGTH))

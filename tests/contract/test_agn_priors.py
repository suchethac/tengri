# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the AGN informative-prior public surface.

Surface protected: the eight AGN prior functions in
``tengri.parameters.agn_priors``, reachable as ``tengri.agn.priors``
(``tengri.components.agn`` re-exports the module, see that package's
``__init__.py``); JIT-compilability and gradient-smoothness of each; the
documented error behavior of ``prior_energy_balance``'s ``mode`` argument.

Exact numeric values against the upstream formulas they implement are
NOT this file's job -- that comparison (and the previous version of this
file's failure to do it, comparing tengri's own formula to itself) lives in
``tests/crossval/test_agn_priors_vs_agnfitter.py``. Any numeric expectation
here is either an exact statement from upstream's OWN definition (e.g. "the
flexible-mode floor is exactly zero once emission >= absorption" -- a fact
about ``PRIORS_AGNfitter.py``, not a copy of tengri's implementation) or
independently derived (``scipy.stats.norm.logpdf``), never a re-typed copy of
the formula under test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from scipy import stats

pytestmark = pytest.mark.contract

from tengri.parameters.agn_priors import (
    AGNFITTER_HARD_REJECT,
    AGNFITTER_PRIOR_DEFAULTS,
    agnfitter_priors,
    gaussian_log_prior,
    prior_agn_fraction,
    prior_energy_balance,
    prior_ir_syn_fraction,
    prior_ir_xrays,
    prior_low_agn_fraction,
    prior_midir_uv,
    prior_stellar_mass,
    prior_uv_xrays,
)

_ALL_PRIOR_NAMES = (
    "gaussian_log_prior",
    "prior_energy_balance",
    "prior_stellar_mass",
    "prior_agn_fraction",
    "prior_low_agn_fraction",
    "prior_ir_syn_fraction",
    "prior_uv_xrays",
    "prior_ir_xrays",
    "prior_midir_uv",
    "agnfitter_priors",
    "AGNFITTER_PRIOR_DEFAULTS",
)


class TestPublicSurface:
    """The module is reachable from the documented public locations."""

    def test_importable_from_parameters_module(self):
        import tengri.parameters.agn_priors as mod

        for name in _ALL_PRIOR_NAMES:
            assert hasattr(mod, name), f"{name} missing from tengri.parameters.agn_priors"

    def test_reachable_as_tengri_agn_priors(self):
        """``tengri.agn.priors`` (the D7-audit-required public namespace)."""
        import tengri

        assert hasattr(tengri.agn, "priors"), "tengri.agn.priors is not exposed"
        for name in _ALL_PRIOR_NAMES:
            assert hasattr(tengri.agn.priors, name)

    def test_agnfitter_priors_defaults_pinned_values(self):
        """``AGNFITTER_PRIOR_DEFAULTS`` is AGNfitter-rX's ``modelsettings``
        default (only energy balance + AGN fraction enabled). Task 11 item 9:
        this constant was missing from ``docs/api/models.rst`` and this test
        file even though ``agnfitter_priors`` already reads every one of its
        own keyword defaults off it -- pinned here against independently
        hardcoded values (not read back off the dict itself) so a value
        drift in the module is caught, not just a key-set drift."""
        assert AGNFITTER_PRIOR_DEFAULTS == {
            "energy_balance": True,
            "energy_balance_mode": "flexible",
            "stellar_mass": False,
            "agn_fraction": True,
            "low_agn_fraction": False,
            "midir_uv": False,
            "uv_xrays": False,
            "ir_xrays": False,
            "ir_syn_fraction": False,
        }

    def test_agnfitter_priors_defaults_are_the_single_source(self):
        """``agnfitter_priors``'s ``enable_*``/``*_mode`` keyword defaults
        must equal ``AGNFITTER_PRIOR_DEFAULTS`` -- the adapter reads every
        one directly off the dict at definition time, so this also guards
        the key-set correspondence itself (a renamed/added/removed key on
        either side that the other missed)."""
        import inspect

        params = inspect.signature(agnfitter_priors).parameters
        mapping = {
            "energy_balance_mode": "energy_balance_mode",
            "enable_energy_balance": "energy_balance",
            "enable_stellar_mass": "stellar_mass",
            "enable_agn_fraction": "agn_fraction",
            "enable_low_agn_fraction": "low_agn_fraction",
            "enable_midir_uv": "midir_uv",
            "enable_uv_xrays": "uv_xrays",
            "enable_ir_xrays": "ir_xrays",
            "enable_ir_syn_fraction": "ir_syn_fraction",
        }
        assert set(mapping.values()) == set(AGNFITTER_PRIOR_DEFAULTS), (
            "AGNFITTER_PRIOR_DEFAULTS keys and agnfitter_priors' enable_* "
            "parameters have drifted apart"
        )
        for param_name, key in mapping.items():
            assert params[param_name].default == AGNFITTER_PRIOR_DEFAULTS[key], (
                f"agnfitter_priors({param_name}=...) default disagrees with "
                f"AGNFITTER_PRIOR_DEFAULTS[{key!r}]"
            )

    def test_hard_reject_is_a_finite_sentinel_not_inf(self):
        """Upstream's own comment (`PRIORS_AGNfitter.py:98`) marks this choice:
        ``return -9999 #-np.inf`` -- a finite value, not -inf (grad-safety)."""
        assert AGNFITTER_HARD_REJECT == -9999.0
        assert jnp.isfinite(AGNFITTER_HARD_REJECT)

    def test_three_import_spellings_resolve_to_the_same_object(self):
        """Fix round 2: ``from tengri.agn.priors import agnfitter_priors`` used
        to raise ``ModuleNotFoundError`` -- ``tengri.agn`` was aliased in
        ``sys.modules`` but ``tengri.agn.priors`` was not, so attribute access
        and the two-level ``from tengri.agn import priors`` both worked
        (the latter falls back to attribute lookup on the already-imported
        package) while the three-level form, which needs ``tengri.agn.priors``
        to resolve as a genuine module first, did not. All three spellings
        must now import the identical function object."""
        import tengri
        from tengri.agn import priors as priors_from_package
        from tengri.agn.priors import agnfitter_priors as agnfitter_priors_direct

        assert tengri.agn.priors.agnfitter_priors is agnfitter_priors_direct
        assert priors_from_package.agnfitter_priors is agnfitter_priors_direct
        assert agnfitter_priors_direct is agnfitter_priors  # this file's own import
        # And the canonical, non-aliased import path agrees too.
        import tengri.parameters.agn_priors as canonical

        assert canonical.agnfitter_priors is agnfitter_priors_direct


class TestGaussianLogPrior:
    """Frozen: includes the normalization constant (unlike a bare -0.5*z^2)."""

    @pytest.mark.parametrize(
        "mu, sigma, par", [(0.0, 0.1, 0.0), (0.0, 0.1, 0.05), (2.0, 2.0, -1.0), (-2.0, 0.5, -2.3)]
    )
    def test_matches_scipy_norm_logpdf(self, mu, sigma, par):
        expected = stats.norm.logpdf(par, loc=mu, scale=sigma)
        got = float(gaussian_log_prior(mu, sigma, par))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_jit_compiles(self):
        f = jax.jit(gaussian_log_prior)
        assert jnp.isfinite(f(0.0, 0.1, 0.05))


class TestEnergyBalance:
    """Frozen branch structure (`PRIORS_AGNfitter.py:78-106`):

    hard reject whenever emission < absorption in BOTH modes; flexible mode
    is otherwise an exact flat 0 (no Gaussian at all); restrictive mode
    otherwise applies a sigma=0.1 Gaussian about equality.
    """

    def test_flexible_mode_is_exactly_zero_once_physical(self):
        """Upstream's flexible branch (`:100`) is a bare ``return 0``, not a
        Gaussian -- any excess emission is exactly zero penalty."""
        lp_at_equality = float(prior_energy_balance(1.0e44, 1.0e44, mode="flexible"))
        lp_with_excess = float(prior_energy_balance(1.0e44, 5.0e44, mode="flexible"))
        assert lp_at_equality == 0.0
        assert lp_with_excess == 0.0

    def test_hard_reject_fires_in_both_modes(self):
        for mode in ("flexible", "restrictive"):
            lp = float(prior_energy_balance(2.0e44, 1.0e44, mode=mode))
            assert lp == AGNFITTER_HARD_REJECT

    def test_restrictive_mode_matches_scipy_at_offset(self):
        ratio = jnp.log10(2.0)  # l_sb_emit = 2 * l_gal_att
        expected = stats.norm.logpdf(float(ratio), loc=0.0, scale=0.1)
        got = float(prior_energy_balance(1.0e44, 2.0e44, mode="restrictive"))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be"):
            prior_energy_balance(1.0e44, 2.0e44, mode="invalid")

    def test_grad_finite_and_nontrivial(self):
        def f(l_sb_emit):
            return prior_energy_balance(1.0e44, l_sb_emit, mode="restrictive")

        grad_val = jax.grad(f)(2.0e44)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0

    def test_jit_compiles(self):
        def f(l_gal_att, l_sb_emit):
            return prior_energy_balance(l_gal_att, l_sb_emit, mode="restrictive")

        got = jax.jit(f)(1.0e44, 2.0e44)
        assert jnp.isfinite(got)


class TestStellarMass:
    def test_matches_scipy_norm_logpdf(self):
        expected = stats.norm.logpdf(6.0, loc=4.5, scale=1.5)
        got = float(prior_stellar_mass(6.0))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_grad_finite(self):
        grad_val = jax.grad(prior_stellar_mass)(6.0)
        assert jnp.isfinite(grad_val)


class TestAGNFraction:
    """Frozen: rest-1500A flux ratio with a redshift-dependent regime split
    (`PRIORS_AGNfitter.py:109-199`), NOT a bolometric floor."""

    def test_hard_reject_in_bright_regime_when_agn_fainter_than_galaxy(self):
        # Very bright 1500A data (large data_flux_1500, dlum=1 -> abs_mag_data
        # ~ -31, well below characteristic_mag(z=2)-1 ~ -21.3) -> QSO/bright
        # regime; bbb_flux < gal_flux -> AGNfrac1500 < 0 -> hard reject.
        lp = float(prior_agn_fraction(1.0e-27, 1.0e-25, 1.0e32, dlum=1.0, redshift=2.0))
        assert lp == AGNFITTER_HARD_REJECT

    def test_grad_finite_in_galaxy_regime(self):
        def f(bbb_flux_1500):
            return prior_agn_fraction(bbb_flux_1500, 1.0e-25, 1.0e-30, dlum=1.0, redshift=2.0)

        grad_val = jax.grad(f)(1.0e-25)
        assert jnp.isfinite(grad_val)

    def test_jit_compiles(self):
        got = jax.jit(prior_agn_fraction)(1.0e-25, 1.0e-25, 1.0e-30, 1.0, 2.0)
        assert jnp.isfinite(got)


class TestLowAGNFraction:
    """Frozen: distinct from :func:`prior_agn_fraction` -- same mean (-2) in
    both regimes, no hard-reject branch, -3 mag threshold offset."""

    def test_never_hard_rejects(self):
        # Even a strongly AGN-dominated (bright) configuration must NOT reject
        # -- prior_low_AGNfraction has no reject branch at all upstream.
        lp = float(prior_low_agn_fraction(1.0e-20, 1.0e-25, 1.0e10, dlum=1.0, redshift=2.0))
        assert jnp.isfinite(lp)
        assert lp != AGNFITTER_HARD_REJECT

    def test_grad_finite(self):
        def f(bbb_flux_1500):
            return prior_low_agn_fraction(bbb_flux_1500, 1.0e-25, 1.0e-30, dlum=1.0, redshift=2.0)

        grad_val = jax.grad(f)(1.0e-25)
        assert jnp.isfinite(grad_val)


class TestIRSynFraction:
    def test_grad_finite(self):
        def f(syn_flux_ir):
            return prior_ir_syn_fraction(1.0e-27, 9.0, 1.0e-30, 12.5, 1.0e-25, syn_flux_ir)

        grad_val = jax.grad(f)(1.0e-25)
        assert jnp.isfinite(grad_val)


class TestUVXrays:
    def test_matches_scipy_at_peak(self):
        beta, gamma = 0.643, 6.8734
        log_l2kev = 28.0
        log_l2500a = (log_l2kev - gamma) / beta  # exact alpha_ox relation -> ratio 0
        expected = stats.norm.logpdf(0.0, loc=0.0, scale=0.4)
        got = float(prior_uv_xrays(log_l2500a, log_l2kev))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_grad_finite(self):
        grad_val = jax.grad(lambda x: prior_uv_xrays(x, 28.0))(30.0)
        assert jnp.isfinite(grad_val)


class TestIRXrays:
    def test_matches_scipy_at_peak(self):
        nulnu_6um = 1.0e41  # x=0
        model = 22.9494264
        expected = stats.norm.logpdf(0.0, loc=0.0, scale=0.5)
        got = float(prior_ir_xrays(model, nulnu_6um))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_grad_finite(self):
        grad_val = jax.grad(lambda x: prior_ir_xrays(x, 1.0e41))(23.0)
        assert jnp.isfinite(grad_val)


class TestMidIRUV:
    """Frozen: x = log10(nu*L_nu(6um)/1e41) -- the SAME x formula as
    prior_ir_xrays (review round 1 fix: a previous version subtracted
    upstream's L_nu-calibrated 27.30103 directly from log10(nulnu_6um)
    without the 13.69897 frequency correction, off by log10(nu_6um) in x)."""

    def test_matches_scipy_at_peak(self):
        nulnu_6um = 1.0e44
        x = jnp.log10(nulnu_6um / 1e41)
        model = float((16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643)
        expected = stats.norm.logpdf(0.0, loc=0.0, scale=0.6)
        got = float(prior_midir_uv(model, nulnu_6um))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_shares_x_with_prior_ir_xrays(self):
        """Both priors call the same private ``_x_from_nulnu_6um`` helper."""
        from tengri.parameters.agn_priors import _x_from_nulnu_6um

        nulnu_6um = 3.0e42
        x = float(_x_from_nulnu_6um(nulnu_6um))
        model_ir_xrays = 22.9494264 + 1.024 * x - 0.047 * x**2
        model_midir_uv = (16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643
        assert float(prior_ir_xrays(model_ir_xrays, nulnu_6um)) == pytest.approx(
            stats.norm.logpdf(0.0, loc=0.0, scale=0.5), abs=1e-9
        )
        assert float(prior_midir_uv(model_midir_uv, nulnu_6um)) == pytest.approx(
            stats.norm.logpdf(0.0, loc=0.0, scale=0.6), abs=1e-9
        )

    def test_grad_finite(self):
        grad_val = jax.grad(lambda x: prior_midir_uv(x, 1.0e44))(45.0)
        assert jnp.isfinite(grad_val)
        assert jnp.any(grad_val != 0.0), (
            "`grad_val` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_disagrees_with_direct_luminosity_comparison(self):
        """D7(c) regression: equal L_mir/L_uv is NOT this prior's peak."""
        lp_equal_luminosities = float(prior_midir_uv(45.0, 10**45.0))
        x = jnp.log10(10.0**45.0 / 1e41)
        true_peak_bbmodel = float((16.2530786 + 1.024 * x - 0.047 * x**2) / 0.643)
        lp_at_true_peak = float(prior_midir_uv(true_peak_bbmodel, 10**45.0))
        assert lp_at_true_peak > lp_equal_luminosities + 50.0

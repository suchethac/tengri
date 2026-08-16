# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #226: 4D α-enhanced SSP path through StellarSEDComponent.

Before the fix, ``StellarSEDComponent.apply`` raised ``NotImplementedError``
when the SSP grid carried a native ``[α/Fe]`` axis (4D shape
``(n_met, n_alpha, n_age, n_wave)``). Users with MIST / Vazdekis
α-enhanced templates couldn't fit at all.

The fix collapses the α axis once via ``interpolate_alpha_only`` and
threads the resulting 3D ssp_flux through the standard DSPS lognormal-MDF
kernel — so the 4D and 3D paths share the same Z bookkeeping.

These tests use a synthetic 4D fixture (the project ships only 3D SSPs)
to pin three invariants:

1. The 4D path no longer raises and produces a finite SED.
2. Setting ``met_alpha_fe`` to a value at one of the grid nodes yields
   the same SED as a 3D run on the corresponding α slice (linear
   interpolation hits grid nodes exactly).
3. Sweeping ``met_alpha_fe`` across the grid moves the SED, i.e. the
   wiring is responsive — the bug was "silently identical regardless
   of α".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platforms", "cpu")

from tengri.components.stellar.sps.dsps_wrapper import SSPData, has_alpha_grid


@pytest.fixture(scope="module")
def synthetic_4d_ssp():
    """Hand-rolled 4D SSP grid.

    Shape ``(n_met=3, n_alpha=3, n_age=8, n_wave=120)`` — small enough for
    a fast test, large enough that the DSPS kernel does real work.

    Each (Z, α) slice is the 3D base (a smooth function of (age, wave))
    multiplied by an α-dependent factor ``1 + 0.4 * α`` so sweeping α
    moves the SED in a deterministic way.
    """
    rng = np.random.default_rng(123)
    n_met, n_alpha, n_age, n_wave = 3, 3, 8, 120
    lgmet = jnp.linspace(-2.0, 0.2, n_met)
    alpha_fe = jnp.array([-0.2, 0.0, 0.4])  # includes α=0 as an exact grid node
    # log10(age/Gyr) from 1 Myr to 10 Gyr
    lg_age_gyr = jnp.linspace(-3.0, 1.0, n_age)
    wave = jnp.linspace(1000.0, 30000.0, n_wave)
    # 3D base: separable in (age, wave); add a small Z-dependent shift
    age_kernel = jnp.exp(-((lg_age_gyr + 2.0) ** 2) / 2.0)
    wave_kernel = jnp.exp(-((jnp.log10(wave) - 3.5) ** 2) / 0.5)
    base = jnp.einsum("a,w->aw", age_kernel, wave_kernel)
    # (n_met, n_age, n_wave), gentle Z-dependence
    base_met = base[None, :, :] * (1.0 + 0.05 * lgmet[:, None, None])
    # (n_met, n_alpha, n_age, n_wave), α-dependent multiplicative shift
    alpha_factor = 1.0 + 0.4 * alpha_fe
    ssp_flux = base_met[:, None, :, :] * alpha_factor[None, :, None, None]
    return SSPData(
        ssp_wave=wave,
        ssp_flux=ssp_flux,
        ssp_lg_age_gyr=lg_age_gyr,
        ssp_lgmet=lgmet,
        ssp_alpha_fe=alpha_fe,
        ssp_mass_remaining=None,
    )


@pytest.fixture(scope="module")
def synthetic_3d_at_alpha_zero(synthetic_4d_ssp):
    """3D slice of the same grid at the α=0 node, for the 4D-vs-3D invariant."""
    ssp_4d = synthetic_4d_ssp
    # α axis is centered at 0 by construction
    ia0 = int(jnp.argmin(jnp.abs(ssp_4d.ssp_alpha_fe)).item())
    return SSPData(
        ssp_wave=ssp_4d.ssp_wave,
        ssp_flux=ssp_4d.ssp_flux[:, ia0, :, :],
        ssp_lg_age_gyr=ssp_4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp_4d.ssp_lgmet,
        ssp_mass_remaining=None,
    )


def _rest_sed(ssp_data, alpha_fe_value: float):
    """Run the orchestrator chain on a const-SFH model and return rest-frame SED."""
    from tengri import Fixed, Parameters, SEDModel

    spec = Parameters(
        met_mode="delta",
        met_alpha_fe=Fixed(alpha_fe_value),
        mean_sfh_type="const",
        sfh_const_log_total_mass=Fixed(0.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_diff=Fixed(0.0),
        dust_tau_bc=Fixed(0.0),
        redshift=Fixed(0.05),
    )
    model = SEDModel(spec, ssp_data)
    p = dict(spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict_rest_sed(p).sed)


@pytest.mark.regression_bug
def test_4d_alpha_path_runs_without_notimplementederror(synthetic_4d_ssp):
    """#226: SSP with ssp_alpha_fe set + met_mode='delta' used to raise."""
    assert has_alpha_grid(synthetic_4d_ssp), "fixture is supposed to be 4D"
    sed = _rest_sed(synthetic_4d_ssp, alpha_fe_value=0.0)
    assert sed.shape == synthetic_4d_ssp.ssp_wave.shape
    assert np.all(np.isfinite(sed)), "SED must be finite"
    assert sed.sum() > 0, "SED must be non-degenerate"


@pytest.mark.regression_bug
def test_4d_at_grid_node_matches_3d_slice(synthetic_4d_ssp, synthetic_3d_at_alpha_zero):
    """At an α grid node, the 4D path must reproduce the corresponding 3D slice.

    Linear interpolation has zero error at grid nodes, so this is a
    bit-exact invariant up to floating-point rounding.
    """
    # α=0 is a grid node by construction
    sed_4d = _rest_sed(synthetic_4d_ssp, alpha_fe_value=0.0)
    sed_3d = _rest_sed(synthetic_3d_at_alpha_zero, alpha_fe_value=0.0)
    # 4D path ignores effective_metallicity at α=0 (logically equivalent
    # to 3D run); 3D path multiplies by effective_metallicity(log_z, 0).
    # effective_metallicity(log_z, 0) == log_z by construction (see
    # dsps_wrapper.effective_metallicity), so both branches feed the
    # same gal_lgmet to DSPS. Difference is dominated by float64
    # round-off in the linear interpolation.
    np.testing.assert_allclose(sed_4d, sed_3d, rtol=1e-10, atol=0.0)


@pytest.mark.regression_bug
def test_4d_alpha_fe_sweeps_change_sed(synthetic_4d_ssp):
    """Sweeping α across the grid must move the SED (the bug was zero motion)."""
    seds = {
        afe: _rest_sed(synthetic_4d_ssp, alpha_fe_value=float(afe))
        for afe in (-0.2, 0.0, 0.2, 0.4)
    }
    # Per-α SED sums across the wave grid, normalized to α=0
    sums = {a: float(s.sum()) for a, s in seds.items()}
    rel_spread = (max(sums.values()) - min(sums.values())) / sums[0.0]
    assert rel_spread > 0.05, (
        f"Sweeping met_alpha_fe across grid only moved SED sum by {rel_spread:.4f} "
        f"(values={sums}); the fixture builds in a 0.4×α multiplicative factor, "
        f"so a >5% spread is the floor."
    )

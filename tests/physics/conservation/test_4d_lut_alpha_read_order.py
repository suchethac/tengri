# SPDX-License-Identifier: BSD-3-Clause
"""The precomputed photometry LUT reads a 4-D [alpha/Fe] library per SSP (#3434).

The undeclared-scatter forward reads every SSP column by monotone PCHIP across Z, at
each [alpha/Fe] node, then mixes the alpha nodes linearly and contracts the age
marginal (owner ruling 2026-11-15, "PCHIP each SSP"). A LUT route that collapses alpha
first and reads Z second is a different number whenever the alpha nodes are not a
constant factor apart.

The check is against the LUT's own table, so the LUT's approximation error (about 4e-5
on this grid, from its filter discretization, and the same on the control) cancels:
the derived ``stellar_phot_lnu_precomp`` must be proportional, with one constant over
filters, to the per-SSP reference built from that table.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import SpectrumPrecomp, WavePrecomp
from tengri.observation import Spectroscopy

pytestmark = pytest.mark.conservation

ALPHA_NODES = jnp.array([-0.2, 0.0, 0.2, 0.4])
#: Ratio spread over filters, float64 round-off.
ROUND_OFF = 1e-10


def _grid(ssp, factor):
    flux4 = ssp.ssp_flux[:, None, :, :] * factor[:, :, None, None]
    return ssp._replace(ssp_flux=flux4, ssp_alpha_fe=ALPHA_NODES)


@pytest.fixture(scope="module")
def ssp_multiplicative():
    """Control: a constant-factor alpha grid, where alpha-first and Z-first agree."""
    ssp = tengri.load_ssp()
    factor = jnp.broadcast_to((1.0 + 0.4 * ALPHA_NODES)[None, :], (ssp.ssp_flux.shape[0], 4))
    return _grid(ssp, factor)


@pytest.fixture(scope="module")
def ssp_non_multiplicative():
    """A grid whose alpha dependence varies with Z, so the two read orders differ."""
    ssp = tengri.load_ssp()
    lgz = jnp.asarray(ssp.ssp_lgmet)
    factor = 1.0 + 0.4 * ALPHA_NODES[None, :] * (1.0 + 0.2 * (lgz[:, None] + 2.0) ** 2)
    return _grid(ssp, factor)


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["galex_fuv", "sdss_u", "sdss_r", "2mass_h"])
    )


def _build_lut(ssp4, obs):
    return SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_frac": Fixed(0.6),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
            "age_kernel": "cic",
        },
        met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        approx=WavePrecomp(),
    )


def _per_ssp_and_alpha_first(lut_table, lgmet, met_z, marginal, alpha_fe):
    """Per-SSP reference and the alpha-first order, both as ``(f,)`` vectors.

    ``lut_table`` is ``(n_met, n_alpha, n_age, n_filt)``. Per-SSP: PCHIP each
    (alpha, age, filter) column across Z, then mix alpha, then contract age.
    Alpha-first: mix alpha on the stored table, then PCHIP across Z.
    """
    from tengri.components.stellar.sps.dsps_wrapper import alpha_bracket

    ia, fa = alpha_bracket(ALPHA_NODES, jnp.asarray(alpha_fe))
    ia, fa = int(ia), float(fa)
    x = np.asarray(lgmet, dtype=float)
    table = np.asarray(lut_table, dtype=float)
    per_node = PchipInterpolator(x, table, axis=0, extrapolate=False)(float(met_z))
    per_ssp = np.einsum("a,af->f", marginal, (1 - fa) * per_node[ia] + fa * per_node[ia + 1])
    mixed = (1 - fa) * table[:, ia] + fa * table[:, ia + 1]
    alpha_first_table = PchipInterpolator(x, mixed, axis=0, extrapolate=False)(float(met_z))
    alpha_first = np.einsum("a,af->f", marginal, alpha_first_table)
    return per_ssp, alpha_first


@pytest.mark.parametrize("grid", ["control_multiplicative", "non_multiplicative"])
def test_photometry_lut_reads_each_ssp_before_alpha(request, obs, grid):
    ssp4 = request.getfixturevalue(
        "ssp_multiplicative" if grid == "control_multiplicative" else "ssp_non_multiplicative"
    )
    model = _build_lut(ssp4, obs)
    derived = model.predict_state({}).derived
    stellar = next(
        c for c in model._build_component_chain() if type(c).__name__.startswith("Stellar")
    )
    lut = stellar._state.ssp_phot_lut
    _, _, _, met_joint, met_z, lgmet = stellar.compute_joint_weights_exact(
        {
            k: jnp.asarray(float(model.spec.get_distribution(k).default))
            for k in model.spec.all_params
            if getattr(model.spec.get_distribution(k), "default", None) is not None
        },
        ssp_data=ssp4,
    )
    marginal = np.asarray(met_joint)[0]
    per_ssp, alpha_first = _per_ssp_and_alpha_first(lut.ssp_phot, lgmet, met_z, marginal, 0.3)
    lut_out = np.asarray(derived["stellar_phot_lnu_precomp"])
    scale_ssp = lut_out / per_ssp
    scale_alpha_first = lut_out / alpha_first
    spread_ssp = scale_ssp.max() / scale_ssp.min() - 1.0
    spread_alpha_first = scale_alpha_first.max() / scale_alpha_first.min() - 1.0
    if grid == "control_multiplicative":
        assert spread_ssp <= ROUND_OFF, f"control spread {spread_ssp:.3e}"
    else:
        assert spread_ssp <= ROUND_OFF, (
            f"LUT vs per-SSP reference ratio spread {spread_ssp:.3e} "
            f"(alpha-first reference spread {spread_alpha_first:.3e})"
        )


def _build_spectrum_lut(ssp4, obs_spec):
    return SEDModel.build(
        ssp_data=ssp4,
        observation=obs_spec,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_frac": Fixed(0.6),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
            "age_kernel": "cic",
        },
        met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        approx=SpectrumPrecomp(),
    )


@pytest.mark.parametrize("grid", ["control_multiplicative", "non_multiplicative"])
def test_spectrum_lut_reads_each_ssp_before_alpha(request, grid):
    ssp4 = request.getfixturevalue(
        "ssp_multiplicative" if grid == "control_multiplicative" else "ssp_non_multiplicative"
    )
    obs_spec = tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_r"]),
        spectroscopy=Spectroscopy(
            wave_obs=jnp.linspace(3600.0, 9800.0, 300), resolution=3000.0, resample="point"
        ),
    )
    model = _build_spectrum_lut(ssp4, obs_spec)
    derived = model.predict_state({}).derived
    stellar = next(
        c for c in model._build_component_chain() if type(c).__name__.startswith("Stellar")
    )
    lut = stellar._state.ssp_spec_lut
    params = {
        k: jnp.asarray(float(model.spec.get_distribution(k).default))
        for k in model.spec.all_params
        if getattr(model.spec.get_distribution(k), "default", None) is not None
    }
    _, _, _, met_joint, met_z, lgmet = stellar.compute_joint_weights_exact(params, ssp_data=ssp4)
    marginal = np.asarray(met_joint)[0]
    per_ssp, alpha_first = _per_ssp_and_alpha_first(lut.ssp_on_pixels, lgmet, met_z, marginal, 0.3)
    out = np.asarray(derived["stellar_spec_lnu_precomp"])
    keep = np.abs(per_ssp) > 0
    spread = np.ptp(out[keep] / per_ssp[keep]) / np.min(out[keep] / per_ssp[keep])
    spread_alpha_first = np.ptp(out[keep] / alpha_first[keep]) / np.min(
        out[keep] / alpha_first[keep]
    )
    assert spread <= ROUND_OFF, (
        f"{grid}: spectrum LUT vs per-SSP ratio spread {spread:.3e} "
        f"(alpha-first reference spread {spread_alpha_first:.3e})"
    )


def _build_free_z_lut(ssp4, obs):
    """Free-redshift twin of :func:`_build_lut`: the ztable photometry route (#3537)."""
    return SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Uniform(0.05, 1.0),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_frac": Fixed(0.6),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
            "age_kernel": "cic",
        },
        met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        approx=WavePrecomp(),
    )


@pytest.mark.parametrize("grid", ["control_multiplicative", "non_multiplicative"])
def test_free_z_photometry_reads_metallicity_before_alpha(request, obs, grid):
    """At a ztable node the free-z route equals the fixed-z route (#3537).

    The fixed-z route reads each SSP across metallicity, then mixes alpha (#3528). The
    free-z ztable used to collapse alpha first; on a non-multiplicative grid that is a
    different number at the same redshift. A multiplicative grid is order-independent.
    """
    ssp4 = request.getfixturevalue(
        "ssp_multiplicative" if grid == "control_multiplicative" else "ssp_non_multiplicative"
    )
    free = _build_free_z_lut(ssp4, obs)
    stellar = next(
        c for c in free._build_component_chain() if type(c).__name__.startswith("Stellar")
    )
    z_grid = np.asarray(stellar._state.ssp_phot_ztable.z_grid)
    z_node = float(z_grid[len(z_grid) // 2])
    got = np.asarray(free.predict_state({"redshift": z_node}).derived["stellar_phot_lnu_precomp"])
    fixed = _build_lut(ssp4, obs)
    fixed = SEDModel.build(
        ssp_data=ssp4,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(z_node),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_frac": Fixed(0.6),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
            "age_kernel": "cic",
        },
        met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        approx=WavePrecomp(),
    )
    ref = np.asarray(fixed.predict_state({}).derived["stellar_phot_lnu_precomp"])
    rel = float(np.max(np.abs(got - ref) / np.abs(ref)))
    assert rel <= 1e-12, f"{grid}: free-z vs fixed-z at a node, max rel {rel:.3e}"

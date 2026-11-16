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

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.forward.sed_model import SpectrumPrecomp, WavePrecomp
from tengri.observation import Spectroscopy

pytestmark = pytest.mark.conservation

ALPHA_NODES = jnp.array([-0.2, 0.0, 0.2, 0.4])
#: Ratio spread over filters, float64 round-off.
ROUND_OFF = 1e-10


def _grid(ssp, factor):
    # A derived grid is not the hosted file it came from: renaming it keeps the nebular
    # pairing from swapping in that file's bare twin, whose axes this 4-D grid no longer
    # shares (#3465).
    flux4 = ssp.ssp_flux[:, None, :, :] * factor[:, :, None, None]
    return ssp._replace(
        ssp_flux=flux4,
        ssp_alpha_fe=ALPHA_NODES,
        source=f"{ssp.source}_alpha_synthetic",
    )


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


def _build_lut(ssp4, obs, approx):
    """Fixed-z model on the 4-D grid; ``approx=None`` is the exact path."""
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
        approx=approx,
    )


@pytest.mark.parametrize("grid", ["control_multiplicative", "non_multiplicative"])
def test_photometry_lut_matches_exact_path_on_the_4d_grid(request, obs, grid):
    """Native slopes (#3541): the photometry LUT is the exact path to round-off.

    Under the native-slope read each SSP is read across Z by the cubic Hermite of its
    node values and native PCHIP slopes, then alpha is mixed, which is the exact order.
    On a non-multiplicative grid the alpha-first order differs; the LUT must not.
    """
    ssp4 = request.getfixturevalue(
        "ssp_multiplicative" if grid == "control_multiplicative" else "ssp_non_multiplicative"
    )
    lut = np.asarray(_build_lut(ssp4, obs, WavePrecomp()).predict_photometry({}), dtype=np.float64)
    exact = np.asarray(_build_lut(ssp4, obs, approx=None).predict_photometry({}), dtype=np.float64)
    rel = float(np.max(np.abs(lut / exact - 1.0)))
    assert rel <= ROUND_OFF, f"{grid}: photometry LUT vs exact, max rel {rel:.3e}"


def _build_spectrum_lut(ssp4, obs_spec, approx):
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
        approx=approx,
    )


@pytest.mark.parametrize("grid", ["control_multiplicative", "non_multiplicative"])
def test_spectrum_lut_matches_exact_path_on_the_4d_grid(request, grid):
    """Native slopes (#3541): the spectrum LUT is the exact path to round-off, per pixel."""
    ssp4 = request.getfixturevalue(
        "ssp_multiplicative" if grid == "control_multiplicative" else "ssp_non_multiplicative"
    )
    obs_spec = tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_r"]),
        spectroscopy=Spectroscopy(
            wave_obs=jnp.linspace(3600.0, 9800.0, 300), resolution=3000.0, resample="point"
        ),
    )
    lut = np.asarray(
        _build_spectrum_lut(ssp4, obs_spec, SpectrumPrecomp()).predict_spectrum({}),
        dtype=np.float64,
    )
    exact = np.asarray(
        _build_spectrum_lut(ssp4, obs_spec, approx=None).predict_spectrum({}), dtype=np.float64
    )
    keep = np.abs(exact) > 0
    rel = float(np.max(np.abs(lut[keep] / exact[keep] - 1.0)))
    assert rel <= ROUND_OFF, f"{grid}: spectrum LUT vs exact, max rel {rel:.3e}"


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
    """At a ztable node the free-z route equals the exact fixed-z path (#3537, #3541).

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
    got = np.asarray(free.predict_photometry({"redshift": z_node}), dtype=np.float64)
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
        approx=None,
    )
    ref = np.asarray(fixed.predict_photometry({}), dtype=np.float64)
    rel = float(np.max(np.abs(got - ref) / np.abs(ref)))
    assert rel <= ROUND_OFF, f"{grid}: free-z LUT vs exact fixed-z at a node, max rel {rel:.3e}"


# ── (4) energy balance: metallicity read before the alpha bracket (#3541, item 4) ──


def _emitter_contribution(ssp4, approx):
    """The dust emitter's on-minus-off contribution to two WISE bands, at a 4D grid."""
    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(["wise_w3", "wise_w4"], convention="bessell")
    )

    def model(emitter):
        return SEDModel.build(
            ssp_data=ssp4,
            observation=obs,
            approx=approx,
            redshift=Fixed(0.0),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            met={"type": "delta", "alpha_fe": Fixed(0.3), "all_params": Fixed(DEFAULT)},
            dust_emission=(
                {"type": "dale2014", "all_params": Fixed(DEFAULT)} if emitter else {"type": "none"}
            ),
        )

    on = np.asarray(model(True).predict_photometry({}), dtype=np.float64)
    off = np.asarray(model(False).predict_photometry({}), dtype=np.float64)
    return on - off


@pytest.mark.parametrize("grid", ["control_multiplicative", "non_multiplicative"])
def test_energy_balance_lut_reads_each_ssp_before_alpha(request, grid):
    """The absorbed-energy LUT reads each SSP across Z, then mixes alpha (#3434).

    The emitter's contribution from the LUT must equal the exact path's, which reads
    metallicity per SSP before the [alpha/Fe] bracket. On a non-multiplicative grid the
    alpha-first order differs.
    """
    ssp4 = request.getfixturevalue(
        "ssp_multiplicative" if grid == "control_multiplicative" else "ssp_non_multiplicative"
    )
    exact = _emitter_contribution(ssp4, None)
    lut = _emitter_contribution(ssp4, WavePrecomp())
    rel = float(np.max(np.abs(lut / exact - 1.0)))
    assert rel <= 1e-7, f"{grid}: energy-balance LUT vs exact, max rel {rel:.3e}"

# SPDX-License-Identifier: BSD-3-Clause
"""``WavePrecomp`` serves the energy (CIGALE) filter convention too (#3505, row B6).

The photometry LUT used to refuse ``convention='energy'`` because only the
photon-counting Bessell weight was threaded through the preintegration. The
weight is now a precompute argument, so the energy LUT must (a) build, (b) agree
with the exact energy path per band, and (c) differ from the Bessell LUT, which
shows the weight is really threaded rather than silently reused.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn.component import AGNSEDComponent
from tengri.components.stellar.sps.precompute import precompute_photometry
from tengri.forward.sed_model import WavePrecomp
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterConvention, FilterCurve

pytestmark = pytest.mark.contract

#: Per-band agreement of the energy LUT with the exact energy path (relative).
LUT_TOL = 1e-3


def _curves():
    """Five top-hat-like bands spanning the synthetic SSP's wavelength range."""
    import jax.numpy as jnp

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    return tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))


def _obs(convention):
    return Observation(photometry=Photometry(filters=_curves(), convention=convention))


def _build(ssp, convention, approx):
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(convention),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=None,
            neb={"type": "none"},
            redshift=Fixed(0.5),
            approx=approx,
        )


def test_energy_waveprecomp_builds_and_matches_the_exact_energy_path(synthetic_ssp_wide):
    """The energy LUT builds (no NotImplementedError) and tracks approx=None per band."""
    exact = _build(synthetic_ssp_wide, FilterConvention.ENERGY, approx=None)
    lut = _build(synthetic_ssp_wide, FilterConvention.ENERGY, approx=WavePrecomp())

    f_exact = np.asarray(exact.predict_photometry({}))
    f_lut = np.asarray(lut.predict_photometry({}))

    assert np.all(np.isfinite(f_lut)) and np.all(f_lut > 0)
    rel = np.abs(f_lut / f_exact - 1.0)
    assert np.all(rel < LUT_TOL), f"energy LUT vs exact per band: {rel}"


def test_energy_lut_differs_from_the_bessell_lut(synthetic_ssp_wide):
    """The weight is threaded: the energy LUT is not the photon-counting one."""
    ssp = synthetic_ssp_wide
    fw = [np.asarray(c.wave) for c in _curves()]
    ft = [np.asarray(c.trans) for c in _curves()]
    kw = {
        "ssp_data": ssp,
        "filter_waves": fw,
        "filter_trans": ft,
        "redshift": 0.5,
        "dl_cm": 1.0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bessell = precompute_photometry(**kw, convention=FilterConvention.BESSELL)
        energy = precompute_photometry(**kw, convention=FilterConvention.ENERGY)

    b = np.asarray(bessell.ssp_phot)
    e = np.asarray(energy.ssp_phot)
    assert b.shape == e.shape
    # The normalized band means move by about the spectral slope across each band
    # (measured max ~1e-2 here); a silently-reused Bessell LUT would agree to round-off.
    rel = np.max(np.abs(e / b - 1.0))
    assert rel > 1e-3, f"energy and Bessell LUTs are indistinguishable (max rel {rel:.3e})"


def test_omitting_the_convention_is_an_error(synthetic_ssp_wide):
    """The convention has no default: a builder that omits it fails, it never gets Bessell."""
    fw = [np.asarray(c.wave) for c in _curves()]
    ft = [np.asarray(c.trans) for c in _curves()]
    with pytest.raises(TypeError, match="convention"):
        precompute_photometry(
            ssp_data=synthetic_ssp_wide,
            filter_waves=fw,
            filter_trans=ft,
            redshift=0.5,
            dl_cm=1.0,
        )


def _agn_build(ssp, convention, approx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(convention),
            redshift=Fixed(0.5),
            approx=approx,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 6.0},
            dust_attenuation={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
                "blr": {"type": "analytic", "all_params": Fixed(DEFAULT), "agn_log_lbol": 13.0},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )


def test_energy_agn_lut_matches_the_exact_energy_path(synthetic_ssp_wide):
    """The AGN filter integrals carry the energy weight too, not the Bessell default.

    The AGN component caches the passbands and integrates them at apply time; before
    #3505 that integral was hard-wired to the photon weight, so an energy observation
    with an AGN would silently publish Bessell AGN fluxes beside energy stellar ones.
    """
    exact = _agn_build(synthetic_ssp_wide, FilterConvention.ENERGY, approx=None)
    lut = _agn_build(synthetic_ssp_wide, FilterConvention.ENERGY, approx=WavePrecomp())

    f_exact = np.asarray(exact.predict_photometry({}))
    f_lut = np.asarray(lut.predict_photometry({}))

    assert np.all(np.isfinite(f_lut)) and np.all(f_lut > 0)
    rel = np.abs(f_lut / f_exact - 1.0)
    assert np.all(rel < LUT_TOL), f"energy AGN LUT vs exact per band: {rel}"

    # The power-law AGN spectrum is smooth enough that the photon and energy band
    # means differ by less than LUT_TOL, so the parity above cannot by itself tell a
    # threaded weight from a hard-wired one. The state the apply integrates with can.
    agn = next(c for c in lut._build_component_chain() if isinstance(c, AGNSEDComponent))
    assert agn._state.band_spec.convention is FilterConvention.ENERGY

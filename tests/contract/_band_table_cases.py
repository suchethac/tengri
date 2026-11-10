# SPDX-License-Identifier: BSD-3-Clause
"""The six SEDModel builds whose band-table outputs the parity fixtures pin.

Each builder returns a model on a real bare-stellar SSP (``fsps_prsc_miles_chabrier.h5``) or
on a synthetic composable-AGN SSP, so the goldens do not depend on a wNE grid. The six cases
cover every band-table provider: dust emission (fixed redshift), free redshift with the radio
and X-ray additive terms, the composable AGN, the CLOUDY grid nebular, the MAPPINGS shock, and
the exact path with ``approx=None``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

#: The bare-stellar SSP every case reads, except the AGN case (synthetic).
SSP_NAME = "fsps_prsc_miles_chabrier.h5"

#: Filters every case photometers through.
FILTERS = ("sdss_g", "sdss_r", "galex_fuv", "2mass_ks")

#: Top-hat bands for the AGN case, optical to far-IR.
AGN_BANDS_AA = (3000.0, 5500.0, 2.2e4, 1.2e5, 2.5e5, 8.0e5)


def _data(name: str) -> str:
    from tengri._data_setup import find_data_str

    path = find_data_str(name)
    if path is None:
        raise FileNotFoundError(f"band-table parity case needs {name} under the data directory")
    return path


def _bare_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(_data(SSP_NAME))


def _synthetic_agn_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 7.0, 1600)
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-2.5, -1.85, -1.2])
    flux = (
        ((5000.0 / wave) ** 2)[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


def _tophat(center_aa: float, frac_width: float = 0.16, n: int = 40):
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(center_aa * (1.0 - frac_width), center_aa * (1.0 + frac_width), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center_aa)}")


def _dpl_sfh():
    return {"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)}


def _dust_case(*, approx):
    return SEDModel.build(
        ssp_data=_bare_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(FILTERS))),
        approx=approx,
        redshift=Fixed(0.1),
        sfh=_dpl_sfh(),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
    )


def build_dust_fixed_z():
    """Two-component dust with dale2014 emission at fixed redshift, under WavePrecomp."""
    return _dust_case(approx=WavePrecomp())


def build_free_z_radio_xray():
    """Free redshift with the radio and X-ray additive terms, under WavePrecomp."""
    return SEDModel.build(
        ssp_data=_bare_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(FILTERS))),
        approx=WavePrecomp(),
        redshift=Uniform(0.05, 0.3),
        sfh=_dpl_sfh(),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "modified_blackbody", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        agn={"type": "composable", "disc": {"type": "powerlaw"}, "all_params": Fixed(DEFAULT)},
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        xray={"type": "simple", "delta_alpha_ox": Uniform(-2.0, -1.0)},
    )


def build_composable_agn():
    """Composable AGN (powerlaw disc, simple torus) over its luminosity axis, under WavePrecomp."""
    bands = tuple(_tophat(c) for c in AGN_BANDS_AA)
    return SEDModel.build(
        ssp_data=_synthetic_agn_ssp(),
        observation=Observation(photometry=Photometry(filters=bands)),
        approx=WavePrecomp(),
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": "powerlaw"},
            "torus": {"type": "simple"},
            "agn_log_lbol": Uniform(8.0, 14.0),
        },
    )


def _synthetic_cloudy_grid() -> str:
    """Write the deterministic synthetic CLOUDY grid once and return its path.

    The tracked tree carries no CLOUDY grid (the converted ones are untracked), so the case
    uses the same synthetic grid and ionizing SSP as the CLOUDY accuracy test.
    """
    from tests._cloudy_grid import write_synthetic_cloudy_grid

    directory = Path(tempfile.gettempdir()) / "tengri_band_table_parity"
    directory.mkdir(parents=True, exist_ok=True)
    return str(write_synthetic_cloudy_grid(directory / "cloudy_grid.h5"))


def build_cloudy_nebular():
    """CLOUDY grid nebular emission on a synthetic ionizing SSP, under WavePrecomp."""
    from tests._cloudy_grid import synthetic_ionizing_ssp

    return SEDModel.build(
        ssp_data=synthetic_ionizing_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(FILTERS))),
        approx=WavePrecomp(),
        redshift=Fixed(0.1),
        sfh=_dpl_sfh(),
        dust_attenuation={"type": "none"},
        neb={
            "type": "cloudy",
            "all_params": Fixed(DEFAULT),
            "grid": _synthetic_cloudy_grid(),
        },
    )


def build_shock():
    """MAPPINGS shock with its H-alpha luminosity set absolutely, under WavePrecomp."""
    return SEDModel.build(
        ssp_data=_bare_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(FILTERS))),
        approx=WavePrecomp(),
        redshift=Fixed(0.1),
        sfh=_dpl_sfh(),
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        shock={
            "norm": "lhalpha",
            "log_lhalpha": Uniform(39.0, 42.0),
            "velocity": Uniform(200.0, 800.0),
            "log_density": Uniform(-1.0, 2.0),
        },
    )


def build_exact():
    """The dust case on the exact wave-grid path (``approx=None``)."""
    return _dust_case(approx=None)


#: The six cases, in golden-file order.
CASES = {
    "dust_fixed_z": build_dust_fixed_z,
    "free_z_radio_xray": build_free_z_radio_xray,
    "composable_agn": build_composable_agn,
    "cloudy_nebular": build_cloudy_nebular,
    "shock": build_shock,
    "exact": build_exact,
}

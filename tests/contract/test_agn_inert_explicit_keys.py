"""Explicit AGN sub-block keys the selected block type never reads are refused.

An explicit per-parameter key that the category owns but the selected type does
not read is accepted by the grammar and then ignored: a silent failure. The
grammar refuses it by name and lists the types that do read the key, so the spec
can be pointed at the block that reads it (design G3, GRAHSP follow-up to #985).

Tier: contract. Builds use a small synthetic SSP (no ``data/`` dependency).
"""

import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

_CENTERS = (1500.0, 5000.0, 2.0e4, 1.0e5, 2.5e5, 2.0e6)


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def ssp() -> SSPData:
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def obs() -> Observation:
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in _CENTERS)))


def _build(ssp_data, observation, **agn_subblocks):
    """Minimal composable AGN build with the given sub-block dicts (disc/nlr/blr fixed)."""
    agn = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "all_params": Fixed(DEFAULT),
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    agn.update(agn_subblocks)
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 1.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        agn=agn,
        redshift=Fixed(1.0),
    )


def test_grahsp_torus_refuses_tor_temp_and_names_mn12(ssp, obs):
    """``torus:grahsp`` does not read ``agn_grahsp_tor_temp``; ``torus:grahsp_mn12`` does."""
    with pytest.raises(ValueError, match=r"grahsp_mn12") as info:
        _build(
            ssp,
            obs,
            torus={"type": "grahsp", "agn_grahsp_tor_temp": Fixed(0.27)},
        )
    assert "agn_grahsp_tor_temp" in str(info.value)


def test_grahsp_torus_refuses_tau_skirtor_and_names_skirtor(ssp, obs):
    """``tau_skirtor`` is inert under ``torus:grahsp``; the SKIRTOR torus reads it."""
    with pytest.raises(ValueError, match=r"skirtor") as info:
        _build(
            ssp,
            obs,
            torus={"type": "grahsp", "agn_tau_skirtor": Fixed(5.0)},
        )
    assert "agn_tau_skirtor" in str(info.value)


def test_non_grahsp_example_refuses_agn_tau_and_names_nenkova(ssp, obs):
    """``agn_tau`` is read by the Nenkova torus only; SKIRTOR ignores it."""
    with pytest.raises(ValueError, match=r"nenkova") as info:
        _build(
            ssp,
            obs,
            torus={"type": "skirtor", "agn_tau": Fixed(5.0)},
        )
    assert "agn_tau" in str(info.value)


def test_key_the_selected_type_reads_is_accepted(ssp, obs):
    """Positive control: the same key on the block that reads it builds."""
    model = _build(
        ssp,
        obs,
        torus={"type": "grahsp_mn12", "agn_grahsp_tor_temp": Fixed(0.27)},
    )
    assert model is not None

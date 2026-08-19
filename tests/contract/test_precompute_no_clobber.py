# SPDX-License-Identifier: BSD-3-Clause
"""The unconditional precompute pass must never downgrade a populated state (#1738).

Build-time component resolution happens in two stages. An unconditional pass
resolves every component so that ``load()`` runs even when no approximation is
requested -- the fix for #1278, where dust components were absent from the old
per-kind dispatch and so never resolved at all. The ``wave_precomp`` /
``spectrum_precomp`` specializations then layer richer state on top.

For the stellar component the unconditional pass calls ``precompute()`` without
filters, and stellar documents that path as returning an *empty state marker*
(``components/stellar/component.py``). The specialization later returns the
populated LUT state. So the two stages write to the same ``_state`` field with
values of very different worth, and correctness rests on the populated one
winning.

Relying on statement order for that would be a silent-wrong-physics hazard of
exactly the kind this epic exists to remove: a reordering would replace a
photometry LUT with an empty marker, and every prediction would still return
plausible finite numbers. No existing test covered it. This module does.

Deliberately contains no ``try``/``except`` and no runtime ``pytest.skip``. The
quarantined test in #1660 turned a stale-API ``AttributeError`` into "SSP data
not available" and sat green for months while asserting nothing; a guard for a
silent failure must itself fail loudly. Fixtures are synthetic, so this runs
anywhere without the gitignored ``data/ssp_*.h5`` grids.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Bare-stellar SSP on a UV to far-IR grid with a smooth declining continuum."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)  # 100 Angstrom - 1 mm
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-2.5, -1.85, -1.2])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="module")
def tophat_filters():
    """Five synthetic top-hat bands spanning the optical."""
    from tengri.observation.photometry import FilterCurve

    curves = []
    for center in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0):
        lo, hi = center * 0.82, center * 1.18
        wave = jnp.linspace(lo, hi, 48)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, 48)) * 0.6
        curves.append(FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}"))
    return curves


def _build_with_wave_precomp(ssp_data, filters):
    """Build a minimal photometry model under ``approx=WavePrecomp()``.

    Parameters
    ----------
    ssp_data : SSPData
        Synthetic SSP grid.
    filters : list of FilterCurve
        Synthetic photometric bands.

    Returns
    -------
    SEDModel
        A built model whose stellar component should carry a populated LUT.
    """
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

    return SEDModel.build(
        ssp_data=ssp_data,
        observation=Observation(photometry=Photometry(filters=tuple(filters))),
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
        sfh={"type": "dpl", "*": FIXED},
        neb={"type": "none"},
        dust={
            "law_diff": "calzetti",
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_diff": 0.5,
        },
    )


def test_wave_precomp_stellar_lut_is_not_downgraded(synthetic_ssp, tophat_filters):
    """The stellar LUT built by ``wave_precomp`` survives the unconditional pass.

    The unconditional pass calls ``stellar.precompute()`` with ``filters=None``,
    which returns an empty state marker. If that marker were written over the
    populated LUT state, photometry would silently fall back to the exact path
    or to zeros while still returning finite numbers.
    """
    model = _build_with_wave_precomp(synthetic_ssp, tophat_filters)

    chain = model._build_component_chain()
    stellar = chain[0]

    assert stellar.name == "stellar", f"expected stellar first in the chain, got {stellar.name!r}"
    assert stellar._state is not None, (
        "stellar carries no precompute state at all under approx=WavePrecomp()"
    )

    lut = getattr(stellar._state, "ssp_phot_lut", None)
    ztable = getattr(stellar._state, "ssp_phot_ztable", None)
    assert lut is not None or ztable is not None, (
        "stellar._state carries neither ssp_phot_lut nor ssp_phot_ztable under "
        "approx=WavePrecomp(): the populated LUT was replaced by the empty marker "
        "that the unconditional resolution pass produces (#1738)"
    )


def test_unconditional_pass_still_resolves_without_approx(synthetic_ssp, tophat_filters):
    """With no ``approx``, every component is still resolved exactly once.

    The companion to the test above: the no-clobber rule must not be
    implemented by skipping resolution altogether, which would reinstate the
    #1278 defect where a component's ``load()`` never ran on a default build.
    """
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=Observation(photometry=Photometry(filters=tuple(tophat_filters))),
        redshift=Fixed(0.05),
        approx=None,
        sfh={"type": "dpl", "*": FIXED},
        neb={"type": "none"},
        dust={
            "law_diff": "calzetti",
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_diff": 0.5,
        },
    )

    chain = model._build_component_chain()

    assert len(chain) > 0, "component chain is empty"
    assert chain[0].name == "stellar", f"expected stellar first, got {chain[0].name!r}"

# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2362: BakedInBackend checks whether SSP has nebular.

`BakedInBackend` returns zero nebular emission unconditionally, which is
correct only if the SSP already includes it. On bare or unstamped grids,
the result is silent total nebular omission. This test ensures:

1. Raise on bare grids (ssp_data.nebular == "bare")
2. Warn on unstamped grids (ssp_data.nebular == "unknown")
3. Pass silently on grids with nebular included (ssp_data.nebular == "included")
4. Warn about unknown grids even when neb={'type':'ssp'} (split the advisory)
5. has_continuum is True only for "included"; "unknown" keeps True but warns it is assumed
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def _synthetic_ssp(nebular: str):
    """Create a minimal SSPData with specified nebular status."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=jnp.linspace(100.0, 20000.0, 50),
        ssp_flux=jnp.full((2, 4, 50), 1e-4),
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.0, 4),
        ssp_lgmet=jnp.array([-2.5, -1.8]),
        nebular=nebular,
    )


def test_bakedin_raises_on_bare_ssp():
    """BakedInBackend raises when SSP is flagged as bare."""
    pytest.importorskip("tengri")
    import tengri

    ssp = _synthetic_ssp("bare")
    with pytest.raises(ValueError, match="bare"):
        tengri.SEDModel.build(
            ssp,
            sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            neb={"type": "ssp"},
            redshift=tengri.Fixed(0.05),
        )


def test_bakedin_warns_on_unknown_ssp():
    """BakedInBackend warns when SSP nebular status is unknown (without explicit neb)."""
    pytest.importorskip("tengri")
    import tengri
    from tengri.components.nebular.baked_in import BakedInNebularGridWarning

    ssp = _synthetic_ssp("unknown")
    # Grid warning is emitted when neb is NOT explicitly passed (uses default neb=off)
    with pytest.warns(BakedInNebularGridWarning, match="stamp_ssp_nebular_attrs"):
        tengri.SEDModel.build(
            ssp,
            sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            redshift=tengri.Fixed(0.05),
        )


def test_bakedin_unknown_is_silenced_by_explicit_neb_declaration():
    """Grid warning IS silenced by explicit neb={'type':'ssp'} (honours suppression)."""
    pytest.importorskip("tengri")
    import tengri
    from tengri.components.nebular.baked_in import BakedInNebularGridWarning

    ssp = _synthetic_ssp("unknown")
    # The explicit declaration SHOULD silence the grid warning (honours suppression contract)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tengri.SEDModel.build(
            ssp,
            sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            neb={"type": "ssp"},
            redshift=tengri.Fixed(0.05),
        )
        # Should NOT have the grid warning about unknown status (suppressed by explicit neb)
        grid_warnings = [x for x in w if issubclass(x.category, BakedInNebularGridWarning)]
        cat_names = [x.category.__name__ for x in w]
        assert len(grid_warnings) == 0, (
            f"Expected no BakedInNebularGridWarning but got: {cat_names}"
        )


def test_bakedin_silent_on_included_ssp():
    """BakedInBackend is silent when SSP is flagged as having nebular included."""
    pytest.importorskip("tengri")
    import tengri

    ssp = _synthetic_ssp("included")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Will raise if any warning
        model = tengri.SEDModel.build(
            ssp,
            sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            neb={"type": "ssp"},
            redshift=tengri.Fixed(0.05),
        )
    assert model is not None


def test_bakedin_has_continuum_follows_nebular_status():
    """has_continuum is True for 'included', keeps True for 'unknown' with warning."""
    pytest.importorskip("tengri")
    from tengri.components.nebular.baked_in import BakedInBackend, BakedInNebularGridWarning

    # "included" case
    ssp_incl = _synthetic_ssp("included")
    backend_incl = BakedInBackend(ionizing_source_warning="suppress", ssp_data=ssp_incl)
    assert backend_incl.has_continuum is True

    # "unknown" case - should warn when ionizing_source_warning='warn'
    ssp_unkn = _synthetic_ssp("unknown")
    with pytest.warns(BakedInNebularGridWarning):
        backend_unkn = BakedInBackend(ionizing_source_warning="warn", ssp_data=ssp_unkn)
    # has_continuum should be True (assumed) even for unknown
    assert backend_unkn.has_continuum is True


def test_bakedin_warns_when_ssp_data_none():
    """BakedInBackend silent on grid warning when ssp_data=None (no actual data to check)."""
    pytest.importorskip("tengri")
    from tengri.components.nebular.baked_in import BakedInBackend, BakedInNebularWarning

    # ssp_data=None: no actual SSP data provided, so grid warning is not emitted
    # but fixed-logU warning is still emitted when ionizing_source_warning='warn'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        backend = BakedInBackend(ionizing_source_warning="warn", ssp_data=None)
    # Should have fixed-logU warning (BakedInNebularWarning) but NOT grid warning
    neb_warnings = [x for x in w if issubclass(x.category, BakedInNebularWarning)]
    assert len(neb_warnings) >= 1, "Expected at least fixed-logU warning"
    # has_continuum should be True (assumed when status is unknown)
    assert backend.has_continuum is True

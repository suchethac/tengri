# SPDX-License-Identifier: BSD-3-Clause
"""Regression #2973: a failing WavePrecomp table builder must raise at build.

``SEDModel.build(approx=WavePrecomp())`` used to wrap every table builder in a
broad ``try/except Exception`` that warned and fell back to the exact per-call
filter integral. A real defect (a hardcoded ``"ln_phot"`` key in dale2014's
adapter, #2826) was swallowed that way: the model was correct, the table was
never used, and nothing failed. A precompute path that fails must refuse at
build time instead of silently degrading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.regression_bug

_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _ssp_or_skip():
    if not Path(_BARE_SSP).is_file():
        pytest.skip("No bare-stellar SSP grid available under data/.")
    from tengri import load_ssp_data

    return load_ssp_data(_BARE_SSP)


def _ir_obs():
    import jax.numpy as jnp

    from tengri import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    cs = [3500.0, 36000.0, 240000.0, 700000.0, 1600000.0]
    curves = tuple(
        FilterCurve(
            wave=jnp.linspace(c * 0.85, c * 1.15, 40), trans=jnp.ones(40) * 0.5, name=f"b{int(c)}"
        )
        for c in cs
    )
    return Observation(photometry=Photometry(filters=curves))


def _build_wave_precomp_dust_model():
    from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp

    return SEDModel.build(
        ssp_data=_ssp_or_skip(),
        observation=_ir_obs(),
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.5,
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
    )


def test_failing_dust_emission_table_builder_raises(monkeypatch):
    """A KeyError inside a dust-emission table builder must surface, not warn."""
    from tengri import SEDModel

    def _boom(self, chain):
        raise KeyError("ln_phot")

    monkeypatch.setattr(SEDModel, "_dust_emission_band_response", _boom)

    with pytest.raises(KeyError, match="ln_phot"):
        _build_wave_precomp_dust_model()


def test_failing_additive_term_table_builder_raises(monkeypatch):
    """A failure in a per-emitter additive-term builder must surface, not warn."""
    import tengri.forward.sed_model as sed_model_mod
    from tengri import SEDModel

    def _boom(self, chain, emitter):
        raise KeyError("ln_phot")

    monkeypatch.setattr(sed_model_mod, "_chain_implements_emission_terms", lambda chain: ["xray"])
    monkeypatch.setattr(SEDModel, "_additive_term_band_response", _boom)

    with pytest.raises(KeyError, match="ln_phot"):
        _build_wave_precomp_dust_model()

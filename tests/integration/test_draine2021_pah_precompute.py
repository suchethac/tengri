"""Integration tests for the Draine+2021 PAHspec precompute adapter.

Exercises the photometry-grid pre-integration path used by SEDModel
(``dust_emission="draine2021_pah"``) and the JIT-compiled lookup
function used at likelihood-evaluation time.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pahspec_smoke.h5"


@pytest.fixture(scope="module")
def filter_curves():
    """Three top-hat IR/sub-mm filter curves spanning the PAH bands."""
    fws = [
        np.linspace(7.0e4, 1.2e5, 50),  # ~7-12 um (PAH-rich)
        np.linspace(2.0e5, 4.0e5, 50),  # ~20-40 um (mid-IR continuum)
        np.linspace(1.0e6, 2.0e6, 50),  # ~100-200 um (FIR peak)
    ]
    fts = [np.ones_like(fw) for fw in fws]
    return fws, fts


@pytest.fixture(scope="module")
def precomp(filter_curves):
    pytest.importorskip("h5py")
    from tengri.components.dust.dust_emission_precompute import (
        precompute_draine2021_pah_photometry,
    )
    from tengri.components.dust.emission_templates import (
        load_draine2021_pahspec_templates,
    )

    templates = load_draine2021_pahspec_templates(str(FIXTURE))
    fws, fts = filter_curves
    return precompute_draine2021_pah_photometry(
        templates,
        fws,
        fts,
        redshift=0.0,
        starlight="mMMP",
        ionization="st",
        size_distribution="std",
        slab=False,
    )


def test_precompute_shape(precomp, filter_curves):
    fws, _ = filter_curves
    # Convention: grid dims first, filter axis last.
    assert precomp["single_u_phot"].shape == (15, len(fws))
    assert precomp["lgU_grid"].shape == (15,)
    assert precomp["starlight"] == "mMMP"


def test_lookup_jit_and_value(precomp):
    from tengri.components.dust.dust_emission_precompute import (
        build_draine2021_pah_photometry_lookup,
    )

    lookup = build_draine2021_pah_photometry_lookup(precomp)
    L_abs = jnp.asarray(1.0e44)
    lgU = jnp.asarray(1.0)

    out = lookup(L_abs, lgU)
    chex.assert_shape(out, (3,))
    arr = np.asarray(out)
    assert np.isfinite(arr).all()
    assert (arr > 0).all()


def test_lookup_scales_linearly_in_L_absorbed(precomp):
    from tengri.components.dust.dust_emission_precompute import (
        build_draine2021_pah_photometry_lookup,
    )

    lookup = build_draine2021_pah_photometry_lookup(precomp)
    a = np.asarray(lookup(jnp.asarray(1.0e44), jnp.asarray(2.0)))
    b = np.asarray(lookup(jnp.asarray(2.0e44), jnp.asarray(2.0)))
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-6)


def test_lookup_at_grid_endpoints(precomp):
    """Querying lookup exactly at lgU grid points must reproduce the
    pre-integrated photometry at those points to high precision."""
    from tengri.components.dust.dust_emission_precompute import (
        build_draine2021_pah_photometry_lookup,
    )

    lookup = build_draine2021_pah_photometry_lookup(precomp)
    grid = np.asarray(precomp["lgU_grid"])
    phot = np.asarray(precomp["single_u_phot"])  # (n_filt, n_lgU)

    L_abs = 1.0e44
    for i_u, lgU in enumerate(grid):
        out = np.asarray(lookup(jnp.asarray(L_abs), jnp.asarray(lgU)))
        np.testing.assert_allclose(out, L_abs * phot[i_u, :], rtol=1e-6)


def test_axis_params_registered():
    from tengri.components.dust.dust_emission_precompute import AXIS_PARAMS

    assert AXIS_PARAMS["draine2021_pah"] == ("dust_lgU",)


def test_build_lookup_dispatch(precomp):
    """build_lookup(model_name='draine2021_pah') must return the
    PAHspec-specific lookup."""
    from tengri.components.dust.dust_emission_precompute import build_lookup

    fn = build_lookup(precomp, model_name="draine2021_pah")
    out = fn(jnp.asarray(1.0e44), jnp.asarray(0.5))
    chex.assert_shape(out, (3,))


def test_precompute_for_model_returns_none_when_missing(monkeypatch, filter_curves):
    """precompute_for_model returns None (not raises) when the global
    HDF5 grid isn't on disk, so SEDModel can fall back to wavelength-
    level evaluation."""
    from tengri.components.dust.dust_emission_precompute import precompute_for_model

    monkeypatch.delenv("TENGRI_PAHSPEC_PATH", raising=False)
    monkeypatch.setenv("TENGRI_PAHSPEC_PATH", "/nonexistent/path.h5")
    fws, fts = filter_curves
    out = precompute_for_model(
        "draine2021_pah",
        fws,
        fts,
        redshift=0.0,
        parameters=None,
    )
    assert out is None


def test_precompute_for_model_with_explicit_path(monkeypatch, filter_curves):
    """When pointed at a real HDF5 file via env var, precompute_for_model
    returns a populated dict."""
    from tengri.components.dust.dust_emission_precompute import precompute_for_model

    monkeypatch.setenv("TENGRI_PAHSPEC_PATH", str(FIXTURE))
    fws, fts = filter_curves
    out = precompute_for_model(
        "draine2021_pah",
        fws,
        fts,
        redshift=0.0,
        parameters=None,
    )
    assert out is not None
    assert "single_u_phot" in out
    assert out["single_u_phot"].shape == (15, len(fws))

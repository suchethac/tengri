# SPDX-License-Identifier: BSD-3-Clause
"""AGN block ↔ single-source physics equivalence (#738 Phase 4).

Each AGN model's physics lives in ONE place (e.g. ``agn/silva04.py::silva04_sed``).
Both the composable **block** adapter (``blocks/torus_blocks.py``, used by the
runner) and the one-file ``SEDModelComponent`` (``silva04_model.py``)
wrap that same primitive — there is no second copy of the physics.

These tests pin that the live block adapters are *faithful* wrappers of the
single-source primitive (only the documented ``L_ν → L_λ`` conversion), so the
two adapters cannot silently diverge in physics. This is the "confirm the
duplicate works" guarantee for the wire-up/co-locate consolidation (#738) — we
consolidate nothing until the equivalence is proven and we delete nothing.

Skirtor and the disc/nlr blocks add documented extras on top of the shared
primitive (polar-dust energy coupling, Type-1/2 masking, l5100 normalization);
their CIGALE/AGNfitter fidelity is covered by the dedicated parity tests, so
they are not pure-conversion cases and are excluded here.
"""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import pytest

# Importing the blocks package triggers @register_agn_block registration.
from tengri.components.agn import blocks  # noqa: F401
from tengri.components.agn.blocks._protocol import resolve_agn_block

pytestmark = pytest.mark.contract

# Frequency-conversion constant — identical literal to blocks/torus_blocks.py.
_C_AA_PER_S = 2.99792458e18

# UV–FIR rest-frame grid spanning the torus thermal bump.
WAVE = jnp.geomspace(1.0e3, 1.0e7, 256)


# Pure-conversion torus blocks: block(wave, log_lbol, l5100, **p) must equal
# primitive_sed(wave, agn_log_lbol=log_lbol, **p) × c / λ², bit-for-bit.
_PURE_CONVERSION_TORUS = [
    (
        "silva04",
        ("tengri.components.agn.silva04", "silva04_sed"),
        {"agn_log_nh_silva": 23.0, "agn_torus_frac": 0.5},
    ),
    (
        "cat3d_wind",
        ("tengri.components.agn.cat3d_wind", "cat3d_wind_sed"),
        {"agn_torus_frac": 0.5},
    ),
    (
        "skirtor_agnfitter",
        ("tengri.components.agn.skirtor_agnfitter", "skirtor_agnfitter_sed"),
        {"agn_torus_frac": 0.5},
    ),
]


@pytest.mark.parametrize("name,primitive_ref,params", _PURE_CONVERSION_TORUS)
def test_torus_block_is_faithful_wrapper_of_single_source_primitive(name, primitive_ref, params):
    """The torus block adapter only converts L_ν→L_λ of its shared primitive."""
    module = importlib.import_module(primitive_ref[0])
    primitive = getattr(module, primitive_ref[1])
    block = resolve_agn_block("torus", name)

    log_lbol = 11.5
    l5100_disc = jnp.asarray(1.0e44)  # ignored by these torus blocks
    try:
        block_out = block(WAVE, log_lbol, l5100_disc, **params)
        prim_lnu = primitive(WAVE, agn_log_lbol=log_lbol, **params)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"{name!r} template grid not on disk: {exc}")

    prim_llambda = prim_lnu * _C_AA_PER_S / WAVE**2
    assert jnp.allclose(block_out, prim_llambda, rtol=1e-10, atol=0.0), (
        f"torus block {name!r} diverged from its single-source physics primitive "
        f"{primitive_ref[1]} — the block must wrap it with only the L_ν→L_λ "
        f"conversion (#738 Phase 4)."
    )


# (block module, one-file component module, shared physics module) — the block
# adapter and the component both import from the SAME ``components.agn.<model>``
# physics module (the block pulls the ``*_sed`` primitive; the component pulls
# its grid loader from the same module), so there is one source of truth per
# model.
_SHARED_MODULE = [
    ("blocks.torus", "silva04_model", "components.agn.silva04"),
    ("blocks.torus", "cat3d_torus_model", "components.agn.cat3d_wind"),
    ("blocks.torus", "skirtor_model", "components.agn.skirtor"),
    ("blocks.disc", "kd18_disc_model", "components.agn.disc"),
    # NLR had a one-file component too (``nlr_model.AGNNebular``), but it was a
    # grammar-unreachable orphan with drifted param names (``agn_nlr_cov_frac``
    # vs the canonical ``agn_nlr_cf``) — the exact silent-divergence footgun
    # this migration removes — so it was deleted (#897). The canonical NLR
    # block ``blocks.nlr_analytic`` still wraps the single-source
    # ``compute_nlr_sed`` from ``components.agn.nlr``; that faithfulness is
    # pinned by ``test_nlr_analytic_wraps_single_source_kernel`` below.
]


@pytest.mark.parametrize("block_mod,component_mod,physics_module", _SHARED_MODULE)
def test_block_and_component_share_one_physics_module(block_mod, component_mod, physics_module):
    """Structural guard: each model's block and one-file component both import
    from the SAME physics module — there is no second copy to drift out of
    sync."""
    import inspect

    block_src = inspect.getsource(importlib.import_module(f"tengri.components.agn.{block_mod}"))
    component_src = inspect.getsource(
        importlib.import_module(f"tengri.components.agn.{component_mod}")
    )
    assert physics_module in block_src, f"{block_mod} no longer imports {physics_module}"
    assert physics_module in component_src, f"{component_mod} no longer imports {physics_module}"


def test_nlr_analytic_wraps_single_source_kernel(monkeypatch):
    """After #897 removed the redundant NLR one-file component, the canonical
    NLR block ``blocks.nlr_analytic`` must wrap the single-source physics
    kernel ``compute_nlr_sed`` from ``components.agn.nlr`` — so there is one
    source of truth for the analytic NLR spectrum and no second copy to drift.

    Verified by: (1) instrumenting compute_nlr_sed to verify the block calls it,
    (2) checking the returned spectrum is finite and non-negative, (3) verifying
    the block is responsive to one of its parameters (agn_nlr_cf).
    """
    from tengri.components.agn.blocks.nlr import nlr_analytic_block
    from tengri.components.agn.nlr import compute_nlr_sed

    # Instrument compute_nlr_sed to record calls
    call_log = []

    def recording_compute_nlr_sed(*args, **kwargs):
        call_log.append(True)
        return compute_nlr_sed(*args, **kwargs)

    block_module = importlib.import_module("tengri.components.agn.blocks.nlr")
    monkeypatch.setattr(block_module, "compute_nlr_sed", recording_compute_nlr_sed)

    wave = jnp.geomspace(1.0e3, 1.0e5, 50)
    l5100 = jnp.asarray(1.0e44)
    agn_log_lbol = 11.5

    # Call the block with two different agn_nlr_cf values (positive control)
    _wav_low, lum_cf_low = nlr_analytic_block(
        wave, agn_log_lbol, l5100, agn_nlr_cf=0.1, agn_nlr_fwhm_kms=800.0
    )
    _wav_high, lum_cf_high = nlr_analytic_block(
        wave, agn_log_lbol, l5100, agn_nlr_cf=0.5, agn_nlr_fwhm_kms=800.0
    )

    assert len(call_log) > 1, "NLR block did not call compute_nlr_sed — the delegation is broken"

    # Verify output is finite and non-negative (proper spectrum)
    assert jnp.all(jnp.isfinite(lum_cf_low)), "NLR block returned inf/nan"
    assert jnp.all(lum_cf_low >= 0), "NLR block returned negative luminosity"
    assert jnp.all(jnp.isfinite(lum_cf_high)), "NLR block returned inf/nan"
    assert jnp.all(lum_cf_high >= 0), "NLR block returned negative luminosity"

    # Positive control: agn_nlr_cf is a covering fraction; changing it must
    # change the output (otherwise the block is insensitive to parameters)
    assert not jnp.allclose(lum_cf_low, lum_cf_high, rtol=1e-6), (
        "NLR block is unresponsive to agn_nlr_cf — either the parameter is "
        "not passed to compute_nlr_sed, or the block discards its output"
    )

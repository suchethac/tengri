# SPDX-License-Identifier: BSD-3-Clause
"""AGN block ↔ single-source physics equivalence (#738 Phase 4).

Each AGN model's physics lives in ONE place (e.g. ``agn/silva04.py::silva04_sed``).
Both the composable **block** adapter (``blocks/torus_blocks.py``, used by the
runner) and the one-file ``SEDModelComponent`` **port** (``silva04_model.py``)
wrap that same primitive — there is no second copy of the physics.

These tests pin that the live block adapters are *faithful* wrappers of the
single-source primitive (only the documented ``L_ν → L_λ`` conversion), so the
two adapters cannot silently diverge in physics. This is the "confirm the
duplicate works" guarantee for the wire-up/co-locate consolidation (#738) — we
consolidate nothing until the equivalence is proven and we delete nothing.

Skirtor and the disc/nlr blocks add documented extras on top of the shared
primitive (polar-dust energy coupling, Type-1/2 masking, l5100 normalisation);
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


# (block module, one-file port module, shared physics module) — the block
# adapter and the port both import from the SAME ``components.agn.<model>``
# physics module (the block pulls the ``*_sed`` primitive; the port pulls its
# grid loader from the same module), so there is one source of truth per model.
_SHARED_MODULE = [
    ("blocks.silva04_torus", "silva04_model", "components.agn.silva04"),
    ("blocks.cat3d_wind_torus", "cat3d_torus_model", "components.agn.cat3d_wind"),
    ("blocks.skirtor_torus", "skirtor_model", "components.agn.skirtor"),
    ("blocks.kubota_done_disc", "kd18_disc_model", "components.agn.disc"),
    ("blocks.nlr_analytic", "nlr_model", "components.agn.nlr"),
]


@pytest.mark.parametrize("block_mod,port_mod,physics_module", _SHARED_MODULE)
def test_block_and_port_share_one_physics_module(block_mod, port_mod, physics_module):
    """Structural guard: each model's block and one-file port both import from
    the SAME physics module — there is no second copy to drift out of sync."""
    import inspect

    block_src = inspect.getsource(importlib.import_module(f"tengri.components.agn.{block_mod}"))
    port_src = inspect.getsource(importlib.import_module(f"tengri.components.agn.{port_mod}"))
    assert physics_module in block_src, f"{block_mod} no longer imports {physics_module}"
    assert physics_module in port_src, f"{port_mod} no longer imports {physics_module}"

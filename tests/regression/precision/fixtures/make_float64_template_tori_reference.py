# SPDX-License-Identifier: BSD-3-Clause
"""Freeze the float64 template-torus outputs of a reference tree (#3101, #3084).

Writes ``float64_template_tori_reference.npz`` beside this script: float64 SEDs and their
forward derivatives in ``agn_log_lbol`` and the sightline, one torus per scaling code path,
plus the anisotropy tables of the anisotropic libraries. Generate it on the tree the float64
results must not move from (the main commit before the float32 fix), with that tree's
``src`` first on ``PYTHONPATH``, and record its commit::

    PYTHONPATH=<reference tree>/src JAX_PLATFORMS=cpu \\
        python tests/regression/precision/fixtures/make_float64_template_tori_reference.py \\
        --sha <commit of the reference tree>

Every array is float64; the large Fritz anisotropy table is kept as a SHA-256 digest of its
bytes (:func:`table_digest`) plus one column.
``test_bug_3101_3084_float32_template_tori.py`` compares the current tree against it bit
for bit. The script reads only the public kernels and loaders, whose signatures the fix
left unchanged.
"""

import argparse
import hashlib
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tengri  # noqa: F401  (enables x64)

OUT = Path(__file__).with_name("float64_template_tori_reference.npz")

LBOLS = (-23.58, 0.0, 6.0, 10.0, 14.0)
WAVE = np.geomspace(1.0e2, 1.0e7, 64)  # Angstrom
FRAC = 0.4

#: name: module, grid finder, grid loader, kernel, sightline key, sightline, extra kwargs.
KERNELS = {
    "cat3d_wind": ("cat3d_wind", "_find_cat3d_grid", "load_cat3d_wind_grid",
                   "cat3d_wind_sed_from_grid", "agn_cos_inc", 0.55,
                   {"agn_a_cat3d": -1.6, "agn_fwd_cat3d": 1.4}),
    "nenkova_agnfitter_2p": ("nenkova_agnfitter_2p", "_find_nenkova_agnfitter_2p_grid",
                             "load_nenkova_agnfitter_2p_grid",
                             "nenkova_agnfitter_2p_sed_from_grid", "agn_cos_inc", 0.55,
                             {"agn_oa_nenkova": 35.0}),
    "skirtor_agnfitter": ("skirtor_agnfitter", "_find_skirtor_agnfitter_grid",
                          "load_skirtor_agnfitter_grid", "skirtor_agnfitter_sed_from_grid",
                          "agn_cos_inc", 0.55, {}),
    "fritz": ("fritz", "_find_fritz_grid", "load_fritz_grid", "fritz_sed_from_grid",
              "agn_fritz_psy", 41.3, {}),
    "silva04": ("silva04", "_find_silva04_grid", "load_silva04_grid", "silva04_sed_from_grid",
                "agn_log_nh_silva", 23.3, {}),
}  # fmt: skip

#: Libraries whose loader stores an anisotropy table.
ANISOTROPY = {
    "nenkova_agnfitter_3p": ("_find_nenkova_agnfitter_3p_grid", "load_nenkova_agnfitter_3p_grid"),
    "skirtor_agnfitter_1p": ("_find_skirtor_agnfitter_1p_grid", "load_skirtor_agnfitter_1p_grid"),
}


def table_digest(table) -> str:
    """SHA-256 of a table's little-endian float64 bytes: an exact, 64-character fingerprint."""
    data = np.ascontiguousarray(np.asarray(table), dtype="<f8")
    return hashlib.sha256(data.tobytes()).hexdigest()


def library_arrays(name):
    """The frozen arrays of one library (a ``KERNELS`` or ``ANISOTROPY`` key), in float64."""
    out = {}
    if name in ANISOTROPY:
        finder, loader = ANISOTROPY[name]
        mod = importlib.import_module(f"tengri.components.agn.{name}")
        grid = getattr(mod, loader)(getattr(mod, finder)())
        out[f"{name}/anisotropy"] = np.asarray(grid.anisotropy)
        return out
    mod_name, finder, loader, kernel, key, sight, extra = KERNELS[name]
    mod = importlib.import_module(f"tengri.components.agn.{mod_name}")
    grid = getattr(mod, loader)(getattr(mod, finder)())
    fn = getattr(mod, kernel)
    wave = jnp.asarray(WAVE)

    def sed(lbol, s):
        return fn(grid, wave, agn_log_lbol=lbol, agn_torus_frac=FRAC, **{key: s}, **extra)

    rows = []
    for lbol in LBOLS:
        a, b = jnp.asarray(lbol), jnp.asarray(sight)
        d_lbol, d_sight = jax.jacfwd(sed, (0, 1))(a, b)
        rows.append(np.stack([np.asarray(sed(a, b)), np.asarray(d_lbol), np.asarray(d_sight)]))
    out[f"{name}/sed_jac"] = np.stack(rows)
    if name == "fritz":
        # The six-axis table is 190 kB as float64: keep its digest, and one sightline
        # column for a readable diff.
        table = np.asarray(mod.fritz_anisotropy(grid).values)
        out["fritz/anisotropy_sha256"] = np.array(table_digest(table))
        out["fritz/anisotropy_column"] = table[0, 0, 0, 0, 0, :]
    return out


def reference_arrays():
    """Every frozen array, keyed ``<library>/<quantity>``."""
    return {k: v for name in (*KERNELS, *ANISOTROPY) for k, v in library_arrays(name).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sha", required=True, help="commit of the reference tree")
    args = parser.parse_args()
    arrays = reference_arrays()
    for key, value in arrays.items():
        assert key.endswith("_sha256") or value.dtype == np.float64, (key, value.dtype)
    np.savez_compressed(
        OUT,
        generated_on=np.array(args.sha),
        lbols=np.asarray(LBOLS),
        wave=WAVE,
        **arrays,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) from {args.sha}")


if __name__ == "__main__":
    main()

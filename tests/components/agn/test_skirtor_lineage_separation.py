# SPDX-License-Identifier: BSD-3-Clause
"""tengri ships two SKIRTOR lineages, and R64 changes exactly one of them.

The libraries, as they sit on disk:

* ``data/skirtor_templates_v3.h5`` -- the full 6-axis grid
  (tau, p, q, oa, R, cos_inc) with SEPARATE ``disk_emission`` and
  ``dust_emission`` and the per-cell ``spectra/norm`` beside them. Read by
  ``components/agn/skirtor.py``'s ``_load_grid_arrays``. This is the lineage
  whose stored inclination normalization R64 began applying, and it reaches
  ONE quantity: ``skirtor_disc_dust_ratio``'s ``R_faceon``.
* ``data/skirtor_mean{1,2,3}p_torus_grid.h5`` -- the averaged-clumpiness
  torus reductions packaged by AGNfitter-rX (Martinez-Ramirez et al. 2024).
  Dust-only, no disc component and no ``norm`` dataset at all; each is read by
  its own module (``skirtor_agnfitter{,_1p,_2p}.py``) through
  ``TorusTemplateGrid`` + ``torus_lnu_from_grid``, which never touch
  ``_load_grid_arrays``.
* ``data/skirtor_templates_v2.h5`` -- an older torus-only file with no
  ``disk_emission`` and no ``norm``; reachable only as the second entry of
  ``skirtor._GRID_SEARCH_PATHS``, i.e. when v3 is absent.

These tests pin the separation itself: the AGNfitter reductions' public
spellings reproduce their own files bit-for-bit, the files carry no
normalization to apply, and a component-split library without a ``norm``
leaves ``R_faceon`` where it was rather than inventing a scale.

References
----------
- Stalevski et al. 2012, MNRAS, 420, 2756; 2016, MNRAS, 458, 2288 (SKIRTOR)
- Martinez-Ramirez et al. 2024, A&A, 688, A46 (AGNfitter-rX)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: The grid the pins are evaluated on: spans the reductions' whole
#: 10 A - 1e8 A axis, so no comparison is made across a truncation.
_WAVE = jnp.asarray(np.geomspace(8.0, 1.0e8, 3000))

#: Every ``skirtor_agnfitter*`` torus spelling -> (HDF5 group, axis dataset
#: names in template-axis order, block kwargs at its fiducial). The kwargs are
#: spelled out rather than left to the registry defaults so a default change
#: cannot silently move the pin.
_AGNFITTER_BLOCKS = {
    "skirtor_agnfitter": (
        "skirtor_mean3p",
        ("oa_axis", "incl_axis", "tv_axis"),
        dict(
            agn_oa_skirtor=40.0,
            agn_incl_skirtor=30.0,
            agn_tv_skirtor=7.0,
            agn_torus_frac=0.5,
        ),
        ("agn_oa_skirtor", "agn_incl_skirtor", "agn_tv_skirtor"),
    ),
    "skirtor_agnfitter_1p": (
        "skirtor_mean1p",
        ("incl_axis",),
        dict(agn_incl_skirtor=30.0, agn_torus_frac=0.5),
        ("agn_incl_skirtor",),
    ),
    "skirtor_agnfitter_2p": (
        "skirtor_mean2p",
        ("oa_axis", "incl_axis"),
        dict(agn_oa_skirtor=40.0, agn_incl_skirtor=30.0, agn_torus_frac=0.5),
        ("agn_oa_skirtor", "agn_incl_skirtor"),
    ),
}


def _agnfitter_path_or_skip(block: str) -> str:
    import importlib

    module, finder = {
        "skirtor_agnfitter": ("skirtor_agnfitter", "_find_skirtor_agnfitter_grid"),
        "skirtor_agnfitter_1p": ("skirtor_agnfitter_1p", "_find_skirtor_agnfitter_1p_grid"),
        "skirtor_agnfitter_2p": ("skirtor_agnfitter_2p", "_find_skirtor_agnfitter_2p_grid"),
    }[block]
    mod = importlib.import_module(f"tengri.components.agn.{module}")
    try:
        return getattr(mod, finder)()
    except FileNotFoundError:
        pytest.skip(f"{block} grid not available")


@pytest.mark.parametrize("block", sorted(_AGNFITTER_BLOCKS))
def test_agnfitter_reduction_equals_its_own_file(block):
    """The public spelling reproduces its own HDF5, evaluated independently.

    The reference is re-derived from the grid file with ``h5py`` and the
    shared ``torus_lnu_from_grid`` evaluator -- not from a stored array and
    not through the block's cached loader -- so it is the FILE's content the
    block is held to. Any normalization leaking in from the CIGALE-lineage
    v3 grid (R64's ``norm(0)/norm(i)`` runs 1.0 to 3.62) shows up here at
    once.
    """
    h5py = pytest.importorskip("h5py")
    from tengri.components.agn._template_grid import TorusTemplateGrid, torus_lnu_from_grid
    from tengri.components.agn.blocks import resolve_agn_block
    from tengri.utils.physics_constants import C_AA

    group, axis_names, kwargs, coord_keys = _AGNFITTER_BLOCKS[block]
    path = _agnfitter_path_or_skip(block)

    with h5py.File(path, "r") as f:
        g = f[group]
        grid = TorusTemplateGrid(
            template=np.asarray(g["template"][:], dtype=np.float64),
            axes=tuple(np.asarray(g[name][:], dtype=np.float64) for name in axis_names),
            wave_grid=np.asarray(g["wavelength"][:], dtype=np.float64),
        )
        assert "norm" not in g, (
            f"{path} carries a 'norm' dataset: this file is an AGNfitter-rX "
            "reduction and is not supposed to have one, so the lineage map "
            "in this module's docstring is out of date"
        )

    # ``torus_lnu_from_grid`` returns L_nu [erg/s/Hz]; every registered torus
    # block converts to the runner's internal L_lambda with C/lambda^2 before
    # returning, so the reference gets the same conversion and nothing else.
    expected = (
        torus_lnu_from_grid(
            grid,
            _WAVE,
            tuple(kwargs[k] for k in coord_keys),
            agn_log_lbol=12.0,
            agn_torus_frac=kwargs["agn_torus_frac"],
        )
        * C_AA
        / _WAVE**2
    )
    got = resolve_agn_block("torus", block)(
        _WAVE, agn_log_lbol=12.0, l5100_disc=jnp.asarray(1.0e44), templates=None, **kwargs
    )
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        rtol=1e-12,
        atol=0.0,
        err_msg=(
            f"torus/{block} no longer reproduces {path}: the AGNfitter-rX "
            "reductions must be untouched by any change to the CIGALE-lineage "
            "skirtor_templates_v3.h5 loader"
        ),
    )


def test_agnfitter_reductions_do_not_read_the_cigale_lineage_loader():
    """Structural: their loaders open their own files and nothing else.

    ``_load_grid_arrays`` is the function R64 taught to read
    ``spectra/norm``. If an AGNfitter reduction ever routed through it, the
    numeric pin above would be the only thing standing between a loader
    change and three silently-moved torus libraries.
    """
    import inspect

    import tengri.components.agn.skirtor_agnfitter as sa3
    import tengri.components.agn.skirtor_agnfitter_1p as sa1
    import tengri.components.agn.skirtor_agnfitter_2p as sa2

    for module in (sa1, sa2, sa3):
        source = inspect.getsource(module)
        assert "_load_grid_arrays" not in source, (
            f"{module.__name__} references _load_grid_arrays, the "
            "CIGALE-lineage v3 reader: the two SKIRTOR lineages are no longer "
            "separate and R64's inclination normalization can reach an "
            "inclination-AVERAGED library that has none"
        )
        assert "skirtor_templates_v" not in source, (
            f"{module.__name__} names a skirtor_templates_v* file; the "
            "AGNfitter-rX reductions live in skirtor_mean*p_torus_grid.h5"
        )


def test_v2_grid_carries_no_component_split_to_normalize():
    """The v3 fallback is torus-only, so ``R_faceon`` never reaches it.

    ``_GRID_SEARCH_PATHS`` falls back to ``skirtor_templates_v2.h5`` when v3
    is absent. That file has ``torus_emission`` only -- no
    ``disk_emission``, no ``dust_emission``, no ``norm`` -- so
    ``_load_raw_disk_dust_grid`` returns ``None`` for it and
    ``skirtor_disc_dust_ratio`` takes its documented unity-ratio fallback
    rather than a factor read off a file that has none.
    """
    h5py = pytest.importorskip("h5py")
    from tengri._data_setup import find_data

    path = find_data("skirtor_templates_v2.h5")
    if path is None:
        pytest.skip("skirtor_templates_v2.h5 not available")
    with h5py.File(path, "r") as f:
        assert "spectra/torus_emission" in f
        for absent in ("spectra/disk_emission", "spectra/dust_emission", "spectra/norm"):
            assert absent not in f, (
                f"{path} now carries {absent}; the v2 fallback was torus-only, "
                "and a component split without a norm would silently take "
                "skirtor_disc_dust_ratio's unity-ratio path"
            )


def test_component_split_without_norm_leaves_r_faceon_unnormalized():
    """A library that splits disk/dust but carries no ``norm`` degrades openly.

    Threading the 4-field ``(disk, dust, wave_grid, axes)`` tuple that
    :class:`~tengri.components.agn.skirtor.SkirtorDiscDustGrid` replaced --
    the shape a caller built before ``norm`` was read -- must reproduce the
    pre-R64 ``int(disk0)/int(dust)`` exactly, not a normalization the code
    invented for it. And a threaded object of neither shape must raise rather
    than unpack into the wrong fields.
    """
    from tengri.components.agn.blocks import resolve_agn_block
    from tengri.components.agn.skirtor import (
        _load_raw_disk_dust_grid,
        skirtor_disc_dust_ratio,
    )

    grid = _load_raw_disk_dust_grid()
    if grid is None:
        pytest.skip("raw SKIRTOR disk/dust grid not available")
    if grid.norm is None:
        pytest.skip("repackaged SKIRTOR grid carries no spectra/norm")

    disc = jnp.asarray(
        resolve_agn_block("disc", "schartmann2005")(_WAVE, agn_log_lbol=12.0, templates=None)
    )
    fiducial = dict(
        agn_tau_skirtor=7.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_oa_skirtor=40.0,
        agn_radius_ratio=20.0,
        agn_cos_inc=float(np.cos(np.deg2rad(30.0))),
    )
    with_norm = float(
        skirtor_disc_dust_ratio(
            _WAVE, disc, jnp.ones_like(_WAVE), _template=grid, **fiducial
        ).R_faceon
    )
    legacy = (grid.disk, grid.dust, grid.wave_grid, grid.axes)
    without_norm = float(
        skirtor_disc_dust_ratio(
            _WAVE, disc, jnp.ones_like(_WAVE), _template=legacy, **fiducial
        ).R_faceon
    )
    assert with_norm / without_norm == pytest.approx(1.029133714, rel=1e-6, abs=0.0), (
        f"R_faceon with norm {with_norm:.9f} against without {without_norm:.9f}: the "
        "ratio is not the file's own norm(0)/norm(i=30) = 1.029133714, so the "
        "no-norm path is not the pre-R64 quantity it claims to be"
    )
    with pytest.raises(TypeError, match="SkirtorDiscDustGrid"):
        skirtor_disc_dust_ratio(
            _WAVE, disc, jnp.ones_like(_WAVE), _template=(grid.disk, grid.dust), **fiducial
        )

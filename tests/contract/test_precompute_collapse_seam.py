# SPDX-License-Identifier: BSD-3-Clause
"""The one auto-collapse seam shared by every ``*_precompute.py`` (#1738).

``collapse_fixed_axes`` replaced a byte-identical block copied into eleven
precompute modules. The duplication is the point: the two invariants tested
here — that a declared axis name is a parameter that can actually exist, and
that the declaration's length matches the grid it indexes — had no single place
to live while the code was eleven copies, and neither was ever checked.

Sibling: ``test_precompute_axis_collapse.py`` drives the real adapters and
asserts the collapsed lookup reproduces the uncollapsed one. This file tests
the shared seam itself, on synthetic grids, so it needs no template files.

Covers:
- a Fixed axis is collapsed, a free axis is not;
- ``defaults`` collapse axes the model never declared (qsogen/GRAHSP path);
- axis names that no ``Parameters`` can contain warn instead of silently
  doing nothing;
- a declaration that disagrees with its grid refuses to collapse rather than
  contracting the wrong dimension.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import DeadPrecomputeAxisWarning
from tengri.forward.precompute.templates import collapse_fixed_axes
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform
from tengri.utils.grid_interp import PreintegratedGrid
from tengri.utils.interpolation import edges_for_grid

pytestmark = pytest.mark.contract

#: Two parameters that a default ``Parameters`` declares and holds Fixed.
AX0, AX1 = "dust_Rv", "dust_delta"

N0, N1, N_FILT = 5, 4, 3


def _grid(n_axes: int = 2, *, phot_grid_dims: int | None = None) -> PreintegratedGrid:
    """A synthetic preintegrated grid; no template files, so this runs in CI.

    ``phot_grid_dims`` defaults to ``n_axes``. Passing fewer reproduces the
    shape ``cb19_precompute`` ships, where the declared axis count exceeds the
    photometry array's grid rank.
    """
    sizes = [N0, N1, 3, 3, 3][:n_axes]
    axes = tuple(np.linspace(1.0, 2.0, n) for n in sizes)
    if phot_grid_dims is None:
        phot_grid_dims = n_axes
    phot = np.arange(
        np.prod(sizes[:phot_grid_dims], dtype=int) * N_FILT, dtype=np.float64
    ).reshape(*sizes[:phot_grid_dims], N_FILT)
    return PreintegratedGrid(
        phot=jnp.asarray(phot),
        moment=None,
        axes=tuple(jnp.asarray(a) for a in axes),
        edges=tuple(edges_for_grid(a) for a in axes),
        effective_wavelengths=jnp.linspace(4000.0, 8000.0, N_FILT),
        effective_wavelengths_rest=jnp.linspace(4000.0, 8000.0, N_FILT),
        # log10 of the linear scale this used to pass as ``flux_scale=1.0``.
        # #1859/#1878 renamed the field because the linear form has no
        # representable float32 value; log10(1.0) = 0.0 keeps the fixture's
        # meaning (an unscaled grid) exactly.
        log10_flux_scale=0.0,
        n_filters=N_FILT,
    )


def test_fixed_axis_is_collapsed_and_free_axis_is_not():
    """The core contract: a Fixed parameter's axis is interpolated away."""
    spec = Parameters(**{AX1: Uniform(-1.0, 1.0)})
    assert AX0 in spec.get_fixed_values(), "fixture assumes dust_Rv is Fixed by default"
    assert AX1 in spec.free_params, "fixture assumes dust_delta was made free"

    out, remaining, collapsed = collapse_fixed_axes(_grid(), (AX0, AX1), spec)

    assert set(collapsed) == {0}, f"expected only axis 0 to collapse, got {collapsed}"
    assert len(remaining) == 1, "the free axis must survive"
    assert out.phot.shape == (N1, N_FILT), (
        f"collapsing axis 0 of a {(N0, N1, N_FILT)} grid must give {(N1, N_FILT)}, "
        f"got {out.phot.shape}"
    )


def test_nothing_collapses_when_every_axis_is_free():
    """All-free axes are an ordinary configuration, not a defect: stay silent."""
    spec = Parameters(**{AX0: Uniform(2.0, 4.0), AX1: Uniform(-1.0, 1.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeadPrecomputeAxisWarning)
        out, remaining, collapsed = collapse_fixed_axes(_grid(), (AX0, AX1), spec)

    assert collapsed == {}, "no Fixed axis means no collapse"
    assert len(remaining) == 2
    assert out.phot.shape == (N0, N1, N_FILT), "the grid must be returned untouched"


def test_unresolvable_axis_names_warn_instead_of_falling_silent():
    """The #1738 failure: names no Parameters can hold, collapsing nothing.

    Six of the eleven modules shipped exactly this — ``cat3d_precompute``
    declares ``cat3d_cos_inc`` while the component declares ``parameter_prefix
    = "agn_"`` and ``cos_inc``, so the live name is ``agn_cos_inc`` — and their
    advertised auto-collapse had never once fired, with nothing raised.
    """
    spec = Parameters()
    names = ("cat3d_cos_inc", "cat3d_a")
    assert set(names).isdisjoint(spec.valid_param_names), "fixture assumes dead names"

    with pytest.warns(DeadPrecomputeAxisWarning, match="cat3d_cos_inc"):
        out, _remaining, collapsed = collapse_fixed_axes(
            _grid(), names, spec, origin="cat3d_precompute"
        )

    assert collapsed == {}
    assert out.phot.shape == (N0, N1, N_FILT), "warning must not change the result"


def test_defaults_collapse_axes_the_model_never_declared():
    """qsogen/GRAHSP path: a name absent from the spec collapses at its default.

    These modules pass ``defaults``, so their axes do collapse even though the
    names are not model parameters — which is why the dead-name warning is
    conditioned on *nothing having collapsed*, not on the names alone.
    """
    spec = Parameters()
    names = ("agn_plslp1", "agn_ebv")
    defaults = {"agn_plslp1": 1.5, "agn_ebv": 1.2}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeadPrecomputeAxisWarning)
        out, remaining, collapsed = collapse_fixed_axes(
            _grid(), names, spec, all_params=defaults, origin="qsogen_precompute"
        )

    assert set(collapsed) == {0, 1}, f"both axes should collapse at defaults, got {collapsed}"
    assert remaining == ()
    assert out.phot.shape == (N_FILT,)


def test_axis_count_disagreeing_with_the_grid_refuses_to_collapse():
    """A positional index into the wrong number of axes must not proceed."""
    spec = Parameters()
    with pytest.raises(ValueError, match="declares 3 axes"):
        collapse_fixed_axes(_grid(n_axes=2), (AX0, AX1, "dust_slope"), spec)


def test_grid_rank_disagreeing_with_axis_count_refuses_to_collapse():
    """cb19's shape: 7 declared axes over a photometry array with 6 grid dims.

    Collapsing the last declared axis there would contract the *filter*
    dimension and return a silently wrong SED, so this refuses.
    """
    spec = Parameters()
    with pytest.raises(ValueError, match="grid dimensions"):
        collapse_fixed_axes(_grid(n_axes=2, phot_grid_dims=1), (AX0, AX1), spec)


def test_no_parameters_object_is_a_silent_no_op():
    """``parameters=None`` cannot know what is Fixed, so it collapses nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeadPrecomputeAxisWarning)
        out, remaining, collapsed = collapse_fixed_axes(_grid(), (AX0, AX1), None)

    assert collapsed == {}
    assert len(remaining) == 2
    assert out.phot.shape == (N0, N1, N_FILT)

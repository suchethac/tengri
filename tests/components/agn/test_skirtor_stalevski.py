# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the ``skirtor_stalevski`` AGN model returns the raw SKIRTOR SED (#756).

The monolithic ``skirtor`` model pairs the SKIRTOR torus with a *power-law* disc,
under-representing the UV/optical disc (~0.28x ProSpect's SKIRTOR_interp at
2000 A). ``skirtor_stalevski`` instead returns the raw Stalevski (2016)
radiative-transfer SED (``create_skirtor_components.total``) — the disc + torus
as published, with no analytic-disc substitution or re-normalization — which
reproduces codes that read SKIRTOR directly (~0.96x ProSpect). These tests pin:

1. the model is registered and swappable;
2. it equals the raw ``create_skirtor_components.total`` (faithful template);
3. its 2000 A disc is far brighter than the power-law ``skirtor`` model.
"""

from __future__ import annotations

import chex
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_C_AA_PER_S = 2.99792458e18


def _grid_or_skip():
    from tengri.components.agn.skirtor import _find_skirtor_grid

    path = _find_skirtor_grid()
    if path is None:
        pytest.skip("SKIRTOR grid not available (data-gated)")
    return path


def test_skirtor_stalevski_registered():
    import warnings

    from tengri.components.agn import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("skirtor_stalevski")
    assert callable(fn)


def test_skirtor_stalevski_is_raw_total():
    """The model SED equals the faithful raw-Stalevski total interpolator.

    Prefers the v4 grid (published RT total); falls back to the v3 component
    ``.total`` if v4 is absent — matching the loader's own fallback.
    """
    _grid_or_skip()
    from tengri.components.agn import resolve_agn_model
    from tengri.components.agn.skirtor import (
        _find_skirtor_grid,
        _find_skirtor_raw_grid,
        create_skirtor_components_from_grid,
        create_skirtor_raw_total_from_grid,
    )

    wave = np.geomspace(1e3, 1e7, 3000)
    kw = dict(
        agn_log_lbol=10.4,
        agn_tau_skirtor=7.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_oa_skirtor=40.0,
        agn_cos_inc=float(np.cos(np.radians(30.0))),
    )
    sed = resolve_agn_model("skirtor_stalevski")(wave, agn_frac=1.0, **kw)
    v4 = _find_skirtor_raw_grid()
    if v4 is not None:
        ref = create_skirtor_raw_total_from_grid(v4)(wave, frac_agn=1.0, **kw)
    else:
        ref = create_skirtor_components_from_grid(_find_skirtor_grid())(
            wave, frac_agn=1.0, **kw
        ).total
    chex.assert_trees_all_close(np.asarray(sed), np.asarray(ref), rtol=1e-5, atol=0.0)


def test_skirtor_stalevski_disc_brighter_than_powerlaw():
    """The raw disc is far brighter at 2000 A than the power-law ``skirtor`` (#756)."""
    _grid_or_skip()
    from tengri.components.agn import resolve_agn_model

    wave = np.geomspace(1e3, 1e7, 3000)
    kw = dict(
        agn_log_lbol=10.4,
        agn_frac=1.0,
        agn_tau_skirtor=7.0,
        agn_oa_skirtor=40.0,
        agn_cos_inc=float(np.cos(np.radians(30.0))),
    )
    raw = np.asarray(resolve_agn_model("skirtor_stalevski")(wave, **kw))
    powerlaw = np.asarray(resolve_agn_model("skirtor")(wave, **kw))
    disc_raw = float(np.interp(2000.0, wave, raw)) * _C_AA_PER_S / 2000.0
    disc_pl = float(np.interp(2000.0, wave, powerlaw)) * _C_AA_PER_S / 2000.0
    # Raw Stalevski disc reads ~0.96x ProSpect vs ~0.28x for the power-law disc:
    # the raw disc must be at least ~2x brighter at the 2000 A disc continuum.
    assert disc_raw > 2.0 * disc_pl, f"raw disc {disc_raw:.2e} not >> power-law {disc_pl:.2e}"

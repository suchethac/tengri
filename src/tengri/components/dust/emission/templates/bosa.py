# SPDX-License-Identifier: BSD-3-Clause
"""Boquien & Salim BOSA dust emission template as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed

__all__ = ["BosaIRSEDComponent"]


class BosaIRSEDComponent(EmissionComponent):
    """Boquien & Salim (2021) BOSA dust IR emission template.

    Wraps the pure closure from the tabulated BOSA template library,
    parameterized by specific star-formation rate (sSFR) instead of
    radiation field parameters.

    The model interpolates in (log L_TIR, log sSFR) space, where L_TIR
    is derived from the absorbed luminosity via energy balance. The free
    parameter is just log sSFR; L_TIR is computed internally. Because
    :attr:`factors_l_ir` is False, the template **shape** tracks the fitted
    L_TIR directly -- not just the overall normalization: ``apply()`` always
    hands ``predict()`` the real absorbed luminosity, so the (log L_TIR,
    log sSFR) grid lookup sees the real budget and selects the
    luminosity-appropriate row, across the packaged grid's full
    log10(L_TIR/Lsun) = 8.5-12.5 span (Boquien & Salim 2021) -- an
    astrophysically realistic galaxy's L_ir (~1e42-1e45 erg/s) lands well
    inside that span once converted to Lsun (#2272). A budget outside the
    span clips to the nearest edge template, same as any other boundary
    behavior on this grid.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Boquien, M. & Salim, S., 2021, "A new approach to estimate
       dust temperatures, masses, and emissivities from far-infrared SED",
       A&A, 653, A149. https://doi.org/10.1051/0004-6361/202140810

    """

    name: str = "bosa"

    # Free parameters (user-facing names, prefix-stripped)
    log_ssfr = Fixed(-10.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("boquien_salim2021",)

    accepts_threaded_templates: ClassVar[bool] = True

    #: BOSA's template SHAPE is looked up on the (log L_TIR, log sSFR) grid
    #: by ``log10(L_ir)`` (see ``create_bosa_from_grid``/``bosa_emission`` in
    #: ``components/dust/emission_templates.py``), so it is not merely scaled
    #: by ``L_ir`` -- it is a genuine function of it. The generic apply()-level
    #: shortcut this opts out of evaluates ``predict()`` once at the unit
    #: luminosity ``L_ir = 1`` and rescales the result in log space (valid
    #: only for a model proportional to ``L_ir``, see
    #: ``tests/contract/test_dust_emission_l_ir_linearity.py``); for BOSA that
    #: would pin the shape lookup at ``log10(1) = 0`` (clipped to the grid's
    #: lowest node) regardless of the real budget, rescaling only the
    #: amplitude afterwards -- correct total power, wrong shape (#2272).
    factors_l_ir: ClassVar[bool] = False

    #: With ``factors_l_ir=False``, ``apply()`` now hands ``predict()`` the
    #: REAL linear ``L_ir`` (~1e43 erg/s, ``inf`` in pure float32) instead of
    #: the unit-luminosity placeholder. ``log_L_ir`` [dex] is the float32-safe
    #: form (published by the attenuator before the overflow-prone linear
    #: cast), used for both the grid-axis lookup and the normalization --
    #: mirrors ``EnergyBalanceSplitIRSEDComponent``, which declares the same
    #: pair for the same reason.
    optional_inputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s", "log_L_ir": "dex"}

    def load(self, wave: jnp.ndarray | None = None):
        """Load the BOSA template dict so it can be threaded, not baked.

        Returns
        -------
        dict or None
            Template arrays, already normalized: ``_normalize_bosa_grid`` is
            a preprocessing step that must not run on traced arrays. ``None``
            when unavailable; the backend then falls back to its module-level
            load, which bakes 4.45 MB (#1649).

        Notes
        -----
        **JIT-compatible**: no, deliberately; runs at build time.
        """
        del wave
        from tengri._data_setup import find_data_str
        from tengri.components.dust.emission_templates import (
            _normalize_bosa_grid,
            load_bosa_templates,
        )

        for fname in ("bosa_templates_v2.h5", "bosa_templates.h5"):
            path = find_data_str(fname)
            if path is None:
                continue
            grid = load_bosa_templates(path)
            if "wavelength_um" in grid and "wavelength_aa" not in grid:
                grid = _normalize_bosa_grid(grid)
            return grid
        return None

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
        log_L_ir: float | None = None,
        templates=None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute BOSA dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: key is "log_ssfr" (or subset if Fixed).
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for a dust emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir : float
            Total absorbed luminosity in erg/s.
        log_L_ir : float, optional
            ``log10(L_ir / (erg/s))`` [dex]. When available (the normal path
            through ``apply()``), used in place of ``log10(L_ir)`` for the
            normalization (float32-safe: ``L_ir`` itself overflows to ``inf``
            in pure float32) and, after converting to the grid's own
            log10(L_TIR/Lsun) axis, for the grid-shape lookup too -- this is
            what lets the shape track the real budget at astrophysical scales
            (#2272). ``None`` preserves the original linear-only, unconverted
            formula for direct callers (e.g. the bit-exact golden-fixture
            regression).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where sed_out is the updated SED and published
            contains {"sed_dust_ir": emission SED in erg/s/Hz}.

        """
        if templates is not None:
            # Closure built over the THREADED arrays: capture of a tracer is
            # fine, capture of a concrete array is what bakes (#1649).
            from tengri.components.dust.emission_templates import create_bosa_from_grid

            sed = create_bosa_from_grid(templates)(
                wave, L_ir, dust_log_ssfr=p["log_ssfr"], log_L_ir=log_L_ir
            )
        else:
            from tengri.components.dust.emission import bosa as bosa_fn

            sed = bosa_fn(wave, L_ir, dust_log_ssfr=p["log_ssfr"], log_L_ir=log_L_ir)
        return sed_in + sed, {"sed_dust_ir": sed}

# SPDX-License-Identifier: BSD-3-Clause
"""DH02_CE01 cold-dust emission library as a SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.

Without this class the grammar type ``dh02_ce01`` resolved on the exploration
path (``model.predict``, which dispatches through the legacy
``DUST_EMISSION_MODELS`` dict) and raised on the inference path
(``model.predict_photometry``, which resolves the ``_REGISTRY`` component):
so an advertised model could be looked at but never fitted (#1777).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent

__all__ = ["DH02CE01IRSEDComponent"]


class DH02CE01IRSEDComponent(EmissionComponent):
    r"""Dale & Helou (2002) + Chary & Elbaz (2001) cold-dust template library.

    The library published with AGNfitter-rX is a **single-axis** grid indexed
    by :math:`\log_{10}(L_{\rm IR}/L_\odot)`. Templates are linearly
    interpolated along that axis to pick the emission *shape*, and the result
    is renormalized by frequency integral so the emitted power equals the
    absorbed power:

    .. math::

        L_\nu(\lambda) = \frac{L_{\rm abs}}{\int T_\nu \, d\nu} \, T_\nu(\lambda)

    where :math:`T_\nu` is the interpolated template [arbitrary units],
    :math:`L_{\rm abs}` is the dust-absorbed luminosity [erg/s] supplied by the
    energy-balance step, and :math:`L_\nu` is the emitted spectral luminosity
    density [erg/s/Hz].

    This component declares **no free parameters**, which is deliberate and
    matches the statement in ``components/grid_support.py``: the library's only
    grid axis is :math:`L_{\rm TIR}`, "derived from L_absorbed by energy
    balance rather than set by the user, so no prior can overhang it". The
    template *shape* tracks the realized luminosity (Dale & Helou 2002): the
    closure receives the actual ``L_ir`` via ``optional_inputs``, converts it
    to the grid's ``log10(L_TIR/L_sun)`` axis, and interpolates the appropriate
    row. The normalization ensures the emitted power equals the absorbed power
    regardless of shape selection.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable via linear interpolation.

    **Template shape tracks L_ir**: the normalized SED shape depends on the
    realized infrared luminosity (the whole point of the Dale & Helou library —
    warmer, broader templates at higher L_IR). Before #2366, the shape was
    pinned to a hardcoded default; it now tracks the fitted budget via the
    ``log_L_ir`` input. Fits to existing data will report different
    ``dust_emission`` SEDs (same total power, different shape), so the
    component output changes for every model using dh02_ce01 (#2366).

    Implements the same template library as AGNfitter-rX
    (Martínez-Ramírez et al. 2024 [3]_); the grid data are repackaged from that
    release by ``scripts/build_dh02_ce01_grid.py`` and cross-validated in
    ``tests/crossval/test_dh02_ce01_vs_agnfitter.py``.

    References
    ----------
    .. [1] Dale, D. A. & Helou, G., 2002, "The Infrared Spectral Energy
       Distribution of Normal Star-forming Galaxies: Calibration at Far-Infrared
       and Submillimeter Wavelengths", ApJ, 576, 159.
       https://doi.org/10.1086/341632
    .. [2] Chary, R. & Elbaz, D., 2001, "Interpreting the Cosmic Infrared
       Background: Constraints on the Evolution of the Dust-enshrouded Star
       Formation Rate", ApJ, 556, 562. https://doi.org/10.1086/321609
    .. [3] Martínez-Ramírez, L. N., et al., 2024, "AGNfitter-rx: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs", A&A, 688, A46.
       https://doi.org/10.1051/0004-6361/202449329
    """

    name: str = "dh02_ce01"

    #: Stated, not inferred. The library's only grid axis is L_TIR, derived from
    #: L_absorbed by energy balance rather than set by the user, so this engine
    #: genuinely reads no parameter -- as the class docstring above says. Saying
    #: so lets ``dust_emission={'type': 'dh02_ce01', 'all_params': FREE}``
    #: narrow to nothing instead of freeing the whole static union, which is 19
    #: dimensions a sampler cannot move (#1482).
    declares_no_parameters: ClassVar[bool] = True

    _citations_tuple: ClassVar[tuple[str, ...]] = (
        "dale_helou2002",
        "chary_elbaz2001",
    )

    accepts_threaded_templates: ClassVar[bool] = True

    #: DH02_CE01's template SHAPE is looked up on the grid by ``log10(L_ir)``
    #: (see ``create_dh02_ce01`` in ``components/dust/emission_templates.py``),
    #: so it is not merely scaled by ``L_ir`` -- it is a genuine function of
    #: the realized luminosity. The generic apply()-level shortcut this opts
    #: out of evaluates ``predict()`` once at unit luminosity and rescales in
    #: log space (valid only for SED ∝ L_ir, see
    #: ``tests/contract/test_dust_emission_l_ir_linearity.py``); for DH02 that
    #: would pin the shape lookup at ``log10(1) = 0`` (clipped to the grid's
    #: minimum node, 8.3) regardless of the real budget, rescaling only the
    #: amplitude afterwards -- correct total power, wrong shape (#2366).
    factors_l_ir: ClassVar[bool] = False

    #: With ``factors_l_ir=False``, ``apply()`` now hands ``predict()`` the
    #: REAL linear ``L_ir`` (~1e43 erg/s, ``inf`` in pure float32) instead of
    #: unit-luminosity placeholder. ``log_L_ir`` [dex] is the float32-safe form
    #: (published by the attenuator before the overflow-prone linear cast),
    #: used for both the grid-axis lookup and the normalization -- mirrors
    #: ``BosaIRSEDComponent``, which declares the same pair for the same reason.
    optional_inputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s", "log_L_ir": "dex"}

    def load(self, wave: jnp.ndarray | None = None):
        """Load the grid so it can be threaded as an argument, not baked.

        Returns
        -------
        dict or None
            Template arrays from :func:`load_dh02_ce01_lnu_grid`, or ``None``
            when the HDF5 file is absent: the backend then falls back to its
            module-level load.

        Notes
        -----
        **JIT-compatible**: no, deliberately; runs at build time.
        """
        del wave
        from tengri._data_setup import find_data_str
        from tengri.components.dust.emission_templates import load_dh02_ce01_lnu_grid

        path = find_data_str("dh02_ce01_grid.h5")
        return None if path is None else load_dh02_ce01_lnu_grid(path)

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
        """Compute DH02_CE01 cold-dust emission.

        Parameters
        ----------
        p : dict
            Parameters with the ``dust_`` prefix stripped. Empty: this model
            declares none; see the class Notes.
        sed_in : array_like, shape (n_wave,)
            SED from upstream [erg/s/Hz]; zeros when this is the first
            emission component.
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        L_ir : float
            Dust-absorbed luminosity to re-radiate [erg/s].
        log_L_ir : float, optional
            ``log10(L_ir / (erg/s))`` [dex]. When available (the normal path
            through ``apply()``), used in place of ``log10(L_ir)`` for the
            grid-shape lookup after converting to the grid's own
            log10(L_TIR/Lsun) axis -- this is what lets the shape track the
            real budget at astrophysical scales (#2366). ``None`` preserves the
            original linear-only, unconverted formula for direct callers.
        templates : dict, optional
            Grid threaded in as a traced argument. When ``None`` the
            module-level lazy loader is used, which captures the library as a
            compile-time constant (1.32 MB, #1649).

        Returns
        -------
        tuple of (ndarray, dict)
            ``(sed_out, published)``: ``sed_out`` has shape ``(n_wave,)``
            [erg/s/Hz], and ``published`` carries ``"sed_dust_ir"`` [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes.
        """
        del p
        if templates is not None:
            # Closure built over the THREADED arrays: capturing a tracer is
            # fine, capturing a concrete array is what bakes (#1649).
            from tengri.components.dust.emission_templates import create_dh02_ce01_from_grid

            sed = create_dh02_ce01_from_grid(templates)(wave, L_ir, log_L_ir=log_L_ir)
        else:
            from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS

            sed = DUST_EMISSION_MODELS["dh02_ce01"](wave, L_ir, log_L_ir=log_L_ir)
        return sed_in + sed, {"sed_dust_ir": sed}

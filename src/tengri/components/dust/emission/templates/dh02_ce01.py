# SPDX-License-Identifier: BSD-3-Clause
"""DH02_CE01 cold-dust emission library as a SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.

Without this class the grammar type ``dh02_ce01`` resolved on the exploration
path (``model.predict``, which dispatches through the legacy
``DUST_EMISSION_MODELS`` dict) and raised on the inference path
(``model.predict_photometry``, which resolves the ``_REGISTRY`` component) —
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
    closure's ``dust_log_lir`` therefore keeps its own default of 10.0, exactly
    as the legacy dispatch has always called it — enabling the registry path
    changes no number.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable via linear interpolation.

    **Known limitation, not introduced here**: the *shape* selected is
    always the :math:`\log_{10}(L_{\rm IR}/L_\odot) = 10` template, because the
    closure's ``dust_log_lir`` default is a constant and nothing derives it
    from ``L_ir``. The normalization does use ``L_ir``, so the emitted power is
    right while the template shape does not track luminosity. AGNfitter-rX
    fits ``irlum`` as a free parameter. Reconciling the two is a physics
    change with a moved number, so it is reported rather than folded into the
    wiring fix that gave this model an inference path at all.

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
    .. [3] Martínez-Ramírez, L. N., et al., 2024, "AGNfitter-rX: Modeling the
       radio-to-X-ray SED of AGN", A&A, 688, A46.
       https://doi.org/10.1051/0004-6361/202449329
    """

    name: str = "dh02_ce01"

    _citations_tuple: ClassVar[tuple[str, ...]] = (
        "dale_helou2002",
        "chary_elbaz2001",
    )

    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None):
        """Load the grid so it can be threaded as an argument, not baked.

        Returns
        -------
        dict or None
            Template arrays from :func:`load_dh02_ce01_lnu_grid`, or ``None``
            when the HDF5 file is absent — the backend then falls back to its
            module-level load.

        Notes
        -----
        **JIT-compatible**: no, deliberately — runs at build time.
        """
        del wave
        from tengri.components.dust.emission.emission import _find_data_file
        from tengri.components.dust.emission_templates import load_dh02_ce01_lnu_grid

        path = _find_data_file("dh02_ce01_grid.h5")
        return None if path is None else load_dh02_ce01_lnu_grid(path)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
        templates=None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute DH02_CE01 cold-dust emission.

        Parameters
        ----------
        p : dict
            Parameters with the ``dust_`` prefix stripped. Empty — this model
            declares none; see the class Notes.
        sed_in : array_like, shape (n_wave,)
            SED from upstream [erg/s/Hz]; zeros when this is the first
            emission component.
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        L_ir : float
            Dust-absorbed luminosity to re-radiate [erg/s].
        templates : dict, optional
            Grid threaded in as a traced argument. When ``None`` the
            module-level lazy loader is used, which captures the library as a
            compile-time constant (1.32 MB, #1649).

        Returns
        -------
        tuple of (ndarray, dict)
            ``(sed_out, published)`` — ``sed_out`` has shape ``(n_wave,)``
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

            sed = create_dh02_ce01_from_grid(templates)(wave, L_ir)
        else:
            from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS

            sed = DUST_EMISSION_MODELS["dh02_ce01"](wave, L_ir)
        return sed_in + sed, {"sed_dust_ir": sed}

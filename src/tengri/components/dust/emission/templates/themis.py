# SPDX-License-Identifier: BSD-3-Clause
"""Jones et al. THEMIS dust emission template as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed

__all__ = ["ThemisIRSEDComponent"]


class ThemisIRSEDComponent(EmissionComponent):
    """Jones et al. (2017) THEMIS/DustEM dust IR emission template.

    Wraps the pure closure from the tabulated THEMIS/DustEM template library.
    Uses the same Draine & Li (2007) mixing formula but with the THEMIS grain
    composition.

    The model mixes single-U (diffuse) and power-law (PDR) components via
    the power-law fraction, with aromatic carbon fraction (qhac) controlling
    PAH-like features. Supports both bilinear interpolation (2D: qhac, umin)
    and trilinear interpolation (3D: qhac, umin, alpha) depending on template
    availability.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Jones, A. P., Ysard, N., Köhler, M., et al., 2017,
       "The THEMIS model: A review", A&A, 602, A46.
       https://doi.org/10.1051/0004-6361/201628997

    """

    name: str = "themis"

    # Free parameters (user-facing names, prefix-stripped)
    umin = Fixed(1.0)
    gamma_dl = Fixed(0.01)
    qhac = Fixed(0.17)
    alpha = Fixed(2.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("jones2017",)

    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None):
        """Load the THEMIS template dict so it can be threaded, not baked.

        Returns
        -------
        dict or None
            Template arrays with the two non-traceable preparation steps
            already applied, or ``None`` when unavailable — the backend then
            falls back to its module-level load, which bakes 39.4 MB.

        Notes
        -----
        **JIT-compatible**: no, deliberately — runs at build time so the arrays
        reach ``predict`` as a traced argument (#1649).
        """
        del wave
        from tengri.components.dust.emission.emission import _find_data_file
        from tengri.components.dust.emission_templates import (
            _normalize_dl07_like_grid,
            _qhac_axis_to_cigale,
            load_themis_templates,
        )

        for fname in ("themis_templates_v2.h5", "themis_templates.h5"):
            path = _find_data_file(fname)
            if path is None:
                continue
            grid = dict(load_themis_templates(path))
            # Both preparation steps read concrete values — a key census, then
            # a concrete max to pick the qhac unit convention — so neither can
            # run once these arrays are traced. Do them here, eagerly.
            if "spectra_single" in grid and "single_u" not in grid:
                grid = _normalize_dl07_like_grid(grid, q_key="qhac_grid")
            grid["qhac_grid_cigale"] = _qhac_axis_to_cigale(grid["qhac_grid"])
            return grid
        return None

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
        templates=None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute THEMIS dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "umin", "gamma_dl", "qhac",
            "alpha" (or subset if some are Fixed).
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for a dust emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir : float
            Total absorbed luminosity in erg/s.

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where sed_out is the updated SED and published
            contains {"sed_dust_ir": emission SED in erg/s/Hz}.

        """
        kwargs = dict(
            dust_umin=p["umin"],
            dust_gamma_dl=p["gamma_dl"],
            dust_qhac=p["qhac"],
            dust_alpha=p["alpha"],
        )
        if templates is not None:
            # Build the interpolation closure over the THREADED arrays. Closing
            # over tracers inside the current trace is fine; what bakes is a
            # closure over *concrete* arrays, which is what the module-level
            # lazy loader gives here (#1649).
            from tengri.components.dust.emission_templates import create_themis_from_grid

            sed = create_themis_from_grid(templates)(wave, L_ir, **kwargs)
        else:
            from tengri.components.dust.emission import themis as themis_fn

            sed = themis_fn(wave, L_ir, **kwargs)
        return sed_in + sed, {"sed_dust_ir": sed}

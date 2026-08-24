# SPDX-License-Identifier: BSD-3-Clause
"""Schreiber et al. (2018) dust emission template as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed

__all__ = ["Schreiber2018IRSEDComponent"]


class Schreiber2018IRSEDComponent(EmissionComponent):
    """Schreiber et al. (2018) dust IR emission template.

    Wraps the pure closure from the tabulated Schreiber et al. (2018) template
    library, parameterized by dust temperature (T_dust) and PAH fraction (f_pah).

    The model interpolates linearly in the T_dust grid and linearly mixes
    continuum and PAH components.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Schreiber, C., Pannella, M., Elbaz, D., et al., 2018,
       "The ALMA Spectroscopic Survey in the Hubble Ultra Deep Field:
       The molecular gas content of galaxies and tension with
       IllustrisTNG and the Santa Cruz Simulations",
       A&A, 609, A30. https://doi.org/10.1051/0004-6361/201731506

    """

    name: str = "schreiber2018"

    # Free parameters (user-facing names, prefix-stripped). Canonical names
    # (#849): ``dust_T`` (was ``dust_tdust``) + ``dust_f_pah`` (was
    # ``dust_fpah``); old spellings resolve via _LEGACY_PARAM_ALIASES.
    T = Fixed(25.0)
    f_pah = Fixed(0.05)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("schreiber2018",)

    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None):
        """Load the Schreiber+2018 template dict so it can be threaded.

        Returns
        -------
        dict or None
            Template arrays, or ``None`` when unavailable: the backend then
            falls back to its module-level load, which bakes 2.41 MB (#1649).

        Notes
        -----
        **JIT-compatible**: no, deliberately: runs at build time.
        """
        del wave
        from tengri._data_setup import find_data_str
        from tengri.components.dust.emission_templates import load_schreiber2018_templates

        for fname in ("schreiber2018_templates_v2.h5", "schreiber2018_templates.h5"):
            path = find_data_str(fname)
            if path is not None:
                return load_schreiber2018_templates(path)
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
        """Compute Schreiber et al. (2018) dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "T", "f_pah"
            (or subset if some are Fixed).
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
        # schreiber2018 has no top-level closure wrapper (only a lazy-loader entry),
        # so resolve it via the loader dict: same pattern as dale2014_cigale.
        from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS

        schreiber_fn = DUST_EMISSION_MODELS["schreiber2018"]
        # The loader's kwargs are ``dust_T`` / ``dust_f_pah``: passing the old
        # ``dust_tdust`` / ``dust_fpah`` names sent them into ``**_kwargs`` where
        # they were silently ignored (the component's temperature/PAH knobs had NO
        # effect). Fixed as part of the #849 name unification.
        kwargs = dict(dust_T=p["T"], dust_f_pah=p["f_pah"])
        if templates is not None:
            # Closure over the THREADED arrays: capture of a tracer is fine;
            # capture of a concrete array is what bakes (#1649).
            from tengri.components.dust.emission_templates import (
                create_schreiber2018_from_grid,
            )

            sed = create_schreiber2018_from_grid(templates)(wave, L_ir, **kwargs)
        else:
            sed = schreiber_fn(wave, L_ir, **kwargs)
        return sed_in + sed, {"sed_dust_ir": sed}

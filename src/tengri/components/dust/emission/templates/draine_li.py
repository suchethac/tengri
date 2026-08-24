# SPDX-License-Identifier: BSD-3-Clause
"""Draine & Li dust emission templates as SEDModelComponents.

Wraps the pure closures from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed

__all__ = ["DraineLi2007IRSEDComponent", "DraineLi2014IRSEDComponent"]


class DraineLi2007IRSEDComponent(EmissionComponent):
    """Draine & Li (2007) dust IR emission template.

    Wraps the pure closure from the tabulated Draine & Li (2007) template
    library, parameterized by minimum radiation field (umin), power-law
    mixing fraction (gamma), and PAH mass fraction (qpah).

    The model mixes single-U (diffuse) and power-law (PDR) components via
    the power-law fraction, with PAH content controlling the relative
    strength of aromatic features.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    **3.3 µm PAH vs FSPS/Prospector (#963)**: tengri carries the original
    Draine & Li (2007) 3.3 µm PAH feature. FSPS ships modified DL07 tables
    with that one feature halved ("3.3um PAH reduced by 50%" per the FSPS
    ``dust/dustem`` headers); everywhere else the two tabulations agree to
    ≤1.2 % at matched (q_PAH, U_min). Bands sampling rest-frame ~3–3.6 µm
    (e.g. WISE W1 at low z) therefore sit higher here than in
    FSPS/Prospector; measured +65 % in the dust-only component, ~+5 % at
    band level for a star-forming galaxy where starlight dominates W1.

    References
    ----------
    .. [1] Draine, B. T. & Li, A., 2007, "Infrared Emission from Dust",
       ApJ, 657, 810. https://doi.org/10.1086/511055

    """

    name: str = "draine_li2007"

    # Free parameters (user-facing names, prefix-stripped)
    umin = Fixed(1.0)
    gamma_dl = Fixed(0.01)
    qpah = Fixed(2.5)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("draine_li2007",)

    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None):
        """Load the DL07 template dict so it can be threaded, not baked.

        Returns
        -------
        dict or None
            Template arrays, or ``None`` when unavailable: the backend then
            falls back to its module-level load, which bakes 3.76 MB (#1649).

        Notes
        -----
        **JIT-compatible**: no, deliberately; runs at build time.
        """
        del wave
        from tengri._data_setup import find_data_str
        from tengri.components.dust.emission_templates import load_draine_li_templates

        for fname in ("dl07_templates_v2.h5", "dl07_templates.h5"):
            path = find_data_str(fname)
            if path is not None:
                return load_draine_li_templates(path)
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
        """Compute Draine & Li (2007) dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "umin", "gamma_dl", "qpah"
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
        kwargs = dict(
            dust_umin=p["umin"],
            dust_gamma_dl=p["gamma_dl"],
            dust_qpah=p["qpah"],
        )
        if templates is not None:
            from tengri.components.dust.emission_templates import create_dl07_from_grid

            sed = create_dl07_from_grid(templates)(wave, L_ir, **kwargs)
        else:
            from tengri.components.dust.emission import draine_li2007 as dl07_fn

            sed = dl07_fn(wave, L_ir, **kwargs)
        return sed_in + sed, {"sed_dust_ir": sed}


class DraineLi2014IRSEDComponent(EmissionComponent):
    """Draine & Li (2014) dust IR emission template (2014 update).

    Wraps the pure closure from the tabulated Draine & Li (2014) template
    library, which extends DL07 with an additional alpha (radiation field
    power-law slope) parameter.

    The model mixes single-U (diffuse) and power-law (PDR) components via
    the power-law fraction, with trilinear interpolation in (qpah, umin, alpha)
    space for the PDR component.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Draine, B. T., Aniano, G., Krause, O., et al., 2014,
       "Dust emission and the cosmic far-infrared background", ApJ, 780, 172.
       https://doi.org/10.1088/0004-637X/780/2/172

    """

    name: str = "draine_li2014"

    # Free parameters (user-facing names, prefix-stripped)
    umin = Fixed(1.0)
    gamma_dl = Fixed(0.01)
    qpah = Fixed(2.5)
    alpha_dl14 = Fixed(2.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("draine2014",)

    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None):
        """Load the DL14 template grid so it can be threaded, not baked.

        Returns
        -------
        dict or None
            Template arrays, or ``None`` when the grid is unavailable: the
            backend then falls back to its module-level load, which bakes.

        Notes
        -----
        **JIT-compatible**: no, deliberately; this runs at build time so the
        arrays reach ``predict`` as a traced argument (#1649).
        """
        del wave
        from tengri._data_setup import find_data_str
        from tengri.components.dust.emission_templates import load_dl14_templates

        for fname in ("dl14_templates_v2.h5", "dl14_templates.h5"):
            path = find_data_str(fname)
            if path is not None:
                return load_dl14_templates(path)
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
        """Compute Draine & Li (2014) dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "umin", "gamma_dl", "qpah",
            "alpha_dl14" (or subset if some are Fixed).
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
            dust_qpah=p["qpah"],
            dust_alpha_dl14=p["alpha_dl14"],
        )
        if templates is not None:
            # Threaded: the grid arrives as a traced argument.
            from tengri.components.dust.emission_templates import dl14_sed_from_grid

            sed = dl14_sed_from_grid(templates, wave, L_ir, **kwargs)
        else:
            from tengri.components.dust.emission import draine_li2014 as dl14_fn

            sed = dl14_fn(wave, L_ir, **kwargs)
        return sed_in + sed, {"sed_dust_ir": sed}

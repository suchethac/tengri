# SPDX-License-Identifier: BSD-3-Clause
"""Schreiber et al. (2018) dust-library emission as an SEDModelComponent.

Wraps the tabulated closure from :mod:`tengri.components.dust.emission_templates`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust._params import DEFAULT_DUST_F_PAH, SCHREIBER_T_K_DEFAULT
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed

__all__ = ["Schreiber2016IRSEDComponent"]


class Schreiber2016IRSEDComponent(EmissionComponent):
    r"""Schreiber et al. (2018) dust library, as packaged by CIGALE (tabulated).

    The library tabulates, for each dust temperature between 15 and 99 K in 1 K
    steps, the spectrum of one kilogram of dust as a dust continuum and a PAH
    component. The PAH **mass** fraction :math:`f_{\rm PAH}` mixes them per
    kilogram and the mixture is renormalised to the absorbed luminosity,

    .. math::

        L_\nu = L_{\rm abs}\,
        \frac{(1 - f_{\rm PAH})\,S_\nu^{\rm cont} + f_{\rm PAH}\,S_\nu^{\rm PAH}}
             {\int [(1 - f_{\rm PAH})\,S_\nu^{\rm cont} + f_{\rm PAH}\,S_\nu^{\rm PAH}]\,d\nu},

    exactly as CIGALE's ``schreiber2016`` module does (the module name keeps the
    2016 preprint year of the library paper). Because the PAH template carries
    about three times the power of the continuum template per kilogram, the PAH
    *power* share is :math:`f R/(1 - f + f R)` with :math:`R \simeq 3.06`; at
    :math:`f_{\rm PAH} \ge 0.2` the SED peaks on the 7.7 micron PAH complex.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: piecewise linear in ``dust_T`` (kinks at the 1 K nodes)
    and linear in ``dust_f_pah``.

    **Missing data**: the template file ``schreiber2016_templates.h5`` is
    required; there is no analytic fallback, and a missing file raises with the
    path and how to regenerate it.

    **Not modelled**: the CMB heating and contrast corrections of the
    analytic emission models are not applied (the library is a local-universe
    template; the same holds for the other tabulated dust emission models).

    References
    ----------
    .. [1] Schreiber, C., Elbaz, D., Pannella, M., Ciesla, L., Wang, T., &
       Franco, M., 2018, "Dust temperature and mid-to-total infrared color
       distributions for star-forming galaxies at 0 < z < 4",
       A&A, 609, A30. arXiv:1710.10276.
       https://doi.org/10.1051/0004-6361/201731506

    """

    name: str = "schreiber2016"

    # Free parameters (user-facing names, prefix-stripped). Both defaults are
    # CIGALE's declared defaults for this library (tdust = 20 K, fpah = 0.05),
    # read from the module constants the closure's own signature reads, so the
    # two cannot drift apart (#2241, #2597).
    T = Fixed(SCHREIBER_T_K_DEFAULT)
    f_pah = Fixed(DEFAULT_DUST_F_PAH)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("schreiber2018",)

    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None):
        """Load the Schreiber et al. (2018) per-kg template dict so it can be threaded.

        Returns
        -------
        dict
            Template arrays (see
            :func:`~tengri.components.dust.emission_templates.load_schreiber2016_templates`).

        Raises
        ------
        FileNotFoundError
            If ``schreiber2016_templates.h5`` cannot be located; the message
            names how to regenerate it.

        Notes
        -----
        **JIT-compatible**: no, deliberately; runs at build time.
        """
        del wave
        from tengri._data_setup import find_data_str
        from tengri.components.dust.emission_templates import load_schreiber2016_templates

        path = find_data_str("schreiber2016_templates.h5")
        if path is None:
            raise FileNotFoundError(
                "Template file 'schreiber2016_templates.h5' not found in data/. "
                "There is no analytic fallback. Regenerate it from CIGALE's database with "
                "`python scripts/regenerate_schreiber2016_from_cigale.py` (requires pcigale), "
                "or restore data/schreiber2016_templates.h5 from the repository."
            )
        return load_schreiber2016_templates(path)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
        templates=None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute the Schreiber et al. (2018) dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "T", "f_pah"
            (or subset if some are Fixed). ``f_pah`` is the PAH mass fraction.
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for a dust emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir : float
            Total absorbed luminosity in erg/s.
        templates : dict, optional
            The threaded template dict from :meth:`load`; when ``None`` the
            registry closure loads the file itself.

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where sed_out is the updated SED and published
            contains {"sed_dust_ir": emission SED in erg/s/Hz}.

        """
        from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
        from tengri.components.dust.emission_templates import create_schreiber2016_from_grid

        kwargs = {"dust_T": p["T"], "dust_f_pah": p["f_pah"]}
        if templates is not None:
            # Closure over the THREADED arrays: capture of a tracer is fine;
            # capture of a concrete array is what bakes (#1649).
            sed = create_schreiber2016_from_grid(templates)(wave, L_ir, **kwargs)
        else:
            sed = DUST_EMISSION_MODELS["schreiber2016"](wave, L_ir, **kwargs)
        return sed_in + sed, {"sed_dust_ir": sed}

# SPDX-License-Identifier: BSD-3-Clause
"""PAH Drude profiles dust emission as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.resolve import require_redshift

__all__ = ["PAHDrudeIRSEDComponent"]


class PAHDrudeIRSEDComponent(EmissionComponent):
    """Smith et al. (2007) PAH Drude profiles: mid-IR PAH building block.

    Wraps the pure closure :func:`~tengri.components.dust.emission.pah_drude`,
    which provides a sum of 18 PAH Drude profiles (normalized to Smith+2007
    SINGS median strengths).

    This is a **PAH-only building block**, not a standalone energy-balanced dust
    emitter: it carries the aromatic-feature forest only (no thermal continuum),
    so its frequency integral is *not* renormalized to ``L_ir``: it is
    scaled by ``L_ir`` but deliberately leaves the bulk of the absorbed
    energy for a continuum component to carry.

    Select a full model (``dale2014``, ``draine_li2007/2014``, ``themis``,
    ``modified_blackbody``, ``casey2012``, ``schreiber2016``) for an
    energy-conserving dust SED. ``pah_drude`` is intended as a diagnostic /
    composition primitive.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` primitives.

    **Gradient-safe**: yes.

    **Not energy-balanced standalone**: this component deliberately leaves
    much of ``L_ir`` unaccounted for, so it should not be used alone
    or cross-validated against energy-balance tests.

    The PAH template is a pure shape (no free parameters). Runtime evaluation
    uses the precomputed lookup from
    :mod:`~tengri.components.dust.dust_analytic_precompute` in the hybrid kernel;
    this component provides the full-wavelength evaluation.

    References
    ----------
    .. [1] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
       infrared galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
       https://doi.org/10.1086/510378

    """

    name: str = "pah_drude"

    #: Not an energy-balanced dust emitter, so ``SEDModel.build`` refuses it as
    #: a standalone ``dust_emission`` type. Measured on the z = 0 forward pass
    #: of ``tests/regression/precision/test_dust_ir_float32.py``'s fixture:
    #: ``|int sed_dust_ir dnu| / L_ir = 1.8925e-04``. It builds, converges and
    #: reports a dust luminosity like any other model while discarding 99.98%
    #: of the absorbed energy — the silent-wrong-answer shape, which is why the
    #: refusal is loud rather than a warning.
    energy_balanced: ClassVar[bool] = False
    standalone_l_ir_fraction: ClassVar[float] = 1.8925e-4

    #: No free parameters for PAH Drude; just a template shape. Stating it makes
    #: the difference between narrowing this engine's wildcard to nothing and
    #: leaving it to free the whole static union: an empty ``_priors`` alone
    #: cannot be told apart from ``energy_balance_split``, whose parameters are
    #: declared in ``components/dust/_params.py`` rather than on the class, so
    #: ``_declared_param_names`` refuses to infer it. Without the marker,
    #: ``'all_params': FREE`` here freed 19 dimensions a sampler cannot move
    #: (#1482).
    declares_no_parameters: ClassVar[bool] = True

    _citations_tuple: ClassVar[tuple[str, ...]] = ("smith2007",)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute PAH Drude profile emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped. Empty for PAH Drude (no free params).
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for a dust emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir : float
            Total absorbed luminosity in erg/s. Scales the template but does not
            normalize it to conserve energy (see Notes above).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where sed_out is the updated SED and published
            contains {"sed_dust_ir": emission SED in erg/s/Hz}.

        """
        from tengri.components.dust.emission import pah_drude as pah_fn

        z = jnp.asarray(require_redshift(p, "components.dust.emission.analytic.pah_drude.predict"))
        sed = pah_fn(wave, L_ir, redshift=z)
        return sed_in + sed, {"sed_dust_ir": sed}

# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR_mean_3p three-parameter clumpy torus (Stalevski et al. 2016)
SEDModelComponent.

Implements the SKIRTOR_mean_3p torus library as averaged by AGNfitter-rX
on the SEDModelComponent contract, enabling use of AGNfitter-faithful torus
templates in the model-building API.

This is an opt-in adapter distinct from the default ``skirtor`` component,
which uses the full-grid SKIRTOR implementation (X-CIGALE faithful).
The two differ in peak wavelength and other properties; choose based on your
fidelity target (see issue #614, #592, and #633 for motivation).

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes,
   "The dust covering factor in active galactic nuclei," MNRAS, 458,
   2288 (2016). arXiv:1602.06954. bibcode:2016MNRAS.458.2288S.
   https://doi.org/10.1093/mnras/stw444
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn._publication import (
    template_sed_and_log_power,
    with_lsun_outputs,
)
from tengri.components.agn._torus_inclination import (
    TORUS_LEDGER_KEYS,
    absent_torus_ledger,
    torus_ledger,
)
from tengri.components.agn.skirtor_agnfitter import create_skirtor_agnfitter_from_grid
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, SEDComponentState, declared_prior

__all__ = ["SKIRTORAgnfitterTorus"]

#: Single source of truth for ``oa_skirtor``/``cos_inc``/``tv_skirtor``'s
#: bounds and defaults: the shared ``agn_oa_skirtor`` / ``agn_cos_inc`` /
#: ``agn_tv_skirtor`` declarations in ``_params.py``. Read once at class-
#: definition time so the class attributes below cannot drift from them the
#: way ``CAT3DTorus``/``Silva04Torus`` did before Task 1 -- these three were
#: numerically equal to the canonical declaration but were still an
#: unprotected literal restatement (Task 1 fix round 1, finding 1).
_OA_SKIRTOR_PRIOR = declared_prior(_AGN_PARAMS, "agn_oa_skirtor")
_COS_INC_PRIOR = declared_prior(_AGN_PARAMS, "agn_cos_inc")
_TV_SKIRTOR_PRIOR = declared_prior(_AGN_PARAMS, "agn_tv_skirtor")
#: ``log_lbol``/``torus_frac`` restated stale literals (default 11.0 vs the
#: canonical 10.0) -- found by ``tools/check_param_restatements.py``
#: (Task 11 item 5 / fix round 1, R25).
_LOG_LBOL_PRIOR = declared_prior(_AGN_PARAMS, "agn_log_lbol")
_TORUS_FRAC_PRIOR = declared_prior(_AGN_PARAMS, "agn_torus_frac")


@dataclass(frozen=True)
class SKIRTORAgnfitterTorusConfig(SEDComponentConfig):
    """Configuration for SKIRTOR_mean_3p torus templates.

    Parameters
    ----------
    grid_path : str or None
        Path to SKIRTOR_mean_3p template grid (HDF5). If None, templates
        are not pre-loaded (deferred to first use in predict).
    """

    grid_path: str | None = None


@dataclass(frozen=True)
class SKIRTORAgnfitterTorusState(SEDComponentState):
    """Cached SKIRTOR_mean_3p template data.

    Attributes
    ----------
    name : str
        Component identifier.
    skirtor_agnfitter_fn : callable or None
        Compiled interpolation function from create_skirtor_agnfitter_from_grid,
        or None if templates are not available.
    """

    name: str = "skirtor_agnfitter"
    skirtor_agnfitter_fn: Any | None = None


@dataclass(frozen=True)
class SKIRTORAgnfitterTorus(SEDModelComponent):
    r"""SKIRTOR_mean_3p clumpy AGN torus (AGNfitter-faithful averaging).

    Three-parameter torus model with half-opening angle, inclination, and
    equatorial optical depth. Provides C¹-continuous gradients via node-exact
    PCHIP interpolation over the 3D parameter space.  This is the
    **AGNfitter-faithful** variant, differing from the default full-grid
    SKIRTOR in peak wavelength and other properties.

    Attributes
    ----------
    name : str
        Component registry key: ``"skirtor_agnfitter"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : SKIRTORAgnfitterTorusConfig
        Frozen configuration (grid path).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    oa_skirtor : Uniform
        Half-opening angle [deg]. [deg, 10–80]
    cos_inc : Uniform
        Cosine of the inclination from the polar axis, the AGN's one inclination
        (1 = face-on). [dimensionless, 0–1] The library is keyed at
        :math:`i = \arccos(\cos i)` [deg] on its 0–90 deg axis.
    tv_skirtor : Uniform
        Equatorial optical depth τ_9.7. [dimensionless, 3–11]
    torus_frac : Uniform
        Fraction of L_bol reprocessed by torus. [dimensionless, 0–1]

    Cross-component outputs
    -----------------------
    log_L_agn_torus, log_L_agn_torus_los, log_L_agn_disc_unemitted : dex
        The torus-only energy ledger [dex re erg/s] (NAMING_CONTRACT section 4c,
        "Template tori"): the angle-integrated torus power ``f L_bol``
        (``f = torus_frac``), the frequency integral of the torus SED emitted along
        the sightline, and the disc share ``(1 - f) L_bol`` this component does not
        emit, so the first and last add up to ``10**log_lbol L_sun``. Here the first
        two are equal: the AGNfitter-rX ``SKIRTOR_mean_3p`` records are not on SKIRTOR's
        radiative-transfer scale: their inclination power ratios scatter 0.2 to 3.9 about 1
        and do not follow the norm-restored SKIRTOR v3 library averaged the same way (3.26
        at oa = 40 deg, tau = 7), so each sightline is normalized on its own. Each has an
        ``L_agn_*_lsun`` [Lsun] sibling; ``-inf``
        is an absent piece.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Gradient-safe**: yes, PCHIP interpolation is C¹ and differentiable
    across the three parameter axes.

    **Requires template grid**: The SKIRTOR_mean_3p template library must be
    built separately and pointed to via ``grid_path`` in config.
    The predict method gracefully returns zero emission if templates
    are unavailable.

    **AGNfitter parity**: This component reproduces AGNfitter-rX's
    SKIRTOR_mean_3p template selection exactly (nearest-neighbor in
    3D parameter space, then per-L_sun normalization).  The upstream
    full-grid SKIRTOR differs in parameter space (5D vs 3D), peak
    wavelength (Stalevski +40µm vs AGNfitter +25µm at matched geometry),
    and should not be conflated (see issue #614).

    Examples
    --------
    Minimal model with SKIRTOR_mean_3p torus::

        from tengri import SEDModel, Fixed, DEFAULT, Uniform, builders
        from tengri.components.agn.skirtor_agnfitter_model import (
            SKIRTORAgnfitterTorus,
            SKIRTORAgnfitterTorusConfig,
        )

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(alpha=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed(DEFAULT)},
            agn=SKIRTORAgnfitterTorus(
                config=SKIRTORAgnfitterTorusConfig(grid_path="path/to/grid.h5")
            ),
        )

    See Also
    --------
    tengri.components.agn.skirtor_agnfitter : template loader and interpolation.
    tengri.components.agn.skirtor_model : default (X-CIGALE-faithful) SKIRTOR.
    """

    name = "skirtor_agnfitter"
    parameter_prefix = "agn_"
    config: SKIRTORAgnfitterTorusConfig = field(default_factory=SKIRTORAgnfitterTorusConfig)

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        _LOG_LBOL_PRIOR.lo,
        _LOG_LBOL_PRIOR.hi,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=_LOG_LBOL_PRIOR.default,
    )
    oa_skirtor = Uniform(
        _OA_SKIRTOR_PRIOR.lo,
        _OA_SKIRTOR_PRIOR.hi,
        description="Half-opening angle (Stalevski et al.)",
        units="deg",
        default=_OA_SKIRTOR_PRIOR.default,
    )
    cos_inc = Uniform(
        _COS_INC_PRIOR.lo,
        _COS_INC_PRIOR.hi,
        description="Cosine of inclination (1 = face-on)",
        units="dimensionless",
        default=_COS_INC_PRIOR.default,
    )
    tv_skirtor = Uniform(
        _TV_SKIRTOR_PRIOR.lo,
        _TV_SKIRTOR_PRIOR.hi,
        description="Equatorial optical depth τ_9.7",
        units="dimensionless",
        default=_TV_SKIRTOR_PRIOR.default,
    )
    torus_frac = Uniform(
        _TORUS_FRAC_PRIOR.lo,
        _TORUS_FRAC_PRIOR.hi,
        description="Torus luminosity fraction of L_bol",
        units="dimensionless",
        default=_TORUS_FRAC_PRIOR.default,
    )

    # Cross-component output
    outputs: ClassVar[dict[str, str]] = with_lsun_outputs(dict.fromkeys(TORUS_LEDGER_KEYS, "dex"))

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Load SKIRTOR_mean_3p template grid if available.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by SKIRTOR; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_skirtor_agnfitter_from_grid, or None
            if template file is not available or grid_path is not set.
        """
        if not self.config.grid_path:
            return None

        try:
            return create_skirtor_agnfitter_from_grid(self.config.grid_path)
        except (FileNotFoundError, OSError, KeyError):
            # Templates not available: predict will return zero emission
            return None

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX SKIRTOR_mean_3p torus prediction.

        Interpolates the SKIRTOR_mean_3p template grid to the requested
        opening angle, inclination, and optical depth; normalizes to the
        user's luminosity scale; and returns the torus SED contribution.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - oa_skirtor: half-opening angle [deg]
            - cos_inc: cosine of the inclination from the polar axis
            - tv_skirtor: equatorial optical depth
            - torus_frac: torus luminosity fraction

        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Unused (AGN torus is self-contained).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where:

            - sed_out: Updated SED (sed_in + torus contribution).
            - published: ``log_L_agn_torus``, ``log_L_agn_torus_los`` and
              ``log_L_agn_disc_unemitted`` [dex re erg/s] with their ``_lsun`` siblings
              (see Cross-component outputs).

        """
        # If templates are not loaded, return zero emission
        if not hasattr(self, "data") or self.data is None:
            return sed_in, absent_torus_ledger()

        skirtor_agnfitter_fn = self.data

        def _torus(wave_eval, log_lbol_eval):
            return skirtor_agnfitter_fn(
                wavelength=wave_eval,
                agn_log_lbol=log_lbol_eval,
                agn_oa_skirtor=p["oa_skirtor"],
                agn_cos_inc=p["cos_inc"],
                agn_tv_skirtor=p["tv_skirtor"],
                agn_torus_frac=p["torus_frac"],
            )

        # Published log_L_agn_torus: the exact integral of the emitted torus SED, in log space.
        # The template is a log-log power law between its native nodes, so the integral is
        # closed form on them.
        torus, log_L_torus = template_sed_and_log_power(
            _torus, skirtor_agnfitter_fn.native_wave, wave, p["log_lbol"]
        )

        return sed_in + torus, torus_ledger(p["log_lbol"], p["torus_frac"], log_L_torus)

# SPDX-License-Identifier: BSD-3-Clause
r"""Hensley & Draine (2023) Astrodust+PAH dust IR emission as SEDModelComponent.

Faithful native reading of the published Astrodust+PAH emission grid
(Harvard Dataverse doi:10.7910/DVN/3B6E6S): a single continuous starlight-
intensity axis :math:`\log_{10} U` (91 points, -3..6) with discrete emission
columns (total / astrodust-only / PAH-only) and optional spinning-dust
microwave emission.  The template is interpolated over ``lgU`` and rescaled so
its frequency integral matches the dust-absorbed luminosity ``L_ir`` (energy
balance).

This is the sole ``astrodust`` model.  The earlier registry entry parameterized
astrodust with the Draine & Li (2007) ``(umin, gamma, qpah)`` knob-set by
translating the HD23 grid into a synthetic ``(umin, qpah)`` table — but that
grid has no ``qpah`` axis (``dust_qpah`` was a silent no-op) and no ``umin``
axis (``umin`` was merely ``10**lgU``).  The faithful ``lgU`` parameterization
below replaces it (see :ref:`the migration note <#871>`).

References
----------
.. [1] Hensley, B. S. & Draine, B. T. 2023, "The Astrodust+PAH Model: A Unified
   Description of the Extinction, Emission, and Polarization from Dust in the
   Diffuse Interstellar Medium", ApJ, 948, 55.  arXiv:2208.12365.
   doi:10.3847/1538-4357/acc370.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from tengri.components.dust.astrodust_hd23 import (
    load_astrodust_hd23_or_raise,
    resample_lnu_on_aa_grid,
)
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig

__all__ = ["AstrodustIRConfig", "AstrodustIRSEDComponent"]


#: Valid ``AstrodustIRConfig.component`` selectors (separate emission columns
#: in the published grid).
_ASTRODUST_COMPONENTS: tuple[str, ...] = ("total", "astrodust", "pah")


@functools.cache
def _cached_astrodust_grid(template_path: str | None):
    """Process-cached raw HD23 grid.

    The published grid is wavelength/lgU only (no free-parameter axis), so it is
    a constant for the whole run. Caching it (a) avoids re-reading the HDF5 on
    every ``predict`` trace and (b) keeps the lazy path in :meth:`predict` leak-
    safe: the cached arrays are ``jnp.asarray`` of NumPy data (concrete, never
    tracers), so building the template inside a JIT trace introduces no escaping
    intermediate.
    """
    return load_astrodust_hd23_or_raise(template_path)


@dataclass(frozen=True)
class AstrodustIRConfig(SEDComponentConfig):
    r"""Configuration for the Astrodust+PAH IR emission component.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"astrodust"``.
    component : {"total", "astrodust", "pah"}
        Which published emission column to re-radiate: the full thermal
        spectrum (``"total"`` = astrodust grains + PAHs), the astrodust-grain
        contribution only, or the PAH contribution only. Default ``"total"``.
    spinning_dust : bool
        Add the (``lgU``-independent) spinning-dust microwave emission on top
        of the thermal template. Default ``False``.
    f_cnm : float
        Cold-neutral-medium filling fraction used to mix the CNM/WNM
        spinning-dust spectra when ``spinning_dust=True``. Must lie in
        ``[0, 1]``. Default ``0.28`` (the published fiducial).
    template_path : str or None
        Override path to the Astrodust+PAH HDF5 grid. When ``None``, resolves
        the ``TENGRI_ASTRODUST_PATH`` env var, then the bundled
        ``data/astrodust_templates.h5``. Default ``None``.
    """

    name: str = "astrodust"
    component: str = "total"
    spinning_dust: bool = False
    f_cnm: float = 0.28
    template_path: str | None = None


class AstrodustIRSEDComponent(EmissionComponent):
    r"""Hensley & Draine (2023) Astrodust+PAH dust IR emission template.

    Re-emits the dust-absorbed luminosity ``L_ir`` with the shape of the
    published HD23 emission grid, interpolated over a single continuous
    starlight-intensity axis :math:`\log_{10} U` and rescaled to enforce
    energy balance.

    Attributes
    ----------
    name : str
        Registry key: ``"astrodust"``.
    parameter_prefix : str
        Domain prefix for parameters: ``"dust_"`` (inherited).
    config : AstrodustIRConfig
        Frozen configuration (emission column, spinning-dust toggle, path).

    Notes
    -----
    **Physical axes**: the published emission grid has exactly one continuous
    physics axis (``lgU``); ``component`` and ``spinning_dust`` select among
    discrete columns and are therefore configuration choices, not free
    parameters.

    **Cross-component contract**:

    - Reads ``state.derived["L_ir"]`` [erg/s] (published by the attenuator).
    - Publishes ``{"sed_dust_ir": L_nu}`` [erg/s/Hz] (the IR emission profile,
      consumed by :meth:`tengri.Posterior.sed_components`).

    **Energy balance**: the thermal template is rescaled so
    :math:`\int L_\nu \, d\nu = L_{\rm ir}` exactly. Spinning dust, when
    included, is added on top scaled by the same thermal-budget factor (its
    bolometric power is small).

    **Template loading**: the HDF5 grid loads in :meth:`load` (precompute, not
    JIT-ed) and caches the wave-resampled, component-selected template on
    ``self.data``; :meth:`predict` is then pure JAX.

    **JIT-compatible**: yes. **Gradient-safe**: yes (linear interpolation).

    **Pipeline ordering**: MUST run after dust attenuation so ``L_ir`` is
    present. Typical order:
    ``[Stellar, Nebular, DustAttenuation, Astrodust, IGM, Radio]``.
    """

    name = "astrodust"
    config: AstrodustIRConfig = AstrodustIRConfig()

    #: ``load`` resamples onto the ``wave`` it is given, so its arrays are
    #: tracers under a trace and concrete constants outside one. Neither may be
    #: cached on the instance: the first leaks a tracer, the second bakes 4.97 MB
    #: into the graph. The build-time resolution pass therefore skips this
    #: component and ``predict`` resolves per call (#1738).
    resolves_templates_at_trace_time: ClassVar[bool] = True

    # Free parameter (user-facing name, prefix-stripped): starlight intensity.
    lgU = Uniform(
        -3.0,
        6.0,
        default=1.0,
        description="log10(U), starlight intensity in local-ISRF (U=1) units",
        units="dex",
    )

    _citations_tuple: ClassVar[tuple[str, ...]] = ("hensley_draine2023",)

    def __init__(self, config: AstrodustIRConfig | None = None) -> None:
        # Override the class-level default config if an instance is provided.
        if config is not None:
            self.config = config

    def _select_component_um(self, templates: Any) -> jnp.ndarray:
        r"""Return the requested emission column on the grid's micron axis.

        Parameters
        ----------
        templates : AstrodustHD23Templates
            Loaded grid container.

        Returns
        -------
        jnp.ndarray, shape ``(n_lgU, n_wave_um)``
            :math:`L_\nu` per H atom [erg/s/Hz/H] for the selected component.

        Raises
        ------
        ValueError
            If ``config.component`` is not in :data:`_ASTRODUST_COMPONENTS`.
        """
        comp = self.config.component
        if comp == "total":
            return templates.L_nu_total
        if comp == "astrodust":
            return templates.L_nu_astrodust
        if comp == "pah":
            return templates.L_nu_pah
        raise ValueError(f"component={comp!r} not in {_ASTRODUST_COMPONENTS}")

    def _compose_spinning_um(self, templates: Any) -> jnp.ndarray:
        r"""Compose the spinning-dust spectrum for the selected component.

        Mixes the per-phase (CNM/WNM) spinning-dust columns at the configured
        cold-neutral-medium filling fraction ``f_cnm``.

        Parameters
        ----------
        templates : AstrodustHD23Templates
            Loaded grid container.

        Returns
        -------
        jnp.ndarray, shape ``(n_wave_um,)``
            Spinning-dust :math:`L_\nu` per H atom [erg/s/Hz/H].

        Raises
        ------
        ValueError
            If ``config.f_cnm`` is outside ``[0, 1]``.
        """
        f_cnm = float(self.config.f_cnm)
        if not (0.0 <= f_cnm <= 1.0):
            raise ValueError(f"f_cnm must be in [0, 1]; got {f_cnm!r}")
        comp = self.config.component
        if comp == "astrodust":
            return (
                f_cnm * templates.L_nu_spdust_Ad_CNM + (1.0 - f_cnm) * templates.L_nu_spdust_Ad_WNM
            )
        if comp == "pah":
            return (
                f_cnm * templates.L_nu_spdust_PAH_CNM
                + (1.0 - f_cnm) * templates.L_nu_spdust_PAH_WNM
            )
        # "total" -> astrodust + PAH at the chosen f_cnm
        return f_cnm * (templates.L_nu_spdust_Ad_CNM + templates.L_nu_spdust_PAH_CNM) + (
            1.0 - f_cnm
        ) * (templates.L_nu_spdust_Ad_WNM + templates.L_nu_spdust_PAH_WNM)

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        r"""Load and resample the HD23 template onto the pipeline grid.

        Called at model-init time (not JIT-ed). Loads the published grid,
        selects the configured emission column, resamples it onto ``wave``,
        integrates the per-``lgU`` normalizations, and (if requested) composes
        the spinning-dust spectrum.

        Parameters
        ----------
        wave : ndarray, shape ``(n_wave,)``, optional
            Rest-frame wavelength grid in Angstrom. When ``None``, the
            component skips (returns ``None``).

        Returns
        -------
        dict or None
            ``{"lgU_grid", "lnu_template", "norm_per_lgU", "lnu_spinning"}``
            when loaded, else ``None``.

        Raises
        ------
        FileNotFoundError
            When the resolved grid path does not exist on disk. Astrodust
            carries no analytic fallback — the published templates are
            required for physically meaningful predictions.
        """
        if wave is None:
            warnings.warn(
                f"AstrodustIR component {self.name!r}: wave_grid is None. Component will skip.",
                UserWarning,
                stacklevel=2,
            )
            return None

        # Missing grid raises loudly (no silent zeros — analytic fallbacks are
        # not suitable for science). The raw grid is process-cached (concrete).
        templates = _cached_astrodust_grid(self.config.template_path)

        wave_aa = jnp.asarray(wave)
        lnu_um = self._select_component_um(templates)
        lnu_template = resample_lnu_on_aa_grid(
            L_nu_um=lnu_um,
            wave_um=templates.wavelength_um,
            wave_aa=wave_aa,
        )
        norms = integrate_lnu_over_nu(lnu_template, wave_aa)

        if self.config.spinning_dust:
            spd_um = self._compose_spinning_um(templates)
            spd_aa = resample_lnu_on_aa_grid(
                L_nu_um=spd_um[None, :],
                wave_um=templates.wavelength_um,
                wave_aa=wave_aa,
            )[0]
        else:
            spd_aa = jnp.zeros_like(wave_aa)

        return {
            "lgU_grid": jnp.asarray(templates.lgU),
            "lnu_template": lnu_template,
            "norm_per_lgU": norms,
            "lnu_spinning": spd_aa,
        }

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute Astrodust+PAH IR emission and add it to the SED.

        Linearly interpolates the precomputed :math:`L_\nu(\lg U)` cube at the
        requested ``dust_lgU`` (clipped to the grid support :math:`[-3, 6]`),
        rescales so its frequency integral equals ``L_ir``, and adds the
        (``lgU``-independent) spinning-dust spectrum if it was included.

        .. math::

            L_\nu(\lambda) = \frac{L_{\rm ir}}{\int \hat L_\nu \, d\nu}\,
            \hat L_\nu(\lambda;\lg U) + s\,L_\nu^{\rm spin}(\lambda),

        where :math:`\hat L_\nu` is the template shape at ``lgU`` and
        :math:`s = L_{\rm ir} / \int \hat L_\nu\, d\nu` is the thermal-budget
        scale [erg/s/Hz for :math:`L_\nu`, erg/s for :math:`L_{\rm ir}`].

        Parameters
        ----------
        p : mapping[str, ndarray]
            Sliced parameters (prefix stripped): ``p["lgU"]`` [dex].
        sed_in : ndarray, shape ``(n_wave,)``
            Input SED [erg/s/Hz] (emission is added to it).
        wave : ndarray, shape ``(n_wave,)``
            Rest-frame wavelength grid [Angstrom].
        **inputs : ndarray
            ``L_ir`` — dust-absorbed luminosity [erg/s].

        Returns
        -------
        tuple[ndarray, dict]
            ``(sed_out, {"sed_dust_ir": sed_emission})``.

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes (linear interp,
        differentiable except at grid boundaries where clipping occurs).
        """
        L_ir = jnp.asarray(inputs.get("L_ir", 0.0))

        data = inputs.get("templates")
        if data is None:
            data = self.load(wave)
        if data is None:
            return sed_in, {}

        lgU_grid = data["lgU_grid"]
        lnu_template = data["lnu_template"]  # (n_lgU, n_wave)
        norm_per_lgU = data["norm_per_lgU"]
        lnu_spinning = data["lnu_spinning"]

        lgU_clipped = jnp.clip(jnp.asarray(p["lgU"]), lgU_grid[0], lgU_grid[-1])
        L_nu_shape = jax.vmap(
            lambda col: jnp.interp(lgU_clipped, lgU_grid, col),
            in_axes=1,
        )(lnu_template)
        norm_at_lgU = jnp.interp(lgU_clipped, lgU_grid, norm_per_lgU)
        scale = jnp.where(norm_at_lgU > 0, L_ir / norm_at_lgU, 0.0)

        # Spinning dust rides the same thermal-budget scale so its amplitude
        # stays tied to the dust mass implied by L_ir.
        sed_emission = L_nu_shape * scale + lnu_spinning * scale
        return sed_in + sed_emission, {"sed_dust_ir": sed_emission}

# SPDX-License-Identifier: BSD-3-Clause
"""AGNSEDComponent: unified AGN emission (disc + torus + lines + jets/corona).

Dispatches to any registered AGN model (``"multicolor_agn"``,
``"kubota_done_full"``, ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``) via
:func:`tengri.components.agn.unified.resolve_agn_model`.

Cross-component reads
---------------------
Nothing required — AGN is a self-contained additive emitter. ``redshift``
is read from :data:`BARE_NAME_ALLOWLIST`.

Cross-component publications
----------------------------
- ``state.derived["L_agn_bol"]`` (scalar, erg/s) — bolometric AGN
  luminosity. Consumed by
  :class:`tengri.components.xray.XRaySEDComponent` and
  :class:`tengri.components.radio.RadioSEDComponent` via their
  documented fallback (``state.derived.get("L_agn_bol", 0.0)``).
- ``state.derived["sed_agn"]`` — the AGN SED contribution
  (erg/s/Hz, shape n_wave) for diagnostics.

Architectural notes
-------------------
``agn_torus_frac`` is **never** auto-derived from ``agn_cos_inc`` /
``theta_torus`` (CLAUDE.md gotcha — gradient discontinuity). It is
read directly from ``params`` as an independent free parameter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.unified import resolve_agn_model
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import L_SUN

__all__ = ["AGNSEDComponent", "AGNSEDComponentConfig"]


@dataclass(frozen=True)
class AGNSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`AGNSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier. Default ``"agn"``.
    model : str
        AGN model registry key. One of ``"multicolor_agn"`` (Kubota & Done
        outer-zone disc + 2-T torus), ``"kubota_done_full"`` (full 3-zone
        disc), ``"adaf"``, ``"unified_nlr_blr"``, ``"skirtor"``,
        ``"silva04"``, ``"cat3d_wind"``, ``"relagn"``, ``"qsogen"``.
        Default ``"multicolor_agn"``.
    """

    name: str = "agn"
    model: str = "multicolor_agn"


@dataclass(frozen=True)
class AGNSEDComponentState(SEDComponentState):
    """State for AGN component.

    Holds optional filter passbands when ``approx=WavePrecomp()``
    is set on the parent SEDModel. The component uses them at
    :meth:`AGNSEDComponent.apply` time to filter-integrate the
    analytically computed AGN SED and publish ``agn_phot_lnu_precomp``.

    Also optionally caches SKIRTOR torus template grids for JIT threading
    so they become Parameter ops rather than baked Constants.
    """

    name: str = "agn"
    filter_waves: Any | None = None
    filter_trans: Any | None = None
    skirtor_templates: Any | None = None


@dataclass(frozen=True)
class AGNSEDComponent:
    """SEDComponent adapter for the unified AGN model registry.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` delegates to a registered
    AGN function which is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_AGN(λ)``,
    matching the convention used by :class:`RadioSEDComponent` and
    :class:`XRaySEDComponent`.
    """

    config: AGNSEDComponentConfig = field(default_factory=AGNSEDComponentConfig)
    name: str = "agn"
    parameter_prefix: str = "agn_"
    _state: AGNSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """AGN sub-block citations (disc, torus, BLR) flow from
        :mod:`tengri.citations.associations`; the wrapper has no
        always-required paper."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.agn._params`. The legacy ``_AGN_PARAMS``
        bucket in :mod:`tengri.parameters._param_defs` is a derived view
        of the same tuple (plus the ``neb_xid`` orphan kept in the
        registry for the Feltre NLR backend).

        The full ``agn_*`` parameter superset is declared so that any
        registered model can run without missing keys. Users freely
        ``Fixed`` whatever a particular model does not consume; the
        AGN registry functions accept ``**kwargs`` and ignore unused
        names.
        """
        return list(_AGN_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this AGN component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("L_agn_bol", "erg/s", "AGN bolometric luminosity"),
            DerivedKey("sed_agn", "erg/s/Hz", "AGN SED contribution on pipeline wave grid"),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> AGNSEDComponentState:
        r"""Cache filter passbands and SKIRTOR templates for precomputation.

        When ``approx=WavePrecomp()`` is set:
        - Stores filter passbands so :meth:`apply` can publish
          ``agn_phot_lnu_precomp`` (filter-integrated AGN photometry).
        - If the AGN model is "skirtor", pre-loads the template grids
          so they thread through JIT as Parameter ops.

        Parameters
        ----------
        ssp_data : Any | None
            Unused; accepted for Protocol uniformity.
        wave_grid : ndarray | None
            Unused; accepted for Protocol uniformity.
        approx : Mapping[str, bool] | None
            Approximation flags. When ``approx.get('wave_precomp')``
            is ``True``, cache templates and filters.
        filters : tuple of (wave, trans) pairs | None
            Filter wavelengths and transmissions to cache.

        Returns
        -------
        AGNSEDComponentState
            State with optionally-populated filter_waves, filter_trans,
            and skirtor_templates.
        """
        del ssp_data, wave_grid
        approx = approx or {}

        filter_waves = None
        filter_trans = None
        skirtor_templates = None

        if approx.get("wave_precomp") and filters is not None:
            filter_waves = tuple(jnp.asarray(fw) for fw, _ in filters)
            filter_trans = tuple(jnp.asarray(ft) for _, ft in filters)

        # SKIRTOR torus models require template grids. Load them here so
        # they're available for JIT threading.
        if self.config.model == "skirtor":
            try:
                from tengri.components.agn.skirtor import _load_skirtor_default

                skirtor_templates = _load_skirtor_default()
            except Exception:
                # If SKIRTOR template loading fails (file not found),
                # gracefully continue without threading. The apply() path
                # will fall back to lazy loading.
                pass

        return AGNSEDComponentState(
            name=self.name,
            filter_waves=filter_waves,
            filter_trans=filter_trans,
            skirtor_templates=skirtor_templates,
        )

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        """Add AGN emission to ``state.sed_intrinsic`` and publish ``L_agn_bol``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialised to zeros of the same shape.
        params : mapping
            Receives ``agn_*`` keys plus the bare ``redshift``.
        template_data : dict | None
            Nested dict with component namespaces ("nebular", "agn", etc)
            carrying template grids/weights for JIT threading. When present,
            SKIRTOR templates are read from ``template_data["agn"]["skirtor"]``
            as a JIT runtime input instead of module-level lazy-loaded cache.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated and ``L_agn_bol``
            published to ``derived``.
        """
        wave = state.wave

        # Convert log10(L_bol/Lsun) → erg/s once for both the model
        # call and the L_agn_bol publication.
        agn_log_lbol = jnp.asarray(params["agn_log_lbol"])
        L_agn_bol = jnp.power(10.0, agn_log_lbol) * L_SUN

        # Resolve the AGN model from the registry. resolve_agn_model is
        # a factory-time lookup but the returned callable is pure JAX
        # so it folds into a JIT trace cleanly. If template_data is
        # provided, extract AGN SKIRTOR template and thread it.
        agn_fn = resolve_agn_model(self.config.model)

        # Phase 4-D: thread SKIRTOR template as JIT runtime input
        skirtor_template = None
        if template_data is not None and isinstance(template_data, dict):
            agn_data = template_data.get("agn")
            if agn_data is not None and isinstance(agn_data, dict):
                skirtor_template = agn_data.get("skirtor")
        # Build kwargs, adding _template for SKIRTOR threading if available
        agn_kwargs = {
            "agn_frac": jnp.asarray(params.get("agn_frac", 1.0)),
            "agn_alpha": jnp.asarray(params.get("agn_alpha", -1.0)),
            "agn_log_mbh": jnp.asarray(params.get("agn_log_mbh", 8.0)),
            "agn_log_ledd": jnp.asarray(params.get("agn_log_ledd", -1.0)),
            "agn_a_spin": jnp.asarray(params.get("agn_a_spin", 0.0)),
            "agn_torus_frac": jnp.asarray(params.get("agn_torus_frac", 0.5)),
            "agn_T_torus": jnp.asarray(params.get("agn_T_torus", 1000.0)),
            "agn_tau_torus": jnp.asarray(params.get("agn_tau_torus", 3.0)),
            "agn_T_hot": jnp.asarray(params.get("agn_T_hot", 1500.0)),
            "agn_T_warm": jnp.asarray(params.get("agn_T_warm", 300.0)),
            "agn_frac_hot": jnp.asarray(params.get("agn_frac_hot", 0.5)),
            "agn_cos_inc": jnp.asarray(params.get("agn_cos_inc", 0.5)),
            "agn_polar_ebv": jnp.asarray(params.get("agn_polar_ebv", 0.0)),
            "agn_polar_oa": jnp.asarray(params.get("agn_polar_oa", 40.0)),
            "agn_ebv_disc": jnp.asarray(params.get("agn_ebv_disc", 0.0)),
        }
        if skirtor_template is not None:
            agn_kwargs["_template"] = skirtor_template

        L_agn = agn_fn(wave, agn_log_lbol=agn_log_lbol, **agn_kwargs)

        # Phase 3c-3d-agn: filter-integrate L_agn through the cached filter
        # passbands and publish ``agn_phot_lnu_precomp`` so predict_via_precomp
        # can include the AGN contribution in the LUT sum.
        derived_overrides = dict(L_agn_bol=L_agn_bol, sed_agn=L_agn)
        if (
            self._state is not None
            and self._state.filter_waves is not None
            and self._state.filter_trans is not None
        ):
            from tengri.observation.photometry import compute_flux_density

            z = jnp.asarray(params.get("redshift", 0.0))
            # Filter-integrate L_agn at the source's z, dl_cm=1.
            # compute_flux_density returns F_nu = (1+z)/(4π·dl²) · Lν_filter,
            # so undo the cosmology factor to recover the bare rest-frame Lν
            # — matches the convention of stellar_phot_lnu_precomp.
            inv_cosmology = 4.0 * jnp.pi * 1.0**2 / (1.0 + z)
            agn_phot_lnu_precomp = (
                jnp.asarray(
                    [
                        compute_flux_density(
                            L_agn,
                            state.wave,
                            fw,
                            ft,
                            redshift=z,
                            dl_cm=jnp.asarray(1.0),
                        )
                        for fw, ft in zip(
                            self._state.filter_waves,
                            self._state.filter_trans,
                            strict=False,
                        )
                    ]
                )
                * inv_cosmology
            )
            derived_overrides["agn_phot_lnu_precomp"] = agn_phot_lnu_precomp

        return state.add_intrinsic(L_agn).with_(
            derived=state.derived.with_(**derived_overrides),
        )

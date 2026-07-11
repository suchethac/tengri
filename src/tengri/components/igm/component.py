# SPDX-License-Identifier: BSD-3-Clause
"""IGMSEDComponent: intergalactic-medium transmission as a SEDComponent.

A thin wrapper around :func:`tengri.components.igm.igm.igm_transmission`
(Inoue et al. 2014) that satisfies the
:class:`tengri.protocols.SEDComponent` contract.

Design choices (mirrored in :mod:`tengri.components.radio.component`):

- ``parameter_prefix = "igm_"`` for the patchy-reionization extras
  (``igm_x_HI``, ``igm_bubble_mpc``, ``igm_z_mid``, ``igm_dz``,
  ``igm_log_nhi``). The bare ``redshift`` is read via
  :data:`tengri.protocols.component.BARE_NAME_ALLOWLIST` — IGM is the
  canonical reason that allowlist exists.
- IGM is *transmissive*: it multiplies :attr:`ForwardState.sed_observed`
  in place. If ``sed_observed`` is ``None`` the component is a no-op
  (useful in unit tests run before the rest→observed redshifting step).
- :meth:`precompute` is a no-op (Inoue's piecewise-power-law fit is
  evaluated lazily inside :func:`igm_transmission`; no redshift-dependent
  grid needs caching).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.igm._params import PARAMS as _IGM_PARAMS
from tengri.components.igm.igm import igm_absorption
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["IGMSEDComponent", "IGMSEDComponentConfig"]


@dataclass(frozen=True)
class IGMSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`IGMSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"igm"``.
    igm_model : str
        Mean-IGM transmission model resolved from the registry: ``"inoue"``
        (Inoue+2014, default), ``"madau"`` (Madau+1995), ``"meiksin06"``
        (Meiksin 2006), or ``"asada25"`` (Inoue + Asada+2025 CGM damping
        wing). Threaded from ``spec.igm_model`` so the observed-frame
        photometry and spectroscopy honor the configured model rather than
        always falling back to Inoue.
    igm_patchy : bool
        Use the patchy-reionization damping-wing model instead of the mean
        IGM. Default ``False``.
    use_dla : bool
        Multiply by a damped-Lyman-α absorber (params read at apply time),
        so photometry/spectroscopy see the DLA — not only
        ``predict_obs_sed``. Default ``False``.
    """

    name: str = "igm"
    igm_model: str = "inoue"
    igm_patchy: bool = False
    use_dla: bool = False


@dataclass(frozen=True)
class IGMSEDComponentState(SEDComponentState):
    r"""Precomputed per-filter IGM band factors for the WavePrecomp LUT path.

    Attributes
    ----------
    band_zgrid : ndarray, shape (n_z,) or None
        Redshift nodes at which ``band_table`` was evaluated. A single node
        for a fixed-redshift model (then the lookup is exact).
    band_table : ndarray, shape (n_z, n_filters) or None
        Filter-averaged IGM transmission :math:`\langle T \rangle_f(z)`,
        dimensionless. ``None`` when the factors cannot be precomputed
        (patchy reionization or a DLA — both carry free parameters, so the
        factor is not a function of redshift alone).
    """

    name: str = "igm"
    band_zgrid: Any | None = None
    band_table: Any | None = None


@dataclass(frozen=True)
class IGMSEDComponent:
    r"""SEDComponent adapter around :func:`igm_transmission`.

    Parameters
    ----------
    config : IGMSEDComponentConfig, optional
        Frozen structural settings. Default :class:`IGMSEDComponentConfig`.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX.
    **Transmissive**: writes ``sed_observed = sed_observed * T(λ)``.
    Components are no-ops when ``sed_observed is None``.
    """

    config: IGMSEDComponentConfig = field(default_factory=IGMSEDComponentConfig)
    name: str = "igm"
    parameter_prefix: str = "igm_"
    _state: IGMSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """IGM transmission backend (Inoue+2014 / Madau+1995) is config-driven;
        see :data:`tengri.citations.associations.IGM_CITATIONS`."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.igm._params` — the CGM damping-wing
        knobs read by :func:`igm_transmission`. The bare ``redshift``
        parameter is read via :data:`BARE_NAME_ALLOWLIST` and not
        declared here.
        """
        return list(_IGM_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this IGM component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        keys = [
            DerivedKey(
                "igm_transmission",
                "",
                "Inoue+2014 transmission T(lambda) on observed-frame grid",
            ),
        ]
        if self._state is not None and self._state.band_table is not None:
            keys.append(
                DerivedKey(
                    "igm_phot_factor",
                    "",
                    "Filter-averaged IGM transmission <T>_f, one per filter "
                    "(WavePrecomp path; frees the full-grid curve for DCE)",
                )
            )
        return tuple(keys)

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> IGMSEDComponentState:
        r"""No-op precompute. IGM transmission is evaluated at apply time.

        The WavePrecomp band factors are built by
        :meth:`precompute_band_factors`, which needs the filter *convention*
        and padded curves — model-level knowledge this signature does not carry.
        """
        del ssp_data, wave_grid, approx, filters
        return IGMSEDComponentState(name=self.name)

    def precompute_band_factors(
        self,
        wave_rest: jnp.ndarray,
        photometry: Any,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
        redshift_spec: Mapping[str, Any] | None = None,
    ) -> IGMSEDComponentState:
        r"""Tabulate the filter-averaged IGM transmission against redshift.

        The LUT photometry path needs one number per filter,
        :math:`\langle T \rangle_f`, yet the runtime evaluated the full
        Inoue+2014 curve on the whole model grid every call and then averaged
        it down — a 5994-point transmission to produce five numbers, which
        cost 12.1 MFLOPs and pinned the full-resolution grid alive, defeating
        the dead-code elimination that *is* the WavePrecomp speedup (#932).

        The band factor depends only on :math:`(z, f)`:

        .. math::

            \langle T \rangle_f(z) =
            \frac{\int T(\lambda_{\rm obs}, z)\, R_f(\lambda)\, w(\lambda)\,
            {\rm d}\lambda}{\int R_f(\lambda)\, w(\lambda)\, {\rm d}\lambda}

        where :math:`T` is the IGM transmission [dimensionless], :math:`R_f`
        the filter response, and :math:`w` the convention weight (ADR-0017:
        :math:`1/\lambda` photon-counting, :math:`1/\lambda^2` energy). The SED
        does *not* enter — the transmission is averaged alone — so the whole
        table moves to build time.

        Evaluated with the same :func:`lnu_filter_integral_batch` quadrature
        the runtime uses, so the result is **bit-identical at the nodes**;
        a fixed-redshift model gets a single node and is therefore exact.

        Parameters
        ----------
        wave_rest : array_like, shape (n_wave,)
            Rest-frame model wavelength grid [Angstrom].
        photometry : Photometry
            Supplies the padded filter curves and the convention.
        redshift_spec : mapping or None
            ``{'mode': 'fixed', 'value': z}`` or
            ``{'mode': 'free', 'z_min':, 'z_max':, 'n_z':}``.

        Returns
        -------
        IGMSEDComponentState
            Carrying ``band_zgrid`` / ``band_table``, or a bare marker when
            the factors are not precomputable.

        Notes
        -----
        **JIT-compatible**: build-time only — call outside any trace.

        **Not precomputable** when ``igm_patchy`` or ``use_dla`` is set: both
        read free parameters (``igm_x_HI``, ``dla_log_n_hi``, …), so
        :math:`T` is no longer a function of redshift alone. Those configs
        keep the exact full-grid path.
        """
        import numpy as np

        from tengri.observation.photometry import lnu_filter_integral_batch, pad_filters

        if self.config.igm_patchy or self.config.use_dla:
            return IGMSEDComponentState(name=self.name)

        if not filters or wave_rest is None or photometry is None:
            return IGMSEDComponentState(name=self.name)

        # Pad exactly as the stellar component does when it publishes
        # ``phot_filter_waves_padded`` — same arrays, same quadrature, so the
        # tabulated factor is bit-identical to the runtime band average.
        fws, fts = zip(*filters, strict=False)
        fw_pad, ft_pad, _ = pad_filters(
            [jnp.asarray(w) for w in fws], [jnp.asarray(t) for t in fts]
        )
        n_filters = int(getattr(photometry, "n_filters", len(filters)))

        spec = dict(redshift_spec or {"mode": "fixed", "value": 0.0})
        if spec.get("mode") == "free":
            zgrid = np.linspace(
                float(spec.get("z_min", 0.001)),
                float(spec.get("z_max", 3.0)),
                int(spec.get("n_z", 100)),
            )
        else:
            zgrid = np.asarray([float(spec.get("value", 0.0))])

        wave_rest = jnp.asarray(wave_rest)
        rows = []
        for z in zgrid:
            trans = igm_absorption(
                wave_rest * (1.0 + z),
                z,
                igm_patchy=False,
                igm_model=self.config.igm_model,
                use_dla=False,
            )
            rows.append(
                lnu_filter_integral_batch(
                    trans, wave_rest, fw_pad, ft_pad, z, convention=photometry.convention
                )[:n_filters]
            )

        return IGMSEDComponentState(
            name=self.name,
            band_zgrid=jnp.asarray(zgrid),
            band_table=jnp.stack(rows),
        )

    def _band_factor(self, z: jnp.ndarray) -> jnp.ndarray | None:
        """Interpolate the precomputed band factors at ``z``; ``None`` if absent."""
        if self._state is None or self._state.band_table is None:
            return None
        zgrid, table = self._state.band_zgrid, self._state.band_table
        if zgrid.shape[0] == 1:  # fixed-redshift model: exact, no interpolation
            return table[0]
        return jnp.stack([jnp.interp(z, zgrid, table[:, f]) for f in range(table.shape[1])])

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Multiply ``state.sed_observed`` by the Inoue+2014 transmission.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_observed`` is
            ``None`` this returns ``state`` unchanged.
        params : mapping
            Receives ``igm_*`` keys plus the bare ``redshift`` from the
            allowlist.

        Returns
        -------
        ForwardState
            New state with ``sed_observed *= T_IGM(λ_obs, z)``.

        Notes
        -----
        Observed-frame wavelength is :math:`\lambda_{\rm obs} = (1 + z)\,\lambda_{\rm rest}`.
        The transmission curve from Inoue+2014 takes observed-frame Å.
        """
        if state.sed_observed is None:
            return state

        z = jnp.asarray(params["redshift"])
        wave_obs = state.wave * (1.0 + z)

        # Single flat dispatch honoring the configured mean-IGM model and DLA
        # (was hardcoded to Inoue with no DLA, so the observed-frame
        # photometry/spectroscopy projection silently ignored both — #932).
        dla_z = params.get("dla_z", 0.0)
        T = igm_absorption(
            wave_obs,
            z,
            igm_x_HI=params.get("igm_x_HI", 0.0),
            igm_bubble_mpc=params.get("igm_bubble_mpc", 10.0),
            igm_patchy=self.config.igm_patchy,
            igm_model=self.config.igm_model,
            use_dla=self.config.use_dla,
            dla_z=dla_z,
            dla_log_n_hi=params.get("dla_log_n_hi", 20.0),
            dla_temp=params.get("dla_temp", 1e4),
            dla_b_turb=params.get("dla_b_turb", 0.0),
        )

        # The LUT photometry path consumes ``igm_phot_factor`` (n_filters,) rather
        # than band-averaging ``T`` (n_wave,) at runtime. Publishing the precomputed
        # factor leaves the full-grid curve — and the SED it multiplies — as dead
        # code, which is what XLA must be able to eliminate for WavePrecomp to be
        # fast at all (#932 regressed this: 108 us -> 1764 us).
        band_factor = self._band_factor(z)
        derived = state.derived.with_(igm_transmission=T)
        if band_factor is not None:
            derived = derived.with_(igm_phot_factor=band_factor)

        return state.with_(sed_observed=state.sed_observed * T, derived=derived)


# Register in the unified component dispatch table so build_components resolves
# the IGM component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["igm"] = IGMSEDComponent

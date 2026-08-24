# SPDX-License-Identifier: BSD-3-Clause
"""IGMSEDComponent: intergalactic-medium transmission as a SEDComponent.

A thin wrapper around :func:`tengri.components.igm.igm.igm_transmission`
(Inoue et al. 2014) that satisfies the
:class:`tengri.protocols.SEDComponent` contract.

Design choices (mirrored in :mod:`tengri.components.radio.component`):

- ``parameter_prefix = "igm_"`` for the patchy-reionization extras
  (``igm_x_HI``, ``igm_bubble_mpc``, ``igm_z_mid``, ``igm_dz``,
  ``igm_log_nhi``). The bare ``redshift`` is read via
  :data:`tengri.protocols.component.BARE_NAME_ALLOWLIST`; IGM is the
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

import jax
import jax.numpy as jnp

from tengri.components.igm._params import PARAMS as _IGM_PARAMS
from tengri.components.igm.igm import igm_absorption
from tengri.components.template_threading import TemplateThreading
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["IGMSEDComponent", "IGMSEDComponentConfig"]

#: Floor on the number of redshift nodes for the IGM transmission tables.
#:
#: The IGM table is far cheaper per node than the SSP×filter ztable; a handful of
#: numbers per redshift, against a full SSP block; so it has no business inheriting
#: the SSP grid's ``n_z``. Linear-interpolation error falls as h², and the IGM factor
#: is the steepest function of z in the model (the Lyman forest thickens rapidly),
#: so the coarse default was the dominant interpolation error: 3.0e-3 in sdss_u at
#: z=3.98 with n_z=100 over z∈[0,4]. Densifying the IGM grid alone cuts that to the
#: 1e-4 level for a few hundred kB.
_IGM_MIN_N_Z = 400


def _igm_n_z(n_z: int) -> int:
    """Redshift nodes for the IGM tables: at least :data:`_IGM_MIN_N_Z`."""
    return max(int(n_z), _IGM_MIN_N_Z)


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
        so photometry/spectroscopy see the DLA: not only
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
        (patchy reionization or a DLA: both carry free parameters, so the
        factor is not a function of redshift alone).
    """

    name: str = "igm"
    band_zgrid: Any | None = None
    band_table: Any | None = None
    spec_zgrid: Any | None = None
    spec_table: Any | None = None


@dataclass(frozen=True)
class IGMSEDComponent(TemplateThreading):
    r"""SEDComponent adapter around :func:`igm_transmission`.

    Parameters
    ----------
    config : IGMSEDComponentConfig, optional
        Frozen structural settings. Default :class:`IGMSEDComponentConfig`.

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` is pure JAX.
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
        ``tengri.components.igm._params``: the CGM damping-wing
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
        and padded curves: model-level knowledge this signature does not carry.
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
        it down; a 5994-point transmission to produce five numbers, which
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
        does *not* enter (the transmission is averaged alone) so the whole
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
        **JIT-compatible**: build-time only; call outside any trace.

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
        # ``phot_filter_waves_padded``: same arrays, same quadrature, so the
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
                _igm_n_z(int(spec.get("n_z", 100))),
            )
        else:
            zgrid = np.asarray([float(spec.get("value", 0.0))])

        wave_rest = jnp.asarray(wave_rest)

        # Same build-time-constant story as the sub-band node table: one
        # `igm_absorption` call per redshift, re-paid on every build. This half
        # is the larger one once the node table is cached (#1453). Its key
        # carries the filter curves and the convolution convention too, because
        # unlike the node table this integrates against the bandpass.
        from tengri.components.igm import _subband_cache

        key = _subband_cache.band_factor_key(
            wave_rest, fws, fts, zgrid, self.config.igm_model, photometry.convention
        )
        cached = _subband_cache.memo_get(key)
        if cached is None:
            cached = _subband_cache.load(key)
            if cached is not None:
                _subband_cache.memo_put(key, cached)
        if cached is not None and cached.shape == (len(zgrid), n_filters):
            return IGMSEDComponentState(
                name=self.name,
                band_zgrid=jnp.asarray(zgrid),
                band_table=jnp.asarray(cached),
            )

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

        band_table = jnp.stack(rows)
        _subband_cache.memo_put(key, np.asarray(band_table))
        _subband_cache.store(key, np.asarray(band_table))
        return IGMSEDComponentState(
            name=self.name,
            band_zgrid=jnp.asarray(zgrid),
            band_table=band_table,
        )

    def precompute_spec_factors(
        self,
        wave_rest: jnp.ndarray,
        spec_wave_obs: jnp.ndarray,
        redshift_spec: Mapping[str, Any] | None = None,
    ) -> IGMSEDComponentState:
        r"""Tabulate the per-pixel IGM transmission against redshift.

        The SpectrumPrecomp twin of :meth:`precompute_band_factors`, and the same
        bug: ``predict_spectrum_via_precomp`` sampled the full-grid Inoue+2014
        curve at each pixel on every call, so the LUT bought nothing (2120 us
        exact vs 2098 us LUT, 1.0x).

        Pixel *i* is sampled at its rest effective wavelength
        :math:`\lambda_{{\rm obs},i}/(1+z)` from a curve defined as
        :math:`T(\lambda_{\rm rest}(1+z), z)`: the redshift cancels out of the
        *wavelength*, leaving the transmission at the **fixed observed instrument
        grid**. So the factor is a function of :math:`(z, i)` alone and moves to
        build time.

        Tabulates the *composed* operation (interp through the rest grid), not a
        direct evaluation at the pixels, so the result is **bit-identical** to the
        runtime; a direct evaluation would be marginally more accurate, but that
        is still a behavior change, and this is meant to be a pure speedup.

        Parameters
        ----------
        wave_rest : array_like, shape (n_wave,)
            Rest-frame model wavelength grid [Angstrom].
        spec_wave_obs : array_like, shape (n_pix,)
            Observed-frame spectrum pixel centers [Angstrom].
        redshift_spec : mapping or None
            ``{'mode': 'fixed', 'value': z}`` or ``{'mode': 'free', ...}``.

        Returns
        -------
        IGMSEDComponentState
            Carrying ``spec_zgrid`` / ``spec_table``, shape (n_z, n_pix).

        Notes
        -----
        **JIT-compatible**: build-time only. Patchy reionization and DLAs read
        free parameters, so the factor is not a function of redshift alone;
        those keep the exact full-grid path.
        """
        import numpy as np

        if self.config.igm_patchy or self.config.use_dla:
            return IGMSEDComponentState(name=self.name)
        if wave_rest is None or spec_wave_obs is None:
            return IGMSEDComponentState(name=self.name)

        spec = dict(redshift_spec or {"mode": "fixed", "value": 0.0})
        if spec.get("mode") == "free":
            zgrid = np.linspace(
                float(spec.get("z_min", 0.001)),
                float(spec.get("z_max", 3.0)),
                _igm_n_z(int(spec.get("n_z", 100))),
            )
        else:
            zgrid = np.asarray([float(spec.get("value", 0.0))])

        wave_rest = jnp.asarray(wave_rest)
        wave_obs = jnp.asarray(spec_wave_obs)
        rows = []
        for z in zgrid:
            trans = igm_absorption(
                wave_rest * (1.0 + z),
                z,
                igm_patchy=False,
                igm_model=self.config.igm_model,
                use_dla=False,
            )
            # Exactly what predict_spectrum_via_precomp did at runtime: sample the
            # rest-grid curve at each pixel's rest effective wavelength.
            rows.append(jnp.interp(wave_obs / (1.0 + z), wave_rest, trans))

        return IGMSEDComponentState(
            name=self.name,
            spec_zgrid=jnp.asarray(zgrid),
            spec_table=jnp.stack(rows),
        )

    def subband_node_transmission(
        self,
        subband_waves_rest: Any,
        z_grid: Any,
    ) -> Any | None:
        r"""Evaluate the IGM transmission at the sub-band quadrature nodes.

        The photometry LUT integrates each filter as a K-point sub-band
        quadrature (#1122), and :meth:`precompute_band_factors` averages
        :math:`T` *alone* over the bandpass: forming
        :math:`\langle S \rangle \langle T \rangle` where the flux needs
        :math:`\langle S T \rangle`. Across GALEX FUV at :math:`z \approx 0.8`
        the transmission runs from ~1 to ~0 *inside* the band, so that
        covariance term reaches −9.5 %. Evaluating :math:`T` at the same nodes
        the dust screen uses folds it into one contraction:

        .. math::

            F_b = \sum_a \sum_k \Phi[a, b, k]\;
                  A_{\rm dust}(\lambda^*[a, b, k])\;
                  T_{\rm IGM}(\lambda^*[a, b, k])

        where :math:`\lambda^*` is the sub-band's flux-weighted centroid [Å].

        The nodes are supplied **metallicity-resolved**, before the SSP grid is
        contracted. That is not incidental: the node published at runtime is a
        metallicity-weighted average whose weights move with the free parameter
        ``met_logzsol``, so :math:`T` at "the node" is a function of
        :math:`(z, Z)`: not of :math:`z` alone. Across the SSP metallicity grid
        the node shifts by up to 68 % of a sub-band width and :math:`T` there by
        up to 1.3 % in GALEX FUV. Evaluating on the met axis and folding *before*
        the contraction is exact, and costs nothing at runtime: the product is a
        build-time constant of the same shape as the tensor it multiplies.

        Parameters
        ----------
        subband_waves_rest : array_like
            Rest-frame quadrature nodes [Angstrom], shape
            ``(n_met, n_age, n_filters, n_subbands)`` for a fixed-redshift model
            or ``(n_z, n_met, n_age, n_filters, n_subbands)`` for the free-z
            z-table.
        z_grid : array_like
            The redshift(s) the nodes were tabulated at; a single value for a
            fixed-redshift model, else the z-table's own grid, whose length must
            match the leading axis of ``subband_waves_rest``.

        Returns
        -------
        ndarray or None
            :math:`T` at each node, dimensionless, same shape as
            ``subband_waves_rest``. ``None`` when the transmission is not a
            function of :math:`(\lambda, z)` alone and so cannot be tabulated.

        Notes
        -----
        **JIT-compatible**: build-time only; call outside any trace.

        **Not precomputable** when ``igm_patchy`` or ``use_dla`` is set: both read
        free parameters (``igm_x_HI``, ``igm_bubble_mpc``, ``dla_log_n_hi``, …),
        so :math:`T` moves with the sampler and freezing it here would silently
        pin a live transmission. Those configs return ``None`` and keep the exact
        full-grid path: the gate fails **safe**.
        """
        import numpy as np

        if self.config.igm_patchy or self.config.use_dla:
            return None
        if subband_waves_rest is None or z_grid is None:
            return None

        waves = np.asarray(subband_waves_rest)
        zs = np.atleast_1d(np.asarray(z_grid, dtype=float))

        if zs.shape[0] == 1 and waves.ndim == 4:
            # Fixed redshift: one constant, no z axis.
            z = float(zs[0])
            trans = igm_absorption(
                jnp.asarray(waves.reshape(-1) * (1.0 + z)),
                z,
                igm_patchy=False,
                igm_model=self.config.igm_model,
                use_dla=False,
            )
            return jnp.asarray(trans).reshape(waves.shape)

        if waves.ndim != 5 or waves.shape[0] != zs.shape[0]:
            # Shapes disagree with the z-table contract: refuse rather than
            # broadcast something plausible into the forward model.
            return None

        # The loop below is one `igm_absorption` call per redshift and is a
        # build-time constant, so it is re-paid on every `SEDModel.build`;
        # ~9 s on a free-redshift model, three identical builds in one process
        # each paying in full (#1453). Cache it on content, the way the
        # photometry z-table computed in the same call already is.
        from tengri.components.igm import _subband_cache

        key = _subband_cache.cache_key(
            waves,
            zs,
            self.config.igm_model,
            igm_patchy=self.config.igm_patchy,
            use_dla=self.config.use_dla,
        )
        cached = _subband_cache.memo_get(key)
        if cached is None:
            cached = _subband_cache.load(key)
            if cached is not None:
                _subband_cache.memo_put(key, cached)
        if cached is not None and cached.shape == waves.shape:
            return jnp.asarray(cached)

        rows = []
        for i, z in enumerate(zs):
            nodes = waves[i]
            trans = igm_absorption(
                jnp.asarray(nodes.reshape(-1) * (1.0 + float(z))),
                float(z),
                igm_patchy=False,
                igm_model=self.config.igm_model,
                use_dla=False,
            )
            rows.append(np.asarray(trans).reshape(nodes.shape))
        table = np.stack(rows)
        _subband_cache.memo_put(key, table)
        _subband_cache.store(key, table)
        return jnp.asarray(table)

    def _interp_table(self, z, zgrid, table):
        """Interpolate a (n_z, n_col) table at ``z``; exact for a single node."""
        if zgrid.shape[0] == 1:  # fixed-redshift model: no interpolation at all
            return table[0]
        return jax.vmap(lambda col: jnp.interp(z, zgrid, col), in_axes=1, out_axes=0)(table)

    def _band_factor(self, z: jnp.ndarray) -> jnp.ndarray | None:
        """Interpolate the precomputed per-filter band factors at ``z``."""
        if self._state is None or self._state.band_table is None:
            return None
        return self._interp_table(z, self._state.band_zgrid, self._state.band_table)

    def _spec_factor(self, z: jnp.ndarray) -> jnp.ndarray | None:
        """Interpolate the precomputed per-pixel transmission at ``z``."""
        if self._state is None or self._state.spec_table is None:
            return None
        return self._interp_table(z, self._state.spec_zgrid, self._state.spec_table)

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
        # photometry/spectroscopy projection silently ignored both: #932).
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
        # factor leaves the full-grid curve (and the SED it multiplies) as dead
        # code, which is what XLA must be able to eliminate for WavePrecomp to be
        # fast at all (#932 regressed this: 108 us -> 1764 us).
        band_factor = self._band_factor(z)
        spec_factor = self._spec_factor(z)
        derived = state.derived.with_(igm_transmission=T)
        if band_factor is not None:
            derived = derived.with_(igm_phot_factor=band_factor)
        if spec_factor is not None:
            derived = derived.with_(igm_spec_factor=spec_factor)
        derived = self._fold_transmission_into_subbands(derived, params, z, dla_z)

        return state.with_(sed_observed=state.sed_observed * T, derived=derived)

    def _fold_transmission_into_subbands(self, derived, params, z, dla_z):
        r"""Fold :math:`T` at the photometry sub-band nodes at runtime (#1149).

        The WavePrecomp photometry path integrates the stellar continuum as a
        K-point sub-band quadrature (#1122) and consumes
        ``stellar_phot_lnu_per_age_subband_igm_precomp``: the same tensor with
        :math:`T` folded in at each node; so the projection captures
        :math:`\langle S T \rangle` rather than :math:`\langle S \rangle
        \langle T \rangle`. For a mean-IGM model that tensor is a build-time
        constant (``tengri.forward.sed_model._fold_igm_into_subbands``,
        #1135). Patchy reionization and DLAs read free parameters, so it is not,
        and the tensor is absent: leaving the projector to band-average
        :math:`\langle T \rangle` over the whole flux. Across a Lyman-break band
        at :math:`z = 7` that covariance gap reached +281 %.

        Here :math:`T` is evaluated at the published nodes on **every call** and
        folded onto the stellar sub-band tensor, so the parametric IGM gets the
        same quadrature the mean IGM does. The nodes and the sub-band flux are
        small ``(n_age, n_filter, n_subbands)`` tensors, so this does not pin the
        dense SED grid the way the ``igm_transmission`` fallback does.

        No-op unless the mean-IGM fold is *absent* (``igm_patchy`` / ``use_dla``)
        and the stellar sub-band tensors were published (WavePrecomp): the exact
        path and every mean-IGM model are untouched.

        Notes
        -----
        The node is metallicity-*contracted* (the flux-weighted centroid across
        the SSP metallicity grid), so folding here carries the ~1 % met-node
        residual #1135 avoids by folding before the contraction: negligible next
        to the +281 % it removes.

        **JIT-compatible**: yes, pure array ops on published derived tensors.
        """
        if derived.get("stellar_phot_lnu_per_age_subband_igm_precomp") is not None:
            return derived  # mean-IGM build-time fold already present (#1135)
        sub_per_age = derived.get("stellar_phot_lnu_per_age_subband_precomp")
        node_waves = derived.get("stellar_subband_waves_rest_precomp")
        if sub_per_age is None or node_waves is None:
            return derived  # exact path publishes no sub-band tensors

        t_nodes = igm_absorption(
            node_waves.reshape(-1) * (1.0 + z),
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
        ).reshape(node_waves.shape)
        return derived.with_(stellar_phot_lnu_per_age_subband_igm_precomp=sub_per_age * t_nodes)


# Register in the unified component dispatch table so build_components resolves
# the IGM component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["igm"] = IGMSEDComponent

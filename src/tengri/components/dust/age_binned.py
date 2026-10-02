# SPDX-License-Identifier: BSD-3-Clause
"""Age-binned dust attenuation: N independent screens, each its own law and age window.

Generalizes the Charlot & Fall (2000) two-component model (birth cloud +
diffuse ISM) to an arbitrary number of screens. Each screen ``i`` carries a
registered attenuation law, a V-band optical depth ``dust_tau_i``, and a log-age
window ``(lo_i, hi_i)`` in ``log10(age / yr)``; an unbounded side (``None``)
extends the window to the end of the age axis. The screens are not required to
partition the age axis -- a screen with window ``(None, None)`` is global, which
is how ``two_component`` is reproduced exactly as the N = 2 case (see
:class:`AgeBinnedDustComponent`).

Two-screen age-dependent dust has a long history under other names: FSPS's
birth-cloud/diffuse split, Prospector's ``dust1``/``dust2``, BAGPIPES's
``dust_birth_cloud``/``dust_general_ism``, and CIGALE's
``dustatt_modified_starburst`` nebular/stellar split all implement the same
Charlot & Fall (2000) two-component geometry this module generalizes.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.dust._age_mixture import (
    interval_cover,
    interval_fractions,
    interval_optical_depth,
    interval_transmission,
    lyc_interval_transmissions,
    mix_intervals,
    nebular_interval_weights,
    weighted_interval_transmission,
    window_boundaries,
)
from tengri.components.dust._params import ATTENUATION_PARAMS, DEFAULT_DUST_ETA_BALANCE
from tengri.components.dust.laws._registry import law_kwarg_names, resolve_dust_law
from tengri.components.lyc import credited_log10_lyc, ionizing_mask
from tengri.components.template_threading import TemplateThreading
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import C_AA, LYMAN_LIMIT_AA
from tengri.utils.scale import log10_magnitude, pow10

__all__ = [
    "AgeBinnedDustComponent",
    "AgeBinnedDustComponentConfig",
]

#: Screen structural entry: (law registry key, lo log10(age/yr) or None, hi
#: log10(age/yr) or None). A plain tuple of tuples, not a dict/dataclass, so
#: it stays hashable inside the frozen component config (a JIT-key field).
ScreenSpec = tuple[str, float | None, float | None]

#: Registry default range/bound_check/free_prior for each shared law-shape
#: stem, keyed by the law-function keyword (``"dust_slope"``, ...). Reused so
#: a per-screen declared parameter (``dust_slope_0``, ...) accepts the same
#: admissible range and free prior as the shared stem two_component declares --
#: only the REGISTRY DEFAULT differs, which is the law's own published value
#: (see :func:`_law_kwarg_default`), not this shared Fixed(0.0)/Fixed(-0.7)/...
_SHARED_LAW_PARAM_DECL: dict[str, ParamDeclaration] = {p.name: p for p in ATTENUATION_PARAMS}


def _edge_yr(log_edge: float | None) -> float | None:
    """A ``log10(age/yr)`` window edge as an age [yr]; ``None`` stays unbounded."""
    return None if log_edge is None else float(10.0**log_edge)


def _law_kwarg_default(law: str, law_kw: str) -> float:
    """The attenuation law's OWN published default for one shape parameter.

    Parameters
    ----------
    law : str
        Dust-law registry key.
    law_kw : str
        Law-function keyword (e.g. ``"dust_bump_strength"``).

    Returns
    -------
    float
        The law function's signature default for ``law_kw``.

    Raises
    ------
    ValueError
        If the law declares ``law_kw`` with no default (should not happen for
        any registered shape parameter; every one is optional with a
        citation-backed default).

    Notes
    -----
    **JIT-compatible**: no (construction-time introspection).
    """
    fn = resolve_dust_law(law).callable
    default = inspect.signature(fn).parameters[law_kw].default
    if default is inspect.Parameter.empty:
        raise ValueError(
            f"Dust law {law!r} declares shape parameter {law_kw!r} with no default; "
            f"age_binned cannot derive a registry default for it."
        )
    return float(default)


def validate_screens(screens_raw: Any) -> tuple[ScreenSpec, ...]:
    """Validate and normalize the grammar's ``screens`` list.

    THE single validator for ``dust_attenuation={'type': 'age_binned', ...}``,
    called from the grammar translator
    (``tengri.parameters.groups._translate_age_binned``) so a build
    through ``SEDModel.build`` and a direct construction of
    :class:`AgeBinnedDustComponentConfig` fail the same way.

    Parameters
    ----------
    screens_raw : list of dict
        Each dict has a required ``'law'`` key (a dust-law registry name) and
        an optional ``'window_log_yr'`` key, a ``(lo, hi)`` pair in
        :math:`\\log_{10}(\\mathrm{age}/\\mathrm{yr})` with either side
        (or the whole key) ``None``/omitted meaning unbounded.

    Returns
    -------
    tuple of (str, float or None, float or None)
        One ``(law, lo, hi)`` entry per screen, in the given order.

    Raises
    ------
    ValueError
        Naming the screen index: ``screens`` empty; a screen is not a dict;
        missing/unknown ``'law'``; ``'window_log_yr'`` is not a 2-tuple; both
        edges given with ``lo >= hi``; or an unrecognized key in the screen
        dict.
    """
    from tengri.components.dust.laws._registry import DUST_LAWS

    if not screens_raw:
        raise ValueError(
            "dust_attenuation type='age_binned' requires a non-empty 'screens' list, "
            "e.g. screens=[{'law': 'calzetti', 'window_log_yr': (None, 7.0)}, "
            "{'law': 'cardelli', 'window_log_yr': (None, None)}]."
        )
    resolved: list[ScreenSpec] = []
    for i, screen in enumerate(screens_raw):
        if not isinstance(screen, Mapping):
            raise ValueError(
                f"dust_attenuation 'screens'[{i}] must be a dict with keys 'law' and "
                f"optionally 'window_log_yr', got {screen!r}."
            )
        law = screen.get("law")
        if law is None:
            raise ValueError(f"dust_attenuation 'screens'[{i}] is missing required key 'law'.")
        if law not in DUST_LAWS:
            raise ValueError(
                f"dust_attenuation 'screens'[{i}]: unknown dust law {law!r}. "
                f"Available: {sorted(DUST_LAWS)}."
            )
        window = screen.get("window_log_yr", (None, None))
        if window is None:
            window = (None, None)
        if not (isinstance(window, (tuple, list)) and len(window) == 2):
            raise ValueError(
                f"dust_attenuation 'screens'[{i}]['window_log_yr'] must be a (lo, hi) "
                f"pair of log10(age/yr) values (either may be None), got {window!r}."
            )
        lo, hi = window
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is not None and hi is not None and not (lo < hi):
            raise ValueError(
                f"dust_attenuation 'screens'[{i}]['window_log_yr']=({lo}, {hi}) must have lo < hi."
            )
        extra = set(screen) - {"law", "window_log_yr"}
        if extra:
            raise ValueError(
                f"dust_attenuation 'screens'[{i}] has unrecognized key(s) "
                f"{sorted(extra)!r}; only 'law' and 'window_log_yr' are accepted "
                f"per screen."
            )
        resolved.append((law, lo, hi))
    return tuple(resolved)


@dataclass(frozen=True, kw_only=True)
class AgeBinnedDustComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`AgeBinnedDustComponent`.

    Attributes
    ----------
    screens : tuple of (str, float or None, float or None)
        One ``(law, lo, hi)`` entry per screen, already validated by
        :func:`validate_screens`. ``law`` is a dust-law registry key; ``lo``/
        ``hi`` are the log-age window edges [dex], ``None`` meaning unbounded.
    transition_width_dex : float
        Dispersal width [dex] of every window edge; ``0`` (default) is the
        hard step. Mirrors
        :attr:`~tengri.components.dust.two_component.DustSEDComponentConfig.transition_width_dex`.
    name : str
        Diagnostic identifier. Default ``"dust"`` (matches the sibling
        attenuation components).
    lyc_reprocessed_by : str
        ``'young'`` (default): only the youngest interval's ionizing photons
        are absorbed by ``neb_fesc``; ``'all'``: every age's (FSPS/CIGALE).
    lyc_in_energy_balance : bool
        Keep the Lyman continuum in the energy-balance integral (FSPS parity).
    lyc_escape_geometry : str
        ``'screened'`` (default), ``'birth_cloud_holes'`` or ``'clear'``: a
        covering fraction ``neb_fesc`` of the youngest interval's light
        bypasses the screens that end (the birth-cloud-like ones).
    fdust_credit_active : bool
        Whether the HII-region dust-heating credit can ever be nonzero
        (resolved at build time by ``SEDModel._fdust_credit_active``).
    """

    screens: tuple[ScreenSpec, ...] = ()
    transition_width_dex: float = 0.0
    name: str = "dust"
    lyc_reprocessed_by: str = "young"
    lyc_in_energy_balance: bool = False
    lyc_escape_geometry: str = "screened"
    fdust_credit_active: bool = True

    @property
    def windows_yr(self) -> tuple[tuple[float | None, float | None], ...]:
        """Per-screen age windows ``(lo_yr, hi_yr)``; ``None`` is unbounded."""
        return tuple((_edge_yr(lo), _edge_yr(hi)) for _law, lo, hi in self.screens)

    @property
    def age_boundaries_yr(self) -> tuple[float, ...]:
        """Sorted finite window edges [yr]: what stellar publishes younger-than fractions for."""
        return window_boundaries(self.windows_yr)

    @property
    def age_boundary_width_dex(self) -> float:
        """Dispersal width [dex] of the survival function at those edges."""
        return float(self.transition_width_dex)


@dataclass(frozen=True)
class AgeBinnedDustComponentState(SEDComponentState):
    """State for the age-binned dust component: name only (no cached arrays).

    Mirrors :class:`~tengri.components.dust.two_component.DustSEDComponentState`:
    attenuation is attenuation-only and holds nothing between ``precompute()``
    and ``apply()``.
    """

    name: str = "dust"


@dataclass(frozen=True)
class AgeBinnedDustComponent(TemplateThreading):
    r"""SEDComponent adapter for N-screen, age-binned dust attenuation.

    Each screen :math:`i` has a registered law :math:`k_i(\lambda)`, a V-band
    optical depth :math:`\tau_i` (``dust_tau_i``), and an age window
    :math:`[b_{{\rm lo},i}, b_{{\rm hi},i})` (the ``window_log_yr`` edges as
    ages; an unbounded side extends the window to the end of the age axis).
    The sorted window edges cut the age axis into intervals :math:`j`; the
    transmission in interval :math:`j` is the product of the screens whose
    windows cover it, and the transmission of an SSP node :math:`a` is the
    mixture of the stellar populations the age kernel put on it:

    .. math::

        T_j(\lambda) = \exp\!\Bigl[-\sum_{i \in {\rm cover}(j)} \tau_i\,k_i(\lambda)\Bigr],
        \qquad
        T_a(\lambda) = \sum_j F_j(a)\,T_j(\lambda),

    with :math:`F_j(a)` the fraction of node :math:`a`'s formed mass in
    interval :math:`j`, the differences of the stellar component's
    ``age_boundary_younger_fraction`` rows
    (:mod:`tengri.components.stellar.age_boundary`).  The :math:`F_j` sum to
    one at every node, so tiling windows partition the stellar mass exactly.
    The default is a hard step at every edge; ``transition_width_dex > 0``
    spreads each edge log-logistically (and integrates it exactly through
    each node's mass).

    **Two-component identity**: ``screens = [{'law': law_bc, 'window_log_yr':
    (None, log10(t_birth))}, {'law': law_diff, 'window_log_yr': (None,
    None)}]`` is the N = 2 case, and reproduces
    :class:`~tengri.components.dust.two_component.DustSEDComponent` bit for
    bit: both call the same mixture (``_age_mixture``).

    **Nebular continuum and lines** are lit by stars of every age, so they see
    the interval mixture weighted by where the ionizing photons come from:

    .. math::

        T_{\rm neb}(\lambda) = \sum_j q_j\,T_j(\lambda), \qquad
        q_j = \frac{\sum_a F_j(a)\,L_{\rm LyC}(a)}{\sum_a L_{\rm LyC}(a)}

    with :math:`L_{\rm LyC}(a)` the published per-age ionizing luminosity.
    A screen with a finite lower edge therefore attenuates the lines by its
    ionizing-weighted share, rather than being refused.  The nebular
    transmission is dispatched in one place, so a dedicated nebular screen
    can replace it without touching the stellar mixture.

    **Lyman continuum**: the gas that reprocesses the ionizing photons
    surrounds the youngest stars, i.e. the interval below the lowest edge;
    ``lyc_reprocessed_by``, ``lyc_in_energy_balance`` and ``lyc_escape_geometry``
    act on it as they do for ``two_component``.  Screens with a finite upper
    edge are this component's birth-cloud screens: a hole geometry lets a
    covering fraction of the youngest light bypass them.

    **Scope** (narrower than ``two_component``): no per-source screen choice
    (``nebular_screen``/``shock_screen``/``agn_screen``) -- shock and AGN
    light pass through unattenuated; no clumpy-geometry ``dust_f_obscuration``
    floor; no Lyman-limit clip; no decoupled nebular law (``law_neb`` is
    refused: a dedicated nebular screen is the way to give HII-region light
    its own curve).

    Precedent: two-screen age-dependent attenuation is implemented under
    other names by FSPS (``dust1``/``dust2``, which splits the continuous age
    integral at ``dust_tesc``), Prospector, BAGPIPES
    (``dust_birth_cloud``/``dust_general_ism``, which splits the age bin
    straddling ``t_bc``) and CIGALE (``dustatt_modified_starburst``, a step at
    ``separation_age``); this component implements the same Charlot & Fall
    (2000) geometry for N screens.

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` is pure JAX once the per-screen
    law lookups (static Python registry calls) complete.
    **LUT**: no ``WavePrecomp``/``SpectrumPrecomp`` support. Both raise at
    model construction when combined with ``dust_attenuation={'type':
    'age_binned'}``; a fit's ``approx="auto"`` policy resolves to the exact
    wave-grid path for this type instead of raising (see
    ``tengri.inference.fitter._auto_approx_config`` /
    ``_resolve_batch_fit_approx``).

    References
    ----------
    .. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of
       Starlight by Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """

    config: AgeBinnedDustComponentConfig = field(default_factory=AgeBinnedDustComponentConfig)
    name: str = "dust"
    parameter_prefix: str = "dust_"

    def citations(self) -> tuple[str, ...]:
        """Structurally generalizes Charlot & Fall (2000) two-component dust;
        per-screen attenuation laws are config-driven via
        :data:`tengri.citations.associations.DUST_LAW_CITATIONS`."""
        return ("charlot_fall2000",)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Dust attenuation-derived quantities: absorbed luminosity and spectra.

        Mirrors :meth:`tengri.components.dust.two_component.DustSEDComponent.outputs`
        (no per-screen L_absorbed split is needed -- ``L_absorbed`` is the
        bolometric integral of intrinsic minus attenuated SED).
        """
        return (
            DerivedKey(
                "L_ir",
                "erg/s",
                "Dust IR budget: L_absorbed * dust_eta_balance (energy balance)",
            ),
            DerivedKey(
                "L_absorbed",
                "erg/s",
                "Absorbed UV/optical/NIR luminosity (LyC-masked, #922)",
            ),
            DerivedKey(
                "log_L_ir",
                "dex",
                "log10(L_ir / (erg/s)); the float32-safe form of L_ir",
            ),
            DerivedKey(
                "log_L_absorbed",
                "dex",
                "log10(L_absorbed / (erg/s)); the ABSORBED stellar+nebular energy "
                "budget, independent of dust_eta_balance",
            ),
            DerivedKey("sed_dust_attenuated", "erg/s/Hz", "Attenuated stellar SED"),
            DerivedKey(
                "log_line_lums_attenuated",
                "dex",
                "log10 of the discrete line catalog after the ionizing-weighted "
                "screen mixture; absent when no photoionized backend published one",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Optional cross-component reads: nebular continuum, line catalog, LyC keys.

        Unlike :class:`~tengri.components.dust.two_component.DustSEDComponent`
        this component does not declare ``sed_shock``/``sed_agn`` as inputs
        (out of scope, see the class docstring): shock and AGN light pass
        through unattenuated by this component.
        """
        return (
            DerivedKey(
                "sed_nebular",
                "erg/s/Hz",
                "Nebular continuum to attenuate (Cue/CloudyGrid); zeros for BakedIn",
            ),
            DerivedKey(
                "line_waves",
                "Angstrom",
                "Discrete nebular line wavelengths (Cue/CloudyGrid); absent for BakedIn",
            ),
            DerivedKey(
                "log_line_lums",
                "dex",
                "INTRINSIC log10 line luminosities to redden; absent for BakedIn",
            ),
            DerivedKey(
                "lyc_transmission",
                "",
                "Stellar LyC survival fraction where(lambda<912, neb_fesc, 1); absent for BakedIn",
            ),
            DerivedKey(
                "lyc_fdust",
                "",
                "Absolute HII-region dust-absorption share, lyc_shares(neb_fesc, "
                "neb_fdust_frac)[1]; absent for BakedIn or neb_fdust_frac at 0",
            ),
            DerivedKey(
                "lyc_fesc",
                "",
                "Raw neb_fesc, read by the hole geometries; absent for BakedIn",
            ),
            DerivedKey(
                "log_L_lyc",
                "dex",
                "RAW LyC luminosity of the whole stellar population, read under "
                "lyc_reprocessed_by='all' for the HII-dust credit",
            ),
        )

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns: indexed per screen.

        Generated from :attr:`AgeBinnedDustComponentConfig.screens` the way
        the nonparametric SFH and metallicity ladders generate indexed
        declarations from their bin count (``sfh_cont_ratio_{i}`` in
        :mod:`tengri.components.stellar.sfh.registry`, ``met_bin_{i}`` in
        :mod:`tengri.components.stellar.sfh.met_registry`): one
        ``dust_tau_{i}`` per screen, plus one ``dust_<lawparam>_{i}`` for
        every shape parameter screen ``i``'s own law declares (narrowed via
        ``tengri.components.dust.laws._registry.law_kwarg_names``, so a
        screen never declares a shape parameter its law does not read).

        ``dust_tau_{i}``'s default/prior matches the corresponding
        ``two_component`` per-screen declaration where one exists (screen 0 ->
        ``dust_tau_bc``'s ``Uniform(0, 4, default=1.0)``, screen 1 ->
        ``dust_tau_diff``'s ``Uniform(0, 3, default=0.3)``); screen 2+ (no
        two_component counterpart) falls back to the birth-cloud-style
        ``Uniform(0, 4, default=1.0)``. Every per-screen law parameter's
        REGISTRY DEFAULT is the law's own published value (see
        :func:`_law_kwarg_default`) -- never the shared
        ``dust_slope``/``dust_bump_strength``/``dust_delta``/``dust_Rv``
        Fixed(0.0)-style default two_component's bare stems carry, since
        every screen here is independent (no "nobody asked, inherit the
        shared value" gating): its admissible range/``free_prior`` is reused
        from the shared stem's own declaration in ``ATTENUATION_PARAMS`` (the
        valid numeric range of e.g. ``dust_Rv`` does not depend on which
        screen it sits on).
        """
        decls: list[ParamDeclaration] = []
        tau_priors: dict[int, Uniform] = {
            0: Uniform(0.0, 4.0, default=1.0),
            1: Uniform(0.0, 3.0, default=0.3),
        }
        default_tau_prior = Uniform(0.0, 4.0, default=1.0)
        for i, (law, lo, hi) in enumerate(self.config.screens):
            window_desc = f"({lo}, {hi})"
            decls.append(
                ParamDeclaration(
                    f"dust_tau_{i}",
                    tau_priors.get(i, default_tau_prior),
                    f"Screen {i} V-band optical depth [dimensionless]; "
                    f"law={law!r}, window_log_yr={window_desc}",
                    lambda lo_, hi_: lo_ >= 0,
                    "must have lo >= 0",
                )
            )
            for law_kw in sorted(law_kwarg_names(law)):
                if law_kw == "redshift":
                    continue
                shared = _SHARED_LAW_PARAM_DECL.get(law_kw)
                decls.append(
                    ParamDeclaration(
                        f"{law_kw}_{i}",
                        Fixed(_law_kwarg_default(law, law_kw)),
                        f"{law_kw} on screen {i} (law={law!r}; its own published default)",
                        bound_check=shared.bound_check if shared else None,
                        bound_error=shared.bound_error if shared else "",
                        free_prior=shared.free_prior if shared else None,
                    )
                )
        return decls

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: dict[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> AgeBinnedDustComponentState:
        """Return an empty state (no precomputed arrays; attenuation-only).

        Accepts the usual Protocol arguments for uniformity but loads
        nothing. Mirrors
        :meth:`tengri.components.dust.two_component.DustSEDComponent.precompute`.
        """
        del ssp_data, wave_grid, approx, filters
        return AgeBinnedDustComponentState(name=self.name)

    def _screen_arrays(
        self, params: Mapping[str, jnp.ndarray], wavelength: jnp.ndarray
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """Per-screen ``(tau_i, k_i(wavelength))``, in screen order.

        The single evaluation of every screen's law: the per-node stellar
        screen, the nebular continuum and the line catalog all build on this,
        so a screen's curve is evaluated with exactly the same parameters
        wherever it is used.
        """
        wavelength = jnp.asarray(wavelength)
        out = []
        for i, (law, _lo, _hi) in enumerate(self.config.screens):
            tau_i = jnp.asarray(params[f"dust_tau_{i}"])
            law_kwargs: dict[str, jnp.ndarray] = {}
            for law_kw in law_kwarg_names(law):
                if law_kw == "redshift":
                    z = params.get("redshift")
                    if z is not None:
                        law_kwargs["redshift"] = jnp.asarray(z)
                    continue
                flat = f"{law_kw}_{i}"
                if flat in params:
                    law_kwargs[law_kw] = jnp.asarray(params[flat])
            out.append((tau_i, resolve_dust_law(law)(wavelength, **law_kwargs)))
        return out

    def _interval_tau(
        self, params: Mapping[str, jnp.ndarray], wavelength: jnp.ndarray
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Optical depth of every age interval and the per-screen ``tau_i * k_i``.

        Returns
        -------
        tau_int : ndarray, shape (n_interval, n_wave)
            Sum of the covering screens' optical depths per interval.
        tau_k : list of ndarray, shape (n_wave,)
            ``tau_i * k_i`` per screen, in screen order.
        """
        tau_k = [tau_i * k_i for tau_i, k_i in self._screen_arrays(params, wavelength)]
        cover = interval_cover(self.config.windows_yr, self.config.age_boundaries_yr)
        return interval_optical_depth(tau_k, cover), tau_k

    def _interval_fractions(self, younger_fraction: jnp.ndarray | None, n_age: int) -> jnp.ndarray:
        """``F_j(a)`` from the published younger-than rows; one global interval if none."""
        if not self.config.age_boundaries_yr:
            return jnp.ones((1, n_age))
        return interval_fractions(jnp.asarray(younger_fraction))

    def compute_transmission(
        self,
        params: Mapping[str, jnp.ndarray],
        wavelength: jnp.ndarray,
        younger_fraction: jnp.ndarray | None,
    ) -> jnp.ndarray:
        r"""Age-resolved transmission :math:`T_a(\lambda) = \sum_j F_j(a)\,T_j(\lambda)`.

        Parameters
        ----------
        params : mapping
            Receives ``dust_tau_i`` and every declared per-screen law
            parameter (see :meth:`declared_parameters`), plus the bare
            ``redshift``.
        wavelength : ndarray, shape (n_wave,)
            Rest-frame wavelengths [Å].
        younger_fraction : ndarray, shape (n_boundary, n_age) or None
            The stellar component's ``age_boundary_younger_fraction`` for
            ``config.age_boundaries_yr``; ``None`` only when every screen is
            global (no boundaries).

        Returns
        -------
        ndarray, shape (n_age, n_wave)
            Transmission in ``[0, 1]``.

        Notes
        -----
        **JIT-compatible**: yes, pure ``jnp`` plus build-time registry lookups
        (``self.config.screens`` is a static Python tuple).
        """
        n_age = 1 if younger_fraction is None else jnp.asarray(younger_fraction).shape[1]
        tau_int, _ = self._interval_tau(params, wavelength)
        t_int = interval_transmission(tau_int, 0.0)
        return mix_intervals(self._interval_fractions(younger_fraction, n_age), t_int)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
        ztable_data: Any | None = None,
    ) -> ForwardState:
        """Apply N-screen age-binned attenuation + energy balance.

        ``ssp_data``/``template_data``/``ztable_data`` are accepted for
        Protocol uniformity but unused: this component reads only ``state``
        and ``params``.

        Parameters
        ----------
        state : ForwardState
            Must carry ``wave`` and stellar publications ``lnu_age`` (n_age,
            n_wave) and ``age_boundary_younger_fraction`` (n_boundary, n_age)
            in ``derived``.
        params : mapping
            Receives ``dust_*`` keys declared by :meth:`declared_parameters`
            plus the bare ``redshift``.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` set to the attenuated SED and
            ``L_ir``/``L_absorbed`` published to ``derived``.
        """
        if "lnu_age" not in state.derived or "ssp_ages_yr" not in state.derived:
            raise ValueError(
                "AgeBinnedDustComponent.apply requires upstream stellar publications "
                "(state.derived['lnu_age'] and ['ssp_ages_yr']). Place "
                "StellarSEDComponent before AgeBinnedDustComponent in the chain."
            )

        wave = state.wave
        lnu_age = jnp.asarray(state.derived["lnu_age"])  # (n_age, n_wave)
        n_age = lnu_age.shape[0]

        # ── 1. Interval mixture: T_a = sum_j F_j(a) T_j ──────────────────
        younger = state.derived.get("age_boundary_younger_fraction")
        if self.config.age_boundaries_yr and younger is None:
            raise ValueError(
                "AgeBinnedDustComponent.apply requires the stellar component's "
                "age_boundary_younger_fraction (state.derived). SEDModel asks stellar for "
                "it at build time via config.age_boundaries_yr; a hand-built chain must "
                "set StellarSEDComponentConfig(age_boundaries_yr=config.age_boundaries_yr, "
                "age_boundary_width_dex=config.age_boundary_width_dex)."
            )
        fractions = self._interval_fractions(younger, n_age)  # (n_interval, n_age)
        # Mass fraction of each node in the youngest interval: the stars the gas
        # around the birth clouds reprocesses (one global interval -> everyone).
        y_young = fractions[0]
        tau_int, tau_k = self._interval_tau(params, wave)
        t_int = interval_transmission(tau_int, 0.0)  # (n_interval, n_wave)
        sed_intrinsic_stellar = jnp.sum(lnu_age, axis=0)

        # ── 2. Lyman-continuum reprocessing + stellar attenuation ────────
        _lyc_t = state.derived.get("lyc_transmission")
        sed_intrinsic_stellar_eb = sed_intrinsic_stellar
        if _lyc_t is None:
            sed_attenuated = jnp.sum(lnu_age * mix_intervals(fractions, t_int), axis=0)
        else:
            _lyc_t = jnp.asarray(_lyc_t)
            sed_intrinsic_stellar = sed_intrinsic_stellar * _lyc_t
            cover = interval_cover(self.config.windows_yr, self.config.age_boundaries_yr)
            covered_raw = jnp.exp(-tau_int[0])
            # The screens that END (finite upper edge) are the birth-cloud ones a
            # hole bypasses; the rest of the youngest interval's screens stay.
            ism_tau = [
                t
                for t, (_lo, hi), covers in zip(tau_k, self.config.windows_yr, cover)
                if hi is None and covers[0]
            ]
            hole_tau = sum(ism_tau[1:], ism_tau[0]) if ism_tau else jnp.zeros_like(wave)
            hole_raw = (
                jnp.exp(-hole_tau)
                if self.config.lyc_escape_geometry == "birth_cloud_holes"
                else jnp.ones_like(wave)
            )
            observed_int, intrinsic_int = lyc_interval_transmissions(
                t_int,
                lyc_t=_lyc_t,
                mode=self.config.lyc_reprocessed_by,
                geometry=self.config.lyc_escape_geometry,
                f_esc=state.derived.get("lyc_fesc", 0.0),
                f_obscuration=0.0,
                ionizing=ionizing_mask(wave),
                covered_raw=covered_raw,
                hole_raw=hole_raw,
            )
            sed_attenuated = jnp.sum(lnu_age * mix_intervals(fractions, observed_int), axis=0)
            sed_intrinsic_stellar_eb = jnp.sum(
                lnu_age * mix_intervals(fractions, intrinsic_int), axis=0
            )

        # ── 3. Nebular continuum + line catalog: ionizing-weighted mixture ──
        # HII regions are lit by stars of every age, so the nebular light sees
        # the interval mixture weighted by each interval's share of the ionizing
        # luminosity, not by mass. ONE dispatch for the nebular transmission: a
        # dedicated nebular screen replaces ``_nebular_transmission`` alone.
        neb_weights = nebular_interval_weights(state.derived, fractions)
        transmission_neb = self._nebular_transmission(params, wave, neb_weights)
        _sed_neb = state.derived.get("sed_nebular")
        sed_neb = jnp.zeros_like(wave) if _sed_neb is None else jnp.asarray(_sed_neb)
        sed_neb_attenuated = sed_neb * transmission_neb

        _line_waves = state.derived.get("line_waves")
        _log_line_lums = state.derived.get("log_line_lums")
        log_line_lums_attenuated = None
        if _line_waves is not None and _log_line_lums is not None:
            transmission_lines = self._nebular_transmission(
                params, jnp.asarray(_line_waves), neb_weights
            )
            log_line_lums_attenuated = jnp.asarray(_log_line_lums) + log10_magnitude(
                transmission_lines
            )

        # ── 4. Combine stellar + nebular SEDs ────────────────────────────
        # Preserve any non-stellar contribution already added to
        # ``sed_intrinsic`` upstream (radio/X-ray/shock/AGN run unattenuated
        # by this component, matching two_component's "none" screen choice;
        # see the class docstring's Scope note).
        if state.sed_intrinsic is None:
            non_stellar_pre_dust = jnp.zeros_like(wave)
        else:
            non_stellar_pre_dust = state.sed_intrinsic - sed_intrinsic_stellar
        non_stellar_other = non_stellar_pre_dust - sed_neb
        sed_total = non_stellar_other + sed_neb_attenuated + sed_attenuated

        # ── 5. Energy balance: integral of (intrinsic - attenuated) dν ───
        # Mirrors two_component's exact-path integral: stellar + nebular
        # intrinsic/attenuated pairs, LyC-masked unless ``lyc_in_energy_balance``.
        nu = C_AA / wave
        from tengri.forward.energy_balance import (
            bolometric_absorbed_log10,
            log10_add_fdust_credit,
            warn_if_corrupt,
        )

        _eb_cutoff = None if self.config.lyc_in_energy_balance else LYMAN_LIMIT_AA
        log_L_absorbed, _ = bolometric_absorbed_log10(
            sed_intrinsic_stellar_eb + sed_neb,
            sed_attenuated + sed_neb_attenuated,
            nu,
            wave=wave,
            lyman_cutoff_aa=_eb_cutoff,
        )
        # HII-region dust-heating credit: the same helper, population and gate
        # as two_component (the credited LyC is that of the population the
        # nebular escape/dust factor was applied to).
        _lyc_fdust = state.derived.get("lyc_fdust") if self.config.fdust_credit_active else None
        if _lyc_fdust is not None:
            _log_l_lyc_credited = credited_log10_lyc(
                state.derived, y_young, self.config.lyc_reprocessed_by
            )
            if _log_l_lyc_credited is not None:
                log_L_absorbed = log10_add_fdust_credit(
                    log_L_absorbed, _log_l_lyc_credited, jnp.asarray(_lyc_fdust)
                )
        warn_if_corrupt(log_L_absorbed, component="age_binned")

        eta_balance = jnp.asarray(params.get("dust_eta_balance", DEFAULT_DUST_ETA_BALANCE))
        eta_positive = eta_balance > 0
        log_L_ir = jnp.where(
            eta_positive,
            log_L_absorbed + jnp.log10(jnp.where(eta_positive, eta_balance, 1.0)),
            -jnp.inf,
        )
        L_absorbed = pow10(log_L_absorbed)
        L_ir = pow10(log_L_ir)

        derived_overrides = dict(
            L_ir=L_ir,
            L_absorbed=L_absorbed,
            log_L_ir=log_L_ir,
            log_L_absorbed=log_L_absorbed,
            sed_dust_attenuated=sed_attenuated,
            # Re-published dust-reddened, mirroring two_component (#2528):
            # downstream consumers summing per-component SEDs then recover
            # the observed total.
            sed_nebular=sed_neb_attenuated,
        )
        if log_line_lums_attenuated is not None:
            derived_overrides["log_line_lums_attenuated"] = log_line_lums_attenuated

        return state.with_(
            sed_intrinsic=sed_total,
            derived=state.derived.with_(**derived_overrides),
        )

    def _nebular_transmission(
        self,
        params: Mapping[str, jnp.ndarray],
        wavelength: jnp.ndarray,
        neb_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Transmission of the nebular continuum / lines: :math:`\sum_j q_j T_j(\lambda)`.

        The one place the nebular screen is chosen: every nebular consumer
        (continuum, line catalog) calls it, and a dedicated nebular screen
        replaces it without touching the stellar mixture.

        Parameters
        ----------
        params : mapping
            Receives the per-screen ``dust_tau_i`` and law parameters.
        wavelength : ndarray, shape (n,)
            Rest-frame wavelengths [Å] (the full grid for the continuum, the
            discrete line wavelengths for the catalog).
        neb_weights : ndarray, shape (n_interval,)
            Share of the ionizing luminosity produced in each age interval
            (``ionizing_interval_weights``).

        Returns
        -------
        ndarray, shape (n,)
            Transmission in ``[0, 1]``.
        """
        tau_int, _ = self._interval_tau(params, wavelength)
        return weighted_interval_transmission(neb_weights, interval_transmission(tau_int, 0.0))


# Register in the unified component dispatch table so the grammar type
# ``dust_attenuation={'type': 'age_binned'}`` resolves via
# _resolve_registry_component (the single dispatch seam, #844), not a
# hardcoded class in build_components.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["age_binned"] = AgeBinnedDustComponent

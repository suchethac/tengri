# SPDX-License-Identifier: BSD-3-Clause
"""Age-binned dust attenuation: N independent screens, each its own law and age window.

Generalizes the Charlot & Fall (2000) two-component model (birth cloud +
diffuse ISM) to an arbitrary number of screens. Each screen ``i`` carries a
registered attenuation law, a V-band optical depth ``dust_tau_i``, and a log-age
window ``(lo_i, hi_i)`` in ``log10(age / yr)``; an unbounded side (``None``)
contributes a factor of 1. The screens are not required to partition the age
axis -- a screen with window ``(None, None)`` is global, which is how
``two_component`` is reproduced exactly as the N = 2 case (see
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

import jax
import jax.numpy as jnp

from tengri.components.dust._params import ATTENUATION_PARAMS, DEFAULT_DUST_ETA_BALANCE
from tengri.components.dust.laws._registry import law_kwarg_names, resolve_dust_law
from tengri.components.template_threading import TemplateThreading
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import C_AA
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


def _age_window_weight(
    log_age: jnp.ndarray,
    lo_log_age: float | None,
    hi_log_age: float | None,
    transition_width_dex: float,
) -> jnp.ndarray:
    r"""Logistic window weight for one age-binned screen.

    .. math::

        w_i(t) = \sigma\!\left(\frac{\log_{10} t - l_{o,i}}{\Delta}\right)
                  \cdot \sigma\!\left(\frac{h_{i,i} - \log_{10} t}{\Delta}\right)

    where :math:`\sigma` is the logistic sigmoid, :math:`\Delta` is
    ``transition_width_dex``, and an unbounded side (``None``) contributes a
    factor of 1 (the corresponding sigmoid is never evaluated, so the weight
    is exactly 1.0 there, not merely close to it).

    Parameters
    ----------
    log_age : ndarray, shape (n_age,)
        :math:`\log_{10}(\mathrm{age}/\mathrm{yr})` grid.
    lo_log_age : float or None
        Lower window edge :math:`l_o` [dex]; ``None`` means unbounded below.
    hi_log_age : float or None
        Upper window edge :math:`h_i` [dex]; ``None`` means unbounded above.
    transition_width_dex : float
        Sigmoid width :math:`\Delta` [dex] (shared across every screen).

    Returns
    -------
    ndarray, shape (n_age,)
        Window weight :math:`w_i(t) \in (0, 1]` [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes; ``lo_log_age``/``hi_log_age`` are static Python
    floats or ``None``, never traced values, so the branches below resolve at
    trace-build time.
    """
    log_age = jnp.asarray(log_age)
    w = jnp.ones_like(log_age)
    if lo_log_age is not None:
        w = w * jax.nn.sigmoid((log_age - lo_log_age) / transition_width_dex)
    if hi_log_age is not None:
        w = w * jax.nn.sigmoid((hi_log_age - log_age) / transition_width_dex)
    return w


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
        Shared logistic transition width :math:`\\Delta` [dex] for every
        screen's window (mirrors
        :attr:`~tengri.components.dust.two_component.DustSEDComponentConfig.transition_width_dex`).
        Default 0.3.
    name : str
        Diagnostic identifier. Default ``"dust"`` (matches the sibling
        attenuation components).
    """

    screens: tuple[ScreenSpec, ...] = ()
    transition_width_dex: float = 0.3
    name: str = "dust"


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
    :math:`(l_{o,i}, h_{i,i})` in :math:`\log_{10}(\mathrm{age}/\mathrm{yr})`.
    The optical depth seen by an SSP of age :math:`t` is

    .. math::

        \tau(t, \lambda) = \sum_i w_i(t) \, \tau_i \, k_i(\lambda), \qquad
        w_i(t) = \sigma\!\left(\frac{\log_{10} t - l_{o,i}}{\Delta}\right)
                  \sigma\!\left(\frac{h_{i,i} - \log_{10} t}{\Delta}\right)

    with :math:`\sigma` the logistic sigmoid, :math:`\Delta` the shared
    ``transition_width_dex``, and an unbounded window side contributing a
    factor of 1. Transmission is :math:`T(t, \lambda) = \exp[-\tau(t,
    \lambda)]`. Windows need not partition the age axis: a screen with window
    :math:`(-\infty, \infty)` (``window_log_yr=(None, None)``) is global.

    **Two-component identity**: ``screens = [{'law': law_bc, 'window_log_yr':
    (None, log10(t_birth))}, {'law': law_diff, 'window_log_yr': (None,
    None)}]`` reproduces
    :class:`~tengri.components.dust.two_component.DustSEDComponent` exactly:
    :math:`w_0(t)` reduces to the same birth-cloud sigmoid
    (``two_component``'s :math:`y(t)`) and :math:`w_1(t) \equiv 1`, so
    :math:`\tau(t,\lambda) = w_0(t)\,\tau_{\rm bc}\,k_{\rm bc}(\lambda) +
    \tau_{\rm diff}\,k_{\rm diff}(\lambda)`, the Charlot & Fall (2000)
    two-component optical depth.

    Nebular continuum and the discrete emission-line catalog are attenuated
    at the youngest age (the gas sits around the youngest stars):

    .. math::

        \tau_{\rm neb}(\lambda) = \lim_{t \to 0} \tau(t, \lambda)
             = \sum_{i:\, l_{o,i} = -\infty} \tau_i \, k_i(\lambda)

    (as :math:`t \to 0`, :math:`\log_{10} t \to -\infty`: a screen with a
    *finite* lower edge has :math:`w_i \to 0` there, so only the screens
    whose window is unbounded below survive). For the two-component mapping
    above this is exactly :math:`\tau_{\rm bc}\,k_{\rm bc}(\lambda) +
    \tau_{\rm diff}\,k_{\rm diff}(\lambda)`, matching
    ``two_component``'s ``nebular_screen="birth_cloud"`` default.

    **Scope** (deliberately narrower than ``two_component``, #2528): no
    per-source screen choice (``nebular_screen``/``shock_screen``/
    ``agn_screen``) -- every screen attenuates the nebular continuum and line
    catalog uniformly via the youngest-age rule above; no clumpy-geometry
    ``dust_f_obscuration`` floor; no Lyman-limit clip; no Lyman-continuum
    escape-fraction (``neb_fesc``) special-casing; no decoupled nebular law
    (``law_neb``/``*_neb`` overrides). These remain ``two_component``
    features. A model wanting them selects ``dust_attenuation={'type':
    'two_component', ...}`` instead.

    Precedent: two-screen age-dependent attenuation is implemented under
    other names by FSPS (``dust1``/``dust2``), Prospector, BAGPIPES
    (``dust_birth_cloud``/``dust_general_ism``), and CIGALE
    (``dustatt_modified_starburst``'s nebular/stellar split); this component
    generalizes the same Charlot & Fall (2000) geometry to N screens.

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
        bolometric integral of intrinsic minus attenuated SED exactly as
        today, see the class docstring).
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
                "log10 of the discrete line catalog after the youngest-age screen; "
                "absent when no photoionized backend published one",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Optional cross-component reads: nebular continuum and line catalog.

        Unlike :class:`~tengri.components.dust.two_component.DustSEDComponent`
        this component does not declare ``sed_shock``/``sed_agn``/
        ``lyc_transmission`` as inputs (out of scope, see the class
        docstring): shock and AGN light pass through unattenuated by this
        component (the ``"none"`` screen choice in ``two_component``'s
        vocabulary), and Lyman-continuum escape is not special-cased.
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
    ) -> list[tuple[jnp.ndarray, jnp.ndarray, float | None, float | None]]:
        """Per-screen ``(tau_i, k_i(wavelength), lo_i, hi_i)``, in screen order.

        The single evaluation of every screen's law: :meth:`compute_transmission`
        (the stellar per-age screen) and the youngest-age nebular/line tau both
        build on this, so a screen's curve is evaluated with exactly the same
        parameters wherever it is used.
        """
        wavelength = jnp.asarray(wavelength)
        out = []
        for i, (law, lo, hi) in enumerate(self.config.screens):
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
            k_i = resolve_dust_law(law)(wavelength, **law_kwargs)
            out.append((tau_i, k_i, lo, hi))
        return out

    def compute_transmission(
        self,
        params: Mapping[str, jnp.ndarray],
        wavelength: jnp.ndarray,
        ssp_ages_yr: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Age-resolved transmission :math:`T(t, \lambda) = \exp[-\tau(t,\lambda)]`.

        Parameters
        ----------
        params : mapping
            Receives ``dust_tau_i`` and every declared per-screen law
            parameter (see :meth:`declared_parameters`), plus the bare
            ``redshift``.
        wavelength : ndarray, shape (n_wave,)
            Rest-frame wavelengths [Å].
        ssp_ages_yr : ndarray, shape (n_age,)
            SSP lookback ages [yr].

        Returns
        -------
        ndarray, shape (n_age, n_wave)
            Transmission in ``[0, 1]``.

        Notes
        -----
        **JIT-compatible**: yes, pure ``jnp`` plus build-time registry lookups
        (``self.config.screens`` is a static Python tuple).
        """
        wavelength = jnp.asarray(wavelength)
        ssp_ages_yr = jnp.asarray(ssp_ages_yr)
        log_age = jnp.log10(jnp.maximum(ssp_ages_yr, 1.0))
        tau = jnp.zeros((ssp_ages_yr.shape[0], wavelength.shape[0]))
        for tau_i, k_i, lo, hi in self._screen_arrays(params, wavelength):
            w_i = _age_window_weight(log_age, lo, hi, self.config.transition_width_dex)
            tau = tau + (w_i[:, None] * tau_i) * k_i[None, :]
        return jnp.exp(-tau)

    def _youngest_screen_tau(
        self, params: Mapping[str, jnp.ndarray], wavelength: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Youngest-age optical depth :math:`\tau_{\rm neb}(\lambda)`.

        See the class docstring for the :math:`t \to 0` derivation: only
        screens unbounded below (``lo is None``) contribute.

        Parameters
        ----------
        params : mapping
            Receives ``dust_tau_i`` and per-screen law parameters.
        wavelength : ndarray, shape (n,)
            Rest-frame wavelengths [Å] (the full grid for the nebular
            continuum, or the discrete line wavelengths for the line
            catalog).

        Returns
        -------
        ndarray, shape (n,)
            Optical depth (not transmission); caller applies ``exp(-tau)``.

        Notes
        -----
        **JIT-compatible**: yes.
        """
        wavelength = jnp.asarray(wavelength)
        tau = jnp.zeros_like(wavelength)
        for tau_i, k_i, lo, _hi in self._screen_arrays(params, wavelength):
            if lo is None:
                tau = tau + tau_i * k_i
        return tau

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
            n_wave) and ``ssp_ages_yr`` (n_age,) in ``derived``.
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
        ssp_ages_yr = jnp.asarray(state.derived["ssp_ages_yr"])

        # ── 1. Age-resolved transmission + stellar attenuation ──────────
        transmission = self.compute_transmission(params, wave, ssp_ages_yr)
        lnu_age_attenuated = lnu_age * transmission
        sed_attenuated = jnp.sum(lnu_age_attenuated, axis=0)
        sed_intrinsic_stellar = jnp.sum(lnu_age, axis=0)

        # ── 2. Nebular continuum + line catalog, youngest-age screen ────
        tau_neb = self._youngest_screen_tau(params, wave)
        transmission_neb = jnp.exp(-tau_neb)
        _sed_neb = state.derived.get("sed_nebular")
        sed_neb = jnp.zeros_like(wave) if _sed_neb is None else jnp.asarray(_sed_neb)
        sed_neb_attenuated = sed_neb * transmission_neb

        _line_waves = state.derived.get("line_waves")
        _log_line_lums = state.derived.get("log_line_lums")
        log_line_lums_attenuated = None
        if _line_waves is not None and _log_line_lums is not None:
            line_wave = jnp.asarray(_line_waves)
            tau_neb_lines = self._youngest_screen_tau(params, line_wave)
            transmission_lines = jnp.exp(-tau_neb_lines)
            log_line_lums_attenuated = jnp.asarray(_log_line_lums) + log10_magnitude(
                transmission_lines
            )

        # ── 3. Combine stellar + nebular SEDs ────────────────────────────
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

        # ── 4. Energy balance: integral of (intrinsic - attenuated) dν ───
        # Mirrors two_component's exact-path integral exactly: stellar +
        # nebular intrinsic/attenuated pairs, LyC-masked (#922). No per-screen
        # L_absorbed split is needed (the brief: "L_absorbed is the integral
        # of intrinsic minus attenuated SED exactly as today").
        nu = C_AA / wave
        from tengri.forward.energy_balance import bolometric_absorbed_log10, warn_if_corrupt

        log_L_absorbed, _ = bolometric_absorbed_log10(
            sed_intrinsic_stellar + sed_neb,
            sed_attenuated + sed_neb_attenuated,
            nu,
            wave=wave,
            lyman_cutoff_aa=912.0,
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


# Register in the unified component dispatch table so the grammar type
# ``dust_attenuation={'type': 'age_binned'}`` resolves via
# _resolve_registry_component (the single dispatch seam, #844), not a
# hardcoded class in build_components.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["age_binned"] = AgeBinnedDustComponent

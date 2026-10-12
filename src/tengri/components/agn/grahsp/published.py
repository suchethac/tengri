# SPDX-License-Identifier: BSD-3-Clause
r"""Bolometric, torus and absorbed powers of the GRAHSP component (``log_L_agn_*`` keys).

The component publishes ``log_L_agn_bol``, ``log_L_agn_torus`` and ``log_L_agn_absorbed``
[dex re erg/s] and their ``L_agn_*_lsun`` [Lsun] siblings. They are formed from the three
linear powers this module returns, which are internal values at the evaluation scale of the
component (keys ``L_agn_bol``, ``L_agn_torus``, ``L_agn_absorbed`` of an internal dict, erg/s,
never published as such).

Every such power is :math:`\int L_\lambda\,\mathrm{d}\lambda` [erg/s] of the emitted
AGN-side spectrum, with :math:`\lambda` in nm, computed on a grid the component owns and never
on the caller's wavelength array. Each piece of the spectrum has its own rule, chosen by the
piece and not by the other options of the configuration, so swapping the torus model, the disc
or the FeII template changes only the rule of the piece that was swapped:

1. **Unit-area Gaussian** (lines, Si feature). For :math:`G(\lambda;c,s)=
   (\sqrt{2\pi}\,s)^{-1}\exp[-(\lambda-c)^2/(2s^2)]`, :math:`\int G\,\mathrm{d}\lambda = 1` and
   :math:`\int_a^\infty G\,\mathrm{d}\lambda = \tfrac12\,\mathrm{erfc}[(a-c)/(\sqrt2\,s)]`, which
   is the Lyman-limit cut of the lines in the bolometric power. With a weight :math:`h`,
   :math:`\int G h\,\mathrm{d}\lambda = \mathbb{E}_{\lambda\sim N(c,s^2)}[h]`. The Si features,
   far from the kink of :math:`h`, use Gauss-Hermite quadrature with nodes
   :math:`c+\sqrt2\,s\,x_i`; the lines, some of which sit within a few sigma of the 1100 nm
   attenuation kink, use segmented Gauss-Legendre cut at the kink.
2. **Log-normal weight** (the Gaussian torus pair). Write
   :math:`u=\log_{10}(\lambda/\mu\mathrm{m})` and :math:`a=\ln 10`. Completing the square,
   :math:`\exp[-(u-\mu)^2/(2W^2)+au]=\exp(a\mu+a^2W^2/2)\,\exp[-(u-\mu')^2/(2W^2)]` with
   :math:`\mu'=\mu+aW^2`. Since :math:`\mathrm{d}\lambda=1000\,a\,10^u\,\mathrm{d}u` [nm],
   :math:`\int \exp[-(u-\mu)^2/(2W^2)]\,\mathrm{d}\lambda = C_0(W)\,10^{\mu}` with
   :math:`C_0(W)=1000\,a\sqrt{2\pi}\,W\,e^{a^2W^2/2}` [nm], the closed form over
   :math:`(0,\infty)`. With a weight :math:`h`, the same expectation is taken over
   :math:`u\sim N(\mu',W^2)`.
3. **FeII forest** (either template). The forest is linear in :math:`\lambda` between its nodes
   and zero outside, so :math:`\int L\,\mathrm{d}\lambda` is the trapezoid on its nodes (exact),
   and under an attenuation :math:`h` a two-point Gauss-Legendre rule on each linear segment,
   exact for cubics.
4. **Tabulated pieces** (the Netzer disc, the Mor & Netzer 2012 torus and silicate template).
   Each is a smooth power law between its native nodes, so the integral is segmented
   Gauss-Legendre in :math:`\ln\lambda` on the native nodes (plus the attenuation break), with
   the kink of the emitted ``max(si, -torus)`` clip placed at its root inside each segment.
5. **Smooth bending power law and Balmer continuum.** Segmented Gauss-Legendre in
   :math:`\ln\lambda` on a log grid that carries the hard edges of the piece.

The quadrature rules and their Gauss-Hermite and Gauss-Legendre orders live in
``grahsp/_publication_rules.py``, where each order carries the measured error of the
approximation it sets; the identity tests in
``tests/physics/test_grahsp_published_luminosities.py`` check every option combination against
an independent dense integral of the emitted SED.

Notes
-----
**JIT-compatible**: yes; shapes are fixed by the configuration. **Gradient-safe**: yes.
**Inclination**: GRAHSP has no inclination parameter. Its bi-attenuation is one line-of-sight
screen applied to the whole AGN side, so ``L_agn_absorbed`` is the energy that screen removes,
there is no viewing geometry to angle-integrate, and the ``_los`` split of the inclination
dependent emitters does not apply.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erfc

from tengri.components.agn.grahsp._publication_rules import (
    _BALMER_HI_NM,
    _DISC_GRID_HI_NM,
    BALMER_PER_DECADE,
    CLIP_STEP_SIGMA,
    LINE_GL_ORDER,
    LINE_SUBSEGMENTS,
    LINE_WINDOW_SIGMA,
    LOG_GL_ORDER,
    LOG_WINDOW_SIGMA,
    SBPL_PER_DECADE,
    SI_GH_ORDER,
    feii_segment_product,
    gauss_hermite,
    line_expectation,
    log_breaks,
    log_gaussian_expectation,
    lognormal_c0,
    split_gaussian_expectation,
)
from tengri.components.agn.grahsp.attenuation import PREVOT_DEFAULTS
from tengri.components.agn.grahsp.balmer import (
    _BALMER_EDGE_NM,
    _BC_CONVOLUTION_THRESHOLD_NM,
    balmer_continuum,
)
from tengri.components.agn.grahsp.bbb import XRAY_FLOOR_NM, floor_disc_xray
from tengri.components.agn.grahsp.bolometric import LYMAN_LIMIT_NM
from tengri.components.agn.grahsp.lines import (
    _HBETA_BROAD_RATIO,
    _HBETA_NARROW_RATIO,
    _LAMBDA_5100_NM,
)
from tengri.components.agn.grahsp.publication import (
    PUBLICATION_HI_NM,
    WINDOW_SIGMA,
    gauss_legendre_segments,
    linewidth_sigma_nm,
    window_edges,
)
from tengri.components.agn.grahsp.torus import SI_DEFAULTS, _log_gaussian_pair

if TYPE_CHECKING:
    from tengri.components.agn.grahsp.component import GRAHSPSEDComponentState

__all__ = [
    "BALMER_PER_DECADE",
    "CLIP_STEP_SIGMA",
    "LINE_GL_ORDER",
    "LINE_SUBSEGMENTS",
    "LINE_WINDOW_SIGMA",
    "LOG_GL_ORDER",
    "LOG_WINDOW_SIGMA",
    "SBPL_PER_DECADE",
    "SI_GH_ORDER",
    "PublishedLuminosities",
    "feii_segment_product",
    "gauss_hermite",
    "line_expectation",
    "log_breaks",
    "log_gaussian_expectation",
    "lognormal_c0",
    "split_gaussian_expectation",
]


def _net_shape_is_nonnegative(
    lo: Array,
    hi: Array,
    torus: tuple,
    si: Array,
    centers: np.ndarray,
    sigmas: np.ndarray,
) -> Array:
    r"""True when the torus-plus-Si shape is certified non-negative on every segment.

    The shape is :math:`T(\lambda) + s\,[A_{\rm em} g_{\rm em} - A_{\rm ab} g_{\rm ab}]`, with
    :math:`T` the sum of two log-Gaussians (each non-negative) and :math:`g` unit-peak
    Gaussians. On a segment :math:`[a,b]` each Gaussian is unimodal, so its minimum is at an
    end and its maximum at the center clipped into the segment. Summing the lower bounds of
    the terms (a negative coefficient takes the maximum) gives a lower bound of the shape.
    When every segment's bound is non-negative the clip :math:`\max(0,-\mathrm{shape})`
    vanishes identically, whatever the parameters: the test cannot return True where the
    shape dips below zero.

    Parameters
    ----------
    lo, hi : array_like, shape (n_seg,)
        Segment ends [nm], ascending.
    torus : tuple
        ``(log_cool_um, cool_width, log_hot_um, hot_width, hot_fcov, spec12)``.
    si : array_like
        Si strength.
    centers, sigmas : ndarray, shape (2,)
        Si emission and absorption centers and widths [nm].

    Returns
    -------
    ndarray, shape ()
        Boolean scalar.
    """
    log_cool, w_cool, log_hot, w_hot, fhot, spec12 = torus
    u = jnp.log10(jnp.stack([lo, hi]) / 1000.0)
    cool = jnp.exp(-0.5 * ((u - log_cool) / w_cool) ** 2)
    hot = fhot * 10.0 ** (log_cool - log_hot) * jnp.exp(-0.5 * ((u - log_hot) / w_hot) ** 2)
    bound = (jnp.min(cool, axis=0) + jnp.min(hot, axis=0)) / spec12
    em_amp = SI_DEFAULTS["si_em_ampl"]
    ab_amp = em_amp * SI_DEFAULTS["si_ratio"]
    for coeff, center, sigma in (
        (si * em_amp, centers[0], sigmas[0]),
        (-si * ab_amp, centers[1], sigmas[1]),
    ):
        ends = jnp.exp(-0.5 * ((jnp.stack([lo, hi]) - center) / sigma) ** 2)
        peak = jnp.exp(-0.5 * ((jnp.clip(center, lo, hi) - center) / sigma) ** 2)
        bound = bound + jnp.where(coeff >= 0.0, coeff * jnp.min(ends, axis=0), coeff * peak)
    return jnp.all(bound >= 0.0)


def _sign_change_roots(fn, lo: Array, hi: Array, iterations: int = 4) -> Array:
    r"""Regula-falsi roots of ``fn`` on the intervals ``[lo, hi]`` with a sign change.

    Intervals without a sign change return their midpoint, so the split is harmless there.
    Used to place the kinks of :math:`\max(0,-f)` at the crossings of :math:`f`; the roots are
    stopped from the gradient, because :math:`\max(0,-f)` is continuous there.

    Parameters
    ----------
    fn : callable
        ``fn(wave_nm) -> ndarray`` (vectorized).
    lo, hi : array_like, shape (n,)
        Interval ends [nm], ascending.
    iterations : int, optional
        Regula-falsi steps.

    Returns
    -------
    ndarray, shape (n,)
        The crossing estimates [nm], inside ``[lo, hi]``.
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    f_lo = fn(lo)
    f_hi = fn(hi)
    crosses = f_lo * f_hi < 0.0
    safe = jnp.where(crosses, f_hi - f_lo, 1.0)
    root = jnp.where(crosses, lo - f_lo * (hi - lo) / safe, 0.5 * (lo + hi))
    a, b, fa, fb = lo, hi, f_lo, f_hi
    for _ in range(iterations):
        f_root = fn(root)
        left = fa * f_root < 0.0
        b = jnp.where(left, root, b)
        fb = jnp.where(left, f_root, fb)
        a = jnp.where(left, a, root)
        fa = jnp.where(left, fa, f_root)
        denom = jnp.where(crosses, fb - fa, 1.0)
        root = jnp.where(crosses, a - fa * (b - a) / denom, 0.5 * (lo + hi))
    return root


class PublishedLuminosities:
    r"""Mixin of :class:`~tengri.components.agn.grahsp.component.GRAHSPSEDComponent`.

    Holds the methods that compute the bolometric, torus and absorbed powers the component
    publishes as ``log_L_agn_bol``, ``log_L_agn_torus`` and ``log_L_agn_absorbed``. It
    reads ``self.config``, ``self._disc_raw``, ``self._torus_at``, ``self._feii_template`` and
    ``self._attenuation`` of the component and defines no state of its own.
    """

    def _published_luminosities(
        self,
        params: Mapping[str, Array],
        templates_state: GRAHSPSEDComponentState,
    ) -> dict[str, Array]:
        r"""Bolometric, torus and absorbed powers, internal dict ``L_agn_bol`` etc. [erg/s].

        Each is the integral of the emitted AGN-side SED under its documented definition, on
        the component's own grid, so it does not depend on the caller's wavelength array:

        .. math::

           L_{\rm bol}^{\rm BBB} = \int_{91.2\,\mathrm{nm}}^{\infty}
               L_\lambda^{\rm BBB+lines+FeII+Balmer}\,\mathrm{d}\lambda, \qquad
           L_{\rm bol}^{\rm TOR} = \int_0^\infty L_\lambda^{\rm TOR}\,\mathrm{d}\lambda, \qquad
           L_{\rm abs} = \int \left(L_\lambda^{\rm int} - L_\lambda^{\rm att}\right)
               \mathrm{d}\lambda,

        where the BBB and torus terms are intrinsic (before the bi-attenuation) and the
        absorbed power is the intrinsic minus the attenuated AGN-side power (Buchner+ 2024
        section 2.1.4-2.1.5, arXiv:2405.19297).

        Parameters
        ----------
        params : mapping
            ``agn_grahsp_*`` keys.
        templates_state : GRAHSPSEDComponentState
            Pre-loaded template tensors.

        Returns
        -------
        dict
            Internal keys ``L_agn_bol``, ``L_agn_torus``, ``L_agn_absorbed`` [erg/s], scalars,
            at the scale ``params["agn_grahsp_l5100"]`` carries.
            ``GRAHSPSEDComponent.apply`` calls this with the evaluation-scale ``l5100`` (the
            true value in float64, the reference 1e10 erg/s in float32) and publishes the
            ``log10`` of each, plus the offset, as ``log_L_agn_bol`` / ``log_L_agn_torus`` /
            ``log_L_agn_absorbed``: every value here is linear in ``l5100``.

        Notes
        -----
        **JIT-compatible**: yes; every rule has a node count fixed by the configuration.
        **Gradient-safe**: yes. ``L_agn_absorbed`` is exactly zero when ``apply_attenuation``
        is off.

        Each piece is integrated by its own rule, independent of the other pieces' options:
        the lines by segmented Gauss-Legendre cut at the 1100 nm attenuation kink, the Si
        feature by Gauss-Hermite, and the Gaussian torus by its closed-form weight and a
        log-normal Gauss-Legendre expectation; the FeII forest (either template) by its
        native nodes; the Netzer disc and the Mor & Netzer torus by segmented Gauss-Legendre
        on their native nodes; the bending power law and Balmer continuum on a log grid. The
        power law runs to :data:`~tengri.components.agn.grahsp.publication.PUBLICATION_HI_NM`.

        GRAHSP has no inclination parameter: its bi-attenuation is one line-of-sight screen
        on the whole AGN side. ``L_agn_absorbed`` is therefore that screen's absorbed power;
        there is no geometry to angle-integrate, and the angle-integrated versus
        line-of-sight split of the inclination-dependent emitters does not apply.
        """
        absorbed = (
            self._published_absorbed(params, templates_state)
            if self.config.apply_attenuation
            else jnp.zeros(())
        )
        return {
            "L_agn_bol": self._published_bol(params, templates_state),
            "L_agn_torus": self._published_torus(params, templates_state),
            "L_agn_absorbed": absorbed,
        }

    def _disc_at(self, wave_nm: Array, params, templates_state) -> Array:
        """Emitted big blue bump [erg/s/nm]: SBPL or Netzer disc with the X-ray floor."""
        return floor_disc_xray(wave_nm, self._disc_raw(wave_nm, params, templates_state))

    def _disc_integral(
        self, params, templates_state, lo_nm: float, weight_fn=None, extra_nm=()
    ) -> Array:
        """Integral of the big blue bump above ``lo_nm`` times ``weight_fn`` [erg/s].

        The SBPL runs on a log grid to ``PUBLICATION_HI_NM``; the Netzer disc on its own
        nodes plus a log grid (it is zero beyond its last node).
        """
        if self.config.disc_model == "netzer":
            nodes = jnp.asarray(templates_state.disc_wave_nm)
            grid = jnp.asarray(log_breaks(lo_nm, _DISC_GRID_HI_NM, SBPL_PER_DECADE, extra_nm))
            breaks = jnp.clip(jnp.sort(jnp.concatenate([nodes, grid])), lo_nm, nodes[-1])
        else:
            breaks = jnp.asarray(log_breaks(lo_nm, PUBLICATION_HI_NM, SBPL_PER_DECADE, extra_nm))
        lam, w = gauss_legendre_segments(breaks)
        value = self._disc_at(lam, params, templates_state)
        return jnp.sum(w * (value if weight_fn is None else value * weight_fn(lam)))

    def _balmer_breaks(self, lo_nm: float, params: Mapping[str, Array]) -> Array:
        """Breakpoints [nm] of the Balmer continuum: log grid, the 250 nm switch, the edge."""
        linewidth = jnp.asarray(params["agn_grahsp_linewidth_kms"])
        edge = jnp.asarray([_BALMER_EDGE_NM])
        windows = window_edges(edge, linewidth_sigma_nm(edge, linewidth))
        fixed = jnp.asarray(log_breaks(lo_nm, _BALMER_HI_NM, BALMER_PER_DECADE))
        breaks = jnp.sort(
            jnp.concatenate([fixed, jnp.asarray([_BC_CONVOLUTION_THRESHOLD_NM]), windows])
        )
        return jnp.clip(breaks, lo_nm, _BALMER_HI_NM)

    def _balmer_at(self, wave_nm: Array, params: Mapping[str, Array]) -> Array:
        """Intrinsic Balmer continuum [erg/s/nm] (type 1 only; the caller gates it)."""
        return balmer_continuum(
            wave_nm=wave_nm,
            l5100=jnp.asarray(params["agn_grahsp_l5100"]),
            a_bc=jnp.asarray(params.get("agn_grahsp_a_bc", 0.0)),
            linewidth_kms=jnp.asarray(params["agn_grahsp_linewidth_kms"]),
        )

    def _line_terms(
        self, params: Mapping[str, Array], templates_state: GRAHSPSEDComponentState
    ) -> tuple[Array, Array, Array]:
        """Line centers [nm], sigmas [nm] and Lambda_5100-scaled amplitudes [erg/s].

        Each line contributes ``amplitude * G(lambda; center, sigma)`` with G of unit area, as
        :func:`gaussian_lines` evaluates it: the amplitude is ``510 nm * line_lumin``.
        """
        cfg = self.config
        l_agn = jnp.asarray(params["agn_grahsp_l5100"]) / _LAMBDA_5100_NM
        a_lines = jnp.asarray(params["agn_grahsp_a_lines"])
        l_broad = _HBETA_BROAD_RATIO * l_agn * a_lines
        l_narrow = _HBETA_NARROW_RATIO * l_agn * a_lines
        if cfg.agn_type == 1:
            lumin = (
                l_broad * templates_state.line_broad + l_narrow * templates_state.line_narrow_sy2
            )
        elif cfg.agn_type == 2:
            lumin = l_narrow * templates_state.line_narrow_sy2
        else:
            lumin = l_narrow * templates_state.line_narrow_liner
        centers = jnp.asarray(templates_state.line_wave_nm)
        sigmas = linewidth_sigma_nm(centers, jnp.asarray(params["agn_grahsp_linewidth_kms"]))
        return centers, sigmas, _LAMBDA_5100_NM * lumin

    def _feii_amplitude(self, params: Mapping[str, Array]) -> Array:
        """FeII forest amplitude a_FeII * L_Hbeta,broad [erg/s/nm per unit template]."""
        l_agn = jnp.asarray(params["agn_grahsp_l5100"]) / _LAMBDA_5100_NM
        l_broad = _HBETA_BROAD_RATIO * l_agn * jnp.asarray(params["agn_grahsp_a_lines"])
        return jnp.asarray(params["agn_grahsp_a_feii"]) * l_broad

    def _torus_scale(self, params: Mapping[str, Array]) -> Array:
        """lambda * L_lambda of the torus at 12 um, ``2.5 * L5100 * fcov`` [erg/s]."""
        return (
            2.5 * jnp.asarray(params["agn_grahsp_l5100"]) * jnp.asarray(params["agn_grahsp_fcov"])
        )

    def _torus_terms(self, params: Mapping[str, Array]) -> tuple[Array, Array, Array, Array]:
        r"""Closed-form amplitudes and log-centers of the two log-normal torus components.

        With :math:`K = (\lambda L_{12\,\mu m})/(12000\,\mathrm{nm})/T(12\,\mu\mathrm{m})` the
        torus normalization, the cool component integrates to :math:`K\,\lambda_{\rm COOL}\,
        C_0(W_{\rm COOL})` and the hot to :math:`K\,f_{\rm hot}\,\lambda_{\rm COOL}\,
        C_0(W_{\rm HOT})` (see the module docstring).

        Returns
        -------
        amp_cool, amp_hot : ndarray
            Closed-form integrals of the components [erg/s].
        mu_cool, mu_hot : ndarray
            :math:`\log_{10}` of the component peak wavelengths [um].
        """
        cool_lam = jnp.asarray(params["agn_grahsp_cool_lam_um"])
        hot_lam = jnp.asarray(params["agn_grahsp_hot_lam_um"])
        w_cool = jnp.asarray(params["agn_grahsp_cool_width"])
        w_hot = jnp.asarray(params["agn_grahsp_hot_width"])
        fhot = jnp.asarray(params["agn_grahsp_hot_fcov"])
        mu_cool = jnp.log10(cool_lam)
        mu_hot = jnp.log10(hot_lam)
        spec12 = _log_gaussian_pair(jnp.log10(12.0), mu_cool, w_cool, mu_hot, w_hot, fhot)
        scale = (self._torus_scale(params) / 12000.0) / spec12
        amp_cool = scale * cool_lam * lognormal_c0(w_cool)
        amp_hot = scale * fhot * cool_lam * lognormal_c0(w_hot)
        return amp_cool, amp_hot, mu_cool, mu_hot

    def _si_terms(self, params: Mapping[str, Array]) -> tuple[Array, Array, Array, Array]:
        """Si feature: prefactor, closed-form integral, feature centers [nm] and sigmas [nm].

        The feature is ``l_torus/12000 * si * (A_em G_em - A_ab G_ab)`` with unit-peak Gaussians
        G and amplitudes from :data:`SI_DEFAULTS`, so with unit-area G the closed integral is
        ``l_torus/12000 * si * sqrt(2 pi) * (A_em sigma_em - A_ab sigma_ab)``.
        """
        pref = (self._torus_scale(params) / 12000.0) * jnp.asarray(params["agn_grahsp_si"])
        em_amp = SI_DEFAULTS["si_em_ampl"]
        ab_amp = em_amp * SI_DEFAULTS["si_ratio"]
        centers = jnp.asarray([SI_DEFAULTS["si_em_lam_nm"], SI_DEFAULTS["si_abs_lam_nm"]])
        sigmas = jnp.asarray([SI_DEFAULTS["si_em_width_nm"], SI_DEFAULTS["si_abs_width_nm"]])
        # Amplitude vector: (+A_em, -A_ab) times the unit-area Gaussian integral sqrt(2 pi) s.
        signed = jnp.asarray([em_amp, -ab_amp]) * math.sqrt(2.0 * math.pi) * sigmas
        return pref, jnp.sum(signed) * pref, centers, sigmas

    def _published_bol(self, params, templates_state) -> Array:
        """Intrinsic BBB + lines + FeII + Balmer above the Lyman limit [erg/s]."""
        cfg = self.config
        type1 = cfg.agn_type == 1
        total = jnp.zeros(())
        if cfg.include_bbb:
            total = total + self._disc_integral(params, templates_state, LYMAN_LIMIT_NM)
        if cfg.include_balmer and type1:
            lam, w = gauss_legendre_segments(self._balmer_breaks(LYMAN_LIMIT_NM, params))
            total = total + jnp.sum(w * self._balmer_at(lam, params))
        if cfg.include_feii and type1:
            wave, lumin = self._feii_template(templates_state)
            total = total + self._feii_amplitude(params) * jnp.trapezoid(lumin, wave)
        if cfg.include_lines:
            centers, sigmas, amps = self._line_terms(params, templates_state)
            cut = 0.5 * erfc((LYMAN_LIMIT_NM - centers) / (math.sqrt(2.0) * sigmas))
            total = total + jnp.sum(amps * cut)
        return total

    def _published_torus(self, params, templates_state) -> Array:
        """Intrinsic torus continuum + Si feature over (0, inf) [erg/s].

        The Gaussian torus and its Si feature are closed-form integrals plus the correction of
        the emitted ``max(si, -torus)`` clip (:meth:`_si_clip_correction`), which binds inside
        the declared prior box (corner si = -4, cool peak 30 um, width 0.2 dex). The
        Mor & Netzer torus integrates on its native nodes (:meth:`_template_torus_integral`).
        """
        cfg = self.config
        if not cfg.include_torus:
            return jnp.zeros(())
        if cfg.torus_model == "mn12":
            return self._template_torus_integral(params, templates_state)
        amp_cool, amp_hot, _, _ = self._torus_terms(params)
        total = amp_cool + amp_hot
        if cfg.include_si:
            total = total + self._si_terms(params)[1]
            total = total + self._si_clip_correction(params)
        return total

    def _template_torus_integral(self, params, templates_state, weight_fn=None) -> Array:
        r"""Integral of the template torus and Si, ``torus + max(si, -torus)``, times a weight.

        Segmented Gauss-Legendre in :math:`\ln\lambda` between the native nodes of the torus
        and Si templates (and the attenuation break when a weight is given). Within a segment
        both templates are smooth power laws, and the clip :math:`\max(si, -torus)` has at
        most the one kink where ``torus + si`` crosses zero; each segment is split there
        (regula-falsi root, stopped from the gradient: the integrand is continuous at the
        root).

        Parameters
        ----------
        params : mapping
            ``agn_grahsp_*`` keys.
        templates_state : GRAHSPSEDComponentState
            Pre-loaded template tensors.
        weight_fn : callable, optional
            ``weight_fn(wave_nm) -> ndarray`` multiplying the integrand (1 - attenuation).

        Returns
        -------
        ndarray, shape ()
            The integral [erg/s].
        """
        cfg = self.config
        nodes = [jnp.asarray(templates_state.torus_mn12_wave_nm)]
        if cfg.include_si:
            nodes.append(jnp.asarray(templates_state.torus_mn12_si_wave_nm))
        if weight_fn is not None:
            nodes.append(jnp.asarray([PREVOT_DEFAULTS["lam_break_nm"]]))
        edges = jnp.sort(jnp.concatenate(nodes))
        lo, hi = edges[:-1], edges[1:]

        def raw(lam):
            torus, si = self._torus_at(lam, params, templates_state)
            return torus + si

        root = _sign_change_roots(raw, lo, hi)
        lam, w = gauss_legendre_segments(jnp.stack([lo, root, hi], axis=1).ravel())
        torus, si = self._torus_at(lam, params, templates_state)
        emitted = torus + jnp.maximum(si, -torus)
        return jnp.sum(w * (emitted if weight_fn is None else emitted * weight_fn(lam)))

    def _si_clip_correction(self, params: Mapping[str, Array], weight_fn=None) -> Array:
        r"""Integral of the Si clip correction :math:`\int\max(0,-(T+S))\,h\,\mathrm{d}\lambda`.

        The emitted torus term uses :math:`\max(\mathrm{si}, -\mathrm{torus})`, which equals
        :math:`\mathrm{si}+\max(0,-\mathrm{torus}-\mathrm{si})`. Both terms carry the factor
        :math:`l_{\rm torus}/12000`, so the correction is :math:`(l_{\rm torus}/12000)` times the
        dimensionless integral of :math:`\max(0,-(T+S))`, where :math:`T` is the torus shape
        normalized to unity at 12 um and :math:`S` the Si shape per unit ``si``. It is nonzero
        only inside the Si feature windows, and is integrated by Gauss-Legendre on each window
        split into segments of :data:`CLIP_STEP_SIGMA` sigma (the clip is a kink of the
        integrand, so each segment must hold at most one crossing, found by a sign test on its
        ends). The quadrature is skipped when :func:`_net_shape_is_nonnegative` certifies that
        the shape is non-negative everywhere, where the correction is exactly zero.

        Parameters
        ----------
        params : mapping
            ``agn_grahsp_*`` keys.
        weight_fn : callable, optional
            ``weight_fn(wave_nm) -> ndarray`` multiplying the integrand (e.g. 1 - attenuation).

        Returns
        -------
        ndarray, shape ()
            The correction [erg/s].
        """
        l_torus = self._torus_scale(params)
        si = jnp.asarray(params["agn_grahsp_si"])
        centers = np.asarray([SI_DEFAULTS["si_em_lam_nm"], SI_DEFAULTS["si_abs_lam_nm"]])
        sigmas = np.asarray([SI_DEFAULTS["si_em_width_nm"], SI_DEFAULTS["si_abs_width_nm"]])
        offsets = np.arange(-WINDOW_SIGMA, WINDOW_SIGMA + 0.5 * CLIP_STEP_SIGMA, CLIP_STEP_SIGMA)
        breaks = jnp.sort(jnp.asarray((centers[:, None] + sigmas[:, None] * offsets).ravel()))
        cool_lam = jnp.asarray(params["agn_grahsp_cool_lam_um"])
        hot_lam = jnp.asarray(params["agn_grahsp_hot_lam_um"])
        w_cool = jnp.asarray(params["agn_grahsp_cool_width"])
        w_hot = jnp.asarray(params["agn_grahsp_hot_width"])
        fhot = jnp.asarray(params["agn_grahsp_hot_fcov"])
        mu_cool, mu_hot = jnp.log10(cool_lam), jnp.log10(hot_lam)
        spec12 = _log_gaussian_pair(jnp.log10(12.0), mu_cool, w_cool, mu_hot, w_hot, fhot)
        em_amp = SI_DEFAULTS["si_em_ampl"]
        ab_amp = em_amp * SI_DEFAULTS["si_ratio"]

        def net_shape(lam):
            """Torus shape plus Si shape (dimensionless); the clip acts where it is negative."""
            torus_shape = (
                _log_gaussian_pair(jnp.log10(lam / 1000.0), mu_cool, w_cool, mu_hot, w_hot, fhot)
                / spec12
            )
            g_em = jnp.exp(-0.5 * ((lam - centers[0]) / sigmas[0]) ** 2)
            g_ab = jnp.exp(-0.5 * ((lam - centers[1]) / sigmas[1]) ** 2)
            return torus_shape + si * (em_amp * g_em - ab_amp * g_ab)

        lo, hi = breaks[:-1], breaks[1:]

        def integral():
            split = _sign_change_roots(net_shape, lo, hi)
            # lo, split, hi per interval: ascending, each crossing a segment boundary.
            lam, weight = gauss_legendre_segments(jnp.stack([lo, split, hi], axis=1).ravel())
            crossing = jnp.maximum(0.0, -net_shape(lam))
            if weight_fn is not None:
                crossing = crossing * weight_fn(lam)
            return (l_torus / 12000.0) * jnp.sum(weight * crossing)

        # The correction is exactly zero where net_shape >= 0 everywhere; certify that with a
        # rigorous lower bound per segment and skip the quadrature (the common case).
        inactive = _net_shape_is_nonnegative(
            lo, hi, (mu_cool, w_cool, mu_hot, w_hot, fhot, spec12), si, centers, sigmas
        )
        return jax.lax.cond(inactive, lambda: jnp.zeros_like(l_torus), integral)

    def _published_absorbed(self, params, templates_state) -> Array:
        """Intrinsic-minus-attenuated AGN-side power [erg/s], one rule per piece."""
        cfg = self.config
        type1 = cfg.agn_type == 1
        split = PREVOT_DEFAULTS["lam_break_nm"]

        def h_fn(wave_nm):
            return 1.0 - self._attenuation(wave_nm, params)

        total = jnp.zeros(())
        if cfg.include_bbb:
            total = total + self._disc_integral(
                params, templates_state, XRAY_FLOOR_NM, h_fn, (split,)
            )
        if cfg.include_balmer and type1:
            lam, w = gauss_legendre_segments(self._balmer_breaks(XRAY_FLOOR_NM, params))
            total = total + jnp.sum(w * self._balmer_at(lam, params) * h_fn(lam))
        if cfg.include_feii and type1:
            wave, lumin = self._feii_template(templates_state)
            total = total + self._feii_amplitude(params) * feii_segment_product(h_fn, wave, lumin)
        if cfg.include_lines:
            centers, sigmas, amps = self._line_terms(params, templates_state)
            total = total + jnp.sum(
                amps
                * split_gaussian_expectation(
                    h_fn,
                    centers,
                    sigmas,
                    split,
                    LINE_GL_ORDER,
                    LINE_SUBSEGMENTS,
                    LINE_WINDOW_SIGMA,
                )
            )
        if not cfg.include_torus:
            return total
        if cfg.torus_model == "mn12":
            return total + self._template_torus_integral(params, templates_state, h_fn)
        amp_cool, amp_hot, mu_cool, mu_hot = self._torus_terms(params)
        for amp, mu, name in (
            (amp_cool, mu_cool, "agn_grahsp_cool_width"),
            (amp_hot, mu_hot, "agn_grahsp_hot_width"),
        ):
            total = total + amp * log_gaussian_expectation(
                h_fn, mu, params[name], split, LOG_GL_ORDER, LOG_WINDOW_SIGMA
            )
        if cfg.include_si:
            pref, _, centers, sigmas = self._si_terms(params)
            total = total + self._si_clip_correction(params, weight_fn=h_fn)
            em_ab = jnp.asarray(
                [SI_DEFAULTS["si_em_ampl"], -SI_DEFAULTS["si_em_ampl"] * SI_DEFAULTS["si_ratio"]]
            )
            weighted = em_ab * math.sqrt(2.0 * math.pi) * sigmas
            expect = line_expectation(h_fn, centers, sigmas, SI_GH_ORDER)
            total = total + pref * jnp.sum(weighted * expect)
        return total

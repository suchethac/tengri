# SPDX-License-Identifier: BSD-3-Clause
"""Independent dense reference for the GRAHSP published ``L_*`` luminosities.

The reference integrates the SED the component EMITS (its own ``apply`` output, converted
back to :math:`L_\\lambda`), not the component's publication rules:

* segmented 16-point Gauss-Legendre in ln(lambda) from 1 nm to 1e15 nm (the power-law tail of
  the big blue bump included), 20 segments per decade;
* a break at every hard edge of the emitted pieces (the X-ray floor, the Lyman limit, the
  Balmer convolution threshold and edge, the SMC attenuation break), at every native node of
  the selected FeII, Netzer-disc and MN12 templates, and at every sigma of the narrow
  features (the emission lines, the Balmer edge window, the silicate Gaussians);
* a break at each crossing of the raw torus-plus-Si shape through zero, refined by ``brentq``
  on the component's own torus functions: ``max(si, -torus)`` has a kink there.

Between consecutive breaks every emitted piece is smooth, so the rule converges geometrically;
the result does not use any quadrature rule, node count or closed form of the component.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

KEYS = ("L_agn_bol", "L_agn_torus", "L_agn_absorbed")

_C_NM_PER_S = 2.99792458e17
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
_C_KMS = 299792.458

#: Upper edge of the reference [nm]; the tail beyond is below 1e-11 of the luminosity (measured).
REFERENCE_HI_NM = 1.0e15
_LO_NM = 1.0
_PER_DECADE = 20
_GL_ORDER = 16
#: Hard edges of the emitted pieces [nm]: X-ray floor, Lyman limit, Balmer threshold and edge,
#: SMC attenuation break.
_HARD_EDGES_NM = (12.4, 91.2, 250.0, 364.6, 1100.0)
#: Silicate feature centers and sigmas [nm] (Mullaney+ 2011 difference template).
_SI_CENTERS_NM = (9841.0, 14224.0)
_SI_SIGMAS_NM = (1025.3, 1163.5)
_N_ROOT_SLOTS = 24
_PAD_NM = 100.0  # unused root slots sit on a break that already exists

#: The option combinations (the non-default options the component offers).
CONFIGS = {
    "default": {},
    "mn12_torus": {"torus_model": "mn12"},
    "netzer_disc": {"disc_model": "netzer"},
    "veroncetty_feii": {"feii_template": "veroncetty2004"},
    "all_options": {
        "torus_model": "mn12",
        "disc_model": "netzer",
        "feii_template": "veroncetty2004",
    },
    "sy2": {"agn_type": 2},
}


def make_component(**overrides):
    """A GRAHSP component with the given configuration overrides."""
    from tengri.components.agn.grahsp.component import (
        GRAHSPSEDComponent,
        GRAHSPSEDComponentConfig,
    )

    cfg = dataclasses.replace(GRAHSPSEDComponentConfig(), **overrides)
    return GRAHSPSEDComponent(config=cfg)


def default_value(prior):
    """Declared default, else the bounds' midpoint (geometric for a log-uniform prior)."""
    if prior.default is not None:
        return float(prior.default)
    lo, hi = prior.bounds
    return float(np.sqrt(lo * hi)) if type(prior).__name__ == "LogUniform" else 0.5 * (lo + hi)


def default_params(comp):
    """Declared defaults as float64 scalars."""
    return {
        d.name: jnp.asarray(default_value(d.prior), dtype=jnp.float64)
        for d in comp.declared_parameters()
    }


def draw_params(comp, rng, extended=False):
    """One draw of every free parameter from its declared prior (fixed ones at the default).

    ``extended`` also draws the structural knobs the box fixes: the Balmer strength, the line
    width and the BBB infrared cutoff.
    """
    out = {name: float(v) for name, v in default_params(comp).items()}
    for d in comp.declared_parameters():
        prior = d.prior
        kind = type(prior).__name__
        if kind == "Fixed":
            continue
        lo, hi = prior.bounds
        out[d.name] = float(
            np.exp(rng.uniform(np.log(lo), np.log(hi)))
            if kind == "LogUniform"
            else rng.uniform(lo, hi)
        )
    if extended:
        out["agn_grahsp_a_bc"] = float(rng.uniform(0.0, 2.0))
        out["agn_grahsp_linewidth_kms"] = float(rng.uniform(1000.0, 20000.0))
        out["agn_grahsp_cutoff_nm"] = float(np.exp(rng.uniform(np.log(1000.0), np.log(3.0e4))))
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in out.items()}


def _gl_segments(breaks_nm):
    """Gauss-Legendre nodes [nm] and weights [nm] on each segment, in ln(lambda)."""
    x, w = np.polynomial.legendre.leggauss(_GL_ORDER)
    a, b = np.log(breaks_nm[:-1]), np.log(breaks_nm[1:])
    half = 0.5 * (b - a)
    lam = np.exp(a[:, None] + half[:, None] * (x[None, :] + 1.0))
    return lam.ravel(), (half[:, None] * w[None, :] * lam).ravel()


class Reference:
    """Dense reference for one configuration; reuses its jitted evaluators across draws."""

    def __init__(self, **overrides):
        self.over = overrides
        self.full = make_component(**overrides)
        self.tpl = self.full.precompute()
        torus_off = {"include_torus": False, "include_si": False, "apply_attenuation": False}
        self._bbb_side = self._evaluator(make_component(**{**overrides, **torus_off}))
        self._torus_side = self._evaluator(
            make_component(
                **{
                    **overrides,
                    "include_bbb": False,
                    "include_lines": False,
                    "include_feii": False,
                    "include_balmer": False,
                    "apply_attenuation": False,
                }
            )
        )
        self._attenuated = self._evaluator(self.full)
        self._raw = jax.jit(self._raw_torus_plus_si)
        self._cache = {}

    def evaluator(self, **switches):
        """``fn(lam_nm, params) -> L_lambda [erg/s/nm]``, this configuration with ``switches``."""
        key = tuple(sorted(switches.items()))
        if key not in self._cache:
            self._cache[key] = self._evaluator(make_component(**{**self.over, **switches}))
        return self._cache[key]

    def _evaluator(self, comp):
        from tengri.protocols import ForwardState

        tpl = comp.precompute()

        @jax.jit
        def f(wave_aa, params):
            state = ForwardState(wave=wave_aa, sed_intrinsic=jnp.zeros_like(wave_aa), derived={})
            return comp.apply(state, params, templates_state=tpl).sed_intrinsic

        def l_lambda(lam_nm, params):
            lnu = np.asarray(f(jnp.asarray(lam_nm * 10.0), params))
            return lnu * _C_NM_PER_S / lam_nm**2

        return l_lambda

    def _raw_torus_plus_si(self, wave_nm, params):
        """Torus plus Si before the ``max(si, -torus)`` clip, from the component's functions."""
        from tengri.components.agn.grahsp import torus as T

        tpl = self.tpl
        common = {"wave_nm": wave_nm, "l5100": 1.0, "fcov": params["agn_grahsp_fcov"]}
        si_on = self.full.config.include_si
        if self.full.config.torus_model == "mn12":
            tor = T.torus_mn12_continuum(
                tor_temp=params["agn_grahsp_tor_temp"],
                tor_cutoff_um=params["agn_grahsp_tor_cutoff_um"],
                mn12_wave_nm=tpl.torus_mn12_wave_nm,
                mn12_avg=tpl.torus_mn12_avg,
                mn12_lo=tpl.torus_mn12_lo,
                mn12_hi=tpl.torus_mn12_hi,
                **common,
            )
            si = (
                T.torus_mn12_si(
                    si=params["agn_grahsp_si"],
                    si_wave_nm=tpl.torus_mn12_si_wave_nm,
                    si_lumin=tpl.torus_mn12_si_lumin,
                    **common,
                )
                if si_on
                else 0.0
            )
        else:
            tor = T.torus_dust_continuum(
                cool_lam_um=params["agn_grahsp_cool_lam_um"],
                cool_width=params["agn_grahsp_cool_width"],
                hot_lam_um=params["agn_grahsp_hot_lam_um"],
                hot_width=params["agn_grahsp_hot_width"],
                hot_fcov=params["agn_grahsp_hot_fcov"],
                **common,
            )
            si = T.si_feature(si=params["agn_grahsp_si"], **common) if si_on else 0.0
        return tor + si

    def crossings(self, params):
        """Zeros of the raw torus-plus-Si shape (clip kinks), padded to a fixed count."""
        if not self.full.config.include_torus:
            return np.full(_N_ROOT_SLOTS, _PAD_NM), 0
        grid = np.geomspace(100.0, 1.0e7, 60001)
        values = np.asarray(self._raw(jnp.asarray(grid), params))
        idx = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]

        def scalar(x):
            return float(self._raw(jnp.asarray([x]), params)[0])

        roots = [brentq(scalar, grid[i], grid[i + 1], xtol=1e-12, rtol=1e-15) for i in idx]
        if len(roots) > _N_ROOT_SLOTS:
            raise AssertionError(f"{len(roots)} clip crossings exceed the reference slots")
        padded = np.full(_N_ROOT_SLOTS, _PAD_NM)
        padded[: len(roots)] = roots
        return padded, len(roots)

    def breakpoints(self, params, roots):
        """Sorted breakpoints [nm] of the reference (fixed count for a given configuration)."""
        cfg = self.full.config
        tpl = self.tpl
        n = round(_PER_DECADE * math.log10(REFERENCE_HI_NM / _LO_NM)) + 1
        parts = [np.geomspace(_LO_NM, REFERENCE_HI_NM, n), np.asarray(_HARD_EDGES_NM), roots]
        feii = tpl.feii_vc04_wave_nm if cfg.feii_template == "veroncetty2004" else tpl.feii_wave_nm
        axes = [feii]
        if cfg.torus_model == "mn12":
            axes += [tpl.torus_mn12_wave_nm, tpl.torus_mn12_si_wave_nm]
        if cfg.disc_model == "netzer":
            axes.append(tpl.disc_wave_nm)
        parts += [np.asarray(a, dtype=float) for a in axes]
        frac = float(params["agn_grahsp_linewidth_kms"]) / _C_KMS * _FWHM_TO_SIGMA
        steps = np.arange(-24, 25) / 2.0  # +/- 12 sigma in half-sigma steps
        centers = np.concatenate([np.asarray(tpl.line_wave_nm, dtype=float), [364.6]])
        parts.append((centers[:, None] * (1.0 + frac * steps[None, :])).ravel())
        si_steps = np.arange(-24, 25) / 2.0
        for c, s in zip(_SI_CENTERS_NM, _SI_SIGMAS_NM):
            parts.append(c + s * si_steps)
        b = np.sort(np.concatenate([np.ravel(p) for p in parts]))
        b = np.clip(b, _LO_NM, REFERENCE_HI_NM)
        return b

    def nodes(self, params):
        """Quadrature nodes and weights [nm] plus the number of clip crossings."""
        roots, n_roots = self.crossings(params)
        lam, weight = _gl_segments(self.breakpoints(params, roots))
        return lam, weight, n_roots

    def values(self, params):
        """Reference ``L_agn_bol``, ``L_agn_torus``, ``L_agn_absorbed`` [erg/s]; plus crossings."""
        lam, w, n_roots = self.nodes(params)
        bbb = self._bbb_side(lam, params)
        torus = self._torus_side(lam, params)
        attenuated = self._attenuated(lam, params)
        # The emitted SED is additive in L_nu: intrinsic = bbb side + torus side.
        out = {
            "L_agn_bol": float(np.sum(w * np.where(lam >= 91.2, bbb, 0.0))),
            "L_agn_torus": float(np.sum(w * torus)),
            "L_agn_absorbed": float(np.sum(w * (bbb + torus - attenuated))),
        }
        return out, n_roots

    def published(self, params, wave_aa):
        """The component's published values on a caller grid [Angstrom], through ``apply``."""
        from tengri.protocols import ForwardState

        if not hasattr(self, "_pub_fn"):
            comp, tpl = self.full, self.tpl

            @jax.jit
            def pub(w, p):
                state = ForwardState(wave=w, sed_intrinsic=jnp.zeros_like(w), derived={})
                derived = comp.apply(state, p, templates_state=tpl).derived
                # The torus and absorbed powers are published as log10 [dex re erg/s] (#2842);
                # the reference compares linear integrals, and 10**-inf is the zero of an
                # absent power.
                return {
                    "L_agn_bol": derived["L_agn_bol"],
                    "L_agn_torus": 10.0 ** derived["log_L_agn_torus"],
                    "L_agn_absorbed": 10.0 ** derived["log_L_agn_absorbed"],
                }

            self._pub_fn = pub
        out = self._pub_fn(jnp.asarray(np.asarray(wave_aa, dtype=float)), params)
        return {k: float(v) for k, v in out.items()}

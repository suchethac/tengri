# SPDX-License-Identifier: BSD-3-Clause
"""Plumbing shared by the AGNfitter-rX reproduction notebook.

Everything here goes through tengri's public API (``SEDModel.build`` and
``model.predict``) or is plain array bookkeeping; the physics being compared
stays in the notebook cells.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from tengri import DEFAULT, Fixed, SEDModel

from . import units as U

#: ``max|tengri/AGNfitter-rX - 1|`` below which a case is called node-exact.
NODE_EXACT_TOL = 1e-2
FIG_DPI = 150


def make_save_fig(figs_dir: Path):
    """Return ``save_fig(filename)``, writing to ``figs_dir`` and leaving the figure open."""
    figs_dir.mkdir(exist_ok=True)

    def save_fig(filename: str) -> None:
        plt.savefig(str(figs_dir / filename), dpi=FIG_DPI, bbox_inches="tight")

    return save_fig


def assert_comparable(arr_ref, arr_t, *, name: str) -> None:
    """Refuse a blank or wildly mis-scaled panel."""
    a_ref = np.asarray(arr_ref)
    a_t = np.asarray(arr_t)
    assert np.isfinite(a_ref).any() and np.isfinite(a_t).any(), f"{name}: NaN-only"
    assert (a_ref > 0).any() and (a_t > 0).any(), f"{name}: zero/negative-only"
    ratio = a_ref.max() / a_t.max()
    assert 1e-3 < ratio < 1e3, f"{name}: y-scale ratio {ratio:.2e} out of range"


def norm_at(wave, L, lam_aa):
    """Scale an SED so ``L(lam_aa) = 1`` (interpolated anchor)."""
    wave = np.asarray(wave)
    L = np.asarray(L)
    order = np.argsort(wave)
    ref = float(np.interp(lam_aa, wave[order], L[order]))
    return L / ref if ref > 0 else L


def norm_peak(L):
    """Scale an SED so its maximum is 1."""
    L = np.asarray(L)
    m = float(np.max(L))
    return L / m if m > 0 else L


def val_at(w, L, lam):
    """``L`` interpolated at ``lam``, for unsorted ``w``."""
    o = np.argsort(np.asarray(w))
    return float(np.interp(lam, np.asarray(w)[o], np.asarray(L)[o]))


def band_power(wave_aa, L_nu, lo_aa, hi_aa) -> float:
    """Power ``|int L_nu d(nu)|`` between two wavelengths [erg/s for L_nu in erg/s/Hz]."""
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    L_nu = np.asarray(L_nu, dtype=np.float64)
    sel = (wave_aa >= lo_aa) & (wave_aa <= hi_aa)
    nu = U.C_ANGSTROM_PER_S / wave_aa[sel]
    order = np.argsort(nu)
    return float(np.trapezoid(L_nu[sel][order], nu[order]))


def node_exact_verdict(rows, title) -> None:
    """One line: how many cases of a window table meet the node-exact tolerance, and the worst."""
    devs = {r["label"]: float(r["max_abs_dev"]) for r in rows}
    n_ok = sum(d < NODE_EXACT_TOL for d in devs.values())
    worst = max(devs, key=devs.get)
    line = (
        f"{title}: {n_ok}/{len(devs)} cases within max|t/a-1| < {NODE_EXACT_TOL:g}; "
        f"worst = {worst} ({devs[worst]:.3g})"
    )
    print(line)


def resolved_params(m) -> None:
    """Print the model's resolved parameter table, so no parameter rests on a silent default."""
    m.spec.summary()


def make_agn_builders(ssp, sfh, no_dust):
    """Builders of isolated AGN sub-blocks through ``SEDModel.build``'s ``agn={...}`` grammar.

    Every build states ``'norm': 'independent'`` (the disc scales on ``agn_log_lbol`` and the
    torus on its own amplitude, AGNfitter-style bookkeeping) and ``atten={'type': 'none'}``.
    Each returns ``(wave_aa, L_nu)`` read from ``model.predict(params).sed.components`` and
    keeps its last model on ``.last_model``.
    """

    def _build(agn, **kw):
        return SEDModel.build(
            ssp_data=ssp,
            sfh=sfh,
            dust_attenuation=no_dust,
            agn=agn,
            neb={"type": "none"},
            redshift=Fixed(0.0),
            **kw,
        )

    def _read(m, key):
        comps = m.predict({}).sed.components
        return np.asarray(comps["wavelength"]), np.asarray(comps[key])

    def _agn(disc, torus, log_lbol, **extra):
        return {
            "type": "composable",
            "disc": disc,
            "torus": torus,
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "atten": {"type": "none"},
            "agn_log_lbol": Fixed(log_lbol),
            "all_params": Fixed(DEFAULT),
            "norm": "independent",
            **extra,
        }

    def disc(disc_type, *, log_lbol=11.0, ebv_disc=None, **disc_params):
        """Isolated accretion-disc SED; ``ebv_disc`` sets ``agn_ebv_disc`` (``EBVbbb`` analog)."""
        spec = {"type": disc_type, "all_params": Fixed(DEFAULT)}
        spec.update({k: Fixed(v) for k, v in disc_params.items()})
        if ebv_disc is not None:
            spec["agn_ebv_disc"] = Fixed(ebv_disc)
        m = _build(_agn(spec, {"type": "none"}, log_lbol))
        disc.last_model = m
        return _read(m, "sed_agn_disc")

    def torus(torus_type, *, log_lbol=11.0, **torus_params):
        """Isolated torus SED."""
        spec = {"type": torus_type, "all_params": Fixed(DEFAULT)}
        spec.update({k: Fixed(v) for k, v in torus_params.items()})
        m = _build(_agn({"type": "none"}, spec, log_lbol))
        torus.last_model = m
        return _read(m, "sed_agn_torus")

    def qsogen_full(*, log_lbol=11.0):
        """tengri's THB21 analog: the qsogen continuum with its broad lines and Fe II.

        qsogen carries both line families in its ``blr`` block (``nlr`` stays off); the
        0.7 um Halpha + [N II] bump that defines THB21 lives there, so the disc-only
        continuum alone does not reproduce it.
        """
        agn = _agn(
            {"type": "qsogen", "all_params": Fixed(DEFAULT)},
            {"type": "none"},
            log_lbol,
            blr={"type": "qsogen", "all_params": Fixed(DEFAULT)},
            feii={"type": "qsogen_balmer", "all_params": Fixed(DEFAULT)},
        )
        m = _build(agn)
        qsogen_full.last_model = m
        return _read(m, "sed_agn")

    def disc_atten(disc_type, atten_type, ebv, **atten_params):
        """Disc SED with a named ``atten`` law at a given E(B-V)."""
        atten = {"type": atten_type, "ebv": Fixed(ebv)}
        atten.update({k: Fixed(v) for k, v in atten_params.items()})
        agn = _agn({"type": disc_type, "all_params": Fixed(DEFAULT)}, {"type": "none"}, 11.0)
        agn["atten"] = atten
        m = _build(agn)
        disc_atten.last_model = m
        return _read(m, "sed_agn_disc")

    return SimpleNamespace(disc=disc, torus=torus, qsogen_full=qsogen_full, disc_atten=disc_atten)

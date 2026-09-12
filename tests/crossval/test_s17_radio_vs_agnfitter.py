# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate ``radio_sfr_bell2003_split`` against AGNfitter-rX's S17_radio tail.

AGNfitter-rX's ``STARBURST`` component (``MODEL_AGNfitter.py``, ``'S17_radio'``
branch) ships the Schreiber+2018 dust continuum with a radio extension built
by applying the Bell (2003) IR-radio correlation to the S17 template's own
tabulated L_IR, split 90%/10% non-thermal/thermal (Martinez-Ramirez+2024,
Sec 3, p.3). The driver's :func:`cold_dust_radio_template` reads that
pre-built radio tail directly from the vendored FITS table (no formula is
re-derived at runtime on the driver side); this file checks that tengri's
:func:`tengri.radio.radio_sfr_bell2003_split` reproduces the SAME spectral
SHAPE across the radio band.

Absolute normalization is NOT compared (see D9/D7 in the colddust_radio
parity audit: the vendored table's node normalization involves an internal
"effective q_IR" that is not 2.64 -- a units/bookkeeping quirk in the
upstream table build, already characterized in the audit and out of scope
here). Both sides are instead normalized at 1.4 GHz and compared at 5 and
10 GHz.

150 MHz is NOT tested against the vendored table
--------------------------------------------------
The vendored ``s17_radio`` table's lowest tabulated frequency node is
~1.0017 GHz (measured directly from ``_s17_radio_tables()``'s ``dust_nu_hz``
array) -- upstream's own S17_radio FITS table does not extend down to
150 MHz at all, despite ``cold_dust_radio_template``'s docstring describing
coverage "down to ~1.4 GHz". A 150 MHz assertion against this vendored data
would either read an out-of-range extrapolation or the clamped boundary
node, neither of which tests anything real; this premise (found while
writing this file, not assumed) is why the crossval assertions below cover
1.4/5/10 GHz only.

References
----------
.. [1] E. F. Bell, "Estimating Star Formation Rates from Infrared and
   Radio Luminosities," ApJ, 586, 794 (2003).
.. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.radio import radio_sfr_bell2003_split, sfr_from_lir

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_COLD_DUST_REF = _DATA_DIR / "agnfitter_cold_dust_reference.h5"

_C_AA_PER_S = 2.99792458e18  # speed of light [Angstrom * Hz]

if not _COLD_DUST_REF.is_file():
    pytest.skip(
        f"AGNfitter cold-dust reference not found: {_COLD_DUST_REF}",
        allow_module_level=True,
    )

_agnfitter_driver = pytest.importorskip(
    "reproduction.agnfitter._drivers.agnfitter_driver",
    reason="reproduction driver package not importable",
)


def _driver_radio_shape_ratio(freq_hz: float, *, ref_freq_hz: float = 1.4e9) -> float:
    """Vendored S17_radio tail: L_nu(freq_hz) / L_nu(ref_freq_hz).

    Interpolates the vendored table log-log in frequency (nodes are closely
    log-spaced, so this is effectively node-exact -- see the module for the
    verification that established this).
    """
    _dust_nu_hz, _dust_sed_lnu, _pah_nu_hz, _pah_sed_lnu, tdust_ax, _lir_conv, fpah_ax = (
        _agnfitter_driver._s17_radio_tables()
    )
    t = len(tdust_ax) // 2
    f = 5
    wave_aa, L_nu = _agnfitter_driver.cold_dust_radio_template(
        tdust=float(tdust_ax[t]), fpah=float(fpah_ax[f])
    )
    nu = _C_AA_PER_S / wave_aa
    order = np.argsort(nu)
    log_nu = np.log10(nu[order])
    log_lnu = np.log10(L_nu[order])

    def _at(target_hz: float) -> float:
        return float(10.0 ** np.interp(np.log10(target_hz), log_nu, log_lnu))

    return _at(freq_hz) / _at(ref_freq_hz)


def _tengri_radio_shape_ratio(freq_hz: float, *, ref_freq_hz: float = 1.4e9) -> float:
    """tengri's bell2003_split: L_nu(freq_hz) / L_nu(ref_freq_hz) (q_ir cancels)."""
    wave_aa = jnp.asarray([_C_AA_PER_S / freq_hz, _C_AA_PER_S / ref_freq_hz])
    L_nu = np.asarray(radio_sfr_bell2003_split(wave_aa, L_ir=1e44, q_ir=2.64))
    return float(L_nu[0] / L_nu[1])


@pytest.mark.parametrize("freq_hz", [5.0e9, 1.0e10], ids=["5GHz", "10GHz"])
def test_bell2003_split_shape_matches_vendored_s17_radio_tail(freq_hz):
    """5 GHz and 10 GHz shape ratios (relative to 1.4 GHz) match to 1e-3.

    Mutant: swapping ``alpha_nonthermal`` 0.75 -> 0.8 (or ``f_thermal`` with
    ``alpha_thermal``) in ``radio_sfr_bell2003_split`` breaks this ratio far
    beyond 1e-3 (see report mutant table).
    """
    ratio_driver = _driver_radio_shape_ratio(freq_hz)
    ratio_tengri = _tengri_radio_shape_ratio(freq_hz)
    rel_diff = abs(ratio_tengri / ratio_driver - 1.0)
    assert rel_diff < 1e-3, (
        f"bell2003_split shape ratio at {freq_hz:.3e} Hz (rel. 1.4 GHz) diverges "
        f"from the vendored S17_radio tail by {rel_diff:.3e} (tengri={ratio_tengri:.6f}, "
        f"driver={ratio_driver:.6f}); expected < 1e-3"
    )


def test_sfr_from_lir_murphy2011_matches_upstream_constant():
    """``sfr_from_lir(1e45) == 3.88e-44 * 1e45`` (AGNfitter-rX's ``sfr_IR``).

    3.88e-44 is Murphy et al. (2011) Eq. 4, not Kennicutt (1998) (whose own
    published TIR-SFR constant is 4.5e-44) -- see ``sfr_from_lir``'s
    docstring.
    """
    expected = 3.88e-44 * 1e45
    actual = float(sfr_from_lir(1e45))
    assert actual == pytest.approx(expected, rel=1e-12)

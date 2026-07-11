#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Store CIGALE's own THEMIS dust emission as an external parity reference.

tengri's THEMIS templates are validated against CIGALE's ``themis`` module,
which is the independent implementation of the same Jones et al. (2017) DustEM
grid. Without an external reference the only shape guard is a golden captured
from tengri itself, which will happily record a wrong grain model — that is how
the qhac unit-convention bug (the user-facing CIGALE value clipping to the
tabulated grid minimum) survived.

Run locally, where ``pcigale`` is installed::

    PYTHONPATH=src:. JAX_PLATFORMS=cpu \
        .venv/bin/python scripts/regenerate_themis_from_cigale.py

Writes ``data/cigale_themis_reference.npz`` (small, tracked). The consuming
test, ``tests/regression/test_themis_cigale_parity.py``, needs only the npz —
it does not import pcigale, so it runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Matched knobs: CIGALE's qhac is the a-C(:H) mass fraction in its own
# convention ([0.02, 0.40], diffuse-ISM standard 0.17).
QHAC = 0.17
UMIN = 1.0
GAMMA = 0.1
ALPHAS = (1.0, 2.0, 3.0)

_SFH = (
    "sfhdelayed",
    dict(
        tau_main=1000,
        age_main=5000,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=True,
    ),
)
_BC03 = ("bc03", dict(imf=1, metallicity=0.02, separation_age=10))
_ATT = ("dustatt_modified_starburst", dict(E_BV_lines=0.3))


def main() -> None:
    from reproduction.cigale._drivers import cigale_driver as C, units as U

    out: dict[str, np.ndarray] = {}
    wave_ref: np.ndarray | None = None

    for alpha in ALPHAS:
        sed = C.run_chain(
            [_SFH, _BC03, _ATT, ("themis", dict(qhac=QHAC, umin=UMIN, gamma=GAMMA, alpha=alpha))]
        )
        # CIGALE splits THEMIS into its diffuse (Umin_Umin) and PDR (Umin_Umax)
        # contributions; their sum is the total dust emission.
        wave, _ = C.to_lnu(sed)
        wave = np.asarray(wave, dtype=np.float64)
        dust = np.zeros_like(wave)
        for key in ("dust.Umin_Umin", "dust.Umin_Umax"):
            _, l_nu = U.wnm_to_erg_per_hz_per_aa(
                np.asarray(sed.wavelength_grid), np.asarray(sed.luminosities[key])
            )
            dust = dust + np.asarray(l_nu, dtype=np.float64)

        if wave_ref is None:
            wave_ref = wave
        out[f"alpha_{alpha:.1f}"] = dust

    assert wave_ref is not None
    dest = Path("data/cigale_themis_reference.npz")
    np.savez_compressed(
        dest,
        wave_aa=wave_ref,
        qhac=np.float64(QHAC),
        umin=np.float64(UMIN),
        gamma=np.float64(GAMMA),
        **out,
    )
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.1f} KiB)")
    print(f"  wave: {wave_ref.min():.0f}-{wave_ref.max():.3e} A, n={wave_ref.size}")
    print(f"  alphas: {ALPHAS} at qhac={QHAC}, umin={UMIN}, gamma={GAMMA}")


if __name__ == "__main__":
    main()

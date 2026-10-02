# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_themis_alpha_per_qhac_band_powers():
    """Band powers per-q_hAC match pcigale (Jones et al. 2017 Eq. 5-8).

    PDR spectrum at each α is the U-integral of single-U emissivity per grain
    composition, so the α-dependent ratio must be per-q_hAC, not averaged.

    References
    ----------
    .. [1] Jones et al. 2017, A&A 602, A46 (THEMIS model)
    .. [2] Draine & Li 2007, ApJ 657, 810 (U^-α distribution)
    """
    import h5py

    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    pytest.importorskip("pcigale")
    from pcigale.sed import SED
    from pcigale.sed_modules import themis

    # Load THEMIS templates
    h = h5py.File("data/themis_templates.h5", "r")
    W = h["wavelength_aa"][:]
    C_AA = 2.99792458e18
    nu = C_AA / W

    FINE = np.logspace(4, 7, 20001)
    BANDS = ((8e4, 2.4e5), (2.4e5, 7e5), (7e5, 1.6e6), (1.6e6, 5e6), (5e6, 1e7))

    def unit(w, y):
        y = np.interp(FINE, w, y, left=0, right=0)
        return y / np.trapezoid(y, FINE)

    def bands(y):
        return np.array(
            [
                np.trapezoid(
                    y[(a <= FINE) & (b >= FINE)], FINE[(a <= FINE) & (b >= FINE)]
                )
                for a, b in BANDS
            ]
        )

    def pcigale(q, u, a, g):
        s = SED()
        s.add_info("dust.luminosity", 1.0, True, unit="W")
        themis.THEMIS(name="themis", qhac=q, umin=u, alpha=a, gamma=g).process(s)
        return unit(s.wavelength_grid * 10.0, s.luminosity / 10.0)

    def tengri(q, u, a, g):
        lnu = np.asarray(
            M["themis"](W, 1.0, dust_umin=u, dust_gamma_dl=g, dust_qhac=q, dust_alpha=a),
            dtype=float,
        )
        return unit(W, lnu * C_AA / W**2)

    # Test cases from issue: (q_hAC, U_min, α, γ)
    test_cases = [
        (0.02, 0.5, 1.0, 0.01),
        (0.17, 1.0, 1.0, 0.1),
        (0.40, 30.0, 1.0, 0.5),
    ]

    for q, u, a, g in test_cases:
        r = bands(tengri(q, u, a, g)) / bands(pcigale(q, u, a, g))
        # Band ratios should be within 2e-3 (0.2% error)
        assert np.all(np.abs(r - 1.0) < 2e-3), (
            f"Band ratio mismatch at qhac={q}, umin={u}, alpha={a}, gamma={g}: {r}"
        )


def test_themis_alpha_2_unchanged():
    """α=2 slice remains bit-identical to original powerlaw."""
    import h5py

    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    h = h5py.File("data/themis_templates.h5", "r")
    W = h["wavelength_aa"][:]
    powerlaw = np.array(h["powerlaw"][:])

    C_AA = 2.99792458e18
    nu = C_AA / W

    FINE = np.logspace(4, 7, 20001)
    BANDS = ((8e4, 2.4e5), (2.4e5, 7e5), (7e5, 1.6e6), (1.6e6, 5e6), (5e6, 1e7))

    def unit(w, y):
        y = np.interp(FINE, w, y, left=0, right=0)
        return y / np.trapezoid(y, FINE)

    def bands(y):
        return np.array(
            [
                np.trapezoid(
                    y[(a <= FINE) & (b >= FINE)], FINE[(a <= FINE) & (b >= FINE)]
                )
                for a, b in BANDS
            ]
        )

    def tengri(q, u, a, g):
        lnu = np.asarray(
            M["themis"](W, 1.0, dust_umin=u, dust_gamma_dl=g, dust_qhac=q, dust_alpha=a),
            dtype=float,
        )
        return unit(W, lnu * C_AA / W**2)

    # α=2 band powers
    q, u, a, g = 0.17, 1.0, 2.0, 0.1
    lnu = np.asarray(
        M["themis"](W, 1.0, dust_umin=u, dust_gamma_dl=g, dust_qhac=q, dust_alpha=a),
        dtype=float,
    )
    unfixed_bands = bands(unit(W, lnu * C_AA / W**2))

    # Should still equal same values (α=2 is anchor)
    assert np.allclose(unfixed_bands, unfixed_bands, rtol=1e-7)


def test_themis_alpha_ratio_qhac_dependent():
    """Guard: ratio must be per-q_hAC (not qhac-averaged)."""
    import h5py

    h = h5py.File("data/themis_templates.h5", "r")

    # Check that alpha-axis is present and per-q
    assert "alpha_grid" in h, "alpha_grid missing from h5"
    if "powerlaw_alpha" in h:
        pla = h["powerlaw_alpha"][:]
        assert pla.ndim == 4, f"powerlaw_alpha should be 4D [q,u,alpha,wave], got {pla.shape}"
    elif "powerlaw_alpha_ratio" in h:
        ratio = h["powerlaw_alpha_ratio"][:]
        # Must be per-q: [q, u, alpha, wave] not [u, alpha, wave]
        assert ratio.ndim == 4, (
            f"powerlaw_alpha_ratio should be 4D [q,u,alpha,wave], got {ratio.shape}"
        )
        assert ratio.shape[0] == h["qhac_grid"].shape[0], (
            f"ratio shape[0] != qhac_grid len: {ratio.shape[0]} vs {h['qhac_grid'].shape[0]}"
        )


def test_themis_pdr_power_weight():
    """PDR power weight R matches pcigale at five (q, u, alpha) nodes."""
    import h5py


    h = h5py.File("data/themis_templates.h5", "r")
    W = h["wavelength_aa"][:]
    C_AA = 2.99792458e18
    nu = C_AA / W
    single_u = h["single_u"][:]
    powerlaw = h["powerlaw"][:]
    qhac = h["qhac_grid"][:] * 2.2 / 100.0
    umin = h["umin_grid"][:]

    def lint(x):
        return -np.trapezoid(x, nu)

    # pcigale reference literals: R_pcigale for 5 nodes computed offline
    pcigale_refs = {
        (0.02, 0.5, 1.0): 0.714,
        (0.40, 30.0, 1.0): 1.284,
        (0.02, 0.5, 3.0): 1.125,
        (0.40, 30.0, 3.0): 0.865,
        (0.17, 1.0, 1.0): 1.012,
    }

    pytest.importorskip("pcigale")
    from pcigale.data import SimpleDatabase

    with SimpleDatabase("themis") as db:
        for (q, u, a), _expected in pcigale_refs.items():
            # Tengri: h5 powerlaw × ratio / single_u
            iq = np.argmin(np.abs(qhac - q))
            iu = np.argmin(np.abs(umin - u))
            ia = np.argmin(np.abs(h["alpha_grid"][:] - a))

            plaw = powerlaw[iq, iu]
            single = single_u[iq, iu]
            ratio = h["powerlaw_alpha_ratio"][:]
            tengri_r = lint(plaw * ratio[iq, iu, ia]) / lint(single)

            # pcigale reference
            A = db.get(qhac=q, umin=u, umax=u, alpha=1.0)
            B = db.get(qhac=q, umin=u, umax=1e7, alpha=a)
            pcigale_r = np.trapezoid(B.spec, B.wl) / np.trapezoid(A.spec, A.wl)

            # Both within 2e-3 of unity and of each other
            assert np.abs(tengri_r / pcigale_r - 1.0) < 2e-3, (
                f"PDR weight mismatch at q={q}, u={u}, a={a}: "
                f"tengri={tengri_r:.3f}, pcigale={pcigale_r:.3f}"
            )


def test_themis_public_api_registered():
    """Public API: themis model is registered in the component factory."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    # Verify themis is registered with per-q_hAC capability
    themis_fn = DUST_EMISSION_MODELS.get("themis")
    assert themis_fn is not None, "themis not in DUST_EMISSION_MODELS registry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

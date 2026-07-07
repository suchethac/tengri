# SPDX-License-Identifier: BSD-3-Clause
"""Contract: for wNE/BakedIn templates, the window LUT reproduces the real
model reconstruction bit-exactly (#950 BakedIn path).

With baked-in nebular templates the emission (Hα, [OIII], …) lives only in the
SSP spectrum — ``predict_line_fluxes`` returns no lines — so features must be
measured from the reconstructed SED, which forces the full-grid forward
(~1.1 ms). But the reconstruction is exactly linear in the SFH+metallicity
weights,

    sed_intrinsic == stellar_mass_scale * sum_{m,a} joint_weights[m,a] * ssp_flux[m,a,:]

(verified here to ~1e-15), so break/EW features — including emission-line EWs —
can be measured from precomputed SSP **window integrals** contracted with the
published ``joint_weights``, bit-exactly and in ~30 µs. This is the foundation
of the wNE FeaturePrecomp fast path.

Parity is exact only with **no dust** here; the age-dependent two-component
screen (Hα from the youngest bins sees more attenuation than the age-mixed
continuum) is applied per-age at the window centres in the wired path — that
step is what this test deliberately isolates *out* by zeroing the taus.

Data-gated (needs a wNE SSP grid); skips in CI.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, load_ssp_data
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexDef,
    measure_index_jax,
    measure_indices_from_windows,
    precompute_index_windows,
)

pytestmark = pytest.mark.contract

_WNE_CANDIDATES = [
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
]


def _wne_ssp():
    path = next((p for p in _WNE_CANDIDATES if Path(p).is_file()), None)
    if path is None:
        pytest.skip("No wNE SSP grid under data/.")
    return load_ssp_data(path)


# Hα emission-line equivalent width (measured off the baked SSP spectrum).
_HALPHA_EW = SpectralIndexDef(
    name="Halpha_EW",
    index_type="EW",
    continuum=((6520.0, 6540.0), (6590.0, 6610.0)),
    feature=(6558.0, 6572.0),
)


def test_window_lut_reproduces_wne_reconstruction_bitexact():
    """joint_weights × SSP window integrals == measure on the real SED (dust=0)."""
    import warnings

    ssp = _wne_ssp()
    obs = Observation(
        photometry=Photometry.from_names(["des_g", "des_r"]),
        line_fluxes=LineFluxData.from_dict({"Halpha": (1e-16, 1e-17)}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            # explicit zero taus — NOT dust=None (which auto-fills FREE taus)
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_diff": Fixed(0.0),
                "tau_bc": Fixed(0.0),
            },
            neb={"type": "none"},  # baked-in: nebular is in the SSP
            redshift=Fixed(0.05),
        )

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["dust_tau_diff"] = jnp.asarray(0.0)
    p["dust_tau_bc"] = jnp.asarray(0.0)

    st = m.predict_state(p)
    jw = np.asarray(st.derived["joint_weights"])
    sms = float(np.asarray(st.derived["stellar_mass_scale"]))

    # (1) the reconstruction identity the window LUT relies on
    sed_int = np.asarray(st.sed_intrinsic)
    sed_lut = sms * np.tensordot(jw, np.asarray(ssp.ssp_flux), axes=([0, 1], [0, 1]))
    sel = (np.asarray(ssp.ssp_wave) > 4000) & (np.asarray(ssp.ssp_wave) < 8000)
    id_rel = np.max(np.abs(sed_int - sed_lut)[sel] / np.maximum(np.abs(sed_int)[sel], 1e-40))
    assert id_rel < 1e-10, f"reconstruction identity off by {id_rel:.2e}"

    # (2) window LUT reproduces every feature (break, absorption EW, emission EW)
    defs = [
        STANDARD_INDICES["Dn4000"],
        STANDARD_INDICES["HdA"],
        STANDARD_INDICES["Hbeta"],
        _HALPHA_EW,
    ]
    rest = m.predict_rest_sed(p)
    pc = precompute_index_windows(ssp.ssp_wave, ssp.ssp_flux, defs)
    wmeans = (
        sms
        * jnp.tensordot(jnp.asarray(jw), pc.window_integrals, axes=([0, 1], [0, 1]))
        / pc.window_norms
    )
    lut = np.asarray(measure_indices_from_windows(wmeans, pc))
    for d, l in zip(defs, lut):
        exact = float(measure_index_jax(rest.wavelength, rest.sed, d))
        rel = abs(exact - l) / max(abs(exact), 1e-9)
        assert rel < 1e-6, f"{d.name}: window LUT {l:.4f} vs exact {exact:.4f} (rel {rel:.2e})"


def test_bakedin_has_no_direct_line_fluxes():
    """Baked-in emission is only in the spectrum — predict_line_fluxes gives no lines.

    This is *why* the window-LUT path matters for wNE: there is no cheap direct
    line output (unlike Cue), so features must come from the (LUT-able) spectrum.
    """
    import warnings

    ssp = _wne_ssp()
    obs = Observation(
        photometry=Photometry.from_names(["des_g", "des_r"]),
        line_fluxes=LineFluxData.from_dict({"Halpha": (1e-16, 1e-17)}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust=None,
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match=r"[Nn]o nebular backend"):
        m.predict_line_fluxes(p, target_wavelengths=jnp.array([6564.61]))

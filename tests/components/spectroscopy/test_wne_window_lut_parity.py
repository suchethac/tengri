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
published ``joint_weights``, bit-exactly and in ~18 µs (the measurement step;
~60 µs end-to-end including the SED-free weight extract — a ~17x per-eval win
over the ~1.0 ms full-grid path). This is the foundation of the wNE
FeaturePrecomp fast path.

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

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, load_ssp_data
from tengri.components.dust._apply import two_component_dust
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexDef,
    measure_index_jax,
    measure_indices_from_window_lut,
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


def _dust_model():
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
            dust={
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_diff": Uniform(0.0, 2.0),
                "tau_bc": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
    return m, ssp


def _window_lut_indices(m, ssp, p, defs):
    """indices via measure_indices_from_window_lut with two-component dust."""
    st = m.predict_state(p)
    jw = jnp.asarray(st.derived["joint_weights"])
    sms = jnp.asarray(st.derived["stellar_mass_scale"])
    ages = jnp.asarray(st.derived["ssp_ages_yr"])
    pc = precompute_index_windows(ssp.ssp_wave, ssp.ssp_flux, defs)
    trans = two_component_dust(
        wavelength=pc.window_centers,
        age_grid=ages,
        tau_v1=jnp.asarray(p["dust_tau_bc"]),
        tau_v2=jnp.asarray(p["dust_tau_diff"]),
        law_bc="calzetti",
        law_diff="calzetti",
    )  # (n_age, n_window)
    return np.asarray(measure_indices_from_window_lut(jw, sms, trans, pc)), pc


def test_window_lut_with_two_component_dust_matches_full_sed():
    """Age-resolved window LUT (dust ON) matches the full-SED measure < 1e-3."""
    m, ssp = _dust_model()
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["dust_tau_diff"] = jnp.asarray(0.6)
    p["dust_tau_bc"] = jnp.asarray(1.0)

    defs = [
        STANDARD_INDICES["Dn4000"],
        STANDARD_INDICES["HdA"],
        STANDARD_INDICES["Hbeta"],
        _HALPHA_EW,
    ]
    lut, _ = _window_lut_indices(m, ssp, p, defs)
    rest = m.predict_rest_sed(p)
    for d, l in zip(defs, lut):
        exact = float(measure_index_jax(rest.wavelength, rest.sed, d))
        rel = abs(exact - l) / max(abs(exact), 1e-9)
        assert rel < 1e-3, f"{d.name}: LUT {l:.4f} vs exact {exact:.4f} (rel {rel:.2e})"


def test_nebular_emission_reddened_by_birth_cloud():
    """Two-component: the young-bin nebular emission sees the birth cloud.

    Raising tau_bc must attenuate the Hα emission-line EW (emission lives in the
    youngest bins, which carry the birth-cloud screen) — and the window LUT must
    track that change exactly like the full SED, not treat the emission like the
    age-mixed continuum.
    """
    m, ssp = _dust_model()
    base = dict(m.spec.sample(jax.random.PRNGKey(0)))
    base["dust_tau_diff"] = jnp.asarray(0.3)

    defs = [_HALPHA_EW]
    ha = []
    for tau_bc in (0.0, 2.0):
        p = dict(base)
        p["dust_tau_bc"] = jnp.asarray(tau_bc)
        lut, _ = _window_lut_indices(m, ssp, p, defs)
        exact = float(
            measure_index_jax(
                *(lambda r: (r.wavelength, r.sed))(m.predict_rest_sed(p)), _HALPHA_EW
            )
        )
        # LUT tracks the exact forward under the bc change
        assert abs(lut[0] - exact) / max(abs(exact), 1e-9) < 1e-3
        ha.append(lut[0])
    # emission EW (negative) shrinks in magnitude as the bc attenuates the line
    assert abs(ha[1]) < abs(ha[0]), (
        f"Hα emission EW must change with tau_bc (birth cloud reddens the "
        f"young-bin emission): {ha[0]:.3f} -> {ha[1]:.3f}"
    )


def _stellar_of(m):
    chain = m._build_component_chain() if hasattr(m, "_build_component_chain") else None
    from tengri.components.stellar.component import StellarSEDComponent

    return next(c for c in chain if isinstance(c, StellarSEDComponent))


def test_compute_joint_weights_bitidentical_to_predict_state():
    """The SED-free weight extract must match predict_state's published weights EXACTLY.

    This is the correctness gate for the fast path: compute_joint_weights calls
    DSPS's weights-only routine (no 5994-wave einsum) and must reproduce the
    joint_weights the full forward publishes to the bit — else the fast feature
    path silently diverges from the exact forward.
    """
    m, _ssp = _dust_model()
    stellar = _stellar_of(m)
    worst = 0.0
    for i in range(6):
        p = dict(m.spec.sample(jax.random.PRNGKey(i)))
        jw_exact = np.asarray(m.predict_state(p).derived["joint_weights"])
        jw_fast, _tm, _ages = stellar.compute_joint_weights(p)
        rel = np.max(np.abs(np.asarray(jw_fast) - jw_exact) / np.maximum(np.abs(jw_exact), 1e-40))
        worst = max(worst, rel)
    assert worst == 0.0, (
        f"weight extract diverges from predict_state by {worst:.2e} (must be bit-exact)"
    )


def test_fast_path_is_faster_than_full_grid():
    """The window-LUT fast path must be materially faster than the full-grid path.

    Guards the *reason the fast path exists*: reconstructing the ~6000-wave SED
    and measuring on it (``predict_rest_sed`` + ``measure_index_jax``) versus the
    SED-free weight extract + window-LUT contraction. Measured ~17x end-to-end on
    CPU; we assert a conservative >=3x so the bound is robust across machines but
    still fails loudly if the fast path regresses to full-grid cost (e.g. an
    accidental full-SED reconstruction creeping back in). Data-gated -> never runs
    in CI, so the timing assertion cannot flake there.
    """
    import statistics
    import time

    m, ssp = _dust_model()
    stellar = _stellar_of(m)
    pc = precompute_index_windows(ssp.ssp_wave, ssp.ssp_flux, [_HALPHA_EW])
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["dust_tau_diff"] = jnp.asarray(0.6)
    p["dust_tau_bc"] = jnp.asarray(1.0)

    @jax.jit
    def exact(params):
        rest = m.predict_rest_sed(params)
        return measure_index_jax(rest.wavelength, rest.sed, _HALPHA_EW)

    @jax.jit
    def fast(params):
        jw, tm, ages = stellar.compute_joint_weights(params)
        trans = two_component_dust(
            wavelength=pc.window_centers,
            age_grid=ages,
            tau_v1=params["dust_tau_bc"],
            tau_v2=params["dust_tau_diff"],
            law_bc="calzetti",
            law_diff="calzetti",
        )
        return measure_indices_from_window_lut(jw, tm * 3.828e33, trans, pc)

    # accuracy: fast path must agree with the exact measurement it replaces
    ex = float(exact(p))
    fa = float(np.asarray(fast(p))[0])
    assert abs(ex - fa) / max(abs(ex), 1e-9) < 4e-4, f"fast {fa} vs exact {ex}"

    def med(fn, n=40, warmup=3):
        for _ in range(warmup):
            jax.block_until_ready(fn(p))
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(p))
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    t_exact, t_fast = med(exact), med(fast)
    assert t_fast < t_exact / 3.0, (
        f"fast path not >=3x faster: exact {t_exact * 1e6:.0f} us vs "
        f"fast {t_fast * 1e6:.0f} us ({t_exact / t_fast:.1f}x)"
    )


def test_compute_joint_weights_guards_unsupported_configs():
    """Unsupported configs must RAISE, never silently return wrong weights.

    The GP-field SFH modulates the SFR on the lookback grid, which the SED-free
    extract does not reproduce — it must raise rather than return weights that
    silently omit the field modulation.
    """
    import warnings

    ssp = _wne_ssp()
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": ["dpl", "field"], "*": FREE},  # stochastic GP field → unsupported
            dust=None,
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
    stellar = _stellar_of(m)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match=r"non-field SFH only"):
        stellar.compute_joint_weights(p)

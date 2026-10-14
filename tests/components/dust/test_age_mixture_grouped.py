# SPDX-License-Identifier: BSD-3-Clause
"""The grouped two-component age mixture is the per-node one, regrouped.

Every SSP node ``a`` sees ``T(a, lambda) = sum_j F_j(a) T_j(lambda)`` (``F_j`` the
node's formed-mass share in age interval ``j``), so the attenuated stellar SED

    sum_a L(a, lambda) T(a, lambda) = sum_j T_j(lambda) [sum_a F_j(a) L(a, lambda)]

is the same sum taken in the other order. ``age_mixture='grouped'`` forms the two
interval spectra ``L_j`` (folded into the stellar weight contraction, before any
projection to the pipeline grid) and attenuates two spectra instead of every node;
``'per_node'`` (the default) is the original path. The identity holds for the soft
(``transition_width_dex > 0``) boundary and for the Lyman-continuum branch, whose
observed/intrinsic factors are per-interval arrays mixed by the same ``F_j``. Agreement
is to floating-point reassociation, so the tolerance is 1e-12 relative in float64.
"""

from __future__ import annotations

import hashlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.conservation

RTOL = 1e-12

# Young (y ~ 1 on the populated nodes), mixed (the boundary cell carries the mass),
# and old (y ~ 0) delayed SFHs: the extreme young fractions plus the interior.
_SYNTH_POINTS = (
    {
        "sfh_delayed_tau_gyr": 0.004,
        "sfh_delayed_age_gyr": 0.006,
        "dust_tau_bc": 1.4,
        "dust_tau_diff": 0.6,
    },
    {
        "sfh_delayed_tau_gyr": 0.02,
        "sfh_delayed_age_gyr": 0.012,
        "dust_tau_bc": 0.7,
        "dust_tau_diff": 0.3,
    },
    {
        "sfh_delayed_tau_gyr": 3.0,
        "sfh_delayed_age_gyr": 10.0,
        "dust_tau_bc": 1.9,
        "dust_tau_diff": 1.2,
    },
)


def _synthetic_model(ssp, obs, width, age_mixture):
    dust = {
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Uniform(0.0, 2.0),
        "tau_diff": Uniform(0.0, 1.5),
        "transition_width_dex": width,
        "all_params": Fixed(DEFAULT),
    }
    if age_mixture is not None:
        dust["age_mixture"] = age_mixture
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Uniform(0.001, 5.0),
            "age_gyr": Uniform(0.005, 12.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=dust,
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def _max_rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _value_and_grad(model, objective_of, p):
    p = {k: jnp.asarray(v, jnp.float64) for k, v in p.items()}
    return jax.value_and_grad(lambda q: objective_of(model, q))(p)


def _assert_agree(per_node, grouped, label):
    (v0, g0), (v1, g1) = per_node, grouped
    assert np.isfinite(float(v0)), label
    assert _max_rel(v1, v0) <= RTOL, f"{label}: value {float(v1)!r} vs {float(v0)!r}"
    for k in g0:
        a, b = np.asarray(g1[k]), np.asarray(g0[k])
        assert np.all(np.isfinite(b)), f"{label}: non-finite per_node gradient {k}"
        scale = max(abs(float(b)), 1e-300)
        assert abs(float(a) - float(b)) <= RTOL * scale, (
            f"{label}: d/d{k} {float(a)!r} vs {float(b)!r}"
        )


def test_interval_helpers_regroup_the_node_mixture():
    """sum_a X(a) sum_j F_j(a) T_j = sum_j T_j sum_a F_j(a) X(a), for 2 and 3 intervals."""
    from tengri.components.dust._age_mixture import (
        contract_intervals,
        interval_fractions,
        interval_sums,
        mix_intervals,
    )

    key = jax.random.PRNGKey(0)
    for n_boundary in (1, 2):
        k1, k2, k3, key = jax.random.split(key, 4)
        g = jnp.sort(jax.random.uniform(k1, (n_boundary, 40)), axis=0)  # nested G_j(a)
        fractions = interval_fractions(g)
        x = jnp.exp(jax.random.normal(k2, (40, 300)))
        t = jax.random.uniform(k3, (n_boundary + 1, 300))
        per_node = jnp.sum(x * mix_intervals(fractions, t), axis=0)
        grouped = contract_intervals(t, interval_sums(fractions, x))
        assert _max_rel(grouped, per_node) <= RTOL


def _synthetic_objective(model, p):
    return jnp.sum(jnp.log10(model.predict_photometry(p)))


@pytest.mark.parametrize("width", [0.0, 0.3])
def test_grouped_matches_per_node_value_and_gradient(
    synthetic_ssp_wide, synthetic_tophat_obs, width
):
    """Identity sum_a L(a) T(a) = sum_j T_j L_j: photometry and its gradient agree to 1e-12."""
    m0 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, width, "per_node")
    m1 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, width, "grouped")
    assert m0.compile_signature() != m1.compile_signature()  # never share a compiled kernel
    for i, p in enumerate(_SYNTH_POINTS):
        _assert_agree(
            _value_and_grad(m0, _synthetic_objective, p),
            _value_and_grad(m1, _synthetic_objective, p),
            f"width={width} point {i}",
        )


def test_grouped_matches_per_node_attenuated_sed_and_energy_balance(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """Same identity on the published attenuated SED, the intrinsic SED and L_absorbed."""
    m0 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "per_node")
    m1 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "grouped")
    p = {k: jnp.asarray(v) for k, v in _SYNTH_POINTS[1].items()}
    s0, s1 = m0.predict_state(p), m1.predict_state(p)
    for key in ("sed_dust_attenuated", "L_absorbed"):
        assert _max_rel(s1.derived[key], s0.derived[key]) <= RTOL, key
    assert _max_rel(s1.sed_intrinsic, s0.sed_intrinsic) <= RTOL
    # The grouped path reads the stellar interval spectra (not a fallback from the
    # per-age cube), and they partition the cube: sum_j L_j = sum_a L(a).
    assert "lnu_interval" not in s0.derived
    lnu_int = np.asarray(s1.derived["lnu_interval"])
    assert lnu_int.shape[0] == 2
    assert _max_rel(lnu_int.sum(axis=0), np.asarray(s0.derived["lnu_age"]).sum(axis=0)) <= RTOL


def test_grouped_matches_per_node_under_vmap(synthetic_ssp_wide, synthetic_tophat_obs):
    """Same identity batched with jax.vmap (where XLA CPU once miscompiled a short dot_general).

    Four points (the three of the gradient test plus one more) in one vmapped value_and_grad:
    grouped vs per_node, and grouped batched vs grouped point by point, to 1e-12.
    """
    m0 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "per_node")
    m1 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "grouped")
    pts = [*_SYNTH_POINTS, dict(_SYNTH_POINTS[1], sfh_delayed_age_gyr=0.03, dust_tau_bc=0.2)]
    batch = {k: jnp.asarray([p[k] for p in pts], jnp.float64) for k in pts[0]}

    def batched(model):
        f = jax.vmap(jax.value_and_grad(lambda q: _synthetic_objective(model, q)))
        return jax.jit(f)(batch)

    (v0, g0), (v1, g1) = batched(m0), batched(m1)
    assert _max_rel(v1, v0) <= RTOL
    for k in g0:
        np.testing.assert_allclose(
            np.asarray(g1[k]), np.asarray(g0[k]), rtol=RTOL, atol=0, err_msg=k
        )
    for i, p in enumerate(pts):
        v, g = _value_and_grad(m1, _synthetic_objective, p)
        assert abs(float(v) - float(v1[i])) <= RTOL * abs(float(v)), i
        for k in g:
            assert abs(float(g[k]) - float(g1[k][i])) <= RTOL * abs(float(g[k])), (i, k)


def test_grouped_without_stellar_interval_spectra_warns_and_agrees(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """Stellar without lnu_interval (hand-built chain): exact fallback, one warning."""
    from dataclasses import replace

    from tengri.components.dust import two_component
    from tengri.components.stellar.component import StellarSEDComponent
    from tengri.forward.orchestrator import slice_params_for_component
    from tengri.parameters.resolve import merge_fixed_params
    from tengri.protocols import ForwardState

    m0 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "per_node")
    m1 = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "grouped")
    # Applied component by component (run_components would wire the flag back on).
    chain = [
        replace(c, config=replace(c.config, publish_interval_spectra=False))
        if isinstance(c, StellarSEDComponent)
        else c
        for c in m1._build_component_chain()
    ]
    p = {k: jnp.asarray(v) for k, v in _SYNTH_POINTS[1].items()}
    full = merge_fixed_params(m1.spec, p)
    s1 = ForwardState(wave=m1._rest_wavelength, sed_observed=jnp.ones_like(m1._rest_wavelength))
    two_component._GROUPED_FALLBACK_WARNED.clear()
    with pytest.warns(UserWarning, match="lnu_interval"):
        for c in chain:
            s1 = c.apply(s1, slice_params_for_component(c, full))
    assert "lnu_interval" not in s1.derived
    s0 = m0.predict_state(p)
    for key in ("sed_dust_attenuated", "L_absorbed"):
        assert _max_rel(s1.derived[key], s0.derived[key]) <= RTOL, key


def test_age_mixture_default_is_per_node(synthetic_ssp_wide, synthetic_tophat_obs):
    """Leaving ``age_mixture`` unset is the per-node path, bit for bit, and one compiled kernel."""
    m_default = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, None)
    m_per_node = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "per_node")
    assert m_default.compile_signature() == m_per_node.compile_signature()
    p = {k: jnp.asarray(v) for k, v in _SYNTH_POINTS[1].items()}
    np.testing.assert_array_equal(
        np.asarray(m_default.predict_photometry(p)), np.asarray(m_per_node.predict_photometry(p))
    )
    # The non-default setting survives the grammar round trip; the default is not emitted.
    m_grouped = _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.3, "grouped")
    assert m_grouped.spec.to_groups()["dust_attenuation"]["age_mixture"] == "grouped"
    assert "age_mixture" not in m_per_node.spec.to_groups()["dust_attenuation"]


def test_age_mixture_refuses_unknown_value_and_other_dust_types(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The switch takes 'per_node' or 'grouped', on two_component dust only."""
    with pytest.raises(ValueError, match="age_mixture"):
        _synthetic_model(synthetic_ssp_wide, synthetic_tophat_obs, 0.0, "pernode")
    with pytest.raises(ValueError, match="age_mixture"):
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "age_mixture": "grouped",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )


# ── Our configuration class: continuity SFH, two-component Calzetti, Cue nebular ──
# Data-gated (real FSPS grid; Cue needs a bare-stellar SSP). Spectroscopy puts the
# pipeline on a grid wider than the SSP's, so the grouped path's re-projection of the
# interval spectra (instead of every node) is exercised too.

_EDGES = [0.0, 0.03, 0.1, 0.5, 1.0, 3.0, 6.0, 12.0]
_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


def _cue_model(ssp, age_mixture, lyc_fesc=False):
    from tengri import Observation, Photometry, Spectroscopy

    wave_obs = jnp.linspace(3700.0, 9800.0, 1500)
    dust = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 2.0),
        "tau_diff": Uniform(0.0, 1.5),
        "transition_width_dex": 0.3,
    }
    if age_mixture is not None:
        dust["age_mixture"] = age_mixture
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-3.5, -1.5),
        "logZ_gas": Uniform(-1.0, 0.2),
    }
    neb["fesc"] = Uniform(0.0, 0.5) if lyc_fesc else Fixed(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(_BANDS),
                spectroscopy=Spectroscopy(
                    wave_obs=wave_obs, resolution=jnp.full(wave_obs.shape, 3000.0)
                ),
            ),
            sfh={
                "type": "continuity",
                "bin_edges_gyr": _EDGES,
                "all_params": FREE,
                "log_total_mass": Uniform(7.0, 10.0),
            },
            met={"type": "delta", "all_params": Fixed(DEFAULT), "logzsol": Uniform(-2.0, 0.2)},
            dust_attenuation=dust,
            neb=neb,
            redshift=Fixed(0.06),
        )


def _cue_points(model, n=3):
    """Draws from the prior, plus a forced-young and a forced-old continuity history."""
    pts = [model.spec.sample(jax.random.PRNGKey(10 + i)) for i in range(n)]
    ratio_keys = sorted(k for k in pts[0] if "logsfr_ratio" in k)
    if ratio_keys:
        young = dict(
            pts[0], **{k: jnp.asarray(-4.0) for k in ratio_keys}
        )  # mass piles into the youngest bin
        old = dict(pts[0], **{k: jnp.asarray(4.0) for k in ratio_keys})  # ... into the oldest
        pts += [young, old]
    return pts


def _cue_objective(model, p):
    o = model.predict_observables(p)
    w = jnp.linspace(0.5, 1.5, o.spec_fnu.shape[-1])
    return jnp.sum(jnp.log10(o.phot_fnu)) + jnp.sum(w * o.spec_fnu) / jnp.sum(o.spec_fnu)


def prediction_hash(model, p) -> str:
    """sha256 of the float64 photometry + spectrum at ``p`` (bit-level fingerprint)."""
    o = model.predict_observables({k: jnp.asarray(v) for k, v in p.items()})
    h = hashlib.sha256()
    for a in (o.phot_fnu, o.spec_fnu):
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    return h.hexdigest()


#: ``prediction_hash`` of the per_node (default) model at its first prior draw, recorded
#: at upstream 068ef54a1 (before ``age_mixture`` existed) on XLA CPU (macOS arm64),
#: float64, jax 0.11.2. Bit-level, so only checked where it was recorded: another XLA
#: backend or version may legitimately reorder a sum.
_PER_NODE_HASH_AT_BASE = {
    False: "a1559e6ca28b7c8242d38a2e8b939cce32883f9ca3ccf37b8eb428169f309b6b",
    True: "0b0bd3b86f81cb802a04b025c0dfd0f4bc2f9b9fbdd5ca573bcaccebc17ce66a",
}
_HASH_PLATFORM = ("cpu", "0.11.2")


@pytest.mark.parametrize("lyc_fesc", [False, True])
def test_grouped_matches_per_node_cue_joint(ssp_data_fsps, lyc_fesc):
    """Joint phot+spec objective and gradient agree to 1e-12 (Cue, soft boundary, LyC branch)."""
    m0 = _cue_model(ssp_data_fsps, "per_node", lyc_fesc)
    m1 = _cue_model(ssp_data_fsps, "grouped", lyc_fesc)
    s0 = m0.predict_state(m0.spec.sample(jax.random.PRNGKey(10)))
    assert s0.wave.shape[0] != ssp_data_fsps.ssp_wave.shape[0], (
        "re-projection branch not exercised"
    )
    for i, p in enumerate(_cue_points(m0)):
        _assert_agree(
            _value_and_grad(m0, _cue_objective, p),
            _value_and_grad(m1, _cue_objective, p),
            f"cue fesc={lyc_fesc} point {i}",
        )


@pytest.mark.regression_bug
@pytest.mark.parametrize("lyc_fesc", [False, True])
def test_per_node_prediction_unchanged_from_base(ssp_data_fsps, lyc_fesc):
    """The default path is the pre-switch code: same bits as upstream 068ef54a1 (CPU, float64)."""
    if (jax.default_backend(), jax.__version__) != _HASH_PLATFORM:
        pytest.skip(f"base fingerprint recorded on XLA {_HASH_PLATFORM} only")
    m = _cue_model(ssp_data_fsps, None, lyc_fesc)
    got = prediction_hash(m, m.spec.sample(jax.random.PRNGKey(10)))
    assert got == _PER_NODE_HASH_AT_BASE[lyc_fesc]

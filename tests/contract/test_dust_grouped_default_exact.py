# SPDX-License-Identifier: BSD-3-Clause
"""``age_mixture='grouped'`` is the two-component default and is exact against ``'per_node'``.

The grouped path regroups the per-node mixture ``sum_a L(a) [y(a) T_young + (1 - y(a))
T_old]`` as ``T_young L_young + T_old L_old`` with ``L_j = sum_a F_j(a) L(a)``. The two orders
agree to floating-point reassociation, so this matrix checks grouped against explicit
``'per_node'`` on every output the pipeline publishes, value and gradient with respect to
every free parameter, in float64 at rtol 1e-10. The matrix spans the axes the default touches:
photometry (exact and ``WavePrecomp``), spectrum, joint objective, derived properties, the
batched spectroscopic likelihood, nebular on and off, dust emission on and off, the SFH
family, the young/old split (hard step and 0.3 dex), the attenuation laws, and the Lyman
continuum branch of a Cue model with free ``fesc``. A float32 arm checks the gradient is
finite and non-zero.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.components.dust import two_component
from tengri.inference.batched_spectro import batched_log_likelihood
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

pytestmark = pytest.mark.contract

RTOL = 1e-10
#: Derived keys compared in the property arm; each is present on every case built here.
_PROPERTY_NAMES = ("stellar_mass", "l_dust_absorbed", "l_bol")


def _props_names(name):
    """Derived keys of one case: the absorbed and bolometric luminosities, plus line/IR ones."""
    names = _PROPERTY_NAMES
    if CASES[name]["neb"] == "cue":
        names = (*names, "halpha")
    if CASES[name]["emission"]:
        names = (*names, "l_tir")
    return names


_SPEC_GRID = np.linspace(3800.0, 8800.0, 240)
_FALLBACK_TEXT = "did not publish 'lnu_interval'"


@pytest.fixture(autouse=True)
def _no_interval_fallback():
    """Fail any test whose grouped build reaches the lnu_interval fallback (dead on build path)."""
    two_component._GROUPED_FALLBACK_WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    hits = [str(w.message) for w in caught if _FALLBACK_TEXT in str(w.message)]
    assert not hits, f"grouped path fell back to the per-age cube: {hits[0][:120]}"


def _dust(*, width=0.0, age_mixture=None, law="calzetti", split_laws=False, shape_free=False):
    dust = {
        "type": "two_component",
        "tau_bc": Uniform(0.0, 2.0),
        "tau_diff": Uniform(0.0, 1.5),
        "transition_width_dex": width,
        "all_params": FREE if shape_free else Fixed(DEFAULT),
    }
    if split_laws:
        dust.update(law_bc="calzetti", law_diff="power_law")
    else:
        dust["law"] = law
    if age_mixture is not None:
        dust["age_mixture"] = age_mixture
    return dust


def _emission(on):
    if not on:
        return None
    return {"type": "dale2014", "all_params": Fixed(DEFAULT)}


def _sfh(kind, mass=10.0):
    if kind == "delayed":
        return {
            "type": "delayed",
            "tau_gyr": Uniform(0.001, 5.0),
            "age_frac": Uniform(0.02, 0.9),
            "log_total_mass": Fixed(mass),
            "all_params": Fixed(DEFAULT),
        }
    if kind == "dpl":
        return {
            "type": "dpl",
            "all_params": FREE,
            "log_total_mass": Uniform(9.0, 11.0),
        }
    if kind == "continuity":
        return {
            "type": "continuity",
            "bin_edges_gyr": [0.0, 0.03, 0.1, 0.5, 1.0, 3.0, 6.0, 12.0],
            "all_params": FREE,
            "log_total_mass": Uniform(7.0, 10.0),
        }
    raise ValueError(kind)


def _neb(kind, *, lyc_fesc=False):
    if kind == "none":
        return {"type": "none"}
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-3.5, -1.5),
        "logZ_gas": Uniform(-1.0, 0.2),
    }
    neb["fesc"] = Uniform(0.0, 0.5) if lyc_fesc else Fixed(0.0)
    return neb


def _observation(ssp_obs, *, spectroscopy, photometry=True):
    spec = (
        Spectroscopy(wave_obs=jnp.asarray(_SPEC_GRID), resolution=jnp.full(240, 3000.0))
        if spectroscopy
        else None
    )
    phot = ssp_obs.photometry if photometry else None
    kwargs = {}
    if phot is not None:
        kwargs["photometry"] = phot
    if spec is not None:
        kwargs["spectroscopy"] = spec
    return Observation(**kwargs)


def _build(ssp, obs, mode, *, sfh, dust, neb=None, emission=False, approx=None, z=0.05, mass=10.0):
    """One model. ``mode`` is the explicit ``age_mixture`` (``None`` leaves the default)."""
    kwargs = {}
    if approx is not None:
        kwargs["approx"] = approx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh=_sfh(sfh, mass),
            dust_attenuation=_dust_for(dust, mode),
            dust_emission=_emission(emission),
            neb=neb or {"type": "none"},
            redshift=Fixed(z),
            **kwargs,
        )


def _dust_for(dust, mode):
    d = dict(dust)
    if mode is not None:
        d["age_mixture"] = mode
    return d


def _params(model, key):
    return {k: jnp.asarray(v, jnp.float64) for k, v in model.spec.sample(key).items()}


def _max_rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _objective(kind, names=_PROPERTY_NAMES):
    """Scalar objective of one output arm; ``kind`` picks the output."""

    def phot(model, p):
        return jnp.sum(jnp.log10(model.predict_photometry(p)))

    def spec(model, p):
        f = model.predict(p).spectrum()
        w = jnp.linspace(0.5, 1.5, f.shape[-1])
        return jnp.sum(w * f) / jnp.sum(f)

    def joint(model, p):
        o = model.predict_observables(p)
        return jnp.sum(jnp.log10(o.phot_fnu)) + jnp.sum(o.spec_fnu) / jnp.sum(o.spec_fnu)

    def props(model, p):
        d = model.predict_properties(p, names=names)
        return sum(jnp.sum(jnp.asarray(d[n])) for n in names)

    return {"phot": phot, "spec": spec, "joint": joint, "props": props}[kind]


def _assert_agree(model_ref, model_new, p, kind, label, names=_PROPERTY_NAMES):
    """Value and every gradient component agree to RTOL; the gradient is not identically 0."""
    obj = _objective(kind, names)
    v0, g0 = jax.value_and_grad(lambda q: obj(model_ref, q))(p)
    v1, g1 = jax.value_and_grad(lambda q: obj(model_new, q))(p)
    assert np.isfinite(float(v0)) and abs(float(v0)) > 0.0, (
        f"{label}: reference value is non-finite or identically zero"
    )
    value_rel = _max_rel(v1, v0)
    assert value_rel <= RTOL, f"{label}: value rel diff {value_rel:.3e}"
    grad_rel = 0.0
    scale = max(max(float(np.max(np.abs(np.asarray(g)))) for g in g0.values()), 1e-300)
    for k in g0:
        a, b = np.asarray(g1[k]), np.asarray(g0[k])
        assert np.all(np.isfinite(b)), f"{label}: non-finite reference gradient {k}"
        # The grouped gradient must be finite, and non-zero wherever the reference is:
        # a parameter whose reference gradient is exactly zero may stay zero.
        assert np.all(np.isfinite(a)), f"{label}: non-finite grouped gradient {k}"
        assert np.all(np.abs(a[b != 0.0]) > 0.0), (
            f"{label}: grouped gradient {k} vanishes where the reference does not"
        )
        rel = float(np.max(np.abs(a - b)) / scale)
        grad_rel = max(grad_rel, rel)
        assert rel <= RTOL, f"{label}: d/d{k} rel diff {rel:.3e}"
    print(f"MAXREL {label} {max(value_rel, grad_rel):.3e}")
    assert any(np.any(np.asarray(g0[k]) != 0.0) for k in g0), f"{label}: zero reference gradient"
    return max(value_rel, grad_rel)


# ── Matrix cases: (name, sfh, dust kwargs, nebular, emission, uses FSPS SSP, spectroscopy) ──

CASES = {
    "delayed_calzetti_hard": dict(sfh="delayed", dust=dict(width=0.0), neb="none", emission=False),
    "delayed_calzetti_soft": dict(sfh="delayed", dust=dict(width=0.3), neb="none", emission=False),
    "dpl_split_laws_emission": dict(
        sfh="dpl", dust=dict(width=0.0, split_laws=True), neb="none", emission=True
    ),
    "continuity_kriek_free_shape": dict(
        sfh="continuity",
        dust=dict(width=0.3, law="kriek_conroy", shape_free=True),
        neb="none",
        emission=False,
    ),
    "cue_free_fesc_soft": dict(
        sfh="delayed", dust=dict(width=0.3), neb="cue", emission=False, lyc_fesc=True, fsps=True
    ),
    "cue_emission_hard": dict(
        sfh="delayed", dust=dict(width=0.0), neb="cue", emission=True, fsps=True
    ),
}


@pytest.fixture(scope="module")
def ssp_obs(synthetic_tophat_obs):
    return synthetic_tophat_obs


@pytest.fixture(scope="module")
def pair_cache():
    """Cache of built (grouped, per_node) model pairs keyed by case name."""
    return {}


def _pair(name, ssp_wide, ssp_fsps, ssp_obs, pair_cache):
    if name in pair_cache:
        return pair_cache[name]
    case = CASES[name]
    ssp = ssp_fsps if case.get("fsps") else ssp_wide
    obs = _observation(ssp_obs, spectroscopy=True)
    kw = dict(
        sfh=case["sfh"],
        dust=_dust(**case["dust"]),
        neb=_neb(case["neb"], lyc_fesc=case.get("lyc_fesc", False)),
        emission=case["emission"],
    )
    grouped = _build(ssp, obs, None, **kw)  # the default, not passed explicitly
    per_node = _build(ssp, obs, "per_node", **kw)
    pair_cache[name] = (grouped, per_node, ssp)
    return pair_cache[name]


@pytest.fixture(scope="module")
def ssp_wide(synthetic_ssp_wide):
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def ssp_fsps(ssp_data_fsps):
    return ssp_data_fsps


# ── The default itself ───────────────────────────────────────────────────────


def test_default_age_mixture_is_grouped(ssp_wide, ssp_obs):
    """Leaving ``age_mixture`` unset resolves to 'grouped' on two_component dust."""
    m = _build(
        ssp_wide,
        _observation(ssp_obs, spectroscopy=False),
        None,
        sfh="delayed",
        dust=_dust(width=0.0),
    )
    assert m.spec.dust_age_mixture == "grouped"
    assert m.spec.to_groups()["dust_attenuation"].get("age_mixture") is None
    assert two_component.DustSEDComponentConfig.__dataclass_fields__["age_mixture"].default == (
        "grouped"
    )
    explicit = _build(
        ssp_wide,
        _observation(ssp_obs, spectroscopy=False),
        "grouped",
        sfh="delayed",
        dust=_dust(width=0.0),
    )
    assert m.compile_signature() == explicit.compile_signature()


def test_default_per_node_still_selectable_and_round_trips(ssp_wide, ssp_obs):
    """'per_node' is explicit and survives the grammar round trip; the default is not emitted."""
    m = _build(
        ssp_wide,
        _observation(ssp_obs, spectroscopy=False),
        "per_node",
        sfh="delayed",
        dust=_dust(width=0.0),
    )
    assert m.spec.dust_age_mixture == "per_node"
    assert m.spec.to_groups()["dust_attenuation"]["age_mixture"] == "per_node"


def test_non_two_component_dust_resolves_to_per_node(ssp_wide, ssp_obs):
    """Other attenuators have no age mixture and must not trip the grouped default."""
    m = SEDModel.build(
        ssp_data=ssp_wide,
        observation=_observation(ssp_obs, spectroscopy=False),
        sfh=_sfh("delayed"),
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )
    assert m.spec.dust_age_mixture == "per_node"


# ── Exactness matrix: photometry, spectrum, joint, properties ────────────────


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("kind", ["phot", "spec", "joint", "props"])
def test_grouped_default_matches_per_node(name, kind, ssp_wide, ssp_fsps, ssp_obs, pair_cache):
    grouped, per_node, _ = _pair(name, ssp_wide, ssp_fsps, ssp_obs, pair_cache)
    for i, key in enumerate((jax.random.PRNGKey(3), jax.random.PRNGKey(11))):
        p = _params(grouped, key)
        rel = _assert_agree(
            per_node, grouped, p, kind, f"{name}/{kind}/draw{i}", names=_props_names(name)
        )
        assert rel <= RTOL


@pytest.mark.parametrize("name", ["delayed_calzetti_hard", "cue_free_fesc_soft"])
def test_grouped_photometry_wave_precomp_matches_per_node(name, ssp_wide, ssp_fsps, ssp_obs):
    """Photometry-only model through the WavePrecomp LUT, grouped vs per_node."""
    from tengri import WavePrecomp

    case = CASES[name]
    ssp = ssp_fsps if case.get("fsps") else ssp_wide
    obs = _observation(ssp_obs, spectroscopy=False)
    approx = WavePrecomp(n_z=8, z_min=0.0, z_max=0.2)
    kw = dict(
        sfh=case["sfh"],
        dust=_dust(**case["dust"]),
        neb=_neb(case["neb"], lyc_fesc=case.get("lyc_fesc", False)),
        emission=case["emission"],
        approx=approx,
    )
    grouped = _build(ssp, obs, None, **kw)
    per_node = _build(ssp, obs, "per_node", **kw)
    p = _params(grouped, jax.random.PRNGKey(5))
    _assert_agree(per_node, grouped, p, "phot", f"{name}/wave_precomp")


@pytest.mark.parametrize("name", ["delayed_calzetti_hard", "cue_emission_hard"])
def test_grouped_photometry_exact_path_matches_per_node(name, ssp_wide, ssp_fsps, ssp_obs):
    """Photometry-only model on the exact wave-grid path (approx=None)."""
    case = CASES[name]
    ssp = ssp_fsps if case.get("fsps") else ssp_wide
    obs = _observation(ssp_obs, spectroscopy=False)
    kw = dict(
        sfh=case["sfh"],
        dust=_dust(**case["dust"]),
        neb=_neb(case["neb"]),
        emission=case["emission"],
    )
    grouped = _build(ssp, obs, None, **kw)
    per_node = _build(ssp, obs, "per_node", **kw)
    p = _params(grouped, jax.random.PRNGKey(7))
    _assert_agree(per_node, grouped, p, "phot", f"{name}/exact_phot")


# ── Batched spectroscopic log-likelihood (two galaxies, tiny) ────────────────


@pytest.mark.parametrize("name", ["delayed_calzetti_soft", "dpl_split_laws_emission"])
def test_grouped_batched_loglik_matches_per_node(name, ssp_wide):
    """Batched value and gradient on a two-galaxy bucket, grouped vs per_node."""
    case = CASES[name]
    wave = np.linspace(4000.0, 7600.0, 120)
    galaxies = []
    for z in (0.05, 0.1):
        bm = gaussian_resolution_bands(jnp.asarray(wave), 3000.0, 3)
        galaxies.append(
            GalaxySpectrum(
                wave=wave,
                flux=np.full(wave.shape, 1.0e-12),
                ivar=np.full(wave.shape, 1.0e24),
                resolution=bm,
                z=z,
            )
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = _batched_template(ssp_wide, None, wave, case)
        per_node = _batched_template(ssp_wide, "per_node", wave, case)
    [(spec, batch, _)] = build_spectro_batches(galaxies, quantum=120, model=grouped)
    stacked = {
        k: jnp.broadcast_to(jnp.asarray(v, jnp.float64), (2,))
        for k, v in grouped.spec.sample(jax.random.PRNGKey(2)).items()
        if k != "redshift"
    }

    def run(model):
        ll = batched_log_likelihood(model, spec)
        return jax.jit(jax.vmap(jax.value_and_grad(ll)))(stacked, batch)

    (v0, g0), (v1, g1) = run(per_node), run(grouped)
    assert _max_rel(v1, v0) <= RTOL
    worst = _max_rel(v1, v0)
    for k in g0:
        scale = max(float(np.max(np.abs(np.asarray(g0[k])))), 1e-300)
        rel = float(np.max(np.abs(np.asarray(g1[k]) - np.asarray(g0[k])))) / scale
        worst = max(worst, rel)
        assert rel <= RTOL, k
    print(f"MAXREL batched/{name} {worst:.3e}")


def _batched_template(ssp, mode, wave, case):
    from tengri.observation.banded import gaussian_resolution_bands as _bands

    bm = _bands(jnp.asarray(wave), 3000.0, 3)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(
            spectroscopy=Spectroscopy(
                wave_obs=jnp.asarray(wave), resample="conserving", resolution_matrix=bm
            )
        ),
        sfh=_sfh(case["sfh"]),
        dust_attenuation=_dust_for(_dust(**case["dust"]), mode),
        dust_emission=_emission(case["emission"]),
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.2),
    )


# ── Lyman continuum branch: Cue with free fesc (loss parity with per_node) ───


def test_grouped_cue_free_fesc_lyc_branch_gradient(ssp_fsps, ssp_obs):
    """The LyC branch: the escape fraction is free, so the gradient flows through it."""
    obs = _observation(ssp_obs, spectroscopy=True)
    kw = dict(sfh="delayed", dust=_dust(width=0.3), neb=_neb("cue", lyc_fesc=True), emission=False)
    grouped = _build(ssp_fsps, obs, None, **kw)
    per_node = _build(ssp_fsps, obs, "per_node", **kw)
    p = _params(grouped, jax.random.PRNGKey(19))
    assert any("fesc" in k for k in p), "fesc is not a free parameter"
    _assert_agree(per_node, grouped, p, "joint", "cue_lyc_fesc/joint")


# ── float32 ──────────────────────────────────────────────────────────────────


def test_grouped_float32_gradient_is_finite_and_nonzero(ssp_fsps, ssp_obs):
    """Float32 mode: the grouped default keeps a finite, non-zero gradient."""
    obs = _observation(ssp_obs, spectroscopy=False)
    with jax.enable_x64(False):
        model = _build(ssp_fsps, obs, None, sfh="delayed", dust=_dust(width=0.3), mass=8.0)
        p = {
            k: jnp.asarray(v, jnp.float32)
            for k, v in model.spec.sample(jax.random.PRNGKey(4)).items()
        }
        obj = _objective("phot")
        value, grads = jax.value_and_grad(lambda q: obj(model, q))(p)
    assert np.isfinite(float(value)) and abs(float(value)) > 0.0, (
        "float32 photometric objective is non-finite or identically zero"
    )
    for k, g in grads.items():
        assert np.all(np.isfinite(np.asarray(g))), k
    assert any(np.any(np.asarray(g) != 0.0) for g in grads.values())

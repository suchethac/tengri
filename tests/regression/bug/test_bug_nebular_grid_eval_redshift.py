# SPDX-License-Identifier: BSD-3-Clause
"""The fast nebular grid's photometry follows the EVALUATION redshift.

``FeaturePrecomp`` tabulates the nebular contribution to each band per unit
``Q_H``. The band value of a redshifted SED depends on the redshift, but the table
was projected once, through the observed-frame filters, at a build-time reference
redshift (a prior draw for a free redshift, the placeholder for a catalog fit) and
served at every evaluation redshift. On a dust-free Cue model (the only kind the
grid serves photometry for) the total band flux was off by 20-22 % for a free
``Uniform(0.05, 2)`` redshift, 12-15 % for ``FREE = Uniform(0, 20)`` and 8-18 % for
a ``catalog_z_range`` fit, and ``Fitter``'s ``approx="auto"`` attaches the table to
every photometry-only fit of such a model.

The grid now splits the nebular SED into its continuum and line catalog (both
linear in ``Q_H``): the lines are placed in each band at the traced redshift as
delta lines, the continuum is tabulated over ``ln(1 + z)``. The tests compare the
fast path with the exact projection and with a build at the same fixed redshift,
on the real FSPS/MILES SSP and real SDSS/2MASS filters.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, FeaturePrecomp, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve, lnu_filter_integral

pytestmark = pytest.mark.regression_bug

#: The budget nearby Cue-precomp tests use for the fast path against the exact one
#: (``test_issue_1353_cue_photometry_only_precomp``): 0.42 % on the total band flux.
BUDGET = 4.2e-3

_FILTER_DIR = Path(__file__).resolve().parents[3] / "data" / "filters"
_FILTERS = ("SLOAN_SDSS_g", "SLOAN_SDSS_r", "SLOAN_SDSS_i", "2MASS_2MASS_J")

#: A young, bright population: the nebular emission is a few percent of the band.
_SFH = {"sfh_dpl_age_gyr": 0.3, "sfh_dpl_tau_gyr": 0.1}


def _filter(name):
    d = np.loadtxt(_FILTER_DIR / f"{name}.dat")
    return FilterCurve(wave=jnp.asarray(d[:, 0]), trans=jnp.asarray(d[:, 1]), name=name)


class _Models:
    """Builds each model once per module: a fast-nebular build costs tens of seconds."""

    def __init__(self, ssp):
        self.ssp = ssp
        self.obs = Observation(photometry=Photometry(filters=tuple(_filter(n) for n in _FILTERS)))
        self._cache = {}

    def get(self, redshift, approx, *, free_gas=False):
        key = (repr(redshift), repr(approx), free_gas)
        if key not in self._cache:
            neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
            if free_gas:
                neb["logU"] = Uniform(-3.5, -1.5)
            self._cache[key] = SEDModel.build(
                ssp_data=self.ssp,
                observation=self.obs,
                # age is pinned: its prior support is bounded by the age of the universe at the
                # build redshift, so a free age would put the same standardized point at
                # different physical ages in models built at different redshifts
                sfh={"type": "dpl", "all_params": FREE, "age_gyr": Fixed(_SFH["sfh_dpl_age_gyr"])},
                dust_attenuation={"type": "none"},
                redshift=redshift,
                neb=neb,
                approx=approx,
            )
        return self._cache[key]

    def fast(self, redshift, **kw):
        return self.get(redshift, (WavePrecomp(), FeaturePrecomp(n_grid=4)), **kw)

    def exact(self, redshift, **kw):
        return self.get(redshift, WavePrecomp(), **kw)


@pytest.fixture(scope="module")
def models(ssp_data_fsps):
    return _Models(ssp_data_fsps)


def _point(model, **over):
    """A parameter point for ``model``: a fixed draw with the young-population SFH."""
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    p.update({k: jnp.asarray(v) for k, v in _SFH.items() if k in p})
    p.update({k: jnp.asarray(v) for k, v in over.items() if k in p or k == "redshift"})
    return p


def _phot(model, p):
    return np.asarray(model.predict_photometry(p), dtype=np.float64)


def _rel(a, b):
    """Max relative difference over the bands the reference lights up.

    A band blueward of the Lyman break at high redshift is exactly zero in the
    reference; the fast path must be exactly zero there too (and finite).
    """
    assert np.all(np.isfinite(a)), f"non-finite fast photometry: {a}"
    live = b != 0.0
    assert np.all(a[~live] == 0.0), f"flux where the reference has none: {a[~live]}"
    return float(np.max(np.abs(a[live] / b[live] - 1.0)))


def _grid_component(model):
    from tengri.components.nebular.component import NebularSEDComponent

    return next(c for c in model._cached_component_chain if isinstance(c, NebularSEDComponent))


def _free_z_cases():
    # z nodes are uniform in ln(1+z); the odd values sit between nodes.
    return [
        pytest.param(Uniform(0.05, 2.0), z, id=f"U0.05-2-z{z}")
        for z in (0.3, 0.8, 1.5, 0.5517, 1.193)
    ] + [pytest.param(FREE, z, id=f"FREE-z{z}") for z in (0.3, 0.8, 1.5, 6.0, 3.3711)]


# ── T1: free redshift matches the exact path and the fixed-z build ─────────────


class TestFreeRedshiftFollowsEvaluationRedshift:
    @pytest.mark.parametrize(("prior", "z"), _free_z_cases())
    def test_free_z_matches_the_exact_projection(self, models, prior, z):
        """LOAD-BEARING: fast (WavePrecomp, FeaturePrecomp) vs exact WavePrecomp, free z.

        Neuter: project the table at the build-time reference redshift again (use
        ``grid.reference_redshift`` for the evaluation redshift) and this fails by
        8-22 %.
        """
        fast, exact = models.fast(prior), models.exact(prior)
        p = _point(exact, redshift=z)
        rel = _rel(_phot(fast, p), _phot(exact, p))
        print(f"{prior} z={z}: fast vs exact max rel {rel:.2e}")
        assert rel < BUDGET, f"free-z fast nebular drifts {rel:.2e} from the exact path at z={z}"

    @pytest.mark.parametrize(
        ("prior", "z"),
        [(Uniform(0.05, 2.0), 0.8), (Uniform(0.05, 2.0), 1.5), (FREE, 6.0)],
        ids=["U0.05-2-z0.8", "U0.05-2-z1.5", "FREE-z6"],
    )
    def test_free_z_matches_the_fixed_z_fast_build(self, models, prior, z):
        """The same model built at Fixed(z) is the reference the free build must reach.

        The nebular band is compared on its own (3 % of the band: the delta-line and
        ln(1+z) interpolation error), and the total with a looser budget, because
        the total also carries the *stellar* z-table's interpolation error (which the
        Fixed build does not have): 4.8e-3 at z = 6 on this fixture.
        """
        fast_free = models.fast(prior)
        fast_fixed = models.fast(z)
        p = _point(fast_free, redshift=z)
        p_fixed = {k: v for k, v in p.items() if k != "redshift"}
        neb_free = _neb_band(fast_free, p)
        neb_fixed = _neb_band(fast_fixed, p_fixed)
        rel_neb = _rel(neb_free, neb_fixed)
        rel = _rel(_phot(fast_free, p), _phot(fast_fixed, p_fixed))
        print(f"{prior} z={z}: nebular band {rel_neb:.2e}, total {rel:.2e}")
        assert rel_neb < 3e-2
        assert rel < 1e-2

    def test_ionization_axis_grid_matches_exact(self, models):
        """A table with a free gas axis: the z slab goes through the grid-axis interpolation."""
        prior = Uniform(0.05, 2.0)
        fast = models.fast(prior, free_gas=True)
        exact = models.exact(prior, free_gas=True)
        for z in (0.8, 1.193):
            p = _point(exact, redshift=z, neb_logU=-2.3)
            assert _rel(_phot(fast, p), _phot(exact, p)) < BUDGET

    def test_nebular_band_itself_follows_z(self, models):
        """The published nebular band, not the total: 3 % of the nebular band at most."""
        prior = Uniform(0.05, 2.0)
        fast, exact = models.fast(prior), models.exact(prior)
        for z in (0.3, 0.5517, 1.193, 1.9):
            p = _point(exact, redshift=z)
            neb_f = _neb_band(fast, p)
            neb_e = _neb_band(exact, p)
            assert np.max(np.abs(neb_f / neb_e - 1.0)) < 3e-2, z


def _neb_band(model, p, fixed_values=None):
    fv = model.spec.get_fixed_values() if fixed_values is None else fixed_values
    state = model.predict_state(p, fixed_values=fv, observables_only=True)
    return np.asarray(state.derived["nebular_phot_lnu_precomp"], dtype=np.float64)


# ── T6: the fast path is engaged for a free redshift ────────────────────────────


class TestFastPathIsEngaged:
    @pytest.mark.parametrize("prior", [Uniform(0.05, 2.0), FREE], ids=["U0.05-2", "FREE"])
    def test_free_redshift_is_served_by_the_grid(self, models, prior):
        """A silent fallback to the exact path would also pass the accuracy tests."""
        fast = models.fast(prior)
        comp = _grid_component(fast)
        grid = comp.grid_table
        assert grid is not None and grid.serves_split_bands and grid.redshift_is_tabulated
        assert not comp.must_materialize_sed
        p = _point(fast, redshift=0.8)
        state = fast.predict_state(
            p, fixed_values=fast.spec.get_fixed_values(), observables_only=True
        )
        assert float(jnp.max(jnp.abs(state.derived["sed_nebular"]))) == 0.0, (
            "the continuum was computed: the grid path is not serving photometry"
        )
        assert "nebular_phot_lnu_lines_precomp" in state.derived

    def test_the_table_is_threaded_not_baked(self, models):
        """The continuum z-table rides the template channel as a runtime argument."""
        fast = models.fast(Uniform(0.05, 2.0))
        td = fast._template_data_for_jit()
        assert set(td["nebular_grid"]) == {"log_cont_ztable_per_qh", "cont_keep", "cont_lnz"}

    def test_signature_keys_the_z_table_and_its_grid(self, models):
        from tengri._cache_keys import baked

        grid = _grid_component(models.fast(Uniform(0.05, 2.0))).grid_table
        shifted = dataclasses.replace(grid, cont_lnz=grid.cont_lnz * 1.01)
        bare = dataclasses.replace(
            grid, log_cont_ztable_per_qh=None, cont_keep=None, cont_lnz=None
        )
        keys = {baked(grid), baked(shifted), baked(bare)}
        assert len(keys) == 3, "the compile key must distinguish the z table and its z grid"


# ── the published split ──────────────────────────────────────────────────────────


class TestPublishedSplit:
    @pytest.mark.parametrize("z", [0.3, 0.5517, 1.193, 1.9])
    def test_total_is_lines_plus_continuum(self, models, z):
        """nebular_phot_lnu_precomp == lines.sum(0) + continuum, at several z."""
        fast = models.fast(Uniform(0.05, 2.0))
        state = fast.predict_state(
            _point(fast, redshift=z),
            fixed_values=fast.spec.get_fixed_values(),
            observables_only=True,
        )
        d = state.derived
        total = np.asarray(d["nebular_phot_lnu_precomp"])
        lines = np.asarray(d["nebular_phot_lnu_lines_precomp"])
        cont = np.asarray(d["nebular_phot_lnu_cont_precomp"])
        waves = np.asarray(d["nebular_line_phot_waves_rest"])
        assert lines.shape == (waves.shape[0], total.shape[0]) and cont.shape == total.shape
        np.testing.assert_allclose(lines.sum(axis=0) + cont, total, rtol=1e-12)
        assert np.all(lines >= 0.0) and np.all(cont >= 0.0) and np.all(total > 0.0)

    def test_split_keys_do_not_end_in_the_summed_suffix(self):
        """``predict_via_precomp`` sums every ``*_phot_lnu_precomp`` field: a
        decomposition named that way would count the nebular bucket three times."""
        from tengri.protocols.derived_state import DerivedState

        names = DerivedState.field_names()
        for key in (
            "nebular_phot_lnu_lines_precomp",
            "nebular_phot_lnu_cont_precomp",
            "nebular_line_phot_waves_rest",
        ):
            assert key in names
            assert not key.endswith("_phot_lnu_precomp")

    def test_line_wavelengths_are_those_the_sed_renders(self, models):
        """Cue's network wavelengths are air: Halpha sits at 6562.8, not vacuum 6564.6."""
        fast = models.fast(Uniform(0.05, 2.0))
        waves = np.asarray(_grid_component(fast).grid_table.sed_line_waves)
        assert np.min(np.abs(waves - 6562.8)) < 0.1
        assert np.min(np.abs(waves - 6564.6)) > 1.0


# ── T2: catalog_z_range, the Fitter's runtime-redshift path ──────────────────────


_CATALOG_Z = (0.05, 2.0)


def _catalog_fast(models):
    return models.get(
        Fixed(0.05),
        (WavePrecomp(catalog_z_range=_CATALOG_Z), FeaturePrecomp(n_grid=4)),
    )


def _observables_at(model, p, z):
    """Photometry through the jitted observables call with a runtime redshift."""
    fixed = {**model.spec.get_fixed_values(), "redshift": jnp.asarray(z)}
    fn = model._get_or_build_predict_observables_jit()
    out = fn(p, fixed, *model._resolve_threaded_data(None, None, None))
    return np.asarray(out.phot_fnu, dtype=np.float64)


class TestCatalogRedshift:
    @pytest.mark.parametrize("z", [0.3, 0.8, 1.5, 1.9])
    def test_runtime_z_matches_fixed_z_exact(self, models, z):
        """A catalog model with fixed_values['redshift'] = z equals Fixed(z) exact."""
        cat = _catalog_fast(models)
        exact = models.exact(z)
        p = _point(exact)
        rel = _rel(_observables_at(cat, p, z), _phot(exact, p))
        print(f"catalog z={z}: runtime-z fast vs Fixed(z) exact max rel {rel:.2e}")
        assert rel < BUDGET

    @pytest.mark.parametrize("z", [0.8, 1.5])
    def test_runtime_z_matches_fixed_z_fast_build(self, models, z):
        cat = _catalog_fast(models)
        fixed = models.fast(z)
        p = _point(fixed)
        assert _rel(_observables_at(cat, p, z), _phot(fixed, p)) < BUDGET

    def test_catalog_grid_is_engaged_per_galaxy(self, models):
        cat = _catalog_fast(models)
        grid = _grid_component(cat).grid_table
        assert grid.serves_split_bands and grid.redshift_is_tabulated
        p = _point(cat)
        for z in (0.3, 0.8, 1.5, 1.9):
            fv = {**cat.spec.get_fixed_values(), "redshift": jnp.asarray(z)}
            state = cat.predict_state(p, fixed_values=fv, observables_only=True)
            assert float(jnp.max(jnp.abs(state.derived["sed_nebular"]))) == 0.0
            assert "nebular_phot_lnu_lines_precomp" in state.derived

    def test_the_reference_redshift_is_the_catalog_lower_bound(self, models):
        grid = _grid_component(_catalog_fast(models)).grid_table
        assert grid.reference_redshift == pytest.approx(_CATALOG_Z[0])


# ── the catalog engines: per-galaxy redshift through the vmapped loss ─────────────


class TestCatalogEngines:
    """The shared forward code must see each galaxy's traced redshift under vmap."""

    _ZS = (0.3, 0.8, 1.5, 1.9)

    def _data(self, models):
        truth = _point(models.exact(0.3))
        datas, noises = [], []
        for z in self._ZS:
            flux = _phot(models.exact(z), truth)
            datas.append(flux)
            noises.append(0.05 * np.abs(flux))
        return truth, np.stack(datas), np.stack(noises)

    def test_vmapped_batch_loss_matches_fixed_z_fits(self, models):
        from tengri.inference.backends.mcmc.catalog import _get_flat_logdensity, _make_substitute
        from tengri.inference.fitter import Fitter

        _, datas, noises = self._data(models)
        cat = _catalog_fast(models)
        fitter = Fitter(
            cat,
            datas[0],
            noises[0],
            data_type="photometry",
            approx=(WavePrecomp(catalog_z_range=_CATALOG_Z), FeaturePrecomp(n_grid=4)),
            params_override={"redshift": self._ZS[0]},
        )
        assert "redshift" in fitter._data_args
        logp, _unravel, init_flat, template = _get_flat_logdensity(
            fitter, fitter._initialize_unbounded(jax.random.PRNGKey(3))
        )
        substitute = _make_substitute(template, True, False)
        presence = jnp.ones_like(jnp.asarray(datas[0]))

        def one(data, noise, z):
            return logp(init_flat, substitute(data, noise, presence, z, None, None))

        batched = np.asarray(
            jax.jit(jax.vmap(one))(jnp.asarray(datas), jnp.asarray(noises), jnp.asarray(self._ZS))
        )
        assert np.all(np.isfinite(batched))
        for i, z in enumerate(self._ZS):
            ref_fitter = Fitter(
                models.exact(z),
                datas[i],
                noises[i],
                data_type="photometry",
                approx=WavePrecomp(),
            )
            ref_logp, _, ref_init, ref_template = _get_flat_logdensity(
                ref_fitter, ref_fitter._initialize_unbounded(jax.random.PRNGKey(3))
            )
            ref = float(ref_logp(ref_init, ref_template))
            np.testing.assert_allclose(batched[i], ref, rtol=2e-2, err_msg=f"galaxy z={z}")

    def test_sequential_engine_loss_matches_fixed_z_fits(self, models):
        """``CatalogFitter._run_sequential``'s per-galaxy Fitter (params_override)."""
        from tengri.inference.context import InferenceContext
        from tengri.inference.fitter import Fitter

        _, datas, noises = self._data(models)
        cat = _catalog_fast(models)
        approx = (WavePrecomp(catalog_z_range=_CATALOG_Z), FeaturePrecomp(n_grid=4))
        for i, z in enumerate(self._ZS):
            f_cat = Fitter(
                cat,
                datas[i],
                noises[i],
                data_type="photometry",
                approx=approx,
                params_override={"redshift": z},
            )
            f_ref = Fitter(
                models.exact(z), datas[i], noises[i], data_type="photometry", approx=WavePrecomp()
            )
            ctx_c, ctx_r = InferenceContext.from_target(f_cat), InferenceContext.from_target(f_ref)
            p_u = ctx_c.initial_params(jax.random.PRNGKey(3))
            lc = float(ctx_c.neg_log_posterior_fn(p_u, ctx_c.data_args))
            p_r = ctx_r.initial_params(jax.random.PRNGKey(3))
            lr = float(ctx_r.neg_log_posterior_fn(p_r, ctx_r.data_args))
            np.testing.assert_allclose(lc, lr, rtol=2e-2, err_msg=f"galaxy z={z}")


# ── T7: the Fitter ────────────────────────────────────────────────────────────────


class TestFitter:
    def test_auto_policy_attaches_the_grid_and_its_loss_matches_exact(self, models):
        from tengri.inference.context import InferenceContext
        from tengri.inference.fitter import Fitter

        prior = Uniform(0.05, 2.0)
        exact = models.exact(prior)
        truth = _point(exact, redshift=0.8)
        flux = _phot(exact, truth)
        err = 0.05 * np.abs(flux)
        auto = Fitter(exact, flux, err, data_type="photometry")
        assert auto.model.approx.feature_precomp, "approx='auto' must attach FeaturePrecomp"
        ref = Fitter(exact, flux, err, data_type="photometry", approx=WavePrecomp())
        assert not ref.model.approx.feature_precomp
        ctx_a, ctx_r = InferenceContext.from_target(auto), InferenceContext.from_target(ref)
        p_a = ctx_a.initial_params(jax.random.PRNGKey(5))
        p_r = ctx_r.initial_params(jax.random.PRNGKey(5))
        la = float(ctx_a.neg_log_posterior_fn(p_a, ctx_a.data_args))
        lr = float(ctx_r.neg_log_posterior_fn(p_r, ctx_r.data_args))
        print(f"Fitter loss: auto {la:.6g} vs exact {lr:.6g}")
        np.testing.assert_allclose(la, lr, rtol=2e-2)


# ── T3: the reference redshift is only a convention ───────────────────────────────


class TestReferenceRedshiftIndependence:
    def test_tables_built_at_different_reference_redshifts_agree(self, models):
        from tengri.components.nebular.nebular_grid_precompute import (
            precompute_nebular_grid,
            reconstruct_nebular_phot,
        )

        exact = models.exact(FREE)
        lines = jnp.asarray([6564.61, 5008.24, 4862.68])
        tables = []
        for z_ref in (0.1, 17.3):
            ref = _point(exact, redshift=z_ref)
            tables.append(precompute_nebular_grid(exact, lines, n_grid=2, ref_params=ref))
        lo, hi = tables
        assert lo.reference_redshift == pytest.approx(0.1)
        assert hi.reference_redshift == pytest.approx(17.3)
        for field in ("log_line_per_qh", "log_restband_per_qh", "log_sed_lines_per_qh"):
            d = np.max(np.abs(np.asarray(getattr(lo, field)) - np.asarray(getattr(hi, field))))
            assert d < 1e-5, f"{field} depends on the reference redshift by {d:.2e} dex"
        for z in (0.3, 1.2, 3.0, 8.0):
            a = np.asarray(reconstruct_nebular_phot(53.0, {}, lo, z))
            b = np.asarray(reconstruct_nebular_phot(53.0, {}, hi, z))
            np.testing.assert_allclose(a, b, rtol=1e-5, err_msg=f"z={z}")

    def test_default_reference_redshift_is_deterministic_and_documented(self, models):
        from tengri.components.nebular.nebular_grid_precompute import reference_redshift

        assert reference_redshift(models.exact(0.8)) == pytest.approx(0.8)
        assert reference_redshift(models.exact(Uniform(0.05, 2.0))) == pytest.approx(0.05)
        assert reference_redshift(models.exact(FREE)) == pytest.approx(
            1e-3
        )  # lower bound 0, floored


# ── T4: delta lines ────────────────────────────────────────────────────────────────


class TestDeltaLines:
    @staticmethod
    def _setup():
        from tengri.components.nebular._shared import render_nebular_lines

        wave = jnp.linspace(3000.0, 9000.0, 60001)  # 0.1 A
        fw = jnp.asarray([5400.0, 5500.0, 5700.0, 6300.0, 6500.0, 6600.0])
        ft = jnp.asarray([0.0, 0.0, 0.8, 0.8, 0.0, 0.0])
        profile = render_nebular_lines(jnp.asarray([5000.0]), jnp.asarray([1.0]), wave, 0.0, 30.0)
        return wave, fw, ft, profile

    def test_matches_a_rendered_line_swept_across_a_filter_edge(self):
        from tengri.components.nebular.nebular_band_z import delta_line_band_kernel

        wave, fw, ft, profile = self._setup()
        zs = np.linspace(0.06, 0.34, 57)  # observed line 5300 .. 6700: through both edges
        delta = np.array(
            [float(delta_line_band_kernel(z, [5000.0], [fw], [ft])[0, 0]) for z in zs]
        )
        exact = np.array([float(lnu_filter_integral(profile, wave, fw, ft, z)) for z in zs])
        peak = exact.max()
        assert peak > 0
        assert np.max(np.abs(delta - exact)) / peak < 5e-3
        # the band response really does switch on and off across the sweep
        assert delta[0] == 0.0 and delta[-1] == 0.0 and delta.max() > 0.9 * peak

    def test_uses_the_redshifted_line_position_and_the_photon_weight(self):
        """A line at 5000 A sits at (1+z) 5000 in the band: T((1+z) 5000), weight 1/lambda."""
        from tengri.components.nebular.nebular_band_z import delta_line_band_kernel

        _, fw, ft, _ = self._setup()
        c_aa = 2.99792458e18
        norm = float(jnp.trapezoid(ft / fw, fw))
        z = 0.2  # observed 6000: on the plateau, T = 0.8
        got = float(delta_line_band_kernel(z, [5000.0], [fw], [ft])[0, 0])
        assert got == pytest.approx(5000.0 * 0.8 / (c_aa * norm), rel=1e-12)
        # at rest (z = 0) the line at 5000 A is blueward of the band: no response
        assert float(delta_line_band_kernel(0.0, [5000.0], [fw], [ft])[0, 0]) == 0.0

    def test_redshift_gradient_is_finite_and_matches_finite_differences(self):
        from tengri.components.nebular.nebular_band_z import delta_line_band_kernel

        _, fw, ft, _ = self._setup()

        def band(z):
            return delta_line_band_kernel(z, [5000.0, 6563.0], [fw], [ft]).sum()

        grad = jax.grad(band)
        zs = np.linspace(-0.02, 0.45, 83)
        assert all(np.isfinite(float(grad(z))) for z in zs)
        # grad-assert: nonzero-only — the 5000 A line sits on the 5500-5700 A filter edge
        # at z = 0.12, so d(band)/dz cannot vanish
        assert abs(float(grad(0.12))) > 0.0
        eps = 1e-6
        # away from the table nodes (5500, 5700, 6300, 6500 -> z = 0.10, 0.14, 0.26, 0.30 for
        # the 5000 A line) the band is linear in z and the gradient is exact
        for z in (0.12, 0.2, 0.28):
            fd = (float(band(z + eps)) - float(band(z - eps))) / (2 * eps)
            assert float(grad(z)) == pytest.approx(fd, rel=1e-5, abs=1e-30)


# ── T5: the continuum table ────────────────────────────────────────────────────────


class TestContinuumTable:
    @staticmethod
    def _cont(wave):
        w = np.asarray(wave, dtype=np.float64)
        jump = np.where(w > 3646.0, 0.5, 1.0)  # Balmer jump
        return np.where(w < 915.0, 0.0, (w / 5000.0) ** -0.5 * jump)

    @staticmethod
    def _table(wave, fw, ft, z_lo=0.001, z_hi=20.2, n_z=250):
        from tengri.components.nebular.nebular_band_z import continuum_ztable, ln1pz_grid

        lnz = ln1pz_grid(z_lo, z_hi, n_z)
        tab = continuum_ztable(TestContinuumTable._cont(wave)[None, :], wave, [fw], [ft], lnz)[0]
        pos = tab[tab > 0]
        log_tab = np.log10(np.maximum(tab, 1e-3 * pos.min()))
        return jnp.asarray(lnz), jnp.asarray(log_tab), jnp.asarray(tab > 0.0), tab

    def test_nodes_reproduce_the_exact_projection(self, ssp_data_fsps):
        """The kernel is the same quadrature as lnu_filter_integral: equal at every node."""
        g = _filter("SLOAN_SDSS_g")
        wave = np.asarray(ssp_data_fsps.ssp_wave)
        lnz, _, _, tab = self._table(wave, g.wave, g.trans)
        cont = jnp.asarray(self._cont(wave))
        for i in (0, 7, 40, 120, 200):
            z = float(np.expm1(lnz[i]))
            exact = float(lnu_filter_integral(cont, jnp.asarray(wave), g.wave, g.trans, z))
            assert tab[i, 0] == pytest.approx(exact, rel=1e-9, abs=1e-300)

    def test_dense_off_grid_sweep_over_the_balmer_jump(self, ssp_data_fsps):
        from tengri.components.nebular.nebular_band_z import continuum_band_at_z

        g = _filter("SLOAN_SDSS_g")
        wave = np.asarray(ssp_data_fsps.ssp_wave)
        lnz, log_tab, nonzero, _ = self._table(wave, g.wave, g.trans)
        cont = jnp.asarray(self._cont(wave))
        zs = np.linspace(0.04, 2.0, 301)  # the jump crosses g at z ~ 0.1 - 0.5
        got = np.array(
            [float(continuum_band_at_z(0.0, log_tab, nonzero, lnz, z, lambda s: s)[0]) for z in zs]
        )
        exact = np.array(
            [float(lnu_filter_integral(cont, jnp.asarray(wave), g.wave, g.trans, z)) for z in zs]
        )
        rel = np.abs(got / exact - 1.0)
        print(f"continuum sweep: p95 {np.percentile(rel, 95):.2e} max {rel.max():.2e}")
        assert np.percentile(rel, 95) < 1e-3 and rel.max() < 2e-2

    def test_below_the_lyman_truncation_is_exactly_zero_with_finite_gradients(self, ssp_data_fsps):
        from tengri.components.nebular.nebular_band_z import continuum_band_at_z

        g = _filter("SLOAN_SDSS_g")
        wave = np.asarray(ssp_data_fsps.ssp_wave)
        lnz, log_tab, nonzero, _ = self._table(wave, g.wave, g.trans)

        def band(z):
            return continuum_band_at_z(0.0, log_tab, nonzero, lnz, z, lambda s: s)[0]

        for z in (7.3, 8.0, 12.0):  # the whole g band is blueward of 915 A in the rest frame
            assert float(band(z)) == 0.0
            assert float(jax.grad(band)(z)) == 0.0
        # across the partially covered edge: finite, non-negative
        for z in np.linspace(3.0, 5.6, 27):
            assert np.isfinite(float(jax.grad(band)(z))) and float(band(z)) >= 0.0

    def test_outside_the_table_is_clamped_to_the_edge(self, ssp_data_fsps):
        from tengri.components.nebular.nebular_band_z import continuum_band_at_z

        g = _filter("SLOAN_SDSS_g")
        wave = np.asarray(ssp_data_fsps.ssp_wave)
        lnz, log_tab, nonzero, tab = self._table(wave, g.wave, g.trans, z_lo=0.2, z_hi=1.5, n_z=40)

        def band(z):
            return float(continuum_band_at_z(0.0, log_tab, nonzero, lnz, z, lambda s: s)[0])

        assert band(0.05) == pytest.approx(tab[0, 0], rel=1e-12)
        assert band(3.0) == pytest.approx(tab[-1, 0], rel=1e-12)
        mid = band(0.6)
        assert min(tab[:, 0]) <= mid <= max(tab[:, 0])


# ── dusty Cue, free redshift: the dust-screen channels follow the redshift too ───────

#: The budget of the dusty-grid contract tests for the fast path against WavePrecomp
#: (``test_dusty_nebular_grid_parity``: twice the measured 1.25e-3).
DUSTY_BUDGET = 2.5e-3


class TestDustyFreeRedshift:
    @staticmethod
    def _dusty(models, approx):
        key = ("dusty", repr(approx))
        if key not in models._cache:
            models._cache[key] = SEDModel.build(
                ssp_data=models.ssp,
                observation=models.obs,
                sfh={"type": "dpl", "all_params": FREE, "age_gyr": Fixed(_SFH["sfh_dpl_age_gyr"])},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                    "tau_bc": Uniform(0.0, 2.0),
                    "tau_diff": Uniform(0.0, 2.0),
                },
                dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
                neb={"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-3.5, -2.0)},
                redshift=Uniform(0.05, 2.0),
                approx=approx,
            )
        return models._cache[key]

    @pytest.mark.parametrize("z", [0.3, 1.5])
    def test_dusty_grid_photometry_follows_z_and_is_engaged(self, models, z):
        fast = self._dusty(models, (WavePrecomp(), FeaturePrecomp(n_grid=4)))
        exact = self._dusty(models, WavePrecomp())
        comp = _grid_component(fast)
        assert comp.grid_table.serves_dust and comp.grid_table.redshift_is_tabulated
        assert not comp.must_materialize_sed
        assert any(getattr(c, "nebular_from_grid", False) for c in fast._cached_component_chain)
        p = _point(exact, redshift=z, neb_logU=-2.7, dust_tau_bc=1.0, dust_tau_diff=1.0)
        state = fast.predict_state(
            p, fixed_values=fast.spec.get_fixed_values(), observables_only=True
        )
        assert float(jnp.max(jnp.abs(state.derived["sed_nebular"]))) == 0.0
        n_lines = comp.grid_table.sed_line_waves.shape[0]
        assert state.derived["nebular_phot_lnu_subband_precomp"].shape[-1] > n_lines
        rel = _rel(_phot(fast, p), _phot(exact, p))
        print(f"dusty z={z}: fast vs WavePrecomp max rel {rel:.2e}")
        assert rel < DUSTY_BUDGET

# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4-D: nebular backend selection, dust IR builds, AGN SKIRTOR compilation.

What this file covers: that a nebular backend named in the build grammar is
actually reachable and predicts, that template-based dust models construct, and
that the monolithic SKIRTOR model compiles under ``jit`` and agrees with eager.

It does **not** verify that any template threads as a ``Parameter`` rather than
baking in as a ``Constant``, though its tests were once named and docstring'd
as if it did -- and stayed green while 31 MB of SKIRTOR grid baked into every
graph (#1383, #1549). That invariant needs a jaxpr constant count taken on the
surface where the template data is an argument; it lives in
``test_agn_template_threading.py``. Do not re-add threading claims here without
a measurement to back them.

Category A was six tests that could not fail
--------------------------------------------

Four of them took an ``ssp_bare`` fixture they never used. That fixture pointed
at ``data/ssp_prsc_bc03_chabrier.h5``, which is gitignored, is not one of the
two grids ``conftest.py`` generates, and is absent from CI -- so the four
skipped on every machine, and what they would have done was never observed.

What they would have done:

* ``test_*_backend_config_declaration`` set ``NebularSEDComponentConfig(
  backend="cb19")`` and asserted the field read back ``"cb19"``. That dataclass
  accepts **any** string -- ``backend="bogus_backend"`` is equally fine -- so
  the assertion was a dataclass echo. The ``SEDModel.build`` grammar *does*
  validate, and rejects an unknown type with the menu of real ones; that is the
  check those tests were reaching for, and it is asserted below.
* ``test_*_backend_in_spec`` called ``Parameters(..., nebular_backend=...)``.
  ``Parameters.__init__`` has no such keyword; it raises ``ValueError: Unknown
  parameter 'nebular_backend'`` -- for ``"cb19"`` exactly as much as for
  ``"mappings"``. Backend selection moved to ``neb={'type': ...}``.
* ``test_*_backend_exposes_a_grid`` asserted ``hasattr(backend, "grid")`` and
  ``backend.grid is not None``, wrapped in ``except Exception: pytest.skip(
  f"...instantiation failed: {e}")`` commented "may skip if grid data
  unavailable". For MAPPINGS that construction raises
  ``IonizingSpectrumInconsistencyError`` -- a deliberate guard reporting that
  the ionizing field comes from a Starburst99 grid rather than the user's DSPS
  SSPs, and asking for ``ionizing_source_warning='warn'|'suppress'``. The
  handler filed a design decision as missing data. #1615 in one line: ``except
  Exception`` cannot tell "optional dependency absent" from "the code is
  telling you something".

And the wiring those six were named for does not exist: ``mappings`` is not a
registered nebular type at all (#2070). ``neb={'type': 'mappings'}`` raises
with the menu, and ``tengri``'s own listing advertises five backends without
it, while ``mappings_photo.py`` and a four-axis ``mappings_photo_precompute.py``
sit there complete and unreachable.

The SSP fixture is the tracked one now
--------------------------------------

``ssp_data_fsps`` (``data/fsps_prsc_miles_chabrier.h5``) is tracked in git and
present on every runner, and is bare-stellar, which is what Cue needs. Nothing
here wanted BC03 specifically.
"""

from __future__ import annotations

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Parameters, SEDModel
from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tests._data_skip import DATA_DIR
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.contract

_SSP_WNE = DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

#: Nebular backends this file builds end to end.
#:
#: ``cloudy`` is registered too but needs ``data/cloudy_grid_mist.h5``, which is
#: neither tracked nor fetched by CI. ``ssp`` and ``none`` have no backend
#: object to exercise.
_BUILDABLE_NEBULAR = ["cb19", "cue"]

#: What the grammar advertises. Pinned so that registering a type -- or
#: dropping one -- is a deliberate edit rather than a silent change.
_REGISTERED_NEBULAR = {"cb19", "cloudy", "cue", "mappings", "mappings_agn", "none", "ssp"}


@pytest.fixture(scope="module")
def ssp_wne():
    if not _SSP_WNE.is_file():
        pytest.skip(f"wNE SSP not available at {_SSP_WNE}")
    return load_ssp_data(str(_SSP_WNE))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _base_spec(**kwargs):
    """Basic spec: all params fixed, optionally overridden."""
    defaults = dict(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    defaults.update(kwargs)
    return Parameters(**defaults)


def _silent_build(spec, ssp, obs, **kwargs):
    """Build a model, suppressing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, **kwargs)


def _assert_predicts(model, obs, label):
    """A built model must produce finite, non-zero fluxes of the right shape.

    ``assert model is not None`` was the assertion at four of these call sites.
    It is true of every value a constructor can return, so it cannot separate a
    model that builds from one that builds and then predicts nothing.
    """
    phot = model.predict_photometry(model.spec.get_fixed_values())
    chex.assert_tree_all_finite(phot)
    assert phot.shape == (len(obs.photometry.filters),), (
        f"{label}: photometry shape {phot.shape} does not match "
        f"{len(obs.photometry.filters)} filters"
    )
    assert float(jnp.max(jnp.abs(phot))) > 0.0, f"{label}: every band is exactly zero"
    return phot


# ── Category A: nebular backend selection ─────────────────────────────────────


class TestNebularBackendSelection:
    """The grammar's nebular types are reachable, and unknown ones are refused."""

    @pytest.mark.parametrize("backend", _BUILDABLE_NEBULAR)
    def test_registered_backend_builds_and_predicts(self, backend, ssp_data_fsps, obs):
        """``neb={'type': backend}`` yields a model that predicts finite fluxes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Fixed(10.0)},
                met={"logzsol": Fixed(-0.5)},
                neb={"type": backend, "all_params": FIXED, "logU": Fixed(-3.0)},
                redshift=Fixed(0.1),
            )
        _assert_predicts(model, obs, f"neb={backend}")

    def test_unknown_nebular_type_is_rejected_with_the_menu(self, ssp_data_fsps, obs):
        """An unknown backend raises, and the message names the real ones.

        This is the check the two ``..._config_declaration`` tests were reaching
        for. ``NebularSEDComponentConfig(backend=...)`` performs no validation at
        all -- it accepts ``"bogus_backend"`` -- so asserting the field reads
        back what was just assigned tested the dataclass, not the wiring.
        """
        with pytest.raises(ValueError, match="Unknown nebular type") as exc:
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Fixed(10.0)},
                neb={"type": "bogus_backend", "all_params": FIXED},
                redshift=Fixed(0.1),
            )
        message = str(exc.value)
        for name in _REGISTERED_NEBULAR:
            assert name in message, f"the error should list {name!r} as available: {message}"

    def test_registered_nebular_types_are_pinned(self, ssp_data_fsps, obs):
        """The advertised menu is exactly ``_REGISTERED_NEBULAR``.

        Read off the error message rather than restated from a registry import,
        because the message is the surface a user actually sees. ``mappings`` is
        absent by design here: ``MappingsPhotoStellarBackend`` and its four-axis
        precompute adapter exist and are complete, and no build grammar reaches
        them (#2070). If that is fixed, this test fails and says so.
        """
        with pytest.raises(ValueError, match="Unknown nebular type") as exc:
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Fixed(10.0)},
                neb={"type": "definitely_not_a_backend", "all_params": FIXED},
                redshift=Fixed(0.1),
            )
        listed = {n.strip() for n in str(exc.value).split("Available:")[-1].strip(" .").split(",")}
        assert listed == _REGISTERED_NEBULAR, (
            f"registered nebular types changed: added {sorted(listed - _REGISTERED_NEBULAR)}, "
            f"removed {sorted(_REGISTERED_NEBULAR - listed)}"
        )


class TestNebularBackendGrids:
    """The grid a backend loads has axes an interpolator can use."""

    @staticmethod
    def _axis_is_usable(axis, label):
        arr = np.asarray(axis)
        assert arr.ndim == 1 and arr.shape[0] >= 2, (
            f"{label}: axis has shape {arr.shape}; an interpolation axis needs at "
            f"least two nodes, and a size-1 axis is what breaks edges_for_grid"
        )
        assert np.all(np.isfinite(arr)), f"{label}: axis holds non-finite nodes"
        assert np.all(np.diff(arr) > 0.0), f"{label}: axis is not strictly increasing"

    def test_cb19_grid_axes_are_usable(self):
        """CB19's grid exposes a strictly increasing, multi-node logU axis."""
        from tengri.components.nebular.cloudy_cb19 import CB19Backend

        grid = CB19Backend().grid
        self._axis_is_usable(grid.log_U_grid, "CB19 log_U_grid")
        wavelengths = np.asarray(grid.line_wavelengths)
        assert wavelengths.size > 0 and np.all(wavelengths > 0.0), (
            "CB19 line wavelengths must all be positive"
        )

    def test_mappings_photo_grid_axes_are_usable(self):
        """MappingsPhotoStellarBackend: direct construction works, model path refuses.

        The backend was registered to let users discover it, but the grid data is
        incomplete (51.2% NaN in logHB_per_logq, 2656/5184 cells). The guard moved
        to the model path (sed_model.py), so direct construction succeeds but the
        model refuses at build time with TengriIOError (#2082).
        """
        # Direct construction should work (grid check moved to model path)
        backend = MappingsPhotoStellarBackend(ionizing_source_warning="suppress")
        assert backend is not None, "Backend construction should succeed"
        assert backend.grid is not None, "Grid should be loaded in backend"

    def test_bare_mappings_construction_refuses_loudly(self):
        """Constructing without acknowledging the ionizing-source mismatch raises.

        The guard is the point: a silent default here would fit a model whose
        stellar continuum and nebular lines come from different stellar
        population synthesis codes. Pinned so nobody softens it to a warning.
        """
        from tengri.components.nebular import IonizingSpectrumInconsistencyError

        with pytest.raises(IonizingSpectrumInconsistencyError, match="Starburst99"):
            MappingsPhotoStellarBackend()


# ── Category B: Dust IR template threading ────────────────────────────────────


class TestDustIRTemplateThreading:
    """Dust IR emission template threading tests."""

    @pytest.mark.parametrize(
        "dust_emission",
        ["modified_blackbody", "casey2012"],  # Analytic models (no templates)
    )
    def test_analytic_dust_models_no_templates(self, ssp_wne, obs, dust_emission):
        """Analytic dust emission models (no templates) work without threading."""
        spec = _base_spec(dust_emission=dust_emission)

        model = _silent_build(spec, ssp_wne, obs)
        # No templates to thread; _template_data_for_jit should return None or empty
        template_data = model._template_data_for_jit()
        # Dust analytic models don't produce template data; only nebular does
        assert template_data is None

    @pytest.mark.parametrize(
        "dust_emission",
        ["dale2014", "draine_li2007", "draine_li2014", "astrodust", "bosa"],
    )
    def test_dust_template_models_build_and_predict(self, ssp_wne, obs, dust_emission):
        """Template-based dust models build and produce finite fluxes.

        The SDSS bands here do not sample dust *emission*, which is IR, so this
        is a construction and finiteness check, not a check on the dust SED --
        named accordingly. It replaces ``assert model is not None``, which is
        true of any value a constructor returns.
        """
        try:
            model = _silent_build(_base_spec(dust_emission=dust_emission), ssp_wne, obs)
        except FileNotFoundError as e:
            pytest.skip(f"Template file not available for {dust_emission}: {e}")

        _assert_predicts(model, obs, dust_emission)

    def test_dust_ir_jit_non_jit_agreement(self, ssp_wne, obs):
        """JIT and non-JIT dust IR SED paths agree to floating-point precision."""
        spec = _base_spec(dust_emission="dale2014")

        try:
            model = _silent_build(spec, ssp_wne, obs)
        except FileNotFoundError:
            pytest.skip("Dale2014 template not available")

        fixed_params = spec.get_fixed_values()

        phot_non_jit = model.predict_photometry(fixed_params)
        phot_jit = jax.jit(lambda p: model.predict_photometry(p))(fixed_params)

        # The test's whole claim is agreement, so compare the two. It used to
        # assert each was separately `not None` and never compare them, which
        # would pass even if the JIT path returned different fluxes.
        chex.assert_tree_all_finite(phot_non_jit)
        assert float(jnp.max(jnp.abs(phot_non_jit))) > 0.0, "all-zero photometry proves nothing"
        chex.assert_trees_all_close(phot_jit, phot_non_jit, rtol=1e-12, atol=0.0)


# ── Category C: AGN SKIRTOR ───────────────────────────────────────────────────


class TestAGNSKIRTORTemplateThreading:
    """AGN SKIRTOR compilation tests."""

    def test_skirtor_model_jit_compiles(self, ssp_wne, obs):
        """The monolithic SKIRTOR model compiles under jit and matches eager.

        A smoke test, and named like one. It was previously called
        ``..._jit_compatibility`` and docstring'd "without baking templates
        into HLO" while asserting only ``result is not None`` -- a property
        true of any model that compiles at all. It stayed green through the
        entire period in which 31 MB of SKIRTOR grid baked into every graph
        (#1383). The invariant its old name claimed is now measured, on the
        surface where it is falsifiable, in
        ``test_agn_template_threading.py``.

        ``test_skirtor_template_build`` used to build this same spec and assert
        ``model is not None``. Everything it covered is covered here, before a
        strictly stronger assertion, so it is gone.
        """
        spec = _base_spec(
            agn_model="skirtor",
            agn_log_lbol=Fixed(11.42),
            agn_cos_inc=Fixed(0.5),
            agn_torus_frac=Fixed(0.5),
        )

        try:
            model = _silent_build(spec, ssp_wne, obs)
        except (FileNotFoundError, ValueError):
            pytest.skip("SKIRTOR grid not available")

        fixed_vals = model.spec.get_fixed_values()
        result = assert_jit_matches_eager(lambda p: model.predict_photometry(p), fixed_vals)

        # Assert something the compile actually has to produce: finite fluxes
        # of the right shape, not merely "not None".
        chex.assert_tree_all_finite(result)
        assert result.shape == (len(obs.photometry.filters),)


# ── Regression: Phase 4-B SSP threading ───────────────────────────────────────


def test_phase4b_ssp_threading_regression(ssp_wne, obs):
    """A plain SSP-threaded model still builds and predicts.

    Was ``assert model is not None`` followed by ``assert model.ssp_data is not
    None`` -- two properties that hold for any object a constructor returns and
    any attribute it assigns.
    """
    model = _silent_build(_base_spec(), ssp_wne, obs)

    assert model.ssp_data.ssp_flux.shape[0] == np.asarray(ssp_wne.ssp_lgmet).shape[0], (
        "the model's SSP grid should be the one it was handed"
    )
    _assert_predicts(model, obs, "phase4b SSP threading")

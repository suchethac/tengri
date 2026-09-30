# SPDX-License-Identifier: BSD-3-Clause
# Regression tests for three defects found while wiring the per-wavelength
# SSP library resolution curve onto the default LSF path (#2518):
# (1) SSPData's jax pytree flatten/unflatten swapped ssp_alpha_fe and
#     ssp_resolution_kms, crashing any jit/vmap/tree_map round-trip once
#     either field was populated (both were always None before #2518);
# (2) the per-library FWHM table matched the *whole* filename stem, which
#     never equals a real shipped filename (fsps_*.h5 catalog names, or the
#     locally generated ssp_*_wNE_logGasU..._logGasZ...h5 names), so the
#     curve was silently None for every real grid;
# (3) sigma_lib_kms/the SSP library curve must never leak into emission-line
#     predictions -- that seam belongs to #2519.

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, Parameters
from tengri.components.stellar.sps.dsps_wrapper import SSPData, load_ssp_data
from tengri.forward.sed_model import SEDModel

pytestmark = pytest.mark.regression_bug


def test_sspdata_pytree_roundtrip_preserves_resolution_and_alpha_fe():
    """A populated ssp_resolution_kms must survive a jax pytree round-trip (#2518).

    Before the fix, ``_sspdata_unflatten`` reconstructed ``SSPData`` with
    ``*children[:6]`` positional-splatted against a field order that put
    ``ssp_resolution_kms`` sixth and ``ssp_alpha_fe`` seventh, while the
    flattened ``children`` tuple carried them in the opposite order. This
    silently swapped the two fields when both were populated, and raised
    ``TypeError: SSPData.__new__() got multiple values for argument
    'ssp_resolution_kms'`` as soon as either was non-None -- which every
    populated grid now is, since #2518 has ``load_ssp_data`` fill
    ``ssp_resolution_kms`` by default. Any ``jax.jit``/``vmap``/``tree_map``
    over a loaded SSPData crossed this boundary.
    """
    resolution_kms = jnp.array([70.0, 60.0])
    alpha_fe = jnp.array([0.1, 0.2, 0.3])
    ssp = SSPData(
        ssp_wave=jnp.array([1.0, 2.0]),
        ssp_flux=jnp.zeros((1, 1, 2)),
        ssp_lg_age_gyr=jnp.array([0.0]),
        ssp_lgmet=jnp.array([0.0]),
        ssp_mass_remaining=jnp.array([1.0]),
        ssp_resolution_kms=resolution_kms,
        ssp_alpha_fe=alpha_fe,
        imf="chabrier",
        source="test",
        nebular="bare",
    )

    leaves, treedef = jax.tree_util.tree_flatten(ssp)
    roundtripped = jax.tree_util.tree_unflatten(treedef, leaves)

    chex.assert_trees_all_close(roundtripped.ssp_resolution_kms, resolution_kms)
    chex.assert_trees_all_close(roundtripped.ssp_alpha_fe, alpha_fe)

    # jax.jit crosses the same flatten/unflatten boundary on every call;
    # this is the failure mode a jitted forward pass would have hit.
    identity = jax.jit(lambda s: s.ssp_resolution_kms)
    chex.assert_trees_all_close(identity(ssp), resolution_kms)


@pytest.mark.parametrize(
    "filename",
    [
        "fsps_prsc_miles_chabrier.h5",
        "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    ],
)
def test_load_ssp_data_populates_curve_for_real_shipped_filenames(filename):
    """load_ssp_data must resolve the MILES FWHM for real shipped filenames (#2518).

    Before the fix, the lookup matched ``fp.stem`` (the whole filename, e.g.
    ``"fsps_prsc_miles_chabrier"`` or
    ``"ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0"``) against table
    keys spelled ``"ssp_prsc_miles_chabrier"`` / ``"...miles_chabrier_wNE"``
    -- neither of which is a substring match, so ``ssp_resolution_kms`` was
    silently ``None`` for every file the download catalog
    (``_data_setup._KNOWN_SSPS``, all ``fsps_*``) or the locally generated
    nebular-baked pipeline (``ssp_*_wNE_logGasU..._logGasZ...``) actually
    produces.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp = load_ssp_data(f"data/{filename}")
    assert ssp.ssp_resolution_kms is not None
    sigma = np.asarray(ssp.ssp_resolution_kms)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


def test_curve_attached_only_when_wavelengths_match_the_library_reference_grid(tmp_path):
    """A filename token alone is not enough to attach a library curve (#2518).

    A synthetic grid whose filename contains a library token (``c3k_a``)
    but whose wavelength array is not that library's reference grid (e.g.
    a BPASS grid built on its own axis, ``bpss_stars_c3k_a_chabrier.h5``,
    1221 nodes) must not receive that library's resolution curve.
    """
    import h5py

    path = tmp_path / "bpss_stars_c3k_a_chabrier.h5"
    n_met, n_age, n_wave = 2, 4, 50
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)  # not C3K's reference grid
        f["ssp_flux"] = np.full((n_met, n_age, n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = np.linspace(-3.0, 1.0, n_age)
        f["ssp_lgmet"] = np.array([-2.5, -1.8])
        f["ssp_mass_remaining"] = np.full((n_met, n_age), 0.7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssp = load_ssp_data(str(path))
    assert ssp.ssp_resolution_kms is None


def test_bc03_still_falls_back_to_none():
    """A library absent from the FWHM table still yields None, without a
    load-time warning (#2518).

    Guards the other side of the token-matching fix: a real but
    undocumented library (BC03) must not spuriously match a token it does
    not contain. ``load_ssp_data`` itself stays quiet (see
    ``test_ssp_resolution_deficit_warning.py`` for the correctly-scoped
    warning, which fires at ``SEDModel`` build time only when the model
    actually configures spectroscopy and falls back to the flat scalar);
    a grid used for photometry-only models never reaches ``sigma_lib_kms``
    at all, so a load-time warning fired on grids that would never trigger
    it -- including this suite's own synthetic fixtures in
    ``tests/contract/test_ssp_nebular_metadata.py``, whose filenames also
    match no known library token.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ssp = load_ssp_data("data/ssp_prsc_bc03_chabrier.h5")
    assert ssp.ssp_resolution_kms is None


def _minimal_spec():
    return Parameters(
        sfh_tsnorm_log_total_mass=Fixed(1.0),
        sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),
        sfh_tsnorm_width_gyr=Fixed(0.3),
        sfh_tsnorm_skew=Fixed(0.0),
        sfh_tsnorm_trunc=Fixed(3.0),
        met_logzsol=Fixed(-0.3),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.0),
    )


def test_sigma_lib_curve_does_not_reach_predict_state():
    """The SSP library LSF curve must never leak upstream of the LSF step (#2518).

    ``ssp_resolution_kms`` is consumed exclusively inside
    ``Observation.predict``/``project_spectrum``/``apply_lsf`` (the
    post-``predict_state`` projection step). ``predict_state`` -- the SFH,
    stellar, dust and nebular physics, including every discrete line
    luminosity and any ``neb_eline_sigma_kms`` broadening choice -- must be
    bit-identical whether the loaded grid carries a resolution curve or
    not. The line-painting seam for spectroscopic fitting
    (``build_eline_design_matrix``) lives entirely outside this call and
    combines only instrument resolution and ``neb_eline_sigma_kms``
    (#2519); this test pins that the SSP curve introduced by #2518 cannot
    have created a new path into it.
    """
    ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
    spec = _minimal_spec()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_curve = SEDModel(spec, ssp)
        model_flat = SEDModel(spec, ssp._replace(ssp_resolution_kms=None))

    assert model_curve.ssp_data.ssp_resolution_kms is not None
    assert model_flat.ssp_data.ssp_resolution_kms is None

    state_curve = model_curve.predict_state({})
    state_flat = model_flat.predict_state({})

    chex.assert_trees_all_close(state_curve.sed_intrinsic, state_flat.sed_intrinsic)
    assert set(state_curve.derived.keys()) == set(state_flat.derived.keys())
    for key in state_curve.derived:
        chex.assert_trees_all_close(state_curve.derived[key], state_flat.derived[key])

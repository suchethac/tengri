# SPDX-License-Identifier: BSD-3-Clause
"""#3467: sed_from_sfh / photometry_from_sfh / spectrum_from_sfh mis-integrate tabulated SFHs.

The simulation helpers wrote their own age weights: SFR held at its edge values
outside the table, mass older than the oldest SSP template dropped, the lookback
origin fixed at 13.7 Gyr whatever the redshift, ``lgmet_scatter`` accepted and
ignored, and the SFR point-sampled at the SSP age nodes. ``Catalog.from_histories``
does none of that. The helpers now build their weights through the same routine
the catalog's forward pass uses (:func:`tabulated_history_weights`), so the two
agree on formed mass and photometry.

The grid is built in the test (oldest template 12.589 Gyr, the PARSEC/MILES edge),
so no data download is needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.conservation

_Z_OBS = 0.5
_LOG_Z = -0.3  # log10(Z/Zsun) used throughout
_OLDEST_TEMPLATE_GYR = 12.589


@pytest.fixture(scope="module")
def grid_ssp():
    """A small SSP grid whose oldest template (12.589 Gyr) is younger than the universe."""
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 7.0, 1600)
    lg_age = jnp.linspace(-3.0, float(np.log10(_OLDEST_TEMPLATE_GYR)), 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])  # absolute log10(Z)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (lg_age - lg_age.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=lg_age,
        ssp_lgmet=lgmet,
    )


def _formed(t_gyr, sfr, ssp, **kwargs):
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh

    out = sed_from_sfh(jnp.asarray(t_gyr), jnp.asarray(sfr), ssp, log_z=_LOG_Z, **kwargs)
    return float(out["stellar_mass"]), out


def _integral(t_gyr, sfr):
    return float(np.trapezoid(sfr, np.asarray(t_gyr) * 1e9))


def test_sfr_outside_the_table_is_zero_not_edge_clamped(grid_ssp):
    """SFR = 1 on cosmic [5, 13.7] Gyr forms 8.70e9 Msun, not 1.26e10 from the clamp."""
    t = np.linspace(5.0, 13.7, 2000)
    formed, _ = _formed(t, np.ones_like(t), grid_ssp)
    assert formed == pytest.approx(8.70e9, rel=5e-3)


def test_mass_older_than_the_oldest_template_is_kept(grid_ssp):
    """Mass older than the 12.589 Gyr template lands on that template, so none is lost."""
    t = np.linspace(0.05, 13.787, 2000)
    sfr = np.ones_like(t)
    formed, out = _formed(t, sfr, grid_ssp)
    assert formed == pytest.approx(_integral(t, sfr), rel=5e-3)
    # Mass sits on the oldest template: the weight there is non-negligible.
    assert float(out["weights"][-1]) > 0.0


def test_redshift_sets_the_lookback_origin(grid_ssp):
    """z = 0.5: formed mass equals the table integral, and the origin is age(z)."""
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh
    from tengri.cosmology import age_at_z

    t_obs = float(age_at_z(_Z_OBS))
    t = np.linspace(0.05, t_obs, 4000)
    sfr = np.exp(-t / 3.0)
    formed, _ = _formed(t, sfr, grid_ssp, redshift=_Z_OBS)
    assert formed == pytest.approx(_integral(t, sfr), rel=5e-3)

    # The default origin is age(z), and an explicit t_obs_gyr still overrides it.
    by_redshift = sed_from_sfh(
        jnp.asarray(t), jnp.asarray(sfr), grid_ssp, log_z=_LOG_Z, redshift=_Z_OBS
    )
    by_explicit = sed_from_sfh(
        jnp.asarray(t), jnp.asarray(sfr), grid_ssp, log_z=_LOG_Z, t_obs_gyr=t_obs
    )
    np.testing.assert_allclose(by_redshift["sed"], by_explicit["sed"], rtol=1e-10)

    # Without a redshift the origin is age(0), so the SED differs from age(z).
    at_zero = sed_from_sfh(jnp.asarray(t), jnp.asarray(sfr), grid_ssp, log_z=_LOG_Z)
    assert not np.allclose(np.asarray(at_zero["sed"]), np.asarray(by_redshift["sed"]), rtol=1e-3)


@pytest.mark.parametrize("centre_gyr", [12.70, 12.7137, 12.75])
def test_a_box_of_sfr_keeps_its_mass_wherever_it_sits(grid_ssp, centre_gyr):
    """1e10 Msun in a 2 Myr box forms 1e10 Msun at any position, not 57.7x or 0."""
    # Nodes on the box edges (and just outside them), so the tabulated history
    # itself integrates to exactly 1e10 Msun; a box between nodes would test the
    # sampling of the input, not the forward model.
    width_gyr = 2e-3
    lo, hi = centre_gyr - width_gyr / 2, centre_gyr + width_gyr / 2
    t = np.unique(np.concatenate([np.linspace(0.05, 13.7, 20000), [lo - 1e-7, lo, hi, hi + 1e-7]]))
    box = np.where((t >= lo) & (t <= hi), 1e10 / (width_gyr * 1e9), 0.0)
    assert _integral(t, box) == pytest.approx(1e10, rel=1e-4)  # 1e-7 Gyr edge ramps
    formed, _ = _formed(t, box, grid_ssp)
    assert formed == pytest.approx(1e10, rel=1e-2)


def test_a_sharp_onset_keeps_its_mass(grid_ssp):
    """A step in SFR at cosmic 5 Gyr forms its full integral, not 13.9 % less."""
    t = np.linspace(0.05, 13.7, 2000)
    sfr = np.where(t >= 5.0, 1.0, 0.0)
    formed, _ = _formed(t, sfr, grid_ssp)
    assert formed == pytest.approx(_integral(t, sfr), rel=1e-2)


def test_lgmet_scatter_changes_the_sed(grid_ssp):
    """lgmet_scatter is applied: 0 and 0.3 dex give different SEDs, not identical ones."""
    t = np.linspace(0.05, 13.7, 2000)
    sfr = np.ones_like(t)
    _, narrow = _formed(t, sfr, grid_ssp, lgmet_scatter=0.0)
    _, wide = _formed(t, sfr, grid_ssp, lgmet_scatter=0.3)
    diff = np.max(np.abs(np.asarray(narrow["sed"]) - np.asarray(wide["sed"])))
    assert diff > 1e-3 * np.max(np.abs(np.asarray(narrow["sed"])))


def test_metallicity_off_the_grid_is_refused(grid_ssp):
    """A scalar log_z outside the SSP grid raises, as Catalog.from_histories does."""
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh

    t = jnp.linspace(0.05, 13.7, 200)
    with pytest.raises(ValueError, match="SSP metallicity grid"):
        sed_from_sfh(t, jnp.ones_like(t), grid_ssp, log_z=-5.0)


def test_metallicity_history_off_the_grid_is_refused(grid_ssp):
    """A Z(t) history with a node outside the SSP grid raises under the default policy."""
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh

    t = jnp.linspace(0.05, 13.7, 200)
    log_z = jnp.full_like(t, _LOG_Z).at[50].set(-5.0)
    with pytest.raises(ValueError, match="SSP metallicity grid"):
        sed_from_sfh(t, jnp.ones_like(t), grid_ssp, log_z=log_z)


def test_off_grid_metallicity_warns_under_warn_policy(grid_ssp):
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh
    from tengri.config.exceptions import OutOfSSPGridWarning

    t = jnp.linspace(0.05, 13.7, 200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sed_from_sfh(t, jnp.ones_like(t), grid_ssp, log_z=-5.0, on_out_of_grid="warn")
    assert any(issubclass(w.category, OutOfSSPGridWarning) for w in caught)


def _catalog_photometry(grid_ssp, t_gyr, sfr, redshift, log_z=_LOG_Z):
    """Predicted r-band photometry and formed mass from Catalog.from_histories."""
    from tengri import (
        DEFAULT,
        Fixed,
        ForwardModel,
        Observation,
        Photometry,
        SEDModel,
    )
    from tengri.inference.catalog import Catalog

    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    sed = SEDModel.build(
        ssp_data=grid_ssp,
        observation=obs,
        redshift=Fixed(redshift),
        sfh={"type": "table"},
        met={"logzsol": Fixed(log_z)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.0,
            "tau_diff": 0.0,
        },
        neb={"type": "none"},
    )
    fwd = ForwardModel.build(sed=sed, observation=obs)
    # A fixed scalar metallicity: the delta path, the same one sed_from_sfh takes.
    cat = Catalog.from_histories(fwd, t_gyr=t_gyr, sfr=sfr[None, :])
    out = cat.simulate(properties=("stellar_mass",), chunk_size=1)
    flux = np.asarray(out.photometry, dtype=float).ravel()
    mass = float(np.asarray(out.properties["stellar_mass"]).ravel()[0])
    return flux, mass


@pytest.mark.parametrize("shape", ["exponential", "box", "rising"])
def test_sed_from_sfh_matches_the_catalog_path(grid_ssp, shape):
    """photometry_from_sfh and Catalog.from_histories agree to 1e-3 on the same history."""
    from tengri import load_filter_set
    from tengri.analysis.simulate import photometry_from_sfh
    from tengri.cosmology import age_at_z

    t_obs = float(age_at_z(_Z_OBS))
    t = np.linspace(0.05, t_obs, 1500)
    if shape == "exponential":
        sfr = np.exp(-t / 3.0)
    elif shape == "box":
        sfr = np.where((t > 2.0) & (t < 3.0), 1.0, 0.0)
    else:
        sfr = t / 5.0

    flux_cat, mass_cat = _catalog_photometry(grid_ssp, t, sfr, _Z_OBS)
    import jax.numpy as jnp

    res = photometry_from_sfh(
        jnp.asarray(t),
        jnp.asarray(sfr),
        grid_ssp,
        load_filter_set(["sdss_r"]),
        log_z=_LOG_Z,
        redshift=_Z_OBS,
    )
    assert float(res["stellar_mass"]) == pytest.approx(mass_cat, rel=1e-3)
    np.testing.assert_allclose(np.asarray(res["flux"]), flux_cat, rtol=1e-3)


def _node_sfh():
    t = np.linspace(0.05, 13.7, 400)
    return t, np.exp(-t / 3.0)


def _node_relative(grid_ssp):
    from tengri.parameters.translate import LOG10_ZSUN

    return np.asarray(grid_ssp.ssp_lgmet, dtype=float) - LOG10_ZSUN


def test_scalar_log_z_at_a_node_gives_that_node_sed(grid_ssp):
    """A fixed Z exactly on an SSP node reads that node's SED (node-exact, #3434)."""
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh
    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

    t, sfr = _node_sfh()
    flux = np.asarray(grid_ssp.ssp_flux)
    for k, log_z in enumerate(_node_relative(grid_ssp)):
        out = sed_from_sfh(jnp.asarray(t), jnp.asarray(sfr), grid_ssp, log_z=float(log_z))
        ages_w = np.asarray(out["weights"], dtype=float)
        expected = LSUN_ERG_PER_S * np.einsum("a,aw->w", ages_w, flux[k])
        np.testing.assert_allclose(np.asarray(out["sed"]), expected, rtol=1e-10)


def test_scalar_log_z_halfway_between_nodes_stays_in_the_envelope(grid_ssp):
    """Between two nodes the cubic may overshoot slightly, but not by more than 2 %."""
    import jax.numpy as jnp

    from tengri.analysis.simulate import sed_from_sfh

    t, sfr = _node_sfh()
    rel = _node_relative(grid_ssp)

    def sed_at(z):
        out = sed_from_sfh(jnp.asarray(t), jnp.asarray(sfr), grid_ssp, log_z=float(z))
        return np.asarray(out["sed"])

    node_seds = np.stack([sed_at(z) for z in rel])
    lo, hi = node_seds.min(axis=0), node_seds.max(axis=0)
    tol = 0.02 * np.abs(node_seds).max(axis=0)
    for k in range(len(rel) - 1):
        sed = sed_at(0.5 * (rel[k] + rel[k + 1]))
        assert np.all(sed >= lo - tol), f"below the envelope between nodes {k} and {k + 1}"
        assert np.all(sed <= hi + tol), f"above the envelope between nodes {k} and {k + 1}"


def test_catalog_fixed_metallicity_declares_no_scatter(grid_ssp):
    """A Fixed-metallicity catalog model reads the SSP grid node-exactly (#3434).

    The grammar fills met_logzsol_scatter with its registry default; that fill is
    not a user declaration, so the catalog path uses the node-exact read, and its
    photometry equals the sed_from_sfh path at the same undeclared scatter.
    """
    import jax.numpy as jnp

    from tengri import (
        DEFAULT,
        Fixed,
        ForwardModel,
        Observation,
        Photometry,
        SEDModel,
        load_filter_set,
    )
    from tengri.analysis.simulate import photometry_from_sfh
    from tengri.cosmology import age_at_z
    from tengri.inference.catalog import Catalog, _stellar_config

    node_rel = _node_relative(grid_ssp)[1]
    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    sed = SEDModel.build(
        ssp_data=grid_ssp,
        observation=obs,
        redshift=Fixed(_Z_OBS),
        sfh={"type": "table"},
        met={"logzsol": Fixed(float(node_rel))},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.0,
            "tau_diff": 0.0,
        },
        neb={"type": "none"},
    )
    fwd = ForwardModel.build(sed=sed, observation=obs)
    assert _stellar_config(fwd).lgmet_scatter_declared is False

    t = np.linspace(0.05, float(age_at_z(_Z_OBS)), 1500)
    sfr = np.exp(-t / 3.0)
    cat = Catalog.from_histories(fwd, t_gyr=t, sfr=sfr[None, :])
    flux = np.asarray(cat.simulate(properties=("stellar_mass",), chunk_size=1).photometry).ravel()
    ref = photometry_from_sfh(
        jnp.asarray(t),
        jnp.asarray(sfr),
        grid_ssp,
        load_filter_set(["sdss_r"]),
        log_z=float(node_rel),
        redshift=_Z_OBS,
    )
    np.testing.assert_allclose(flux, np.asarray(ref["flux"]), rtol=1e-3)


def test_per_age_history_with_undeclared_scatter_is_one_hot_on_nodes(grid_ssp):
    """An evolving Z with no declared scatter reads each age's node Z node-exactly (#3434).

    Every SSP age is given a metallicity that sits on an SSP node; with the
    scatter undeclared, each age's column of joint weights is one-hot on that node.
    """
    from tengri.components.stellar.component import _joint_weights_cic_met_table

    ssp_lgmet = np.asarray(grid_ssp.ssp_lgmet, dtype=float)
    lg_age = np.asarray(grid_ssp.ssp_lg_age_gyr, dtype=float)
    ssp_ages_yr = 10.0 ** (lg_age + 9.0)
    n_age = ssp_ages_yr.shape[0]
    pattern = np.array([0, 2, 1])[np.arange(n_age) % 3]
    lgmet_on_ssp_ages = ssp_lgmet[pattern]

    joint, _ = _joint_weights_cic_met_table(
        ssp_ages_yr,
        np.ones(n_age),
        ssp_ages_yr,
        float(ssp_ages_yr[-1] / 1e9),
        lgmet_on_ssp_ages,
        0.0,
        ssp_lgmet,
        declared=False,
    )
    joint = np.asarray(joint)
    for a in range(n_age):
        col = joint[:, a]
        if col.sum() <= 0.0:
            continue
        expected = np.zeros(ssp_lgmet.shape[0])
        expected[pattern[a]] = 1.0
        np.testing.assert_allclose(col / col.sum(), expected, atol=1e-9)

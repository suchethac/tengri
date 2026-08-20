# SPDX-License-Identifier: BSD-3-Clause
"""Regressions from the component-matrix sweep (build+predict every registry entry).

- Lazy dust-emission templates used to resolve INSIDE the first model's jit
  trace and cache trace-staged arrays in the module registry: the second
  model sharing the slot (e.g. dl07 → dl07_tabulated) died with
  UnexpectedTracerError.
- Three SFH types (constant_then_exponential, db, dbp) built fine and died
  at the first predict with NotImplementedError — the build-time gate and
  the stellar component's runtime allowlist had drifted apart.
- Shipped filter curves are imperfect (duplicate wavelength rows in
  HST_ACS_WFC_F814W, negative transmission in SPIRE/MIPS) — the loader
  must sanitize, since duplicates silently break ascending-grid interp.
"""

import os
import warnings

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.contract

_DL07 = os.path.exists("data/dl07_templates.h5") or os.path.exists("data/dl07_templates_v2.h5")


@pytest.mark.skipif(not _DL07, reason="dl07 templates absent (data-gated)")
def test_lazy_dust_templates_survive_cross_model_use(synthetic_ssp_wide, synthetic_tophat_obs):
    def build_and_predict(emission):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                redshift=Fixed(0.1),
                sfh={"type": "delayed", "*": FIXED},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.5),
                    "tau_diff": Fixed(0.3),
                    "*": FIXED,
                },
                dust_emission={"type": emission, "*": FIXED},
            )
            params = model.spec.sample(jax.random.PRNGKey(1))
            return np.asarray(model.predict_photometry(params))

    first = build_and_predict("dl07")
    second = build_and_predict("dl07_tabulated")  # shares the lazily resolved slot
    assert np.isfinite(first).all() and np.isfinite(second).all()


@pytest.mark.parametrize("sfh_type", ["constant_then_exponential", "db", "dbp"])
def test_runtime_unsupported_sfh_types_fail_at_build(sfh_type):
    from tengri.parameters.groups import parse_groups

    with pytest.raises(ValueError, match="not yet validated"):
        parse_groups(sfh={"type": sfh_type, "*": FIXED}, redshift=Fixed(0.1))


def test_shipped_filter_curves_are_sanitized():
    from tengri.observation.filters import load_filter

    for name in ("hst_f814w", "herschel_spire_250", "spitzer_mips_70"):
        try:
            fc = load_filter(name)
        except KeyError:
            continue  # registry name differs across revisions
        wave = np.asarray(fc.wave)
        trans = np.asarray(fc.trans)
        assert (np.diff(wave) > 0).all(), f"{name}: wavelengths not strictly ascending"
        assert (trans >= 0).all(), f"{name}: negative transmission survived sanitization"


def test_duplicate_and_negative_rows_sanitized(tmp_path):
    from tengri.observation.filters import _load_filter_file

    path = tmp_path / "messy.dat"
    path.write_text("1000 0.0\n2000 0.5\n2000 0.3\n3000 -0.01\n4000 0.0\n")
    wave, trans = _load_filter_file(path)
    assert (np.diff(wave) > 0).all()
    np.testing.assert_allclose(trans[wave == 2000], [0.4])  # duplicates averaged
    assert (trans >= 0).all()


def test_cloudy_grammar_accepts_grid_key():
    """neb={'type': 'cloudy', 'grid': ...} reaches Parameters.cloudy_grid_path.

    Before this fix the grammar had no way to pass the grid at all (only the
    flat expert kwarg), and the missing-grid error listed the packaged
    src/tengri/data/ directory — which never contains CLOUDY grids.
    """
    from tengri.parameters.groups import parse_groups

    params = parse_groups(
        neb={"type": "cloudy", "*": FIXED, "grid": "data/cloudy_grid_mist.h5"},
        redshift=Fixed(0.1),
    )
    assert params.cloudy_grid_path == "data/cloudy_grid_mist.h5"
    assert params.nebular_mode == "cloudy"


def test_cloudy_missing_grid_error_names_the_grammar_key(monkeypatch):
    from tengri.parameters.groups import parse_groups
    from tengri.parameters.parameters import Parameters

    monkeypatch.setattr(Parameters, "_default_cloudy_grid", staticmethod(lambda: None))
    with pytest.raises(ValueError, match=r"neb=\{'type': 'cloudy', 'grid'"):
        parse_groups(neb={"type": "cloudy", "*": FIXED}, redshift=Fixed(0.1))

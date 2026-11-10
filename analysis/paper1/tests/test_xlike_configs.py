# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for X-like configurations.

Tests the five X-like builders (cigale_like, prospector_like, bagpipes_like,
beagle_like, dense_basis_like) against the external code parity specifications.
Key invariants pinned:
- XLIKE keys disjoint from CONFIG_KEYS, CONFIG_KEYS unchanged
- Metadata completeness and parity flags
- Builder argument structure (SFH dicts, dust params, settings)
- Argparse acceptance in fit_one
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.config_metadata import XLIKE_CONFIGS, XLIKE_KEYS
from paper1.configs import CONFIG_KEYS, CONFIGS, config_II

pytestmark = pytest.mark.unit


class TestXlikeMetadata:
    """Test X-like configuration metadata."""

    def test_keys_disjoint_from_grid_configs(self):
        """XLIKE keys disjoint from CONFIG_KEYS, CONFIG_KEYS unchanged."""
        assert CONFIG_KEYS == ["I", "II", "III", "IV", "V", "VI"]
        assert set(XLIKE_KEYS).isdisjoint(set(CONFIG_KEYS))

    def test_xlike_keys_in_metadata(self):
        """All XLIKE_KEYS have corresponding entries in XLIKE_CONFIGS."""
        for key in XLIKE_KEYS:
            assert key in XLIKE_CONFIGS
            assert XLIKE_CONFIGS[key]["key"] == key

    def test_all_xlike_have_code_names(self):
        """All X-like configs map to the five external codes."""
        codes = {cfg["code"] for cfg in XLIKE_CONFIGS.values()}
        expected = {"CIGALE", "Prospector", "BAGPIPES", "BEAGLE", "Dense_Basis"}
        assert codes == expected

    def test_parity_flags(self):
        """Parity flags: True for first three, False for last two."""
        parity_true = {"cigale_like", "prospector_like", "bagpipes_like"}
        parity_false = {"beagle_like", "dense_basis_like"}
        for key in parity_true:
            assert XLIKE_CONFIGS[key]["parity_check"] is True
        for key in parity_false:
            assert XLIKE_CONFIGS[key]["parity_check"] is False

    def test_mismatches_present(self):
        """Each config lists at least 2 differences (rows rest on upstream code, not a count)."""
        for key, cfg in XLIKE_CONFIGS.items():
            mismatches = cfg.get("mismatches", [])
            assert len(mismatches) >= 2, f"{key} has only {len(mismatches)} mismatches"

    def test_mass_definition(self):
        """Mass definition correctly reflects code convention."""
        assert XLIKE_CONFIGS["cigale_like"]["mass_definition"] == "surviving"
        assert XLIKE_CONFIGS["prospector_like"]["mass_definition"] == "formed"
        assert XLIKE_CONFIGS["bagpipes_like"]["mass_definition"] == "surviving"
        assert XLIKE_CONFIGS["beagle_like"]["mass_definition"] == "surviving"
        assert XLIKE_CONFIGS["dense_basis_like"]["mass_definition"] == "surviving"

    def test_fiducial_table1_structure(self):
        """Each config has Pacifici+2023 Table 1 row (fiducial_table1 dict)."""
        table1_keys = {"sfh", "ssp", "nebular", "dust_att", "dust_em", "agn", "sampler"}
        for _key, cfg in XLIKE_CONFIGS.items():
            assert "fiducial_table1" in cfg
            assert set(cfg["fiducial_table1"]) == table1_keys


class TestXlikeBuildKwargs:
    """Test that builder functions construct correct argument dicts (no JAX build).

    Uses monkeypatch to intercept SEDModel.build and capture its kwargs,
    verifying structure without materializing JAX models. ``SEDModel.build``
    is a ``classmethod``; assigning a plain function to the class attribute
    (what ``monkeypatch.setattr`` does) means a call routed through the class
    -- ``SEDModel.build(**kwargs)`` -- does NOT auto-bind a leading ``cls``,
    so the stand-in below takes no ``cls`` parameter. A stand-in that declared
    one would raise ``TypeError: missing 1 required positional argument:
    'cls'`` on every call, before any assertion below ran.
    """

    def test_cigale_like_sfh_structure(self, monkeypatch):
        """cigale_like has delayed SFH with free tau_gyr, age_frac; two-component dust."""
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import cigale_like

        captured_kwargs = {}

        def mock_build(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        # Create minimal test inputs
        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        observation = MagicMock()

        cigale_like(ssp_data, observation, z=1.0)

        assert "sfh" in captured_kwargs
        sfh = captured_kwargs["sfh"]
        assert sfh["type"] == "delayed"
        assert "tau_gyr" in sfh
        assert sfh["tau_gyr"].hi > sfh["tau_gyr"].lo > 0  # Uniform

        assert "dust_attenuation" in captured_kwargs
        dust_att = captured_kwargs["dust_attenuation"]
        assert dust_att["type"] == "two_component"
        # Both screens must be leitherer02 -- CIGALE's own attenuation law --
        # not one screen defaulting to a different law by omission.
        assert dust_att["law_bc"] == "leitherer02"
        assert dust_att["law_diff"] == "leitherer02"

    def test_prospector_like_continuity_sfh(self, monkeypatch):
        """prospector_like has continuity SFH matching config_I structure."""
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import prospector_like

        captured_kwargs = {}

        def mock_build(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        observation = MagicMock()

        prospector_like(ssp_data, observation, z=1.0)

        sfh = captured_kwargs["sfh"]
        assert sfh["type"] == "continuity"
        # Check for ratio parameters
        ratio_keys = [k for k in sfh if k.startswith("ratio_")]
        assert len(ratio_keys) > 0, "continuity SFH should have ratio_* parameters"

    def test_bagpipes_like_matches_config_ii_dpl_sfh(self, monkeypatch):
        """bagpipes_like has dpl SFH with free alpha, beta, tau_gyr, age_frac.

        It must match config_II's dpl SFH exactly: both are meant to be the
        same double-power-law family with the same bounds (age-conditioned
        tau_gyr cap included), and nothing else in the code pins the two
        together, so a comparison against config_II's own captured kwargs is
        the only thing that would catch them drifting apart.
        """
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import bagpipes_like

        captured_kwargs = {}

        def mock_build(**kwargs):
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        observation = MagicMock()

        bagpipes_like(ssp_data, observation, z=1.0)
        sfh = dict(captured_kwargs["sfh"])

        assert sfh["type"] == "dpl"
        assert "alpha" in sfh
        assert "beta" in sfh
        assert "tau_gyr" in sfh
        assert "age_frac" in sfh

        # config_II uses the same mocked SEDModel.build, so this second call
        # reuses (and overwrites) captured_kwargs -- read bagpipes_like's sfh
        # dict above before calling it.
        config_II(ssp_data, observation, z=1.0)
        config_ii_sfh = dict(captured_kwargs["sfh"])

        assert sfh == config_ii_sfh, (
            "bagpipes_like's dpl SFH has drifted from config_II's -- they are "
            "meant to share the same double-power-law family and bounds"
        )

    def test_beagle_like_dust_none(self, monkeypatch):
        """beagle_like has no dust emission, in the exact form the builder passes."""
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import beagle_like

        captured_kwargs = {}

        def mock_build(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        observation = MagicMock()

        beagle_like(ssp_data, observation, z=1.0)

        dust_em = captured_kwargs["dust_emission"]
        # The builder passes exactly {"type": "none"} -- no "all_params" wildcard,
        # no fixed IR parameters -- since a dust-IR-free model has none to govern.
        assert dust_em == {"type": "none"}

    def test_dense_basis_like_leaves_the_registry_untouched(self, monkeypatch):
        """dense_basis_like passes the redshift and mutates no registry setting.

        The reference age for the dense_basis time quantiles is derived from the
        model's redshift by tengri, so the builder must hand over ``redshift``
        and must not write the dense_basis registry entry (global state shared
        by every model in the process).
        """
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import dense_basis_like

        from tengri import SFH_REGISTRY

        captured_kwargs = {}

        def mock_build(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        settings_before = dict(SFH_REGISTRY["dense_basis"].settings)

        dense_basis_like(ssp_data, MagicMock(), z=1.0)

        assert dict(SFH_REGISTRY["dense_basis"].settings) == settings_before
        assert captured_kwargs["sfh"]["type"] == "dense_basis"
        assert "settings" not in captured_kwargs["sfh"]
        assert captured_kwargs["redshift"].value == pytest.approx(1.0)


class TestFitOneArgparse:
    """Test that fit_one.py accepts X-like keys."""

    def test_fit_one_argparse_accepts_xlike_keys(self):
        """fit_one's --config argparse accepts all X-like keys."""

        # Construct args that would be passed to fit_one
        # We only test argparse acceptance, not actual fitting
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--galaxy", type=int, required=True)
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            choices=sorted(list(CONFIGS.keys()) + list(XLIKE_CONFIGS.keys())),
        )
        parser.add_argument("--out", type=Path, required=True)

        # Test that each X-like key is accepted
        for key in XLIKE_KEYS:
            args = parser.parse_args(["--galaxy", "79", "--config", key, "--out", "/tmp"])
            assert args.config == key


class _FakeGridPrior:
    """Stand-in for the ``met_prior_for`` return value: exposes ``lo`` and ``hi``."""

    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi


class TestBeagleLikeParameterization:
    """BEAGLE-like priors and components, read from the captured SEDModel.build call."""

    @staticmethod
    def _capture(monkeypatch, z: float = 1.0) -> dict:
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import beagle_like

        captured = {}

        def mock_build(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        beagle_like(ssp_data, MagicMock(), z=z)
        return captured

    def test_tau_v_free_mu_fixed_no_bc_or_diffuse_tau(self, monkeypatch):
        """tau_V is a LogUniform(1e-3, 5) prior, mu is fixed at 0.4, no tau_bc/tau_diff."""
        from tengri import Fixed, LogUniform

        dust = self._capture(monkeypatch)["dust_attenuation"]

        assert isinstance(dust["tau_v"], LogUniform)
        assert dust["tau_v"].lo == pytest.approx(1e-3)
        assert dust["tau_v"].hi == pytest.approx(5.0)
        assert isinstance(dust["mu"], Fixed)
        assert dust["mu"].value == pytest.approx(0.4)
        assert "tau_bc" not in dust
        assert "tau_diff" not in dust

    def test_age_prior_lower_bound_is_0p1_gyr_over_age_at_z(self, monkeypatch):
        """age_frac is log-uniform from 0.1 Gyr / age(z) to 1."""
        from tengri import LogUniform
        from tengri.cosmology import age_at_z

        sfh = self._capture(monkeypatch, z=1.0)["sfh"]

        assert isinstance(sfh["age_frac"], LogUniform)
        assert sfh["age_frac"].lo == pytest.approx(0.1 / age_at_z(1.0))
        assert sfh["age_frac"].hi == pytest.approx(1.0)

    def test_beagle_met_prior_is_intersection_of_beagle_and_grid(self, monkeypatch):
        """Metallicity bounds are max/min of BEAGLE's shifted range and the grid prior."""
        from unittest.mock import MagicMock

        from paper1 import xlike_configs
        from paper1.xlike_configs import beagle_met_prior

        # BEAGLE range in log10(Z/Zsun): [-2.2, 0.24] + log10(0.01524) - LOG10_ZSUN ~ +0.0307
        beagle_lo, beagle_hi = -2.2 + 0.0307, 0.24 + 0.0307
        ssp_data = MagicMock()

        monkeypatch.setattr(xlike_configs, "met_prior_for", lambda _ssp: _FakeGridPrior(-3.0, 1.0))
        wide = beagle_met_prior(ssp_data)
        assert wide.lo == pytest.approx(beagle_lo, abs=1e-3)
        assert wide.hi == pytest.approx(beagle_hi, abs=1e-3)

        monkeypatch.setattr(xlike_configs, "met_prior_for", lambda _ssp: _FakeGridPrior(-1.0, 0.1))
        narrow = beagle_met_prior(ssp_data)
        assert narrow.lo == pytest.approx(-1.0)
        assert narrow.hi == pytest.approx(0.1)

        monkeypatch.setattr(xlike_configs, "met_prior_for", lambda _ssp: _FakeGridPrior(0.5, 1.0))
        with pytest.raises(ValueError, match="does not overlap"):
            beagle_met_prior(ssp_data)

    def test_ssfr_upper_bound_and_recent_timescale(self, monkeypatch):
        """log_ssfr spans [-12, -7.3]; the recent-episode timescale is fixed at 10 Myr."""
        from tengri import Fixed, Uniform

        sfh = self._capture(monkeypatch)["sfh"]

        assert sfh["type"] == ["delayed", "recent_ssfr"]
        assert isinstance(sfh["log_ssfr"], Uniform)
        assert sfh["log_ssfr"].lo == pytest.approx(-12.0)
        assert sfh["log_ssfr"].hi == pytest.approx(-7.3)
        assert isinstance(sfh["timescale_myr"], Fixed)
        assert sfh["timescale_myr"].value == pytest.approx(10.0)

    def test_neb_is_gutkin16_with_free_logu(self, monkeypatch):
        """Nebular is gutkin16 with logU free on [-4, -1]."""
        from tengri import Uniform

        neb = self._capture(monkeypatch)["neb"]

        assert neb["type"] == "gutkin16"
        assert isinstance(neb["logU"], Uniform)
        assert neb["logU"].lo == pytest.approx(-4.0)
        assert neb["logU"].hi == pytest.approx(-1.0)

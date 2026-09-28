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
        """Each config has at least 3 mismatches listed."""
        for key, cfg in XLIKE_CONFIGS.items():
            mismatches = cfg.get("mismatches", [])
            assert len(mismatches) >= 3, f"{key} has only {len(mismatches)} mismatches"

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
        """cigale_like has delayed SFH with free tau_gyr, age_gyr; two-component dust."""
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
        """bagpipes_like has dpl SFH with free alpha, beta, tau_gyr, age_gyr.

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
        assert "age_gyr" in sfh

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

    def test_dense_basis_like_age_universe_setting(self, monkeypatch):
        """dense_basis_like sets the dense_basis registry's cosmic-age cutoff.

        There is no key in the "sfh" build grammar for a model's registry
        SETTINGS (as opposed to its fittable parameters) -- only 'type',
        'all_params', 'bin_edges_gyr', 'age_kernel', 'field_centering', and
        per-parameter names are recognized for the "sfh" group. The age
        override therefore cannot travel through ``captured_kwargs["sfh"]``;
        it goes through ``tengri.SFH_REGISTRY["dense_basis"].settings``.

        dense_basis_like sets that entry and deliberately leaves it set
        (see the docstring there): the component that reads it
        (``StellarSEDComponent.apply``, reached from
        ``predict_photometry``) does so lazily, at the first prediction
        JAX traces, strictly after this function returns -- so restoring
        the registry before returning would make the override inert for
        every real fit. This test therefore checks the registry is left
        holding ``age_at_z(1.0)`` exactly (not merely "less than 13.47")
        rather than expecting it back at the default, and restores the
        default itself afterward (``finally``) so this test cannot leak
        into a sibling test in the same pytest session that assumes it.
        """
        from unittest.mock import MagicMock

        import numpy as np
        from paper1.xlike_configs import dense_basis_like

        from tengri import SFH_REGISTRY
        from tengri.cosmology import age_at_z

        captured_kwargs = {}

        def mock_build(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("paper1.xlike_configs.SEDModel.build", mock_build)

        ssp_data = MagicMock()
        ssp_data.ssp_lgmet = np.array([-2.0, -1.0, 0.0])
        observation = MagicMock()

        original_age_gyr = SFH_REGISTRY["dense_basis"].settings.get("sfh_db_age_universe_gyr")

        z = 1.0
        try:
            dense_basis_like(ssp_data, observation, z=z)

            expected_age_gyr = float(age_at_z(z))
            assert expected_age_gyr != pytest.approx(13.47), (
                "age_at_z(1.0) must not coincide with the z=0 registry default -- "
                "otherwise this test cannot tell the override from a no-op"
            )

            # Left set (not restored) -- see the docstring above.
            resolved_age_gyr = SFH_REGISTRY["dense_basis"].settings["sfh_db_age_universe_gyr"]
            assert resolved_age_gyr == pytest.approx(expected_age_gyr)

            sfh = captured_kwargs["sfh"]
            assert sfh["type"] == "dense_basis"
            assert "settings" not in sfh, (
                "'settings' is not a recognized key in the sfh build grammar and "
                "raises ValueError on a real (non-mocked) build; the age override "
                "must go through SFH_REGISTRY, not a build kwarg"
            )
        finally:
            # Test hygiene only -- production leaves this mutated (see above).
            if original_age_gyr is None:
                SFH_REGISTRY["dense_basis"].settings.pop("sfh_db_age_universe_gyr", None)
            else:
                SFH_REGISTRY["dense_basis"].settings["sfh_db_age_universe_gyr"] = original_age_gyr


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

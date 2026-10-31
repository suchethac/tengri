# SPDX-License-Identifier: BSD-3-Clause
"""The runner's ``--batch-model`` mode: one compiled model, every galaxy's fit.

No SSP data and no real fit: the model builder, the forward model and the data
classes are replaced by fakes, so what is tested is the batch runner's own code
path (build count, union bands and presence, per-galaxy fit kwargs, derived draws
at each galaxy's redshift, cell output and skipping).

Pinned invariants:
- ``build_model`` runs exactly once per factorial model, at ``Z_REF_FACTORIAL`` with
  ``prior_z=Z_REF_FACTORIAL`` and ``catalog_z_range=(1.0, 1.11)``, whatever the
  number of galaxies;
- every fit call for galaxy g carries ``params={"redshift": z_g}`` (ForwardModel.fit's
  name for the Fitter's ``params_override``) and ``presence`` = galaxy g's band mask;
- the union band order is first appearance across galaxies in the given order; absent
  bands carry flux 0, error 1, presence 0;
- derived draws are evaluated at the galaxy's own redshift;
- one cell per galaxy with its own ``z`` and a ``batch`` block; an existing cell is
  skipped without ``force``;
- ``build_model``'s default path (no ``prior_z``, no ``catalog_z_range``) is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import paper1.bma_evidence as be
import paper1.bma_space as bma_space
from paper1 import _bma_keys as bk

pytestmark = pytest.mark.unit

GALAXIES = (79, 267, 500)
Z = {79: 0.9, 267: 1.05, 500: 1.02}
# Bands per galaxy; 267 lacks hst_f160w, 500 lacks hst_f814w and hst_f125w.
BANDS = {
    79: ["hst_f814w", "hst_f160w"],
    267: ["hst_f814w", "hst_f125w"],
    500: ["hst_f160w"],
}
UNION = ["hst_f814w", "hst_f160w", "hst_f125w"]
EXPECTED_PRESENCE = {
    79: [1.0, 1.0, 0.0],
    267: [1.0, 0.0, 1.0],
    500: [0.0, 1.0, 0.0],
}
N_RESTARTS = 2
N_DRAWS = 700  # more than the 500-draw cap


def _galaxy_arrays(gid: int):
    names = BANDS[gid]
    fnu = np.array([(i + 1) * 1e-27 * (gid % 7 + 1) for i in range(len(names))])
    sigma = np.full(len(names), 1e-28)
    return names, fnu, sigma


MODEL = {
    "sfh": "dpl",
    "ssp": bk.SSP_LABELS[0],
    "attenuation": "calzetti",
    "dust_emission": bk.DUST_EMISSION,
    "nebular": bk.NEBULAR,
}


class _FakeSED:
    """Stands in for SEDModel: a free parameter ``x`` and two properties."""

    spec = SimpleNamespace(free_params=["x"], get_fixed_values=lambda: {"redshift": 1.10})
    available_properties = ("stellar_mass", "stellar_mass_surviving", "sfr_100myr", "sfr_10myr")
    predict_state_calls: ClassVar[list[dict]] = []

    def _ensure_property_catalog(self):
        return None

    @property
    def _property_catalog(self):
        def make(name):
            def fn(state, params):
                import jax.numpy as jnp

                return jnp.exp(params["x"]) * state["z"]

            return SimpleNamespace(fn=fn)

        return {n: make(n) for n in self.available_properties}

    def predict_state(self, params, fixed_values=None):
        # The state carries the redshift the properties are evaluated at.
        type(self).predict_state_calls.append(
            {"params": dict(params), "fixed_values": fixed_values}
        )
        z = float(fixed_values["redshift"]) if fixed_values else 1.10
        return {"z": z}


class _FakeForward:
    """Records every fit call; answers per method."""

    calls: ClassVar[list[dict]] = []

    def fit(self, data, *, key, method, **kwargs):
        type(self).calls.append({"method": method, "data": data, **kwargs})
        samples = {"x": np.linspace(0.0, 1.0, N_DRAWS)}
        if method == "map":
            return SimpleNamespace(diagnostics={"final_loss": 5.0}, samples=None)
        assert method == "laplace"
        diagnostics = {
            "newton_decrement": 0.01,
            "n_clipped_eigenvalues": 0,
            "condition_number": 1e3,
        }
        return SimpleNamespace(log_evidence=-12.0, diagnostics=diagnostics, samples=samples)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    import paper1.configs as configs

    import tengri

    _FakeForward.calls = []
    _FakeSED.predict_state_calls = []
    build_calls: list[dict] = []

    def fake_build(components, ssp_data, observation, *args, **kwargs):
        build_calls.append({"components": components, "args": args, **kwargs})
        return _FakeSED()

    monkeypatch.setattr(configs, "load_ssp_for", lambda config: object())
    monkeypatch.setattr(bma_space, "build_model", fake_build)
    monkeypatch.setattr(
        tengri.Photometry,
        "from_names",
        staticmethod(lambda names: SimpleNamespace(names=tuple(names))),
    )
    monkeypatch.setattr(tengri, "Observation", lambda **kwargs: object())
    monkeypatch.setattr(tengri, "Data", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        tengri.ForwardModel, "build", classmethod(lambda cls, **kw: _FakeForward())
    )
    monkeypatch.setattr(be, "get_galaxy_data", lambda gid: _galaxy_data(gid))
    return SimpleNamespace(forward=_FakeForward, builds=build_calls, tmp=tmp_path)


def _galaxy_data(gid: int):
    names, fnu, sigma = _galaxy_arrays(gid)
    phot = {"names": names, "fnu": fnu, "fnu_err": sigma}
    return Z[gid], phot, fnu, sigma


def _run(out_dir: Path, galaxies=GALAXIES, force=False):
    return be.fit_model_over_galaxies(
        MODEL,
        list(galaxies),
        n_restarts=N_RESTARTS,
        seed=7,
        out_dir=out_dir,
        force=force,
    )


def _calls_for(fakes, fluxes_row: list[float]):
    """The fit calls whose data carries the given (first) flux row."""
    out = []
    for call in fakes.forward.calls:
        flux = np.asarray(call["data"].photometry[0])
        if np.allclose(flux, fluxes_row, rtol=0, atol=0):
            out.append(call)
    return out


def _union_fluxes_for(gid: int) -> list[float]:
    names, fnu, _ = _galaxy_arrays(gid)
    row = [0.0] * len(UNION)
    for name, value in zip(names, fnu, strict=True):
        row[UNION.index(name)] = float(value)
    return row


def test_build_model_runs_once_for_all_galaxies_at_the_reference_redshift(fakes, tmp_path):
    _run(tmp_path)
    assert len(fakes.builds) == 1
    call = fakes.builds[0]
    assert call["args"] == ()
    assert call["z"] == bma_space.Z_REF_FACTORIAL == pytest.approx(1.10)
    assert call["prior_z"] == pytest.approx(1.10)
    assert call["catalog_z_range"] == (1.0, 1.11)


def test_every_fit_call_carries_its_galaxy_redshift_and_presence(fakes, tmp_path):
    _run(tmp_path)
    assert len(fakes.forward.calls) == len(GALAXIES) * (N_RESTARTS + 1)
    for gid in GALAXIES:
        calls = _calls_for(fakes, _union_fluxes_for(gid))
        assert len(calls) == N_RESTARTS + 1, gid
        for call in calls:
            assert call["params"] == {"redshift": Z[gid]}
            np.testing.assert_array_equal(call["presence"], EXPECTED_PRESENCE[gid])


def test_union_band_order_is_first_appearance_and_absent_bands_are_neutral(fakes, tmp_path):
    _run(tmp_path)
    for gid in GALAXIES:
        names, fnu, sigma = _galaxy_arrays(gid)
        flux, err = _calls_for(fakes, _union_fluxes_for(gid))[0]["data"].photometry
        for name, f, s in zip(names, fnu, sigma, strict=True):
            col = UNION.index(name)
            assert flux[col] == pytest.approx(f)
            assert err[col] >= s  # the systematic floor only ever raises the error
        for col, name in enumerate(UNION):
            if name not in names:
                assert flux[col] == 0.0 and err[col] == 1.0


def test_union_bands_are_observed_in_first_appearance_order(fakes, monkeypatch, tmp_path):
    seen: list[tuple] = []

    import tengri

    def record(names):
        seen.append(tuple(names))
        return SimpleNamespace(names=tuple(names))

    monkeypatch.setattr(tengri.Photometry, "from_names", staticmethod(record))
    _run(tmp_path)
    assert seen == [tuple(UNION)]


def test_derived_draws_are_evaluated_at_each_galaxys_own_redshift(fakes, tmp_path):
    _run(tmp_path)
    by_z = {}
    for call in _FakeSED.predict_state_calls:
        by_z.setdefault(call["fixed_values"]["redshift"], 0)
        by_z[call["fixed_values"]["redshift"]] += 1
    assert set(by_z) == {Z[g] for g in GALAXIES}
    for call in _FakeSED.predict_state_calls:
        assert set(call["params"]) == {"x"}  # free parameters only


def test_one_cell_per_galaxy_with_its_own_z_and_the_batch_block(fakes, tmp_path):
    _run(tmp_path)
    key = bk.model_key(MODEL)
    for gid in GALAXIES:
        cell = json.loads((tmp_path / str(gid) / f"{key}.json").read_text())
        assert cell["galaxy"] == gid
        assert cell["z"] == pytest.approx(Z[gid])
        assert cell["error"] is None and cell["valid"] is True
        assert cell["batch"] == {
            "union_bands": UNION,
            "presence": EXPECTED_PRESENCE[gid],
            "catalog_z_range": [1.0, 1.11],
            "prior_z": 1.10,
        }
        npz = np.load(tmp_path / str(gid) / f"{key}.npz")
        assert set(npz.files) == {
            "log_stellar_mass_formed",
            "log_stellar_mass_survived",
            "log_sfr_100myr",
            "log_sfr_10myr",
        }


def test_an_existing_cell_is_skipped_without_force(fakes, tmp_path):
    key = bk.model_key(MODEL)
    (tmp_path / "267").mkdir(parents=True)
    (tmp_path / "267" / f"{key}.json").write_text("{}")
    _run(tmp_path)
    assert _calls_for(fakes, _union_fluxes_for(267)) == []
    assert _calls_for(fakes, _union_fluxes_for(79))  # the other galaxies still fit
    assert json.loads((tmp_path / "267" / f"{key}.json").read_text()) == {}


def test_force_refits_an_existing_cell(fakes, tmp_path):
    key = bk.model_key(MODEL)
    (tmp_path / "267").mkdir(parents=True)
    (tmp_path / "267" / f"{key}.json").write_text("{}")
    _run(tmp_path, force=True)
    assert json.loads((tmp_path / "267" / f"{key}.json").read_text())["galaxy"] == 267


def test_all_cells_present_means_no_build_at_all(fakes, tmp_path):
    key = bk.model_key(MODEL)
    for gid in GALAXIES:
        (tmp_path / str(gid)).mkdir(parents=True)
        (tmp_path / str(gid) / f"{key}.json").write_text("{}")
    _run(tmp_path)
    assert fakes.builds == [] and fakes.forward.calls == []


def test_build_model_default_path_uses_z_for_the_sfh_and_plain_wavepre(monkeypatch):
    """The unchanged path: prior_z and catalog_z_range default to the old behavior."""
    import tengri
    from tengri import WavePrecomp

    captured: dict = {}
    seen_z: list[float] = []

    class _Stop(Exception):
        pass

    def fake_build(cls, **kwargs):
        captured.update(kwargs)
        raise _Stop

    def fake_sfh(ssp_data, z):
        seen_z.append(z)
        return {}

    monkeypatch.setitem(bma_space._SFH_BUILDERS, "dpl", fake_sfh)
    monkeypatch.setattr(tengri.SEDModel, "build", classmethod(fake_build))
    comps = {
        "sfh": "dpl",
        "ssp": bk.SSP_LABELS[0],
        "attenuation": "calzetti",
        "dust_emission": bk.DUST_EMISSION,
        "nebular": bk.NEBULAR,
    }
    with pytest.raises(_Stop):
        bma_space.build_model(comps, object(), object(), 0.7)
    assert seen_z == [0.7]
    assert captured["approx"] == WavePrecomp()
    assert captured["approx"].catalog_z_range is None


def test_build_model_batch_path_passes_prior_z_and_catalog_range(monkeypatch):
    import tengri
    from tengri import WavePrecomp

    captured: dict = {}
    seen_z: list[float] = []

    class _Stop(Exception):
        pass

    def fake_build(cls, **kwargs):
        captured.update(kwargs)
        raise _Stop

    def fake_sfh(ssp_data, z):
        seen_z.append(z)
        return {}

    monkeypatch.setitem(bma_space._SFH_BUILDERS, "dpl", fake_sfh)
    monkeypatch.setattr(tengri.SEDModel, "build", classmethod(fake_build))
    comps = {
        "sfh": "dpl",
        "ssp": bk.SSP_LABELS[0],
        "attenuation": "calzetti",
        "dust_emission": bk.DUST_EMISSION,
        "nebular": bk.NEBULAR,
    }
    with pytest.raises(_Stop):
        bma_space.build_model(
            comps,
            object(),
            object(),
            0.7,
            catalog_z_range=(1.0, 1.11),
            prior_z=1.10,
        )
    assert seen_z == [1.10]  # the SFH priors use prior_z, not the model's own z
    assert captured["approx"] == WavePrecomp(catalog_z_range=(1.0, 1.11))


@pytest.mark.parametrize(
    "extra, message",
    [(["--route", "nss"], "laplace"), (["--set", "named_all"], "factorial")],
)
def test_batch_refuses_other_routes_and_model_sets(monkeypatch, capsys, extra, message):
    key = bk.model_key(MODEL)
    monkeypatch.setattr(
        be,
        "fit_model_over_galaxies",
        lambda *a, **k: pytest.fail("batch must not fit a refused combination"),
    )
    monkeypatch.setattr(sys, "argv", ["bma_evidence.py", "--batch-model", key, *extra])
    with pytest.raises(SystemExit) as exc:
        be.main()
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "--batch-model" in err and message in err


def test_batch_cli_runs_the_default_locked_sample(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_fit(model_dict, galaxies, **kwargs):
        calls.append({"galaxies": list(galaxies), **kwargs})
        return {}

    key = bk.model_key(MODEL)
    monkeypatch.setattr(be, "fit_model_over_galaxies", fake_fit)
    monkeypatch.setattr(be, "DEFAULT_OUT", {**be.DEFAULT_OUT, "laplace": str(tmp_path)})
    monkeypatch.setattr(
        sys, "argv", ["bma_evidence.py", "--batch-model", key, "--set", "factorial"]
    )
    assert be.main() == 0
    assert len(calls) == 1
    assert len(calls[0]["galaxies"]) == 20
    assert all(isinstance(g, int) for g in calls[0]["galaxies"])


def test_batch_cli_takes_an_explicit_galaxy_list(monkeypatch, tmp_path):
    calls: list[dict] = []
    key = bk.model_key(MODEL)
    monkeypatch.setattr(
        be,
        "fit_model_over_galaxies",
        lambda model_dict, galaxies, **kw: calls.append(list(galaxies)) or {},
    )
    monkeypatch.setattr(be, "DEFAULT_OUT", {**be.DEFAULT_OUT, "laplace": str(tmp_path)})
    monkeypatch.setattr(
        sys,
        "argv",
        ["bma_evidence.py", "--batch-model", key, "--galaxies", "79,267"],
    )
    assert be.main() == 0
    assert calls == [[79, 267]]

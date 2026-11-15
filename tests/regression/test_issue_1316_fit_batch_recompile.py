# SPDX-License-Identifier: BSD-3-Clause
"""#1316: fit_batch(redshift_col=...) serves per-galaxy redshift through one tabulated model.

The old behavior rebuilt the model at every row's redshift (one compile per galaxy) and
warned about it. The row's redshift is now a runtime input of one model whose
catalog_z_range covers the table, so no per-row rebuild happens and there is no
recompile warning. The per-row checkpoint loop (``output_dir``) is the path that used
to rebuild; these tests pin that it no longer does.
"""

import warnings

import pytest

pytestmark = pytest.mark.regression_bug


def _catalog():
    return [
        {"z": 0.02, "flux_0": 1.0, "flux_1": 2.0, "flux_2": 3.0,
         "err_0": 0.1, "err_1": 0.2, "err_2": 0.3},
        {"z": 0.05, "flux_0": 1.5, "flux_1": 2.5, "flux_2": 3.5,
         "err_0": 0.1, "err_1": 0.2, "err_2": 0.3},
        {"z": 0.1, "flux_0": 2.0, "flux_1": 3.0, "flux_2": 4.0,
         "err_0": 0.1, "err_1": 0.2, "err_2": 0.3},
    ]  # fmt: skip


def _model(synthetic_ssp, simple_observation):
    from tengri import Fixed, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl"},
        redshift=Fixed(0.0),
    )


def test_no_recompile_warning_and_no_per_row_rebuild(
    synthetic_ssp, simple_observation, tmp_path, monkeypatch
):
    """Per-row redshift is a runtime override: ``with_fixed_redshift`` is never called."""
    from tengri import SEDModel
    from tengri.forward import convenience

    model = _model(synthetic_ssp, simple_observation)

    def _refuse_rebuild(self, redshift):
        raise AssertionError(f"per-row rebuild at z={redshift} (the recompile path)")

    monkeypatch.setattr(SEDModel, "with_fixed_redshift", _refuse_rebuild)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        posts = convenience.fit_batch(
            model,
            _catalog(),
            flux_cols=["flux_0", "flux_1", "flux_2"],
            err_cols=["err_0", "err_1", "err_2"],
            redshift_col="z",
            method="map",
            n_steps=2,
            verbose=False,
            output_dir=str(tmp_path),
        )
    recompile = [w for w in caught if "recompiles" in str(w.message)]
    assert recompile == [], [str(w.message) for w in recompile]
    assert len(posts) == 3
    assert model._catalog_z_range is None, "the caller's model was mutated"

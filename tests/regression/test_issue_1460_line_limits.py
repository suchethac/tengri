# SPDX-License-Identifier: BSD-3-Clause
"""#1460: emission-line limits must survive the ``Data`` seam.

``Data.lines`` accepted only ``{name: (value, err)}``, so an emission-line
upper limit could not be expressed at all — and the rebuild in
``ForwardModel.fit`` (``forward/forward_model.py``) read only ``[0]`` and
``[1]`` before ``dataclasses.replace``-ing the schema's ``line_fluxes``,
which **silently discarded** any ``is_upper_limit`` the user had declared on
the ``Observation``.

Downstream that is silent, not loud, in exactly the way #1321 describes for
the photometry half: ``LineFluxData.limit_mask`` returns ``None`` when no
flag is set, so ``Fitter._build_data_args`` never adds
``line_flux_limit_mask``, and ``inference/likelihood.py`` takes the plain
Gaussian branch. A non-detection is then fit as a measurement, pulling the
model toward flux the galaxy demonstrably does not have.

The vocabulary here is the one ``LineFluxData.from_dict`` already uses —
``{name: (flux, err[, "upper"|"lower"])}`` — so there is one spelling for a
line limit, not two.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _line_obs(base, *, declared_limit: bool):
    """An Observation declaring Halpha (optionally flagged as an upper limit)."""
    from tengri.observation.line_flux_data import LineFluxData
    from tengri.observation.line_list import LineList

    lfd = LineFluxData(
        names=("Halpha",),
        wavelengths=jnp.asarray([6564.61]),
        fluxes=jnp.asarray([1.0e-16]),
        errors=jnp.asarray([1.0e-17]),
        is_upper_limit=jnp.asarray([True]) if declared_limit else None,
    )
    return dataclasses.replace(base, lines=LineList.from_names(["Halpha"]), line_fluxes=lfd)


def _bind(forward, data):
    """Run the real fit path, capturing the Fitter without running the fit."""
    import tengri.inference.fitter as fitmod

    real = fitmod.Fitter
    captured = {}

    class _Spy(real):
        def __init__(self, model, *a, **kw):
            super().__init__(model, *a, **kw)
            captured["fitter"] = self
            captured["observation"] = getattr(model, "observation", None)
            captured["model"] = model
            raise RuntimeError("stop-after-binding")

    fitmod.Fitter = _Spy
    try:
        forward.fit(data, method="map")
    except RuntimeError as exc:
        if "stop-after-binding" not in str(exc):
            raise
    finally:
        fitmod.Fitter = real
    return captured


@pytest.fixture
def forward_with_lines(synthetic_ssp, synthetic_tophat_obs):
    from tengri import FIXED, Fixed, ForwardModel, SEDModel

    obs = _line_obs(synthetic_tophat_obs, declared_limit=False)
    sed = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        redshift=Fixed(0.1),
    )
    return ForwardModel.build(sed=sed, observation=obs)


# --------------------------------------------------------------------------
# The seam itself
# --------------------------------------------------------------------------


def test_data_lines_accepts_a_limit_marker() -> None:
    """``(value, err, 'upper')`` must validate — the from_dict spelling."""
    from tengri import Data
    from tengri.observation.line_list import LineList

    class _Obs:
        lines = LineList.from_names(["Halpha"])

    v = Data(lines={"Halpha": (1.0e-16, 1.0e-17, "upper")}).validate_against(_Obs())
    assert v.line_values["Halpha"][2] == "upper"


@pytest.mark.parametrize("marker", ["UPPER", "up", "limit", 1, True])
def test_invalid_limit_marker_is_rejected(marker) -> None:
    """A bad marker must teach, not leak ``too many values to unpack``."""
    from tengri import Data
    from tengri.observation.line_list import LineList

    class _Obs:
        lines = LineList.from_names(["Halpha"])

    with pytest.raises(ValueError, match=r"(?i)upper.*lower|lower.*upper") as exc:
        Data(lines={"Halpha": (1.0e-16, 1.0e-17, marker)}).validate_against(_Obs())
    assert "Halpha" in str(exc.value), "the error must name the offending line"


# --------------------------------------------------------------------------
# End to end: the flag has to reach the objective, not just the record
# --------------------------------------------------------------------------


def test_limit_declared_in_data_reaches_the_fitter(forward_with_lines) -> None:
    """A limit passed via Data must arrive as a flag on the bound Observation."""
    from tengri import Data

    n_phot = int(forward_with_lines.observation.n_data_phot)
    data = Data(
        photometry=(np.full(n_phot, 1e-17), np.full(n_phot, 1e-18)),
        lines={"Halpha": (1.0e-16, 1.0e-17, "upper")},
    )
    bound = _bind(forward_with_lines, data)["observation"]
    flags = bound.line_fluxes.is_upper_limit
    assert flags is not None, "the upper-limit flag never reached the Fitter"
    assert bool(np.asarray(flags)[0]), "Halpha was not flagged as an upper limit"


def test_limit_mask_reaches_the_objective(forward_with_lines) -> None:
    """``line_flux_limit_mask`` is the gate that selects the censored term.

    ``inference/likelihood.py`` branches on the presence of this key; without
    it the plain Gaussian likelihood runs and the limit is a detection.
    """
    from tengri import Data

    n_phot = int(forward_with_lines.observation.n_data_phot)
    data = Data(
        photometry=(np.full(n_phot, 1e-17), np.full(n_phot, 1e-18)),
        lines={"Halpha": (1.0e-16, 1.0e-17, "upper")},
    )
    cap = _bind(forward_with_lines, data)
    args = cap["fitter"]._build_data_args(cap["model"])
    assert "line_flux_limit_mask" in args, (
        "the censored-likelihood gate is absent — the fit runs the plain "
        "Gaussian term and treats the limit as a detection"
    )
    assert float(np.asarray(args["line_flux_limit_mask"])[0]) == 1.0


# --------------------------------------------------------------------------
# The silent-discard half
# --------------------------------------------------------------------------


def test_schema_declared_limits_are_not_silently_discarded(
    synthetic_ssp, synthetic_tophat_obs
) -> None:
    """Limits on the Observation must not vanish when Data.lines omits them.

    The deprecation on ``Observation(line_fluxes=...)`` sends users to
    ``Data(lines=...)``; before #1460 that migration dropped their flags with
    no warning. Either the flags survive or the user is told — never silence.
    """
    from tengri import FIXED, Data, Fixed, ForwardModel, SEDModel

    obs = _line_obs(synthetic_tophat_obs, declared_limit=True)
    sed = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        redshift=Fixed(0.1),
    )
    forward = ForwardModel.build(sed=sed, observation=obs)
    n_phot = int(forward.observation.n_data_phot)
    data = Data(
        photometry=(np.full(n_phot, 1e-17), np.full(n_phot, 1e-18)),
        lines={"Halpha": (1.0e-16, 1.0e-17)},  # no marker
    )

    with pytest.raises(ValueError, match=r"(?i)upper|limit") as exc:
        _bind(forward, data)
    msg = str(exc.value)
    assert "Halpha" in msg, "the error must name the line whose flag would be lost"
    assert "Data" in msg, "the error must say where to put the flag instead"

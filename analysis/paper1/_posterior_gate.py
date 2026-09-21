# SPDX-License-Identifier: BSD-3-Clause
"""May a mock posterior carry the paper's figure filename?

Split out of ``fig01_mock_joint_infer.py`` so it can be tested without
importing tengri, JAX and matplotlib, the same reason ``_adoption.py`` and
``_figure_style.py`` are separate. Nothing here imports anything beyond the
standard library.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The convergence bar a posterior must clear before this figure may carry the
#: paper's filename -- the same three numbers Section 7 adopts a CANDELS cell
#: on, applied here to the mock.
#:
#: ``GATE_ESS_MIN`` is the effective sample size below which posterior
#: quantiles are not trustworthy (Vehtari et al. 2021, "Rank-normalization,
#: folding, and localization", Bayesian Analysis 16, 667; arXiv:1903.08008).
GATE_RHAT_MAX = 1.01
GATE_ESS_MIN = 400.0
GATE_MAX_DIVERGENCES = 0


def posterior_gate(npz_path: Path) -> tuple[bool, list[str], dict]:
    """Does this posterior clear the bar that lets the figure be published?

    Reads the sidecar ``.json`` the sampler writes beside the draws and returns
    ``(passed, reasons, diagnostics)``. ``reasons`` is empty exactly when
    ``passed``.

    **A missing diagnostic fails, and that is the whole point.** Written the
    obvious way -- ``if rhat > GATE_RHAT_MAX: refuse`` -- a ``None`` compares
    false and sails through, because to that comparison an unrecorded
    diagnostic and a passing one are the same value. This is not hypothetical:
    the posterior on disk when this gate was written carries
    ``rhat_max = None``, having been produced before ``fit_mock_joint.py``
    called the R-hat accessor by its real name, so its R-hat was never written
    down. Recomputed from its own draws it is 1.0550 -- the failing side of
    this bar. The one file that most needed refusing is precisely the one a
    presence-blind check would have waved through under the paper's filename.
    """
    sidecar = npz_path.with_suffix(".json")
    if not sidecar.is_file():
        return False, [f"no diagnostics beside the draws ({sidecar.name} is missing)"], {}
    try:
        diagnostics = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"{sidecar.name} cannot be read: {exc}"], {}

    reasons: list[str] = []

    rhat = diagnostics.get("rhat_max")
    if rhat is None:
        reasons.append("rhat_max was never recorded, so convergence is unverified")
    elif float(rhat) >= GATE_RHAT_MAX:
        reasons.append(f"rhat_max {float(rhat):.4f} >= {GATE_RHAT_MAX}")

    ess = diagnostics.get("ess_min")
    if ess is None:
        reasons.append("ess_min was never recorded")
    elif float(ess) < GATE_ESS_MIN:
        reasons.append(f"ess_min {float(ess):.1f} < {GATE_ESS_MIN:.0f}")

    divergences = diagnostics.get("divergences")
    if divergences is None:
        reasons.append("the divergence count was never recorded")
    elif int(divergences) > GATE_MAX_DIVERGENCES:
        reasons.append(f"{int(divergences)} divergences > {GATE_MAX_DIVERGENCES}")

    return not reasons, reasons, diagnostics


#: The name the paper's \includegraphics points at. Only a posterior that
#: clears the gate may carry it.
PUBLISHED_NAME = "fig01_mock_joint_infer.pdf"


def figure_name(have_posterior: bool, gate_passed: bool) -> str:
    """Which filename this render has earned.

    Three states, and the distinction between the last two is the point. With
    no posterior the figure is truth only; with one that misses the bar it is a
    real inference that is not yet publishable; with one that clears it, it is
    the paper's figure. An incomplete figure carrying the final name is one
    ``\\includegraphics`` away from being published as the real thing, and
    nothing downstream would notice, because the name is the only thing the
    manuscript reads.
    """
    if not have_posterior:
        return "fig01_mock_joint_infer_truthonly.pdf"
    if not gate_passed:
        return "fig01_mock_joint_infer_provisional.pdf"
    return PUBLISHED_NAME

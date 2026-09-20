# SPDX-License-Identifier: BSD-3-Clause
"""Is the precomputed path good enough for the mock the paper fits?

The mock is fit with ``approx=WavePrecomp()``, so every number the
demonstration reports is a number produced through an approximation. #2439
measured that approximation wrong by up to 915% in a band where nebular
emission dominates, and this mock's nebular emission is ~38% of the optical
photometry, so the question is not academic.

The scale that matters is the measurement error, not percent. A 2% bias on a
band with 0.5% errors is a 4-sigma systematic that the fit has no way to
recognize as a systematic: it will absorb it into whichever parameters can
reproduce it, silently and with tight credible intervals. Percent alone cannot
say whether that has happened; ``error / sigma`` can.

Run::

    python -m paper1.verify_mock_precompute

Writes ``results/mock_precompute_bias.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import tengri
from tengri import SEDModel

from . import verify_mock_listing as V
from .fig_mock_joint_infer import TRUTH_NPZ

OUT_JSON = Path(__file__).parent / "results" / "mock_precompute_bias.json"

#: Above this, in units of the measurement error, the approximation is a
#: systematic the fit would absorb into parameters rather than a rounding
#: detail. Well below the ~1 sigma at which a bias starts competing with the
#: noise the posterior is shaped by.
BIAS_TOLERANCE_SIGMA = 0.25


def build_pair():
    """The mock as fit (``WavePrecomp``) and the same model exactly.

    The exact arm is built by forcing ``approx=None`` through the SAME builder
    rather than by writing a second configuration: a hand-copied reference
    model can differ from the one under test in some way nobody intended, and
    then the comparison measures the difference between two configurations
    instead of the cost of the approximation.
    """
    ssp = tengri.load_ssp(V.SSP_NAME)
    obs = V.build_joint_observation()
    lut = V.build_mock_model(ssp, obs)

    original = SEDModel.build

    def forced_exact(**kwargs):
        kwargs["approx"] = None
        return original(**kwargs)

    SEDModel.build = forced_exact
    try:
        exact = V.build_mock_model(ssp, obs)
    finally:
        SEDModel.build = original
    return lut, exact


def main() -> int:
    if not TRUTH_NPZ.exists():
        print(f"no mock at {TRUTH_NPZ}; run `python -m paper1.fig_mock_joint_infer` first")
        return 1
    with np.load(TRUTH_NPZ, allow_pickle=True) as handle:
        truth_npz = {k: handle[k] for k in handle.files}

    truth = dict(
        zip(
            [str(x) for x in truth_npz["free_params"]],
            [float(v) for v in truth_npz["truth_values"]],
            strict=True,
        )
    )
    detected = np.asarray(truth_npz["detected"], dtype=bool)
    phot_sig = np.asarray(truth_npz["phot_sig"])
    spec_sig = np.asarray(truth_npz["spec_sig"])
    filters = [str(f) for f in truth_npz["filters"]]

    lut, exact = build_pair()
    p_lut = np.asarray(lut.predict_photometry(truth))
    p_exact = np.asarray(exact.predict_photometry(truth))
    s_lut = np.asarray(lut.predict(truth).spectrum())
    s_exact = np.asarray(exact.predict(truth).spectrum())

    rows = []
    print(f"{'band':<14}{'exact':>12}{'LUT':>12}{'err %':>10}{'err/sigma':>12}  detected")
    print("-" * 68)
    for i, name in enumerate(filters):
        if p_exact[i] <= 0:
            continue
        pct = 100.0 * (p_lut[i] - p_exact[i]) / p_exact[i]
        nsig = float((p_lut[i] - p_exact[i]) / phot_sig[i])
        rows.append(
            {"band": name, "err_pct": pct, "err_sigma": nsig, "detected": bool(detected[i])}
        )
        print(
            f"{name:<14}{p_exact[i]:>12.3e}{p_lut[i]:>12.3e}{pct:>9.3f}%{nsig:>12.3f}  {detected[i]}"
        )

    spec_sigma = (s_lut - s_exact) / spec_sig
    worst = max(abs(r["err_sigma"]) for r in rows if r["detected"])
    chi2_added = float(
        np.sum(((p_lut[detected] - p_exact[detected]) / phot_sig[detected]) ** 2)
        + np.sum(spec_sigma**2)
    )
    print(
        f"\nspectrum: max |err|/sigma {np.max(np.abs(spec_sigma)):.3f}, "
        f"rms {np.sqrt(np.mean(spec_sigma**2)):.3f}"
    )
    print(f"worst detected band: {worst:.3f} sigma (tolerance {BIAS_TOLERANCE_SIGMA})")
    print(f"chi2 contributed by the approximation alone: {chi2_added:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "bands": rows,
                "worst_detected_sigma": worst,
                "spectrum_max_sigma": float(np.max(np.abs(spec_sigma))),
                "chi2_added": chi2_added,
                "tolerance_sigma": BIAS_TOLERANCE_SIGMA,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_JSON}")

    if worst > BIAS_TOLERANCE_SIGMA:
        print(
            f"\nFAIL: the approximation moves a detected band by {worst:.3f} sigma. "
            "Refit with approx=None, or report the demonstration's recovery as "
            "conditional on the precomputed path."
        )
        return 1
    print("\n[ok] the precomputed path is not a systematic at this mock's noise level")
    return 0


if __name__ == "__main__":
    sys.exit(main())

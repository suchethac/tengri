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
from tengri import SEDModel, SpectrumPrecomp, WavePrecomp

from . import verify_mock_listing as V
from .fig_mock_joint_infer import TRUTH_NPZ

OUT_JSON = Path(__file__).parent / "results" / "mock_precompute_bias.json"

#: Above this, in units of the measurement error, the approximation is a
#: systematic the fit would absorb into parameters rather than a rounding
#: detail. Well below the ~1 sigma at which a bias starts competing with the
#: noise the posterior is shaped by.
BIAS_TOLERANCE_SIGMA = 0.25


def approximated_channels(approx) -> set[str]:
    """Which observation channels ``approx`` actually replaces with a LUT.

    ``WavePrecomp`` is the photometric LUT and ``SpectrumPrecomp`` the
    spectroscopic one; a joint model needs both to approximate both channels.
    Read this off the value the builder passed, never assume it: a guard that
    reports a channel the model never approximated reports a difference that is
    zero for a structural reason, and a structural zero is indistinguishable,
    on the page, from a measured one.
    """
    if approx is None:
        return set()
    configs = approx if isinstance(approx, tuple | list) else (approx,)
    channels: set[str] = set()
    for cfg in configs:
        if isinstance(cfg, WavePrecomp):
            channels.add("photometry")
        elif isinstance(cfg, SpectrumPrecomp):
            channels.add("spectrum")
    return channels


def spectrum_bias(channels, s_lut, s_exact, spec_sig):
    """The spectrum arm's bias, or ``None`` when there is no spectrum LUT.

    Returns ``(max_sigma, chi2, note)``. ``max_sigma`` is ``None`` -- not 0.0 --
    when the model approximates photometry only: both arms then evaluate the
    spectrum on the exact path, so their difference cannot be anything but
    zero and nothing about the spectrum has been checked.
    """
    if "spectrum" not in channels:
        return (
            None,
            0.0,
            "spectrum: not approximated (no SpectrumPrecomp), so both arms compute "
            "it on the exact path; nothing about the spectrum is measured here",
        )
    resid = (np.asarray(s_lut) - np.asarray(s_exact)) / np.asarray(spec_sig)
    worst = float(np.max(np.abs(resid)))
    return (
        worst,
        float(np.sum(resid**2)),
        f"spectrum: max |err|/sigma {worst:.3f}, rms {np.sqrt(np.mean(resid**2)):.3f}",
    )


def build_pair():
    """The mock as fit (``WavePrecomp``) and the same model exactly.

    The exact arm is built by forcing ``approx=None`` through the SAME builder
    rather than by writing a second configuration: a hand-copied reference
    model can differ from the one under test in some way nobody intended, and
    then the comparison measures the difference between two configurations
    instead of the cost of the approximation.

    Returns ``(lut, exact, approx)``, where ``approx`` is the value the builder
    actually passed. The channel report below is derived from it rather than
    restated here, so it cannot drift away from the model under test.
    """
    ssp = tengri.load_ssp(V.SSP_NAME)
    obs = V.build_joint_observation()

    original = SEDModel.build
    seen: list = []

    def capture(**kwargs):
        seen.append(kwargs.get("approx"))
        return original(**kwargs)

    SEDModel.build = capture
    try:
        lut = V.build_mock_model(ssp, obs)
    finally:
        SEDModel.build = original
    if len(seen) != 1:
        raise SystemExit(
            f"the mock builder called SEDModel.build {len(seen)} times; this guard "
            "reads the approximation off a single call and cannot say which model "
            "the numbers below belong to."
        )

    def forced_exact(**kwargs):
        kwargs["approx"] = None
        return original(**kwargs)

    SEDModel.build = forced_exact
    try:
        exact = V.build_mock_model(ssp, obs)
    finally:
        SEDModel.build = original
    return lut, exact, seen[0]


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

    lut, exact, approx = build_pair()
    channels = approximated_channels(approx)
    if "photometry" not in channels:
        print(
            f"FAIL: the mock was built with approx={approx!r}, which carries no "
            "WavePrecomp. Every band difference below would then be zero because "
            "both arms take the same path, and this guard would pass without "
            "having measured the photometric LUT at all."
        )
        return 1
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

    spectrum_max, spectrum_chi2, spectrum_note = spectrum_bias(channels, s_lut, s_exact, spec_sig)
    worst = max(abs(r["err_sigma"]) for r in rows if r["detected"])
    chi2_added = float(
        np.sum(((p_lut[detected] - p_exact[detected]) / phot_sig[detected]) ** 2) + spectrum_chi2
    )
    print(f"\napproximated channels: {', '.join(sorted(channels))}")
    print(spectrum_note)
    print(f"worst detected band: {worst:.3f} sigma (tolerance {BIAS_TOLERANCE_SIGMA})")
    print(f"chi2 contributed by the approximation alone: {chi2_added:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "bands": rows,
                "worst_detected_sigma": worst,
                "approximated_channels": sorted(channels),
                "spectrum_max_sigma": spectrum_max,
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

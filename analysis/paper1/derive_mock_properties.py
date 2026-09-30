# SPDX-License-Identifier: BSD-3-Clause
"""Recovery of the mock's derived properties, in units of the posterior's own width.

Section 3 asks how closely the joint fit recovers the known truth in stellar
mass, star formation rate and AGN bolometric luminosity. Its source comment
names ``mock_joint_mcmc_nuts.npz`` against ``mock_joint_truth.npz`` -- and that
pair cannot answer it. Only one of the three is a free parameter.
``agn_log_lbol`` is sampled directly, but the continuity SFH carries log-SFR
*ratios* plus a total mass, so the star formation rate is derived, and the
truth file stores ``free_params``/``truth_values`` and nothing else. Neither
file contains an SFR or a surviving stellar mass.

They are recoverable without refitting: every draw of all 36 free parameters is
on disk, so the properties can be evaluated afterwards. This does that.

**Formed mass and surviving mass are both reported, and the paper must say
which it means.** ``stellar_mass`` in tengri is the time-integral of the SFH;
``stellar_mass_surviving`` subtracts what the population has returned. On this
mock's truth they differ by 0.195 dex -- larger than the recovery offsets this
script exists to measure, and the quantity published codes report is the
surviving one.

**The offset is quoted in sigma and in dex, and the width is printed beside
both.** A posterior wide enough to cover anything recovers the truth to a
fraction of a sigma while saying nothing, and sigma alone cannot distinguish
that from a sharp posterior centered correctly.

CLI::

    python -m paper1.derive_mock_properties [--max-draws N] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_POSTERIOR = RESULTS / "mock_joint_mcmc_nuts.npz"
DEFAULT_TRUTH = RESULTS / "mock_joint_truth.npz"
DEFAULT_OUT = RESULTS / "mock_joint_recovery.json"

#: Evaluated per draw through the forward model.
DERIVED = ("stellar_mass", "stellar_mass_surviving", "sfr_100myr")

#: Read straight off the draws: these are sampled, not derived.
SAMPLED = ("agn_log_lbol",)

#: Enough to pin a median and a width at the ESS the publication gate demands
#: (400). Every draw costs a full forward evaluation of the kitchen-sink model,
#: about 1.5 s, so the whole chain would be an hour for no extra information.
DEFAULT_MAX_DRAWS = 400


def _properties(model, params: dict, names: tuple[str, ...]) -> dict:
    """Derived properties, or a refusal naming what the model did publish.

    No defaults. A missing ``sfr_100myr`` silently standing in as 1.0 would
    reach Section 3 as log SFR = 0.0, an unremarkable galaxy and therefore
    invisible; the backend sweep carried exactly that default until it was
    removed.
    """
    published = model.predict_properties(params, names=names)
    missing = [n for n in names if n not in published]
    if missing:
        raise SystemExit(
            f"the model published no {missing}; this script will not substitute "
            f"values for them. Available: {sorted(published)}"
        )
    return {n: float(published[n]) for n in names}


def _thin(n_total: int, max_draws: int) -> np.ndarray:
    """Evenly spaced indices, deterministic, spanning the whole chain."""
    if n_total <= max_draws:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, max_draws).round().astype(int)


def summarize(truth_value: float, draws: np.ndarray) -> dict:
    """Recovery of one quantity, stated three ways so none can hide the others."""
    median = float(np.median(draws))
    p16, p84 = (float(x) for x in np.percentile(draws, [16, 84]))
    width = 0.5 * (p84 - p16)
    offset = median - truth_value
    return {
        "truth": truth_value,
        "median": median,
        "p16": p16,
        "p84": p84,
        "width": width,
        "offset": offset,
        # A width of zero would make this infinite; report None rather than a
        # number, because "recovered to inf sigma" is not a measurement.
        "offset_in_sigma": (offset / width) if width > 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--max-draws", type=int, default=DEFAULT_MAX_DRAWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    for path in (args.posterior, args.truth):
        if not path.is_file():
            raise SystemExit(f"missing {path}; run the fit before deriving from it")

    import tengri  # deferred so --help costs nothing

    from . import verify_mock_listing as V
    from ._provenance import code_provenance, provenance_line, publishable

    truth_npz = np.load(args.truth, allow_pickle=True)
    post_npz = np.load(args.posterior, allow_pickle=True)

    truth_names = [str(x) for x in truth_npz["free_params"]]
    post_names = [str(x) for x in post_npz["free_params"]]
    if truth_names != post_names:
        raise SystemExit(
            "the truth and the posterior declare different free parameters, so "
            "their values cannot be paired:\n"
            f"  only in truth:     {sorted(set(truth_names) - set(post_names))}\n"
            f"  only in posterior: {sorted(set(post_names) - set(truth_names))}"
        )

    truth_params = {
        n: float(v) for n, v in zip(truth_names, truth_npz["truth_values"], strict=True)
    }

    ssp = tengri.load_ssp(V.SSP_NAME)
    model = V.build_mock_model(ssp, V.build_joint_observation())

    truth_props = _properties(model, truth_params, DERIVED)

    n_total = len(post_npz[truth_names[0]])
    index = _thin(n_total, args.max_draws)
    print(f"evaluating {len(index)} of {n_total} draws")

    columns: dict[str, list[float]] = {name: [] for name in DERIVED}
    for count, i in enumerate(index, start=1):
        draw = {n: float(post_npz[n][i]) for n in truth_names}
        props = _properties(model, draw, DERIVED)
        for name in DERIVED:
            columns[name].append(props[name])
        if count % 50 == 0:
            print(f"  {count}/{len(index)}")

    # Formed mass is the SFH's time-integral and surviving mass subtracts what
    # the population returned, so one cannot exceed the other. If it does, the
    # draws and the properties have been paired wrongly.
    formed = np.asarray(columns["stellar_mass"])
    survived = np.asarray(columns["stellar_mass_surviving"])
    if not np.all(formed >= survived):
        raise SystemExit(
            f"surviving mass exceeds formed mass in "
            f"{int(np.sum(survived > formed))} of {len(formed)} draws; the "
            "pairing between draws and properties is wrong"
        )

    report: dict = {}
    for name in DERIVED:
        # Masses and SFR are reported in dex, which is how Section 3 quotes
        # them and the scale on which an offset is comparable across them.
        report[f"log_{name}"] = summarize(
            float(np.log10(truth_props[name])), np.log10(np.asarray(columns[name]))
        )
    for name in SAMPLED:
        if name not in post_names:
            raise SystemExit(f"{name} is not among the sampled parameters: {post_names}")
        report[name] = summarize(truth_params[name], np.asarray(post_npz[name])[index])

    provenance = code_provenance(tengri)

    payload = {
        "quantities": report,
        "n_draws_evaluated": len(index),
        "n_draws_available": n_total,
        "formed_minus_surviving_dex_at_truth": float(
            np.log10(truth_props["stellar_mass"]) - np.log10(truth_props["stellar_mass_surviving"])
        ),
        # Published, not raw: the absolute paths belong in the log line
        # below, never in a tracked file. See `_provenance.publishable`.
        "provenance": publishable(provenance),
    }

    width = max(len(k) for k in report)
    print(
        f"\n{'quantity':<{width}} {'truth':>9} {'median':>9} {'offset':>9} {'+-1sig':>8} {'n_sig':>7}"
    )
    print("-" * (width + 46))
    for key, row in report.items():
        nsig = "n/a" if row["offset_in_sigma"] is None else f"{row['offset_in_sigma']:+.2f}"
        print(
            f"{key:<{width}} {row['truth']:>9.4f} {row['median']:>9.4f} "
            f"{row['offset']:>+9.4f} {row['width']:>8.4f} {nsig:>7}"
        )
    print(
        f"\nformed - surviving at truth: {payload['formed_minus_surviving_dex_at_truth']:.4f} dex"
    )
    print(provenance_line(provenance))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

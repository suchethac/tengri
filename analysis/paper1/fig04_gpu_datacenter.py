#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CPU against a datacenter GPU: the main-text fig:gpu.

Reads ``results/sherlock_h100_batch.json``, committed measurements from an
NVIDIA H100 80GB and the same node's Xeon 8462Y+. Nothing here is transcribed:
the previous figure kept its numbers as literals copied out of a bench report,
and the caption then named hardware the literals were not from.

**Two panels, and the second is not decoration.** Dividing wall time by batch
size turns a constant into a 1/n slope, and a 1/n slope reads as scaling. The
H100's per-call time is flat at 13.5-14.0 ms from batch 1 to 2048 -- it
evaluates 2048 galaxies in the wall time it takes for one -- so the whole
measured range is launch-latency bound and the per-galaxy fall is amortization,
not throughput. Panel (b) is where that is visible rather than inferred.

**What this figure may not claim**, from the file's own ``caveats``: no
asymptotic per-galaxy cost. 6.83 us/gal at batch 2048 is a lower bound that is
still falling. And the f64/f32 agreement on the H100 is a latency result, not a
statement about its float64 units.

The ``aa_control`` block records the spread of repeated identical
measurements. Any ratio this figure annotates is checked against it, because a
ratio smaller than the repeat noise is not a measurement.

CLI: python -m analysis.paper1.fig04_gpu_datacenter [--out PDF]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "sherlock_h100_batch.json"
DEFAULT_OUT = HERE / "figures" / "fig04_gpu_datacenter.pdf"

#: Device identity drives hue so a later card cannot repaint these; precision
#: drives the line style. Okabe-Ito, checked for color-vision deficiency.
STYLE = {
    "cpu_f64": ("#0072B2", "-"),
    "cpu_f32": ("#0072B2", "--"),
    "gpu_f64": ("#D55E00", "-"),
    "gpu_f32": ("#D55E00", "--"),
}
ARMS = tuple(STYLE)


def device_labels(payload: dict) -> dict[str, str]:
    """Legend labels from the file's provenance, never hardcoded.

    Two datasets render through this module and they are different
    machines; a literal label here would put an H100 in the legend of the
    consumer figure, which is the exact defect this module exists to end.
    """
    prov = payload.get("provenance") or {}
    cpu, gpu = prov.get("cpu"), prov.get("gpu")
    if not cpu or not gpu:
        raise KeyError("provenance must name both cpu and gpu for the legend")
    return {
        "cpu_f64": f"{cpu}, float64",
        "cpu_f32": f"{cpu}, float32",
        "gpu_f64": f"{gpu}, float64",
        "gpu_f32": f"{gpu}, float32",
    }


def load(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    for required in ("forward", "gradient", "forward_ms_per_call", "aa_control", "caveats"):
        if required not in payload:
            raise KeyError(f"{Path(path).name} has no {required!r} block")
    device_labels(payload)
    return payload


def noise_floor(payload: dict, arm: str) -> float:
    """Worst repeat-to-repeat ratio for one arm, as a fraction above unity."""
    control = payload["aa_control"]
    if control is None:
        raise TypeError("this campaign recorded no A/A control")
    spreads = control.get(arm)
    if not spreads:
        raise KeyError(f"aa_control has no {arm!r}")
    return max(float(v) for v in spreads) - 1.0


def resolvable(payload: dict, ratio: float, *arms: str) -> bool:
    """Is a measured ratio larger than the repeat noise of the arms behind it?

    A ratio inside the noise is not a small effect, it is no effect that has
    been measured, and annotating one would put a number on the figure that a
    rerun could reverse the sign of.
    """
    floor = max(noise_floor(payload, a) for a in arms)
    return abs(ratio - 1.0) > 2.0 * floor


def crossover(batch, cpu, gpu) -> tuple[float, float] | None:
    """The bracketing batches where the GPU first becomes the cheaper arm."""
    cpu, gpu = np.asarray(cpu, float), np.asarray(gpu, float)
    below = np.where(gpu < cpu)[0]
    if not below.size or below[0] == 0:
        return None
    i = int(below[0])
    return float(batch[i - 1]), float(batch[i])


def build(payload: dict) -> tuple[plt.Figure, dict]:
    fwd, per_call = payload["forward"], payload["forward_ms_per_call"]
    batch = np.asarray(fwd["batch"], float)

    fig, (ax_gal, ax_call) = plt.subplots(1, 2, figsize=(7.1, 3.1))
    labels = device_labels(payload)
    for arm in ARMS:
        color, dash = STYLE[arm]
        label = labels[arm]
        ax_gal.loglog(batch, fwd[arm], dash, color=color, marker="o", ms=3, lw=1.2, label=label)
        ax_call.loglog(batch, per_call[arm], dash, color=color, marker="o", ms=3, lw=1.2)

    stats: dict = {}
    # Brackets, not a filled span: the two precisions cross in adjacent
    # octaves, and overlapping spans merge into one gray block that reads as a
    # single wide crossover region rather than as two distinct ones.
    for prec, height in (("f64", 0.90), ("f32", 0.78)):
        span = crossover(batch, fwd[f"cpu_{prec}"], fwd[f"gpu_{prec}"])
        stats[f"crossover_{prec}"] = span
        if span:
            for edge in span:
                ax_gal.axvline(edge, color="0.75", lw=0.6, ls=":", zorder=0)
            ax_gal.annotate(
                f"{prec} crossover\n{int(span[0])}-{int(span[1])}",
                xy=(float(np.sqrt(span[0] * span[1])), height),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=5.5,
                color="0.35",
            )

    i = int(np.argmax(batch))
    # A null aa_control is a DECLARATION that this campaign never measured
    # repeat spread, as distinct from the key being missing, which means
    # someone forgot. Declared-unmeasured renders, but annotates no ratio:
    # without a noise floor there is nothing to say a ratio survives.
    has_control = payload["aa_control"] is not None
    stats["aa_control_recorded"] = has_control
    for device, arms in (("gpu", ("gpu_f64", "gpu_f32")), ("cpu", ("cpu_f64", "cpu_f32"))):
        ratio = float(fwd[arms[0]][i]) / float(fwd[arms[1]][i])
        stats[f"{device}_f64_over_f32"] = ratio
        if not has_control:
            stats[f"{device}_resolvable"] = None
            continue
        ok = resolvable(payload, ratio, *arms)
        stats[f"{device}_resolvable"] = ok
        if not ok:
            raise ValueError(
                f"{device} f64/f32 = {ratio:.4f} is inside the aa_control noise floor; "
                "this figure will not annotate a ratio a rerun could invert"
            )

    # Say that the two H100 curves coincide, as a title rather than an arrow:
    # an annotation inside the axes lands on the legend, and a reader who sees
    # one orange line reasonably concludes an arm failed to plot.
    gpu_ratio = stats["gpu_f64_over_f32"]
    if has_control and abs(gpu_ratio - 1.0) < 0.10:
        ax_gal.set_title(
            f"float32 and float64 coincide on this GPU "
            f"({100 * (gpu_ratio - 1):.1f}% apart at batch {int(batch[-1])})",
            fontsize=6.5,
            color=STYLE["gpu_f64"][0],
        )
    elif not has_control:
        ax_gal.set_title(
            "no A/A repeat control in this campaign; no ratio annotated",
            fontsize=6.5,
            color="0.45",
        )

    flat = np.asarray(per_call["gpu_f64"], float)
    spread = float(flat.max() / flat.min())
    stats["gpu_per_call_spread"] = spread
    note = (
        f"GPU flat: {flat.min():.1f}-{flat.max():.1f} ms\n"
        f"{int(batch[-1])} galaxies cost what 1 does"
        if spread < 1.20
        else f"GPU leaves the latency floor:\n{flat.min():.1f} to {flat.max():.1f} ms per call"
    )
    ax_call.annotate(
        note,
        xy=(batch[len(batch) // 2], float(flat[len(batch) // 2])),
        xytext=(0.06, 0.74),
        textcoords="axes fraction",
        fontsize=5.5,
        color=STYLE["gpu_f64"][0],
        arrowprops={"arrowstyle": "->", "color": STYLE["gpu_f64"][0], "lw": 0.6},
    )

    ax_gal.set_xlabel("batch size (galaxies)")
    ax_gal.set_ylabel(r"forward cost per galaxy  [$\mu$s]")
    ax_call.set_xlabel("batch size (galaxies)")
    ax_call.set_ylabel("wall time per call  [ms]")
    ax_gal.legend(fontsize=6, loc="lower left", frameon=False)
    for ax in (ax_gal, ax_call):
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.tight_layout()
    return fig, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    payload = load(args.data)
    fig, stats = build(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.out}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("  caveats carried by the data file:")
    for c in payload["caveats"]:
        print(f"    - {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

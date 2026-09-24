# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Where a GPU is worth using
#
# Two figures, one for a datacenter card and one for a consumer card, drawn by
# the same renderer from two committed measurement files. Nothing here is
# retyped: the numbers are read from the JSON the benchmark wrote.
#
# Each figure has two panels, and the second is the one that matters. Dividing
# wall time by batch size turns a constant into a $1/n$ slope, and a $1/n$
# slope looks like scaling. The undivided panel shows whether the device is
# actually doing more work as the batch grows.

# %%
import json
import sys
from pathlib import Path

REPO = Path.cwd()
while not (REPO / "analysis" / "paper1").is_dir() and REPO.parent != REPO:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from analysis.paper1 import fig04_gpu_datacenter as gpu

RESULTS = REPO / "analysis" / "paper1" / "results"
OUT = REPO / "paper1_figures" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "fig_gpu_datacenter": RESULTS / "sherlock_h100_batch.json",
    "fig_gpu_consumer": RESULTS / "consumer_gpu_batch.json",
}

# %% [markdown]
# ## Draw both
#
# The renderer refuses a file missing any required block, and refuses to
# annotate a precision ratio that is smaller than the spread of repeated
# identical measurements. A campaign that recorded no such spread declares it
# with an explicit `null`, and its figure is drawn without a ratio rather than
# with an undefendable one.

# %%
summaries = {}
for name, path in DATASETS.items():
    payload = gpu.load(path)
    figure, stats = gpu.build(payload)
    figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    summaries[name] = (payload, stats)
    print(f"{name}: {json.dumps(stats, default=str)}")

# %% [markdown]
# ## What each card is doing
#
# The per-call spread says whether the device left the launch-latency floor
# inside the measured range. A spread near one means every batch cost the same
# wall time, so the per-galaxy fall is amortization of a fixed cost rather than
# throughput, and the smallest per-galaxy number is a lower bound still falling.

# %%
for _name, (payload, stats) in summaries.items():
    prov = payload["provenance"]
    print(f"\n{prov.get('gpu')} with {prov.get('cpu')}, JAX {prov.get('jax')}")
    print(f"  per-call spread over the sweep : {stats['gpu_per_call_spread']:.2f}x")
    print(f"  float64 / float32 on the GPU   : {stats['gpu_f64_over_f32']:.3f}")
    print(f"  repeat control recorded        : {stats['aa_control_recorded']}")
    print(f"  CPU-GPU crossover, float64     : {stats['crossover_f64']}")
    print(f"  CPU-GPU crossover, float32     : {stats['crossover_f32']}")

# %% [markdown]
# ## What these figures may not be used to claim
#
# Carried in the data files themselves, so they travel with the numbers.

# %%
for name, (payload, _stats) in summaries.items():
    print(f"\n{name}:")
    for caveat in payload["caveats"]:
        print(f"  - {caveat}")

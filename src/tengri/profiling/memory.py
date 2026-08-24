# SPDX-License-Identifier: BSD-3-Clause
"""Memory profiling for tengri.

Tracks JAX array footprints (device memory) and process RSS (host memory)
for model components and data structures.

Usage
-----
>>> from tengri.profiling.memory import profile_memory, MemoryReport
>>> report = profile_memory(model)
>>> print(report)
>>> report.to_csv("profiling/outputs/memory.csv")
"""

from __future__ import annotations

import dataclasses
import resource
from typing import Any

# ── Data containers ───────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    """Memory footprint of a single data structure."""

    name: str
    shape: str
    f64_mb: float
    f32_mb: float | None = None
    category: str = ""  # "ssp", "precomp", "nebular", "filter", "dust"


@dataclasses.dataclass
class MemoryReport:
    """Complete memory profiling report."""

    entries: list[MemoryEntry]
    rss_mb: float  # current process RSS

    @property
    def total_f64_mb(self) -> float:
        """Sum of float64 memory across all entries [MB]."""
        return sum(e.f64_mb for e in self.entries)

    @property
    def total_f32_mb(self) -> float:
        """Sum of float32 memory across all entries [MB]."""
        return sum(e.f32_mb for e in self.entries if e.f32_mb is not None)

    def summary(self) -> str:
        """Human-readable memory report."""
        lines = []
        lines.append("MEMORY FOOTPRINT")
        lines.append("=" * 85)
        lines.append(f"{'Data Structure':<45s} {'Shape':<20s} {'f64 (MB)':>10s} {'f32 (MB)':>10s}")
        lines.append("-" * 85)

        # Group by category
        categories = {}
        for e in self.entries:
            cat = e.category or "other"
            categories.setdefault(cat, []).append(e)

        for cat, entries in categories.items():
            if len(categories) > 1:
                lines.append(f"\n  [{cat.upper()}]")
            for e in entries:
                f32_str = f"{e.f32_mb:.3f}" if e.f32_mb is not None else "—"
                lines.append(f"  {e.name:<43s} {e.shape:<20s} {e.f64_mb:>10.3f} {f32_str:>10s}")

        lines.append("-" * 85)
        f32_total = f"{self.total_f32_mb:.1f}" if self.total_f32_mb > 0 else "—"
        lines.append(f"  {'TOTAL':<43s} {'':20s} {self.total_f64_mb:>10.1f} {f32_total:>10s}")
        lines.append(f"\n  Process RSS: {self.rss_mb:.1f} MB")
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write memory report to CSV."""
        import csv

        fieldnames = ["name", "shape", "f64_mb", "f32_mb", "category"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                writer.writerow(
                    {
                        "name": e.name,
                        "shape": e.shape,
                        "f64_mb": f"{e.f64_mb:.3f}",
                        "f32_mb": f"{e.f32_mb:.3f}" if e.f32_mb is not None else "",
                        "category": e.category,
                    }
                )

    def __repr__(self) -> str:
        return self.summary()


# ── Helpers ───────────────────────────────────────────────────────


def _arr_mb(arr: Any) -> float:
    """Memory of a JAX/numpy array in MB."""
    if arr is None:
        return 0.0
    if hasattr(arr, "nbytes"):
        return arr.nbytes / 1e6
    return 0.0


def _shape_str(arr: Any) -> str:
    """Shape string for display."""
    if arr is None:
        return "—"
    if hasattr(arr, "shape"):
        return "×".join(str(s) for s in arr.shape)
    return "—"


def get_rss_mb() -> float:
    """Current process RSS in MB."""
    # macOS returns bytes, Linux returns KB
    import platform

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / 1e6  # bytes → MB
    return usage / 1e3  # KB → MB


# ── Profile model memory ──────────────────────────────────────────


def profile_memory(model) -> MemoryReport:
    """Profile memory footprint of a Model and its data structures.

    Parameters
    ----------
    model : Model
        A tengri Model instance.

    Returns
    -------
    MemoryReport
        Detailed memory breakdown.
    """
    entries = []
    ssp = model.ssp_data

    # --- SSP data ---
    f64 = _arr_mb(ssp.ssp_flux)
    entries.append(
        MemoryEntry(
            name="SSP templates",
            shape=_shape_str(ssp.ssp_flux),
            f64_mb=f64,
            f32_mb=f64 / 2,
            category="ssp",
        )
    )

    entries.append(
        MemoryEntry(
            name="SSP wavelength grid",
            shape=_shape_str(ssp.ssp_wave),
            f64_mb=_arr_mb(ssp.ssp_wave),
            f32_mb=_arr_mb(ssp.ssp_wave) / 2,
            category="ssp",
        )
    )

    entries.append(
        MemoryEntry(
            name="SSP metallicity grid",
            shape=_shape_str(ssp.ssp_lgmet),
            f64_mb=_arr_mb(ssp.ssp_lgmet),
            category="ssp",
        )
    )

    entries.append(
        MemoryEntry(
            name="SSP age grid",
            shape=_shape_str(ssp.ssp_lg_age_gyr),
            f64_mb=_arr_mb(ssp.ssp_lg_age_gyr),
            category="ssp",
        )
    )

    # Alpha-enhanced SSPs (4D)
    if hasattr(ssp, "ssp_alpha_fe") and ssp.ssp_alpha_fe is not None:
        entries.append(
            MemoryEntry(
                name="SSP alpha grid",
                shape=_shape_str(ssp.ssp_alpha_fe),
                f64_mb=_arr_mb(ssp.ssp_alpha_fe),
                category="ssp",
            )
        )

    # --- Precompute LUTs ---
    # The legacy ``PrecomputedData`` container (fixed-z photometry, spectroscopy,
    # z-table, dust-age-weights arrays) was retired (#620). The active LUTs now
    # live on the per-component cached state under ``approx=WavePrecomp()`` /
    # ``approx=SpectrumPrecomp()``; per-array memory reporting for them is a
    # separate follow-up if needed.

    # --- Filters ---
    if model.filter_waves is not None:
        total_filter = 0.0
        for fw_i, ft_i in zip(model.filter_waves, model.filter_trans):
            total_filter += _arr_mb(fw_i) + _arr_mb(ft_i)
        n_filt = len(model.filter_waves)
        entries.append(
            MemoryEntry(
                name=f"Filter curves ({n_filt} bands)",
                shape=f"{n_filt} × ~{len(model.filter_waves[0])} pts",
                f64_mb=total_filter,
                category="filter",
            )
        )

    # --- Nebular (CUE weights) ---
    neb = getattr(model, "_nebular_backend", None)
    if neb is not None and hasattr(neb, "_weights"):
        cue_total = 0.0
        w = neb._weights
        for field_name in w._fields:
            val = getattr(w, field_name)
            if hasattr(val, "nbytes"):
                cue_total += val.nbytes / 1e6
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if hasattr(item, "nbytes"):
                        cue_total += item.nbytes / 1e6
        entries.append(
            MemoryEntry(
                name="CUE neural emulator weights",
                shape="(16 sub-nets + cont.)",
                f64_mb=cue_total,
                category="nebular",
            )
        )

    return MemoryReport(entries=entries, rss_mb=get_rss_mb())


# ── Memory scaling ────────────────────────────────────────────────

"""Benchmark visualization module.

Generates analysis plots from benchmark episode data:
  - Plant comparison: survival time & count per plant type
  - Plant-cell heatmap: distribution of each plant type across the grid
  - Cell statistics: most-planted type per cell, total plant count per cell

All plots use the default matplotlib white-background style.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


PALETTE = [
    "#4ecdc4", "#ff6b6b", "#ffe66d", "#a8e6cf", "#ff8b94",
    "#b8a9c9", "#f0b27a", "#85c1e9", "#82e0aa", "#f1948a",
]


# ═══════════════════════════════════════════════════════════════════════════
# public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_benchmark_plots(
    result: Any,
    output_dir: str,
    rows: int,
    cols: int,
) -> list[str]:
    """Generate all benchmark analysis plots.

    Args:
        result: ``EvaluationResult`` with ``.details`` and ``.extra`` populated.
        output_dir: Directory to write PNG files into.
        rows: Grid row count.
        cols: Grid column count.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    plant_stats = (result.extra or {}).get("plant_stats") or {}
    placements = plant_stats.pop("_placements", []) if isinstance(plant_stats, dict) else []

    # Also collect placements from per-episode details if available
    if not placements:
        for detail in result.details:
            ps = (detail.extra or {}).get("plant_stats") or {}
            if isinstance(ps, dict):
                placements.extend(ps.get("placements", []))

    paths: list[str] = []

    if plant_stats:
        path = _plant_comparison(plant_stats, output_dir)
        if path:
            paths.append(path)

    if placements and rows > 0 and cols > 0:
        path = _plant_cell_heatmaps(placements, rows, cols, output_dir)
        if path:
            paths.append(path)

        path = _cell_statistics(placements, rows, cols, output_dir)
        if path:
            paths.append(path)

    plt.close("all")
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# plant comparison — bar charts
# ═══════════════════════════════════════════════════════════════════════════

def _plant_comparison(
    plant_stats: dict[str, Any],
    output_dir: str,
) -> str | None:
    """Side-by-side bar chart: survival time & plant count per plant type."""
    entries = [
        (v.get("name", k), v.get("survival_steps_mean", 0.0), int(v.get("count_total", 0)))
        for k, v in plant_stats.items()
        if isinstance(v, dict) and not k.startswith("action:") and not k.startswith("_")
    ]
    if not entries:
        return None

    entries.sort(key=lambda x: -x[1])  # sort by survival
    names = [e[0] for e in entries]
    survival_vals = [e[1] for e in entries]
    count_vals = [e[2] for e in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(names) * 0.5)))
    fig.suptitle("Plant Comparison — Survival Time & Placement Count",
                 fontsize=14, fontweight="bold", y=0.98)

    x = range(len(names))
    colours_s = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    colours_c = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    # Left: mean survival steps
    bars1 = ax1.barh(list(x), survival_vals, color=colours_s, height=0.6)
    ax1.set_yticks(list(x))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Mean Survival Steps", fontsize=10)
    ax1.set_title("Survival Time by Plant Type", fontsize=11)
    ax1.invert_yaxis()
    for bar, val in zip(bars1, survival_vals):
        ax1.text(bar.get_width() + ax1.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center", fontsize=8)
    ax1.grid(axis="x", alpha=0.3)

    # Right: count
    bars2 = ax2.barh(list(x), count_vals, color=colours_c, height=0.6)
    ax2.set_yticks(list(x))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Total Planted Count", fontsize=10)
    ax2.set_title("Placement Count by Plant Type", fontsize=11)
    ax2.invert_yaxis()
    for bar, val in zip(bars2, count_vals):
        ax2.text(bar.get_width() + ax2.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "plant_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# plant-cell heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def _plant_cell_heatmaps(
    placements: list[dict[str, int]],
    rows: int,
    cols: int,
    output_dir: str,
) -> str | None:
    """One heatmap subplot per plant type showing placement distribution."""
    if not placements:
        return None

    # Aggregate: plant_type → (row, col) → count
    plant_cells: dict[int, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    for p in placements:
        pt = int(p.get("plant_type", p.get("plant_id", -1)))
        r = int(p.get("row", -1))
        c = int(p.get("col", -1))
        if pt < 0 or r < 0 or c < 0:
            continue
        plant_cells[pt][(r, c)] += 1

    if not plant_cells:
        return None

    plant_types = sorted(plant_cells.keys())
    n_plants = len(plant_types)
    n_cols_plot = min(4, n_plants)
    n_rows_plot = max(1, (n_plants + n_cols_plot - 1) // n_cols_plot)

    fig, axes = plt.subplots(
        n_rows_plot, n_cols_plot,
        figsize=(4 * n_cols_plot + 1, 4 * n_rows_plot + 1),
        squeeze=False,
    )
    fig.suptitle("Plant Placement Distribution Heatmaps",
                 fontsize=14, fontweight="bold", y=0.99)

    # Global colour scale
    all_counts = [v for pc in plant_cells.values() for v in pc.values()]
    vmax = max(all_counts) if all_counts else 1

    # Plant name lookup (hard-coded common ones; extensible)
    plant_names = _plant_name_map()

    for idx, plant_type in enumerate(plant_types):
        ax = axes[idx // n_cols_plot][idx % n_cols_plot]
        grid = np.zeros((rows, cols))
        for (r, c), count in plant_cells[plant_type].items():
            if 0 <= r < rows and 0 <= c < cols:
                grid[r, c] = count

        im = ax.imshow(grid, cmap="YlOrRd", origin="upper", aspect="equal",
                       vmin=0, vmax=vmax)
        name = plant_names.get(plant_type, f"ID {plant_type}")
        ax.set_title(f"{name} (total={int(grid.sum())})", fontsize=10)
        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.tick_params(labelsize=7)

        # Annotate non-zero cells
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                if val > 0:
                    ax.text(c, r, str(int(val)), ha="center", va="center",
                            fontsize=7, color="white" if val > vmax / 2 else "black")

    # Hide unused subplots
    for idx in range(n_plants, n_rows_plot * n_cols_plot):
        axes[idx // n_cols_plot][idx % n_cols_plot].set_visible(False)

    # Shared colour bar
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.84])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Plant Count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    path = os.path.join(output_dir, "plant_cell_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# cell statistics
# ═══════════════════════════════════════════════════════════════════════════

def _cell_statistics(
    placements: list[dict[str, int]],
    rows: int,
    cols: int,
    output_dir: str,
) -> str | None:
    """Two heatmaps: (a) most-planted type per cell, (b) total plant count per cell."""
    if not placements:
        return None

    # Aggregate: (row, col) → {plant_type: count}
    cell_counts: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for p in placements:
        pt = int(p.get("plant_type", p.get("plant_id", -1)))
        r = int(p.get("row", -1))
        c = int(p.get("col", -1))
        if pt < 0 or r < 0 or c < 0:
            continue
        cell_counts[(r, c)][pt] += 1

    if not cell_counts:
        return None

    plant_names = _plant_name_map()

    # Build grids
    dominant_grid = np.full((rows, cols), -1, dtype=int)    # plant_type or -1
    total_grid = np.zeros((rows, cols), dtype=int)           # total plant count
    for (r, c), pcounts in cell_counts.items():
        if 0 <= r < rows and 0 <= c < cols:
            total_grid[r, c] = sum(pcounts.values())
            dominant_grid[r, c] = max(pcounts, key=pcounts.get)

    # Collect all plant types that appear as dominant somewhere
    all_types = sorted(set(int(dominant_grid[r, c]) for r in range(rows) for c in range(cols)
                           if dominant_grid[r, c] >= 0))

    # Build a qualitative colour mapping
    type_colours, type_labels, legend_handles = _build_dominant_legend(all_types, plant_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cell-Level Planting Statistics",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Left: dominant plant per cell ──
    cmap_dominant = matplotlib.colors.ListedColormap(
        ["#ffffff"] + [type_colours.get(t, "#999999") for t in all_types]
    )
    bounds = [-1.5] + [t - 0.5 for t in all_types] + [all_types[-1] + 0.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap_dominant.N)

    masked = np.ma.masked_where(dominant_grid < 0, dominant_grid)
    im1 = ax1.imshow(masked, cmap=cmap_dominant, norm=norm, origin="upper", aspect="equal")
    ax1.set_title("Most-Planted Type per Cell", fontsize=11)
    ax1.set_xlabel("Column", fontsize=9)
    ax1.set_ylabel("Row", fontsize=9)
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(rows))
    ax1.tick_params(labelsize=8)

    # Annotate each cell
    for r in range(rows):
        for c in range(cols):
            pt = int(dominant_grid[r, c])
            if pt >= 0:
                label = plant_names.get(pt, str(pt))[:6]
                ax1.text(c, r, label, ha="center", va="center",
                         fontsize=7, color="white")

    # ── Right: total plant count per cell ──
    vmax2 = max(int(total_grid.max()), 1)
    im2 = ax2.imshow(total_grid, cmap="YlOrRd", origin="upper", aspect="equal",
                     vmin=0, vmax=vmax2)
    ax2.set_title("Total Plants Placed per Cell", fontsize=11)
    ax2.set_xlabel("Column", fontsize=9)
    ax2.set_ylabel("Row", fontsize=9)
    ax2.set_xticks(range(cols))
    ax2.set_yticks(range(rows))
    ax2.tick_params(labelsize=8)

    for r in range(rows):
        for c in range(cols):
            val = int(total_grid[r, c])
            if val > 0:
                ax2.text(c, r, str(val), ha="center", va="center",
                         fontsize=8, color="white" if val > vmax2 / 2 else "black")

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Plant Count", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)

    # Legend for dominant type
    if legend_handles:
        ax1.legend(
            handles=legend_handles,
            title="Plant Types",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title_fontsize=8,
            framealpha=0.8,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "cell_statistics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _plant_name_map() -> dict[int, str]:
    """Mapping from plant type ID to short display name."""
    return {
        0:  "Peashooter",
        1:  "Sunflower",
        2:  "CherryBomb",
        3:  "Wall-nut",
        4:  "PotatoMine",
        5:  "SnowPea",
        6:  "Chomper",
        7:  "Repeater",
        8:  "PuffShroom",
        9:  "SunShroom",
        10: "FumeShroom",
        11: "GraveBuster",
        12: "HypnoShrm",
        13: "ScaredyShrm",
        14: "IceShroom",
        15: "DoomShroom",
        16: "LilyPad",
        17: "Squash",
        18: "Threepeater",
        19: "TangleKelp",
        20: "Jalapeno",
        21: "Spikeweed",
        22: "Torchwood",
        23: "Tall-nut",
        24: "SeaShroom",
        25: "Plantern",
        26: "Cactus",
        27: "Blover",
        28: "SplitPea",
        29: "Starfruit",
        30: "Pumpkin",
        31: "MagnetShrm",
        32: "CabbagePult",
        33: "FlowerPot",
        34: "KernelPult",
        35: "CoffeeBean",
        36: "Garlic",
        37: "Umbrella",
        38: "Marigold",
        39: "MelonPult",
        40: "GatlingPea",
        41: "TwinSunflr",
        42: "GloomShrm",
        43: "Cattail",
        44: "WinterMelon",
        45: "GoldMagnet",
        46: "Spikerock",
        47: "CobCannon",
        48: "Imitater",
    }


def _build_dominant_legend(
    all_types: list[int],
    plant_names: dict[int, str],
) -> tuple[dict[int, str], list[str], list[Any]]:
    """Build colour map, labels, and legend handles for dominant plant types."""
    type_colours: dict[int, str] = {}
    type_labels: list[str] = []
    handles: list[Any] = []
    for i, pt in enumerate(all_types):
        colour = PALETTE[i % len(PALETTE)]
        type_colours[pt] = colour
        name = plant_names.get(pt, f"ID {pt}")
        type_labels.append(name)
        handles.append(
            matplotlib.patches.Patch(color=colour, label=name)
        )
    return type_colours, type_labels, handles

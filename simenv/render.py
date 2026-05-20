"""
SimPVZ episode replay using matplotlib animation.

Usage:
    from simenv.render import replay_episode
    replay_episode(env.render_data, fps=10)
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

# Use a font that supports CJK (fallback to sans-serif)
try:
    matplotlib.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass

# Color scheme
COLORS = {
    "Sunflower": "#FFD700",
    "Peashooter": "#228B22",
    "Wallnut": "#8B4513",
    "Potatomine": "#FF4444",
    "Zombie": "#808080",
    "Zombie_cone": "#FF8C00",
    "Zombie_bucket": "#C0C0C0",
    "Zombie_flag": "#FF6347",
    "Pea": "#32CD32",
    "Mower": "#FF0000",
    "background": "#90C850",
    "grid_line": "#6B8E23",
    "house": "#F5DEB3",
}

# Short display names (ASCII-compatible, no CJK font needed)
NAMES_CN = {
    "Sunflower": "Sunflower",
    "Peashooter": "Peashooter",
    "Wallnut": "Wall-nut",
    "Potatomine": "Mine",
    "Zombie": "Zombie",
    "Zombie_cone": "Cone Zombie",
    "Zombie_bucket": "Bucket Zombie",
    "Zombie_flag": "Flag Zombie",
    "Pea": "Pea",
    "Mower": "Mower",
}


def replay_episode(render_data, fps=10, save_path=None, title="SimPVZ Episode Replay"):
    """
    Replay a captured episode.

    Args:
        render_data: list of frame dicts from SimPVZEnv.render_data
        fps: playback speed (frames per second)
        save_path: if set, save as GIF/MP4 instead of showing
        title: window title
    """
    if not render_data:
        print("No render data to replay.")
        return

    n_lanes = 5
    lane_length = 9
    cell_size = 1.0

    fig, (ax_grid, ax_info) = plt.subplots(
        1, 2, figsize=(14, 7),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(title, fontsize=14)

    # --- Grid panel ---
    ax_grid.set_xlim(-1.0, lane_length + 0.5)
    ax_grid.set_ylim(-0.5, n_lanes)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.invert_yaxis()

    # Background
    ax_grid.add_patch(
        Rectangle((-0.5, -0.5), lane_length + 1, n_lanes,
                   facecolor=COLORS["background"], zorder=0))
    # House zone
    ax_grid.add_patch(
        Rectangle((-1.0, -0.5), 0.5, n_lanes,
                   facecolor=COLORS["house"], edgecolor="black", zorder=0))

    # Draw grid lines
    for i in range(lane_length + 1):
        ax_grid.axvline(i - 0.5, color=COLORS["grid_line"], lw=0.5, zorder=1)
    for j in range(n_lanes + 1):
        ax_grid.axhline(j - 0.5, color=COLORS["grid_line"], lw=0.5, zorder=1)

    # Dynamic elements (created once, updated each frame)
    plant_patches = []
    zombie_patches = []
    projectile_patches = []

    # --- Info panel ---
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    def draw_frame(frame_data):
        # Clear old patches
        for p in plant_patches + zombie_patches + projectile_patches:
            p.remove()
        plant_patches.clear()
        zombie_patches.clear()
        projectile_patches.clear()

        # Draw plants
        for lane in range(n_lanes):
            for name, pos, hp in frame_data["plants"][lane]:
                color = COLORS.get(name, "#00FF00")
                px = pos - 0.35
                py = lane - 0.35
                p = FancyBboxPatch(
                    (px, py), 0.7, 0.7,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="black", lw=0.5, zorder=3)
                ax_grid.add_patch(p)
                plant_patches.append(p)
                # Health bar
                max_hp = {"Sunflower": 300, "Peashooter": 300, "Wallnut": 4000,
                          "Potatomine": 300}.get(name, 300)
                hp_ratio = max(0, hp / max_hp)
                if hp_ratio < 0.99:
                    ax_grid.add_patch(Rectangle(
                        (px, py - 0.1), 0.7, 0.08,
                        facecolor="red", zorder=4))
                    plant_patches.append(ax_grid.patches[-1])
                    ax_grid.add_patch(Rectangle(
                        (px, py - 0.1), 0.7 * hp_ratio, 0.08,
                        facecolor="green", zorder=5))
                    plant_patches.append(ax_grid.patches[-1])

        # Draw zombies
        for lane in range(n_lanes):
            for name, pos, offset, hp in frame_data["zombies"][lane]:
                color = COLORS.get(name, "#808080")
                px = pos + offset - 0.35
                py = lane - 0.25
                z = FancyBboxPatch(
                    (px, py), 0.7, 0.5,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="black", lw=0.5, zorder=3)
                ax_grid.add_patch(z)
                zombie_patches.append(z)

        # Draw projectiles
        for lane in range(n_lanes):
            for name, pos, offset in frame_data["projectiles"][lane]:
                color = COLORS.get(name, "#32CD32")
                px = pos + offset
                py = lane + 0.15
                c = Circle((px, py), 0.12, facecolor=color,
                           edgecolor="black", lw=0.3, zorder=4)
                ax_grid.add_patch(c)
                projectile_patches.append(c)

        # --- Info panel ---
        ax_info.clear()
        ax_info.axis("off")
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        lines = [
            f"Time:  {frame_data.get('time', 0)}s",
            f"Sun:   {frame_data.get('sun', 0)}",
            f"Score: {frame_data.get('score', 0)}",
            f"Lives: {frame_data.get('lives', 1)}",
            "",
            "Cooldowns (s):",
        ]
        cd_map = {"sunflower": "SF", "peashooter": "Pea", "wall-nut": "Wall", "potatomine": "Mine"}
        cooldowns = frame_data.get("cooldowns", {})
        for name, cd in cooldowns.items():
            abbr = cd_map.get(name, name[:4])
            lines.append(f"  {abbr}: {cd}s")
        lines.append("")
        lines.append("Field:")
        for lane in range(n_lanes):
            row_parts = []
            for pname, ppos, php in frame_data["plants"][lane]:
                abbr = cd_map.get(pname.lower(), pname[:4])
                row_parts.append(f"{abbr}@{ppos}")
            for zname, zpos, zoff, zhp in frame_data["zombies"][lane]:
                abbr = {"Zombie": "Z", "Zombie_cone": "ZC", "Zombie_bucket": "ZB", "Zombie_flag": "ZF"}.get(zname, "Z")
                row_parts.append(f"{abbr}@{zpos}")
            if row_parts:
                lines.append(f"  Ln{lane}: {', '.join(row_parts)}")

        for i, line in enumerate(lines):
            y = 0.95 - i * 0.028
            ax_info.text(0.05, y, line, fontsize=10, fontfamily="monospace",
                         verticalalignment="top")

        return plant_patches + zombie_patches + projectile_patches

    # Create animation
    ani = animation.FuncAnimation(
        fig, draw_frame, frames=render_data,
        interval=1000 // fps, blit=False, repeat=False,
    )

    if save_path:
        print(f"Saving replay to {save_path} ...")
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            ani.save(save_path, writer="ffmpeg", fps=fps)
        else:
            ani.save(save_path + ".gif", writer="pillow", fps=fps)
        print("Done.")
    else:
        plt.show()

    plt.close(fig)

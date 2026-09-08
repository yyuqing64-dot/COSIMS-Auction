#!/usr/bin/env python3
"""Fig. 1: COSIMS-Auction dual-time-scale mechanism overview."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Rectangle, Polygon

OUT_DIR = REPO_ROOT / "results" / "figures"
OUT_PNG = OUT_DIR / "fig1_mechanism_overview.png"
OUT_PDF = OUT_DIR / "fig1_mechanism_overview.pdf"

BLUE = "#0B4F9C"
BLUE_LIGHT = "#F4FAFF"
TEAL = "#087A7E"
TEAL_LIGHT = "#F6FEFD"
INK = "#171717"
MUTED = "#666666"
FILL = "#F9FCFF"


def rounded(ax, xy, w, h, text="", fc="white", ec=INK, lw=1.1, r=0.008, fs=9.0, color=INK, weight="normal"):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
                fontsize=fs, color=color, fontweight=weight, linespacing=1.10, zorder=4)
    return box


def arrow(ax, start, end, color=INK, lw=1.2, style="solid", rad=0.0, ms=12):
    arr = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        linestyle=style, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=3, zorder=2,
    )
    ax.add_patch(arr)
    return arr


def line(ax, pts, color=INK, lw=1.1, style="solid", z=1):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=style, zorder=z)


def clock_icon(ax, x, y, color):
    ax.add_patch(Circle((x, y), 0.018, fill=False, edgecolor=color, lw=1.5, linestyle=(0, (3, 2)), zorder=4))
    ax.plot([x, x], [y, y + 0.010], color=color, lw=1.3, zorder=4)
    ax.plot([x, x + 0.010], [y, y], color=color, lw=1.3, zorder=4)
    ax.add_patch(Arc((x + 0.020, y + 0.012), 0.012, 0.012, theta1=150, theta2=260, color=color, lw=1.1, zorder=4))


def trigger_card(ax, x, y, w, h, label, kind):
    rounded(ax, (x, y), w, h, fc=FILL, ec=TEAL, lw=0.9, r=0.006)
    icon_x = x + 0.018
    icon_y = y + h / 2 - 0.012
    if kind == "bars":
        for i, bh in enumerate([0.012, 0.022, 0.032]):
            ax.add_patch(Rectangle((icon_x + i * 0.008, icon_y), 0.004, bh, facecolor="#BFE2E1", edgecolor=TEAL, lw=0.7, zorder=5))
    elif kind == "graph":
        pts = [(icon_x, icon_y + 0.004), (icon_x + 0.010, icon_y + 0.022), (icon_x + 0.020, icon_y + 0.015), (icon_x + 0.030, icon_y + 0.033)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=TEAL, lw=1.0, zorder=5)
        for px, py in pts:
            ax.add_patch(Circle((px, py), 0.0032, facecolor="#BFE2E1", edgecolor=TEAL, lw=0.7, zorder=5))
    else:
        tri = Polygon([(icon_x + 0.016, icon_y + 0.034), (icon_x, icon_y), (icon_x + 0.032, icon_y)], closed=True, facecolor="#BFE2E1", edgecolor=TEAL, lw=0.8, zorder=5)
        ax.add_patch(tri)
        ax.text(icon_x + 0.016, icon_y + 0.013, "!", ha="center", va="center", fontsize=6.5, color=TEAL, fontweight="bold", zorder=6)
    ax.text(x + 0.052, y + h / 2, label, ha="left", va="center", fontsize=6.9, color=INK, linespacing=1.05, zorder=5)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header.
    rounded(ax, (0.350, 0.890), 0.300, 0.075, fc="#FBFBFB", ec=INK, lw=1.1, r=0.010)
    ax.text(0.500, 0.935, "COSIMS-Auction", ha="center", va="center", fontsize=15.0, color=INK, fontweight="bold", zorder=5)
    ax.text(0.500, 0.908, "Dual-time-scale coordination mechanism", ha="center", va="center", fontsize=10.5, color=INK, zorder=5)
    line(ax, [(0.500, 0.890), (0.500, 0.860), (0.270, 0.860)], color=INK, lw=1.1)
    line(ax, [(0.500, 0.860), (0.730, 0.860)], color=INK, lw=1.1)
    arrow(ax, (0.270, 0.860), (0.270, 0.828), color=INK, lw=1.1, ms=9)
    arrow(ax, (0.730, 0.860), (0.730, 0.828), color=INK, lw=1.1, ms=9)

    # Panels.
    rounded(ax, (0.035, 0.515), 0.930, 0.300, fc=BLUE_LIGHT, ec=BLUE, lw=1.15, r=0.010)
    rounded(ax, (0.035, 0.075), 0.930, 0.385, fc=TEAL_LIGHT, ec=TEAL, lw=1.15, r=0.010)

    clock_icon(ax, 0.067, 0.785, BLUE)
    ax.text(0.095, 0.785, "Slow time scale: OT-based global reference planning", ha="left", va="center", fontsize=13.0, color=BLUE, fontweight="bold")
    ax.text(0.858, 0.785, "Updated once per epoch", ha="left", va="center", fontsize=9.2, color=MUTED, style="italic")
    clock_icon(ax, 0.067, 0.430, TEAL)
    ax.text(0.095, 0.430, "Fast time scale: trigger-aware online decision", ha="left", va="center", fontsize=13.0, color=TEAL, fontweight="bold")
    ax.text(0.850, 0.430, "Per-task online scheduling", ha="left", va="center", fontsize=9.2, color=MUTED, style="italic")

    # Slow time scale.
    rounded(ax, (0.070, 0.690), 0.100, 0.052, "Historical\narrivals", fc=FILL, ec=BLUE, fs=9.2)
    rounded(ax, (0.205, 0.690), 0.100, 0.052, "Edge service\nrates", fc=FILL, ec=BLUE, fs=9.2)
    rounded(ax, (0.340, 0.690), 0.100, 0.052, "Propagation\nmatrix", fc=FILL, ec=BLUE, fs=9.2)
    rounded(ax, (0.145, 0.570), 0.215, 0.052, "OT-based global planning", fc=FILL, ec=BLUE, fs=10.0, color=BLUE)
    rounded(ax, (0.452, 0.570), 0.170, 0.052, r"Reference matrix $P^{(e)}$", fc=FILL, ec=BLUE, fs=9.8, color=BLUE)
    rounded(ax, (0.690, 0.570), 0.205, 0.052, r"Top-$K$ candidate sets $C_i^{(e)}$", fc=FILL, ec=BLUE, fs=9.8, color=BLUE)

    line(ax, [(0.120, 0.690), (0.120, 0.665), (0.252, 0.665)], color=BLUE, lw=1.0)
    line(ax, [(0.255, 0.690), (0.255, 0.637)], color=BLUE, lw=1.0)
    line(ax, [(0.390, 0.690), (0.390, 0.665), (0.252, 0.665)], color=BLUE, lw=1.0)
    arrow(ax, (0.252, 0.665), (0.252, 0.622), color=BLUE, lw=1.0)
    arrow(ax, (0.360, 0.596), (0.452, 0.596), color=BLUE, lw=1.0)
    arrow(ax, (0.622, 0.596), (0.690, 0.596), color=BLUE, lw=1.0)

    # Interfaces to fast path.
    line(ax, [(0.537, 0.570), (0.537, 0.525), (0.215, 0.525), (0.215, 0.358)], color=BLUE, lw=1.0, style=(0, (4, 3)))
    arrow(ax, (0.215, 0.358), (0.215, 0.325), color=TEAL, lw=1.0, style=(0, (4, 3)))
    ax.text(0.250, 0.490, "global reference", fontsize=8.4, color=TEAL, ha="left", va="center")
    ax.text(0.185, 0.372, "global\nreference", fontsize=8.1, color=TEAL, ha="center", va="center")
    line(ax, [(0.792, 0.570), (0.792, 0.330)], color=BLUE, lw=1.0, style=(0, (4, 3)))
    arrow(ax, (0.792, 0.330), (0.792, 0.302), color=BLUE, lw=1.0, style=(0, (4, 3)))
    ax.text(0.804, 0.492, "bounded\ncandidate\ninterface", fontsize=8.4, color=BLUE, ha="left", va="center")

    # Fast path, spaced enough for arrows.
    y = 0.265
    rounded(ax, (0.055, y), 0.075, 0.058, "Arriving\ntask $k$", fc=FILL, ec=TEAL, fs=8.7)
    rounded(ax, (0.165, y), 0.100, 0.058, "OT-preferred\nedge node", fc=FILL, ec=TEAL, fs=8.4)
    rounded(ax, (0.315, y), 0.125, 0.058, "Runtime deviation\ndetection", fc=FILL, ec=TEAL, fs=8.4)
    rounded(ax, (0.525, 0.330), 0.085, 0.058, "Default\nOT path", fc=FILL, ec=TEAL, fs=8.2)
    rounded(ax, (0.505, 0.180), 0.120, 0.060, r"Auction correction\nwith $C_i^{(e)}$", fc=FILL, ec=TEAL, fs=7.9)
    rounded(ax, (0.695, y-0.010), 0.085, 0.066, "Selected\nedge candidate", fc=FILL, ec=TEAL, fs=7.8)
    rounded(ax, (0.810, y-0.010), 0.098, 0.066, "Edge-cloud\ncompletion-time\ncomparison", fc=FILL, ec=TEAL, fs=7.3)
    rounded(ax, (0.930, y-0.016), 0.060, 0.078, "Final\nexecution\ndecision $d_k$", fc=FILL, ec=TEAL, lw=2.0, fs=7.0)

    arrow(ax, (0.130, y+0.029), (0.165, y+0.029), color=TEAL, lw=1.0)
    arrow(ax, (0.265, y+0.029), (0.315, y+0.029), color=TEAL, lw=1.0)
    arrow(ax, (0.440, y+0.029), (0.465, y+0.029), color=TEAL, lw=1.0)
    line(ax, [(0.465, y+0.029), (0.465, 0.359), (0.525, 0.359)], color=TEAL, lw=1.0)
    arrow(ax, (0.465, 0.359), (0.525, 0.359), color=TEAL, lw=1.0)
    line(ax, [(0.465, y+0.029), (0.465, 0.210), (0.505, 0.210)], color=TEAL, lw=1.0)
    arrow(ax, (0.465, 0.210), (0.505, 0.210), color=TEAL, lw=1.0)
    ax.text(0.476, 0.374, "No trigger", fontsize=8.0, color=TEAL, ha="left", va="center")
    ax.text(0.463, 0.190, "Trigger\nfired", fontsize=8.0, color=TEAL, ha="left", va="center")

    line(ax, [(0.610, 0.359), (0.640, 0.359), (0.670, y+0.022), (0.695, y+0.022)], color=TEAL, lw=1.0)
    arrow(ax, (0.670, y+0.022), (0.695, y+0.022), color=TEAL, lw=1.0)
    line(ax, [(0.625, 0.210), (0.645, 0.210), (0.670, y+0.022)], color=TEAL, lw=1.0)
    arrow(ax, (0.780, y+0.022), (0.810, y+0.022), color=TEAL, lw=1.0)
    arrow(ax, (0.908, y+0.022), (0.930, y+0.022), color=TEAL, lw=1.0, ms=10)

    # Trigger group: icon no longer overlaps labels.
    rounded(ax, (0.205, 0.105), 0.260, 0.095, fc="none", ec=TEAL, lw=1.0, r=0.014)
    trigger_card(ax, 0.222, 0.123, 0.072, 0.056, "Arrival\nsurge", "bars")
    trigger_card(ax, 0.307, 0.123, 0.074, 0.056, "Local\ncong.", "graph")
    trigger_card(ax, 0.394, 0.123, 0.056, 0.056, "Ref.\nmisfit", "tri")
    line(ax, [(0.335, 0.200), (0.335, 0.230), (0.378, 0.230), (0.378, y)], color=TEAL, lw=1.0, style=(0, (4, 3)))
    arrow(ax, (0.378, 0.230), (0.378, y), color=TEAL, lw=1.0, style=(0, (4, 3)), ms=9)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.025)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()

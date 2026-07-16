#!/usr/bin/env python3
"""Generate experiment-result figures from the frozen 12-month report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

BLUE = "#286A8A"
BLUE_LIGHT = "#D8E9F1"
GREEN = "#267454"
GREEN_LIGHT = "#D8ECE2"
AMBER = "#B16A19"
RED = "#A63D40"
RED_LIGHT = "#F2DADB"
INK = "#25343D"
MID = "#687780"
GRID = "#D7DEE2"
GRAY = "#8C989F"


# Frozen source:
# shared/ashare_open_stocks_glm52_20260714/reports/
# ashare_open_flow_vs_memskill_20260714_memfix02_full.md
MONTHS = [
    "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
]

MONTHLY_RETURNS = {
    "MiroFlow": [-3.02, -12.16, 15.00, -0.98, -2.15, -3.51, 7.77, 2.59, -8.17, 4.95, 6.57, 16.61],
    "MiroMemSkill": [3.48, -7.24, 20.41, 4.59, -3.88, 0.74, -2.54, 0.01, -12.08, 3.93, 12.95, 17.34],
    "沪深300": [-2.51, -4.14, 21.16, 1.73, -2.33, -2.80, 1.89, 1.32, -8.99, 7.34, 0.55, 2.79],
    "ETF核心": [-3.73, -5.17, 21.92, 5.32, -2.45, -4.68, 3.73, 2.94, -10.62, 7.75, 0.27, 2.66],
    "全市场等权（不可交易）": [-4.25, -5.89, 23.30, 10.39, 2.51, -6.87, 4.85, 5.84, -13.67, 15.38, 2.38, 3.32],
}

# Official totals use the evaluator's unrounded monthly returns. The plotted
# paths are reconstructed from the two-decimal monthly values published above.
OFFICIAL_TOTAL_RETURNS = {
    "MiroFlow": 21.28,
    "MiroMemSkill": 38.18,
    "沪深300": 13.98,
    "ETF核心": 15.41,
    "全市场等权（不可交易）": 36.91,
}

# total return (%), maximum drawdown magnitude (%), total fees (CNY)
RISK_RETURN = {
    "沪深300": (13.98, 10.81, 0),
    "ETF核心": (15.41, 11.24, 23597),
    "成长规则": (12.34, 13.51, 22496),
    "低PE Top-4": (25.10, 18.95, 25738),
    "全市场等权\n（不可交易）": (36.91, 13.67, 0),
    "MiroFlow": (21.28, 14.81, 16928),
    "MiroMemSkill": (38.18, 17.01, 19934),
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID)
    ax.tick_params(colors=MID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75, zorder=0)


def cumulative_return(monthly_returns: list[float]) -> np.ndarray:
    values = np.asarray(monthly_returns, dtype=float) / 100.0
    return (np.cumprod(1.0 + values) - 1.0) * 100.0


def performance_overview() -> None:
    fig = plt.figure(figsize=(13.2, 8.0))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.35, 1.0],
        hspace=0.42,
        wspace=0.30,
    )
    nav_ax = fig.add_subplot(gs[0, :])
    delta_ax = fig.add_subplot(gs[1, 0])
    risk_ax = fig.add_subplot(gs[1, 1])

    for ax in (nav_ax, delta_ax, risk_ax):
        style_axis(ax)

    x = np.arange(len(MONTHS))
    line_styles = {
        "MiroMemSkill": (GREEN, 2.8, "-", 4),
        "MiroFlow": (BLUE, 2.4, "-", 4),
        "全市场等权（不可交易）": (AMBER, 1.7, "--", 3),
        "ETF核心": (GRAY, 1.5, "-.", 3),
        "沪深300": (MID, 1.4, ":", 3),
    }
    for name, (color, width, linestyle, marker_size) in line_styles.items():
        path = cumulative_return(MONTHLY_RETURNS[name])
        nav_ax.plot(
            x,
            path,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            marker="o",
            markersize=marker_size,
            label=f"{name}  {OFFICIAL_TOTAL_RETURNS[name]:+.2f}%",
            zorder=3 if name in {"MiroMemSkill", "MiroFlow"} else 2,
        )
    nav_ax.axhline(0, color=INK, linewidth=0.9)
    nav_ax.set_xticks(x, [month.replace("20", "") for month in MONTHS])
    nav_ax.set_ylabel("累计净收益（%）")
    nav_ax.set_title("A  12个月顺序复利净值路径", loc="left", fontweight="bold", color=INK)
    nav_ax.legend(
        frameon=False,
        ncols=3,
        loc="upper left",
        fontsize=8.6,
        handlelength=2.7,
        columnspacing=1.3,
    )
    nav_ax.annotate(
        "最终差距 +16.90pp",
        xy=(11, cumulative_return(MONTHLY_RETURNS["MiroMemSkill"])[-1]),
        xytext=(8.7, 47),
        color=GREEN,
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.1},
    )

    flow = np.asarray(MONTHLY_RETURNS["MiroFlow"])
    mem = np.asarray(MONTHLY_RETURNS["MiroMemSkill"])
    differences = mem - flow
    colors = [GREEN_LIGHT if value >= 0 else RED_LIGHT for value in differences]
    edges = [GREEN if value >= 0 else RED for value in differences]
    bars = delta_ax.bar(x, differences, color=colors, edgecolor=edges, linewidth=1.0, zorder=2)
    delta_ax.axhline(0, color=INK, linewidth=0.9)
    delta_ax.axhline(
        differences.mean(),
        color=BLUE,
        linewidth=1.2,
        linestyle="--",
        label=f"月均差 {differences.mean():+.2f}pp",
    )
    for bar, value in zip(bars, differences, strict=True):
        if abs(value) >= 4.0:
            delta_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.45 if value >= 0 else -0.45),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.7,
                color=INK,
            )
    delta_ax.set_xticks(x, [month[5:] for month in MONTHS], rotation=0)
    delta_ax.set_ylabel("MemSkill − MiroFlow（pp）")
    delta_ax.set_title("B  逐月配对收益差（7胜5负）", loc="left", fontweight="bold", color=INK)
    delta_ax.legend(frameon=False, loc="lower right", fontsize=8.4)

    fees = np.asarray([values[2] for values in RISK_RETURN.values()], dtype=float)
    fee_sizes = 65.0 + 190.0 * fees / max(fees.max(), 1.0)
    point_colors = {
        "MiroMemSkill": GREEN,
        "MiroFlow": BLUE,
        "全市场等权\n（不可交易）": AMBER,
    }
    offsets = {
        "沪深300": (-3, -15),
        "ETF核心": (4, -14),
        "成长规则": (4, 7),
        "低PE Top-4": (-24, 7),
        "全市场等权\n（不可交易）": (-49, -1),
        "MiroFlow": (-35, 8),
        "MiroMemSkill": (-35, 8),
    }
    for (name, (ret, drawdown, fee)), size in zip(RISK_RETURN.items(), fee_sizes, strict=True):
        color = point_colors.get(name, GRAY)
        risk_ax.scatter(
            drawdown,
            ret,
            s=size,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        risk_ax.annotate(
            name,
            (drawdown, ret),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=7.8,
            color=INK,
        )
    risk_ax.set_xlim(9.5, 20.5)
    risk_ax.set_ylim(8, 42)
    risk_ax.set_xlabel("最大回撤幅度（%，越左越好）")
    risk_ax.set_ylabel("累计净收益（%）")
    risk_ax.set_title("C  收益−风险−成本", loc="left", fontweight="bold", color=INK)
    risk_ax.text(
        0.98,
        0.04,
        "圆面积表示总费用\n动量 Top-4：收益 −64.98%，回撤 72.76%（图外）",
        transform=risk_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=MID,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": GRID,
            "linewidth": 0.7,
        },
    )

    fig.suptitle(
        "全A股开放池主实验：收益路径、月度一致性与风险代价",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.018,
        "数据：冻结的 2024-07 至 2025-06 月度确定性回放；100万元顺序复利，包含100股整手与双边费用。全市场等权为不可交易风格参考。",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=MID,
    )
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.075, right=0.98)

    for suffix, kwargs in (
        ("pdf", {}),
        ("png", {"dpi": 220}),
    ):
        fig.savefig(
            ROOT / f"main_experiment_overview.{suffix}",
            bbox_inches="tight",
            pad_inches=0.08,
            metadata={"Creator": "MiroPlot / Matplotlib"} if suffix == "pdf" else None,
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    configure()
    performance_overview()


if __name__ == "__main__":
    main()

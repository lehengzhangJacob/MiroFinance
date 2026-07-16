#!/usr/bin/env python3
"""Generate the Hermes evolution figures used by the research report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent

BLUE = "#2F6B8A"
BLUE_LIGHT = "#DCEAF2"
GREEN = "#2D7D5A"
GREEN_LIGHT = "#DDEFE6"
AMBER = "#A96518"
AMBER_LIGHT = "#F6E7D2"
RED = "#A33A3A"
RED_LIGHT = "#F3DEDE"
INK = "#26343D"
MID = "#65737C"
GRID = "#D5DDE2"
GRAY = "#87939A"
GRAY_LIGHT = "#E9EDF0"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_box(
    ax: plt.Axes,
    center: tuple[float, float],
    text: str,
    *,
    width: float = 2.15,
    height: float = 0.92,
    face: str = BLUE_LIGHT,
    edge: str = BLUE,
    fontsize: float = 9.2,
) -> FancyBboxPatch:
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        linespacing=1.25,
        zorder=4,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID,
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.25,
            "linestyle": "--" if dashed else "-",
            "mutation_scale": 11,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": connectionstyle,
        },
        zorder=2,
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        ROOT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"Creator": "MiroPlot / Matplotlib", "Title": stem},
    )
    fig.savefig(
        ROOT / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)


def evolution_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.25))
    ax.set_xlim(-0.3, 12.9)
    ax.set_ylim(-0.45, 5.2)
    ax.axis("off")

    ax.text(
        0.0,
        5.0,
        "A  回测驱动、可审计的 Skill 自进化闭环",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )

    # Time split ribbon
    ribbon_y = 4.35
    segments = [
        (0.0, 6.0, BLUE_LIGHT, BLUE, "Train 12个月：生成反馈"),
        (6.0, 3.0, AMBER_LIGHT, AMBER, "Dev 6个月：筛选候选"),
        (9.0, 3.0, GREEN_LIGHT, GREEN, "Holdout 6个月：一次性门禁"),
    ]
    for x, width, face, edge, label in segments:
        patch = FancyBboxPatch(
            (x, ribbon_y),
            width,
            0.43,
            boxstyle="round,pad=0.015,rounding_size=0.05",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            ribbon_y + 0.215,
            label,
            ha="center",
            va="center",
            color=edge,
            fontsize=8.7,
            fontweight="bold",
        )

    top_y = 3.22
    low_y = 1.70
    nodes = {
        "snapshot": (0.95, top_y),
        "train": (3.45, top_y),
        "feedback": (5.95, top_y),
        "candidates": (8.45, top_y),
        "l0": (11.45, top_y),
        "dev": (9.75, low_y),
        "best": (7.15, low_y),
        "holdout": (4.55, low_y),
        "registry": (1.45, low_y),
    }

    add_box(
        ax,
        nodes["snapshot"],
        "冻结点时快照\n任务 + 只读回放库",
        face=GRAY_LIGHT,
        edge=GRAY,
    )
    add_box(ax, nodes["train"], "当前 Active Skill\n运行 12 个 Train 月")
    add_box(
        ax,
        nodes["feedback"],
        "结构化失败反馈\n亏损月・集中度・无效输出",
    )
    add_box(
        ax,
        nodes["candidates"],
        "GLM 反思式变异\n生成 3 个不可变候选",
        face=AMBER_LIGHT,
        edge=AMBER,
    )
    add_box(
        ax,
        nodes["l0"],
        "L0 静态门禁\n结构・长度・无硬编码",
        width=2.25,
        face=AMBER_LIGHT,
        edge=AMBER,
    )
    add_box(
        ax,
        nodes["dev"],
        "Dev 配对回测\n收益差 − 回撤惩罚",
        face=AMBER_LIGHT,
        edge=AMBER,
    )
    add_box(ax, nodes["best"], "仅 Dev 最优候选\n获得封存集租约")
    add_box(
        ax,
        nodes["holdout"],
        "一次性 Holdout\n有效性・回撤・配对收益",
        face=GREEN_LIGHT,
        edge=GREEN,
        width=2.35,
    )
    add_box(
        ax,
        nodes["registry"],
        "内容寻址 Registry\n晋升・备份・回滚・血缘",
        width=2.45,
        face=GREEN_LIGHT,
        edge=GREEN,
    )

    arrow(ax, (2.05, top_y), (2.34, top_y))
    arrow(ax, (4.55, top_y), (4.84, top_y))
    arrow(ax, (7.05, top_y), (7.34, top_y))
    arrow(ax, (9.55, top_y), (10.30, top_y))
    arrow(
        ax,
        (11.45, 2.75),
        (10.55, 2.17),
        connectionstyle="arc3,rad=-0.08",
    )
    arrow(ax, (8.65, low_y), (8.26, low_y))
    arrow(ax, (6.05, low_y), (5.75, low_y))
    arrow(ax, (3.36, low_y), (2.70, low_y), color=GREEN)

    reject = add_box(
        ax,
        (8.35, 0.35),
        "拒绝并归档证据\n保持当前 Active 不变",
        width=2.65,
        height=0.70,
        face=RED_LIGHT,
        edge=RED,
        fontsize=8.8,
    )
    arrow(
        ax,
        (11.45, 2.75),
        (9.18, 0.70),
        color=RED,
        dashed=True,
        connectionstyle="arc3,rad=0.20",
    )
    arrow(
        ax,
        (4.55, 1.22),
        (7.03, 0.35),
        color=RED,
        dashed=True,
        connectionstyle="arc3,rad=-0.15",
    )
    ax.text(2.95, 1.91, "通过", color=GREEN, fontsize=8.3, fontweight="bold")
    ax.text(9.45, 0.91, "失败", color=RED, fontsize=8.3, fontweight="bold")
    ax.text(5.45, 0.77, "失败", color=RED, fontsize=8.3, fontweight="bold")

    ax.text(
        0.0,
        -0.12,
        "隔离契约：相同模型、任务顺序与冻结数据库；每个 arm 独立 DB；运行时 memory 关闭；唯一变量是 Skill 正文。",
        ha="left",
        va="center",
        fontsize=8.9,
        color=MID,
    )
    save(fig, "hermes_evolution_pipeline")


def label_bars(ax: plt.Axes, bars, *, suffix: str = "%") -> None:
    for bar in bars:
        value = float(bar.get_height())
        offset = 2.0 if value >= 0 else -3.0
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}{suffix}",
            ha="center",
            va=va,
            fontsize=8.3,
            color=INK,
            fontweight="bold",
        )


def evolution_robustness() -> None:
    fig = plt.figure(figsize=(12.6, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 0.82, 1.15], wspace=0.42)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MID)
        ax.tick_params(colors=MID)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.axhline(0, color=INK, linewidth=0.9, zorder=1)

    # Panel A: first formal R1 result
    groups = np.arange(2)
    width = 0.34
    baseline = [29.35, -1.89]
    r1 = [83.53, 38.95]
    bars_a = ax1.bar(
        groups - width / 2,
        baseline,
        width,
        label="当轮当前 Skill",
        color=GRAY_LIGHT,
        edgecolor=GRAY,
        linewidth=1.0,
        zorder=2,
    )
    bars_b = ax1.bar(
        groups + width / 2,
        r1,
        width,
        label="R1 候选",
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        linewidth=1.1,
        zorder=2,
    )
    label_bars(ax1, bars_a)
    label_bars(ax1, bars_b)
    ax1.set_xticks(groups, ["Dev\n2025-07—12", "Holdout\n2026-01—06"])
    ax1.set_ylim(-12, 95)
    ax1.set_ylabel("累计净收益（%）")
    ax1.set_title("A  R1 单次正式结果", loc="left", fontweight="bold", color=INK)
    ax1.legend(frameon=False, fontsize=8.2, loc="upper left")

    # Panel B: exact same skill hash on the same six months, two rollouts
    repeats = [38.95, -16.52]
    bars = ax2.bar(
        np.arange(2),
        repeats,
        width=0.55,
        color=[BLUE_LIGHT, RED_LIGHT],
        edgecolor=[BLUE, RED],
        linewidth=1.1,
        zorder=2,
    )
    label_bars(ax2, bars)
    ax2.set_xticks([0, 1], ["首次", "复跑"])
    ax2.set_ylim(-28, 52)
    ax2.set_ylabel("同一 Holdout 总收益（%）")
    ax2.set_title("B  同一 R1 的运行噪声", loc="left", fontweight="bold", color=INK)
    ax2.text(
        0.5,
        -24,
        "相同 SHA / 相同月份",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MID,
    )

    # Panel C: sequential walk-forward evidence
    wf = [-5.1611, -2.2001, 5.5522]
    colors = [RED_LIGHT if value < 0 else GREEN_LIGHT for value in wf]
    edges = [RED if value < 0 else GREEN for value in wf]
    bars = ax3.bar(
        np.arange(3),
        wf,
        width=0.58,
        color=colors,
        edgecolor=edges,
        linewidth=1.1,
        zorder=2,
    )
    for bar, value in zip(bars, wf, strict=True):
        offset = 0.35 if value >= 0 else -0.35
        va = "bottom" if value >= 0 else "top"
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}pp",
            ha="center",
            va=va,
            fontsize=8.2,
            color=INK,
            fontweight="bold",
        )
    ax3.set_xticks([0, 1, 2], ["WF-R2\n1—2月", "WF-R3\n3—4月", "WF-R4\n5—6月"])
    ax3.set_ylim(-7.5, 8.2)
    ax3.set_ylabel("候选−R1 月均配对差（pp）")
    ax3.set_title("C  三段滚动 Holdout", loc="left", fontweight="bold", color=INK)
    ax3.text(
        1.0,
        7.1,
        "顺序复合：R1 7.33%  vs.  候选 4.26%\n候选合计落后 3.07pp",
        ha="center",
        va="top",
        fontsize=8.2,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": GRAY_LIGHT,
            "edgecolor": GRAY,
            "linewidth": 0.8,
        },
    )

    fig.text(
        0.01,
        0.985,
        "Hermes 技能进化：强单次信号与稳健性复现",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.065, right=0.99)
    save(fig, "hermes_evolution_robustness")


def main() -> None:
    configure()
    evolution_pipeline()
    evolution_robustness()


if __name__ == "__main__":
    main()

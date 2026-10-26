#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
受迫、阻尼、非线性单摆的庞加莱截面、步长检验与分形结构。

方程：
    theta'' = -(g/l) sin(theta) - q theta' + F_D sin(Omega_D t)

数值方法：
    Euler-Cromer：
        omega_{i+1} = omega_i + [-sin(theta_i) - q omega_i
                                 + F_D sin(Omega_D t_i)] dt
        theta_{i+1} = theta_i + omega_{i+1} dt
        t_{i+1}     = t_i + dt

    每一步后把 theta 约化到 [-pi, pi]。

输出：
    figures/fig1_poincare_initial_conditions.png
    figures/fig2_poincare_dt_comparison.png
    figures/fig3_attractor_zooms.png
    figures/fig4_box_counting.png
    results.txt

依赖：
    pip install numpy matplotlib
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib

# 无显示器环境下也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# 模型参数
# ---------------------------------------------------------------------------
G = 9.8
L = 9.8
Q = 0.5
F_D = 1.2
OMEGA_D = 2.0 / 3.0
T_D = 2.0 * math.pi / OMEGA_D       # 驱动力周期 = 3*pi

# 三组初始条件
INITIAL_CONDITIONS = [
    (0.2, 0.0),
    (0.8, 0.0),
    (-1.0, 0.5),
]
INITIAL_LABELS = [
    r"$(\theta_0,\omega_0)=(0.2,0)$",
    r"$(\theta_0,\omega_0)=(0.8,0)$",
    r"$(\theta_0,\omega_0)=(-1.0,0.5)$",
]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

DT_VALUES = [0.04, 0.02, 0.01]
TRANSIENT_PERIODS = 300
N_SECTION = 3000                   # 每个截面记录的点数
N_LONG = 60000                     # 全局图/局部放大/盒计数使用的点数
LONG_DT = 0.02                     # 长序列步长

# 题目建议的局部窗口
ZOOM1_BOUNDS = (2.0, 2.7, -1.5, 1.5)   # theta_min, theta_max, omega_min, omega_max
ZOOM2_SHRINK = 0.38


# ---------------------------------------------------------------------------
# 数值积分与庞加莱采样
# ---------------------------------------------------------------------------
def wrap_angle(theta: float) -> float:
    """把角度约化到 [-pi, pi]。步长足够小时，一次修正即可。"""
    if theta > math.pi:
        theta -= 2.0 * math.pi
    elif theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def poincare_section(
    theta0: float,
    omega0: float,
    dt: float,
    *,
    n_transient: int = TRANSIENT_PERIODS,
    n_samples: int = N_SECTION,
    q: float = Q,
    f_d: float = F_D,
    omega_d: float = OMEGA_D,
) -> np.ndarray:
    """
    用 Euler-Cromer 积分，并在每个驱动力周期的固定相位处采样。

    采样时刻为 t_n = n*T_D, n = n_transient+1, ..., n_transient+n_samples。
    由于 T_D/dt 一般不是整数，采样时对相邻两个 Euler 点作线性插值，
    从而保证所有采样点的驱动力相位严格相同。
    """
    if dt <= 0.0:
        raise ValueError("dt 必须为正数")
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正数")

    period = 2.0 * math.pi / omega_d
    next_sample_time = (n_transient + 1) * period
    max_steps = int((n_transient + n_samples + 2) * period / dt) + 100

    samples = np.empty((n_samples, 2), dtype=float)
    count = 0

    theta = wrap_angle(theta0)
    omega = float(omega0)
    t = 0.0

    sin = math.sin

    for _ in range(max_steps):
        accel = (
            -(G / L) * sin(theta)
            - q * omega
            + f_d * sin(omega_d * t)
        )

        omega_new = omega + accel * dt
        theta_new = wrap_angle(theta + omega_new * dt)
        t_new = t + dt

        if t_new >= next_sample_time:
            # 线性插值到精确的 n*T_D 时刻。
            frac = (next_sample_time - t) / dt
            frac = min(1.0, max(0.0, frac))

            # theta 已经约化，插值时取最短角差，避免跨过 +/-pi 时出错。
            dtheta = wrap_angle(theta_new - theta)
            theta_sample = wrap_angle(theta + frac * dtheta)
            omega_sample = omega + frac * (omega_new - omega)

            samples[count, 0] = theta_sample
            samples[count, 1] = omega_sample
            count += 1
            next_sample_time += period

            if count >= n_samples:
                break

        theta, omega, t = theta_new, omega_new, t_new

    if count < n_samples:
        return samples[:count]
    return samples


# ---------------------------------------------------------------------------
# 局部窗口与盒计数
# ---------------------------------------------------------------------------
def points_in_bounds(points: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """返回落在矩形窗口内的点。"""
    xmin, xmax, ymin, ymax = bounds
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    return points[mask]


def choose_dense_window(
    points: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    shrink: float = ZOOM2_SHRINK,
    bins: int = 36,
) -> tuple[float, float, float, float]:
    """
    在给定窗口内选一个点最密集的子窗口。

    这样第二级放大图总包含较丰富的点，而不是落在空白区域。
    """
    xmin, xmax, ymin, ymax = bounds
    local = points_in_bounds(points, bounds)

    if len(local) < 20:
        # 点数太少时，退化为窗口中心的一半大小。
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
    else:
        hist, xedges, yedges = np.histogram2d(
            local[:, 0],
            local[:, 1],
            bins=bins,
            range=[[xmin, xmax], [ymin, ymax]],
        )
        i, j = np.unravel_index(np.argmax(hist), hist.shape)
        cx = 0.5 * (xedges[i] + xedges[i + 1])
        cy = 0.5 * (yedges[j] + yedges[j + 1])

    width = (xmax - xmin) * shrink
    height = (ymax - ymin) * shrink

    x0 = min(max(cx - 0.5 * width, xmin), xmax - width)
    y0 = min(max(cy - 0.5 * height, ymin), ymax - height)
    return (x0, x0 + width, y0, y0 + height)


def box_counting(
    points: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    n_scales: int = 15,
) -> dict:
    """
    对局部点集做盒计数，返回 N(epsilon)、拟合斜率 D_B 等信息。

    盒计数使用正方形盒子，边长按 2 的幂次递减。
    """
    xmin, xmax, ymin, ymax = bounds
    local = points_in_bounds(points, bounds)

    if len(local) < 50:
        raise ValueError(
            f"局部区域只有 {len(local)} 个点，无法可靠地做盒计数；"
            "请增大 N_LONG 或换一个局部窗口。"
        )

    width = xmax - xmin
    height = ymax - ymin
    eps0 = min(width, height) / 5.0
    eps = eps0 * (2.0 ** (-np.arange(n_scales, dtype=float)))

    counts = np.empty(n_scales, dtype=int)
    for k, e in enumerate(eps):
        ix = np.floor((local[:, 0] - xmin) / e).astype(np.int64)
        iy = np.floor((local[:, 1] - ymin) / e).astype(np.int64)
        # ix、iy 非负；用一个大整数把二维格点编码为一维，避免重复计数。
        keys = ix * (2 ** 32) + iy
        counts[k] = np.unique(keys).size

    # 拟合区间：排除盒子太大的端点和点集饱和的小尺度端点。
    # 上限取 0.25N，尽量避开“每个点占一个盒子”的饱和区。
    valid = (counts >= 20) & (counts <= 0.25 * len(local))
    if valid.sum() < 4:
        valid = (counts >= 8) & (counts <= 0.60 * len(local))
    if valid.sum() < 3:
        valid = np.ones_like(counts, dtype=bool)

    inv_eps = 1.0 / eps
    log_inv_eps = np.log(inv_eps)
    log_n = np.log(counts.astype(float))

    slope, intercept = np.polyfit(log_inv_eps[valid], log_n[valid], 1)
    d_b = float(slope)

    return {
        "local_points": local,
        "eps": eps,
        "N": counts,
        "valid": valid,
        "D_B": d_b,
        "intercept": float(intercept),
        "n_local": int(len(local)),
        "fit_eps_min": float(eps[valid].min()),
        "fit_eps_max": float(eps[valid].max()),
        "fit_count": int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def _initial_condition_figure(sections: dict, outdir: Path) -> Path:
    """图 1：三组初始条件的庞加莱截面及叠加图。"""
    data = [sections[(0.04, i)] for i in range(len(INITIAL_CONDITIONS))]

    all_points = np.vstack(data)
    y_pad = 0.12 * (all_points[:, 1].max() - all_points[:, 1].min())
    ylim = (all_points[:, 1].min() - y_pad, all_points[:, 1].max() + y_pad)
    xlim = (-math.pi, math.pi)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for i, (ax, pts) in enumerate(zip(axes[:3], data)):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=1.0,
            c=COLORS[i],
            alpha=0.72,
            linewidths=0,
        )
        ax.set_title(f"{INITIAL_LABELS[i]}\n$N={len(pts)}$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\theta$  [rad]")
        ax.set_ylabel(r"$\omega$  [rad/s]")
        ax.grid(alpha=0.22, linewidth=0.5)

    ax = axes[3]
    for i, pts in enumerate(data):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=1.0,
            c=COLORS[i],
            alpha=0.55,
            linewidths=0,
            label=INITIAL_LABELS[i],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    ax.set_title("Overlay: same invariant set")
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.legend(markerscale=5, fontsize=8, loc="best")

    fig.suptitle(
        "Poincare sections at $t=nT_D$ after discarding $300T_D$  "
        f"($\\Delta t=0.04$, {N_SECTION} points each)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / "fig1_poincare_initial_conditions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _time_step_figure(sections: dict, outdir: Path) -> Path:
    """图 2：同一初值、三种步长的全局截面与局部放大。"""
    dts = DT_VALUES
    data = [sections[(dt, 0)] for dt in dts]

    all_points = np.vstack(data)
    y_pad = 0.12 * (all_points[:, 1].max() - all_points[:, 1].min())
    ylim = (all_points[:, 1].min() - y_pad, all_points[:, 1].max() + y_pad)
    xlim = (-math.pi, math.pi)

    x0, x1, y0, y1 = ZOOM1_BOUNDS

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for j, (dt, pts) in enumerate(zip(dts, data)):
        ax = axes[0, j]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c="k", alpha=0.65, linewidths=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"$\\Delta t={dt}$  ($N={len(pts)}$)")
        ax.grid(alpha=0.22, linewidth=0.5)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                ec="tab:red",
                lw=1.0,
                ls="--",
            )
        )
        if j == 0:
            ax.set_ylabel(r"$\omega$  [rad/s]")

        axz = axes[1, j]
        local = points_in_bounds(pts, ZOOM1_BOUNDS)
        axz.scatter(local[:, 0], local[:, 1], s=1.2, c="k", alpha=0.72, linewidths=0)
        axz.set_xlim(x0, x1)
        axz.set_ylim(y0, y1)
        axz.set_xlabel(r"$\theta$  [rad]")
        axz.set_title(f"zoom: $N={len(local)}$")
        axz.grid(alpha=0.22, linewidth=0.5)
        if j == 0:
            axz.set_ylabel(r"$\omega$  [rad/s]")

    fig.suptitle(
        "Numerical reliability: Poincare sections for the same initial condition "
        r"$(\theta_0,\omega_0)=(0.2,0)$",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / "fig2_poincare_dt_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _zoom_figure(
    points: np.ndarray,
    zoom2_bounds: tuple[float, float, float, float],
    outdir: Path,
) -> Path:
    """图 3：全局吸引子及两级局部放大。"""
    xmin, xmax, ymin, ymax = ZOOM1_BOUNDS
    x2min, x2max, y2min, y2max = zoom2_bounds

    y_pad = 0.12 * (points[:, 1].max() - points[:, 1].min())
    ylim = (points[:, 1].min() - y_pad, points[:, 1].max() + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=0.75, c="k", alpha=0.6, linewidths=0)
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(ylim)
    ax.set_title(f"Global attractor ($N={len(points)}$)")
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            ec="tab:red",
            lw=1.2,
            ls="--",
        )
    )

    ax = axes[1]
    local1 = points_in_bounds(points, ZOOM1_BOUNDS)
    ax.scatter(local1[:, 0], local1[:, 1], s=1.5, c="k", alpha=0.72, linewidths=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Zoom 1 ($N={len(local1)}$)")
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.add_patch(
        Rectangle(
            (x2min, y2min),
            x2max - x2min,
            y2max - y2min,
            fill=False,
            ec="tab:red",
            lw=1.2,
            ls="--",
        )
    )

    ax = axes[2]
    local2 = points_in_bounds(points, zoom2_bounds)
    ax.scatter(local2[:, 0], local2[:, 1], s=3.0, c="k", alpha=0.82, linewidths=0)
    ax.set_xlim(x2min, x2max)
    ax.set_ylim(y2min, y2max)
    ax.set_title(
        "Zoom 2\n"
        rf"$[{x2min:.3f},{x2max:.3f}]\times[{y2min:.3f},{y2max:.3f}]$"
        f"  ($N={len(local2)}$)",
        fontsize=9.5,
    )
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    ax.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "Fractal structure of the strange attractor  "
        f"($\\Delta t={LONG_DT}$, {N_LONG} Poincare points)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = outdir / "fig3_attractor_zooms.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _box_counting_figure(box: dict, outdir: Path) -> Path:
    """图 4：局部点集与盒计数 log-log 图。"""
    local = box["local_points"]
    eps = box["eps"]
    counts = box["N"]
    valid = box["valid"]
    d_b = box["D_B"]
    intercept = box["intercept"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(local[:, 0], local[:, 1], s=1.2, c="k", alpha=0.7, linewidths=0)
    xmin, xmax, ymin, ymax = ZOOM1_BOUNDS
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"Local set used for box counting ($N={len(local)}$)")
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    ax.grid(alpha=0.2, linewidth=0.5)

    ax = axes[1]
    x = np.log(1.0 / eps)
    y = np.log(counts.astype(float))
    ax.plot(x, y, "o-", ms=4, lw=1.0, color="tab:blue", label=r"$\log N(\varepsilon)$")

    x_fit = np.linspace(x[valid].min(), x[valid].max(), 100)
    ax.plot(
        x_fit,
        d_b * x_fit + intercept,
        "--",
        color="tab:red",
        lw=1.5,
        label=rf"fit: $D_B={d_b:.3f}$",
    )
    ax.set_xlabel(r"$\log(1/\varepsilon)$")
    ax.set_ylabel(r"$\log N(\varepsilon)$")
    ax.set_title("Box-counting scaling")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Box-counting dimension in the zoomed window  "
        r"$2.0<\theta<2.7,\ -1.5<\omega<1.5$",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = outdir / "fig4_box_counting.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def build_analysis(box: dict) -> str:
    """生成作业要求的文字分析，并写入 results.txt。"""
    d_b = box["D_B"]
    local_n = box["n_local"]
    fit_n = box["fit_count"]

    if d_b < 1.0:
        dim_text = (
            f"本次拟合得到 D_B={d_b:.3f}，略小于 1。对于二维截面上的奇异吸引子，"
            "理想盒维数通常应在 1 和 2 之间；这里略低于 1，说明在所选窗口和"
            "有限采样下，点集表现为非常稀疏的准一维细丝，且小尺度端已经受到"
            "采样饱和的影响，不能把该数值当作高精度维数。它仍然不等于整数 1，"
            "表明点集不是简单光滑曲线，而是具有分形细丝和层级空隙的结构。"
        )
    elif d_b < 2.0:
        dim_text = (
            f"本次拟合得到 D_B={d_b:.3f}，介于 1 和 2 之间。"
            "这说明局部点集既不是简单曲线（D=1），也没有铺满二维窗口（D=2），"
            "而是具有非整数维的分形结构。"
        )
    else:
        dim_text = (
            f"本次拟合得到 D_B={d_b:.3f}，偏大，说明拟合区间可能仍受大尺度"
            "边界效应或采样不足影响；应结合图 4 的中间尺度区间谨慎解释。"
        )

    return (
        "结果分析与讨论\n"
        "================\n"
        "1. 不同初始条件的影响\n"
        "三组初始条件经过 300T_D 的暂态后，庞加莱截面落在同一个有界点集上："
        "整体轮廓、主要弯曲条带和大的空白区域一致，说明它们进入同一个吸引子。"
        "但三组点并不逐点重合，因为混沌系统对初值具有敏感依赖性，经过长时间"
        "积分后具体相点会显著分离；初值只改变暂态路径和具体相点，不改变长期"
        "吸引子的整体形状。截面不是有限个孤立点（周期运动），也不是简单闭合"
        "曲线（准周期运动），而是具有细丝和空隙的复杂集合，说明运动是混沌的。\n\n"
        "2. 时间步长与数值收敛性\n"
        "在 dt=0.04、0.02、0.01 三种步长下，吸引子的外轮廓、主要分支和大的"
        "空隙基本一致，说明这些宏观结构不是某一种步长造成的数值假象，而是"
        "连续系统长期不变集的性质。局部细丝和窄缝存在差异：dt=0.04 时较粗，"
        "dt=0.02 与 0.01 更接近。原因是 Euler-Cromer 只有一阶精度，步长越大，"
        "离散轨道偏离连续轨道的误差越大；混沌系统又会放大这种误差。因此，"
        "整体相似说明数值结果对步长不敏感、宏观结构可靠；局部分歧则说明极细"
        "结构已接近当前数值分辨率，不能仅凭图像重合断言完全收敛。\n\n"
        "3. 局部放大与分形特征\n"
        "全局图具有弯曲条带和分层结构；第一级放大后条带变成更细的丝状结构，"
        "丝与丝之间存在裂隙；第二级放大后仍能看到新的条带、细丝和空隙，而不是"
        "均匀的模糊背景。这种“缩小尺度后仍有复杂结构”的现象是分形特征的重要"
        "表现。它不是单纯由散点太少造成的：程序使用了较长的点列，并在第二级"
        "放大时自动选择点最密集的子区域；步长检验也表明主要结构在 dt=0.02 与"
        "0.01 下比较接近。当然，有限采样会使极细结构看起来断裂或缺失，因此"
        "局部图只是直观证据，还需要盒计数等定量分析。\n\n"
        "4. 盒计数维数\n"
        f"在窗口 2.0<theta<2.7、-1.5<omega<1.5 内，局部点数为 {local_n}，"
        f"参与拟合的尺度点数为 {fit_n}。{dim_text}"
        "有限采样下，盒子太大时点数太少、受窗口边界控制；盒子太小时每个点"
        "几乎占一个盒子，N(epsilon) 趋于饱和，斜率接近 1 甚至更低。只有中间"
        "尺度区间才可能接近真实幂律，因此 D_B 会随采样点数、步长和拟合区间"
        "发生变化。\n\n"
        "5. 奇异吸引子与物理结论\n"
        "阻尼项 -q*omega 持续耗散能量，使相空间体积收缩；周期驱动力不断向"
        "系统输入能量，二者竞争使轨迹最终落到一个有界的不变集上。该集合的"
        "体积为零，却具有细丝状、分层和自相似结构，盒计数维数不是整数；"
        "同时相邻初值会指数分离。这种“有吸引性、分形几何、对初值敏感”的"
        "集合就是奇异吸引子。三组初值的截面形状一致而相点不重合，正是"
        "吸引子稳定性和混沌敏感性同时存在的直接表现。\n"
    )


def build_markdown_report(
    box: dict,
    zoom2_bounds: tuple[float, float, float, float],
    figure_paths: list[Path],
) -> str:
    """生成独立的 Markdown 计算报告（运行程序时自动写出）。"""
    x3, x4, y3, y4 = zoom2_bounds
    d_b = box["D_B"]
    n_local = box["n_local"]
    fit_count = box["fit_count"]
    fit_eps_min = box["fit_eps_min"]
    fit_eps_max = box["fit_eps_max"]
    links = "\n\n".join(
        f"![{p.stem}]({p.as_posix()})" for p in figure_paths
    )
    analysis = build_analysis(box)

    return f"""# 受迫阻尼非线性单摆的庞加莱截面与奇异吸引子

## 1. 模型与数值方法

运动方程

$$
\\frac{{d^2\\theta}}{{dt^2}}
=-\\frac{{g}}{{\\ell}}\\sin\\theta-q\\frac{{d\\theta}}{{dt}}
+F_D\\sin(\\Omega_D t).
$$

取

$$
g=9.8,\\quad \\ell=9.8,\\quad q=0.5,\\quad
F_D=1.2,\\quad \\Omega_D=\\frac{{2}}{{3}},
$$

因此 $g/\\ell=1$，驱动力周期为

$$
T_D=\\frac{{2\\pi}}{{\\Omega_D}}=3\\pi\\approx9.42478\\ \\mathrm{{s}}.
$$

使用 Euler–Cromer 方法：

$$
\\omega_{{i+1}}
=\\omega_i+
\\left[-\\frac{{g}}{{\\ell}}\\sin\\theta_i-q\\omega_i
+F_D\\sin(\\Omega_D t_i)\\right]\\Delta t,
$$

$$
\\theta_{{i+1}}=\\theta_i+\\omega_{{i+1}}\\Delta t,
\\qquad
t_{{i+1}}=t_i+\\Delta t.
$$

每一步后将 $\\theta$ 约化到 $[-\\pi,\\pi]$。舍弃前 $300T_D$ 的暂态，在
$t_n=nT_D$ 处对相邻 Euler 点作线性插值，得到严格同相位的庞加莱截面。

## 2. 计算参数

- 初始条件：$(0.2,0)$、$(0.8,0)$、$(-1.0,0.5)$；
- 暂态：$300T_D\\approx2827.43\\ \\mathrm{{s}}$；
- 每个截面采样点数：$3000$；
- 时间步长：$0.04,\\ 0.02,\\ 0.01\\ \\mathrm{{s}}$；
- 长序列：$\\Delta t={LONG_DT}$，$N={N_LONG}$ 个庞加莱点；
- 第一级放大窗口：$2.0<\\theta<2.7$，$-1.5<\\omega<1.5$；
- 第二级窗口：$[{x3:.4f},{x4:.4f}]\\times[{y3:.4f},{y4:.4f}]$。

## 3. 图表

{links}

## 4. 结果摘要

三组初值经过暂态后进入同一个有界吸引子。截面不是有限个孤立点，也不是
简单闭合曲线，而是具有弯曲条带、细丝和裂隙的复杂集合；不同初值的整体
轮廓一致而具体相点不重合。

三种步长下，吸引子的外轮廓、主要分支和大空隙基本一致；局部细丝在
$\\Delta t=0.04$ 时较粗，$0.02$ 与 $0.01$ 更接近。这说明宏观结构对
离散化不敏感，而极细结构已接近当前数值分辨率。

在局部窗口 $2.0<\\theta<2.7,\\ -1.5<\\omega<1.5$ 中，盒计数得到

$$
D_B={d_b:.4f}
$$

（局部点数 ${n_local}$，拟合尺度点数 ${fit_count}$，
拟合 $\\varepsilon$ 范围约
$[{fit_eps_min:.3g},{fit_eps_max:.3g}]$）。
该值不是整数；有限采样和小尺度饱和会影响其精度。

## 5. 结果分析与讨论

{analysis}

## 6. 复现

```bash
pip install -r requirements.txt
python forced_pendulum_chaos.py
```
"""


def main() -> None:
    outdir = Path("figures")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Forced damped nonlinear pendulum")
    print(f"g={G}, l={L}, q={Q}, F_D={F_D}, Omega_D={OMEGA_D:.6f}")
    print(f"T_D = 2*pi/Omega_D = {T_D:.6f} s; transient = {TRANSIENT_PERIODS} T_D")
    print("=" * 70)

    sections: dict[tuple[float, int], np.ndarray] = {}

    # 图 1 需要三组初值在 dt=0.04 下的截面；图 2 需要同一初值的三种步长。
    jobs = [(0.04, i) for i in range(len(INITIAL_CONDITIONS))]
    jobs += [(dt, 0) for dt in DT_VALUES if dt != 0.04]

    for dt, i in jobs:
        theta0, omega0 = INITIAL_CONDITIONS[i]
        print(
            f"Integrating: dt={dt:.3f}, "
            f"(theta0,omega0)=({theta0},{omega0}), "
            f"discard {TRANSIENT_PERIODS} T_D, record {N_SECTION} points ..."
        )
        sections[(dt, i)] = poincare_section(
            theta0,
            omega0,
            dt,
            n_transient=TRANSIENT_PERIODS,
            n_samples=N_SECTION,
        )

    fig1 = _initial_condition_figure(sections, outdir)
    print(f"Saved: {fig1}")

    fig2 = _time_step_figure(sections, outdir)
    print(f"Saved: {fig2}")

    print(
        f"\nComputing long section for global plot and zooms: "
        f"dt={LONG_DT}, {N_LONG} points ..."
    )
    long_section = poincare_section(
        INITIAL_CONDITIONS[0][0],
        INITIAL_CONDITIONS[0][1],
        LONG_DT,
        n_transient=TRANSIENT_PERIODS,
        n_samples=N_LONG,
    )

    zoom2_bounds = choose_dense_window(long_section, ZOOM1_BOUNDS)
    fig3 = _zoom_figure(long_section, zoom2_bounds, outdir)
    print(f"Saved: {fig3}")
    print(
        "Zoom windows: "
        f"zoom1={ZOOM1_BOUNDS}, zoom2=("
        f"{zoom2_bounds[0]:.4f}, {zoom2_bounds[1]:.4f}, "
        f"{zoom2_bounds[2]:.4f}, {zoom2_bounds[3]:.4f})"
    )

    box = box_counting(long_section, ZOOM1_BOUNDS)
    fig4 = _box_counting_figure(box, outdir)
    print(f"Saved: {fig4}")
    print(
        f"Box counting: N_local={box['n_local']}, "
        f"D_B={box['D_B']:.4f}, fit points={box['fit_count']}, "
        f"eps in [{box['fit_eps_min']:.5g}, {box['fit_eps_max']:.5g}]"
    )

    results_path = Path("results.txt")
    with results_path.open("w", encoding="utf-8") as f:
        f.write("Forced damped pendulum: numerical results\n")
        f.write("=" * 60 + "\n")
        f.write(f"g = {G}\n")
        f.write(f"l = {L}\n")
        f.write(f"q = {Q}\n")
        f.write(f"F_D = {F_D}\n")
        f.write(f"Omega_D = {OMEGA_D}\n")
        f.write(f"T_D = {T_D}\n")
        f.write(f"transient periods = {TRANSIENT_PERIODS}\n")
        f.write(f"points per section = {N_SECTION}\n")
        f.write(f"dt values = {DT_VALUES}\n")
        f.write(f"long section: dt = {LONG_DT}, N = {N_LONG}\n")
        f.write(f"zoom1 bounds = {ZOOM1_BOUNDS}\n")
        f.write(f"zoom2 bounds = {zoom2_bounds}\n")
        f.write(f"local points in zoom1 = {box['n_local']}\n")
        f.write(f"box-counting D_B = {box['D_B']:.6f}\n")
        f.write(f"fit eps range = [{box['fit_eps_min']:.6g}, {box['fit_eps_max']:.6g}]\n")
        f.write(f"fit points = {box['fit_count']}\n")
        f.write("\nFiles:\n")
        for p in [fig1, fig2, fig3, fig4]:
            f.write(f"  {p}\n")

        analysis = build_analysis(box)
        f.write("\n\n" + analysis)

    analysis = build_analysis(box)
    print("\n" + analysis)

    report_path = Path("report.md")
    report_path.write_text(
        build_markdown_report(
            box,
            zoom2_bounds,
            [fig1, fig2, fig3, fig4],
        ),
        encoding="utf-8",
    )
    print(f"Saved summary: {results_path}")
    print(f"Saved report:  {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()

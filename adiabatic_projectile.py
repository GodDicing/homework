#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绝热空气密度模型下炮弹的最小初速度发射问题。

依赖：
    pip install numpy matplotlib

物理模型：
    rho(y) / rho0 = (1 - a*y/T0)^alpha,  1 - a*y/T0 <= 0 时取 0
    B2*(y) / m    = (B2/m) * rho(y)/rho0
    ax = -(B2*(y)/m) * v * vx
    ay = -g -(B2*(y)/m) * v * vy

数值方法：
    显式欧拉法，dt = 0.01 s；
    目标命中判据为轨迹到目标点的距离小于 2 m。

继承结构：
    Projectile                  基类：公共参数与欧拉迭代
    └── AdiabaticDragProjectile 子类：绝热密度与高度相关阻力
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 公共常量
# ---------------------------------------------------------------------------
TARGET = (1000.0, 100.0)       # 目标位置 (m)
G = 9.8                        # 重力加速度 (m/s^2)
DT = 0.01                      # 欧拉步长 (s)
HIT_RADIUS = 2.0               # 命中半径 (m)
MAX_TIME = 120.0               # 单次模拟最长时间 (s)

THETA_MIN, THETA_MAX = 10.0, 80.0
V0_MIN, V0_MAX = 100.0, 1000.0


@dataclass
class Trajectory:
    """一次模拟的结果。"""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    hit: bool
    hit_time: Optional[float]
    max_height: float
    min_distance: float

    @property
    def flight_time(self) -> float:
        """命中时返回命中时刻；未命中时返回落地/终止时刻。"""
        return self.hit_time if self.hit_time is not None else float(self.t[-1])

    def y_at_x(self, x_query: float) -> Optional[float]:
        """用线性插值返回轨迹在 x=x_query 处的 y 值。"""
        if x_query < self.x[0] or x_query > self.x[-1]:
            return None

        j = int(np.searchsorted(self.x, x_query))
        if j == 0:
            return float(self.y[0])
        if j >= len(self.x):
            return float(self.y[-1])

        x0, x1 = float(self.x[j - 1]), float(self.x[j])
        y0, y1 = float(self.y[j - 1]), float(self.y[j])
        if x1 == x0:
            return y1
        w = (x_query - x0) / (x1 - x0)
        return y0 + w * (y1 - y0)


class Projectile:
    """
    炮弹基类。

    只负责公共参数和欧拉迭代；具体的加速度由子类通过 acceleration()
    重写。因此主程序不需要重复编写欧拉循环。
    """

    def __init__(
        self,
        v0: float,
        theta_deg: float,
        *,
        g: float = G,
        dt: float = DT,
        target: tuple[float, float] = TARGET,
        hit_radius: float = HIT_RADIUS,
        max_time: float = MAX_TIME,
    ) -> None:
        if v0 <= 0.0:
            raise ValueError("v0 必须为正数")
        if dt <= 0.0:
            raise ValueError("dt 必须为正数")

        self.v0 = float(v0)
        self.theta_deg = float(theta_deg)
        self.theta = math.radians(self.theta_deg)
        self.g = float(g)
        self.dt = float(dt)
        self.target = (float(target[0]), float(target[1]))
        self.hit_radius = float(hit_radius)
        self.max_steps = max(1, int(max_time / dt))

        self.vx0 = self.v0 * math.cos(self.theta)
        self.vy0 = self.v0 * math.sin(self.theta)

    def acceleration(
        self, x: float, y: float, vx: float, vy: float
    ) -> tuple[float, float]:
        """
        返回 (ax, ay)。

        基类默认只受重力；子类必须重写此方法以加入阻力。
        """
        return 0.0, -self.g

    def simulate(self, stop_on_hit: bool = True) -> Trajectory:
        """用显式欧拉法模拟炮弹运动。"""
        xt, yt = self.target
        dt = self.dt
        hit_r2 = self.hit_radius * self.hit_radius

        x, y = 0.0, 0.0
        vx, vy = self.vx0, self.vy0
        t = 0.0

        ts = [t]
        xs = [x]
        ys = [y]
        vxs = [vx]
        vys = [vy]

        hit = False
        hit_time: Optional[float] = None
        min_d2 = (x - xt) ** 2 + (y - yt) ** 2

        for _ in range(self.max_steps):
            # 显式欧拉：先用当前状态计算加速度
            ax, ay = self.acceleration(x, y, vx, vy)

            x_new = x + vx * dt
            y_new = y + vy * dt
            vx_new = vx + ax * dt
            vy_new = vy + ay * dt
            t_new = t + dt

            # 检查本步欧拉线段到目标的最短距离，避免高速时步长跨过目标。
            sx = x_new - x
            sy = y_new - y
            seg_len2 = sx * sx + sy * sy
            if seg_len2 > 0.0:
                s = ((xt - x) * sx + (yt - y) * sy) / seg_len2
                s = min(1.0, max(0.0, s))
            else:
                s = 0.0

            cx = x + s * sx
            cy = y + s * sy
            d2 = (cx - xt) ** 2 + (cy - yt) ** 2
            if d2 < min_d2:
                min_d2 = d2

            if d2 < hit_r2 and not hit:
                hit = True
                hit_time = t + s * dt

                if stop_on_hit:
                    # 把最接近目标的那个点也加入轨迹，便于绘图。
                    if s > 1.0e-12:
                        ts.append(hit_time)
                        xs.append(cx)
                        ys.append(cy)
                        vxs.append(vx + s * (vx_new - vx))
                        vys.append(vy + s * (vy_new - vy))
                    break

            # 记录欧拉步的端点
            ts.append(t_new)
            xs.append(x_new)
            ys.append(y_new)
            vxs.append(vx_new)
            vys.append(vy_new)

            x, y, vx, vy, t = x_new, y_new, vx_new, vy_new, t_new

            # 落地，或已经越过目标（vx 始终大于 0）
            if y < 0.0 or x > xt + self.hit_radius:
                break

        return Trajectory(
            t=np.asarray(ts, dtype=float),
            x=np.asarray(xs, dtype=float),
            y=np.asarray(ys, dtype=float),
            vx=np.asarray(vxs, dtype=float),
            vy=np.asarray(vys, dtype=float),
            hit=hit,
            hit_time=hit_time,
            max_height=float(max(ys)),
            min_distance=math.sqrt(min_d2),
        )


class AdiabaticDragProjectile(Projectile):
    """
    绝热空气密度模型下的炮弹。

    rho(y) = rho0 * (1 - a*y/T0)^alpha
    B2*(y) = B2 * rho(y)/rho0
    """

    def __init__(
        self,
        v0: float,
        theta_deg: float,
        *,
        mass: float = 10.0,
        B2: float = 4.0e-4,       # B2/m = 4.0e-5 m^-1
        rho0: float = 1.225,
        T0: float = 300.0,
        lapse_rate: float = 6.5e-3,
        alpha: float = 2.5,
        **kwargs,
    ) -> None:
        super().__init__(v0, theta_deg, **kwargs)

        if mass <= 0.0:
            raise ValueError("mass 必须为正数")
        if rho0 <= 0.0:
            raise ValueError("rho0 必须为正数")
        if T0 <= 0.0:
            raise ValueError("T0 必须为正数")

        self.mass = float(mass)
        self.B2 = float(B2)
        self.rho0 = float(rho0)
        self.T0 = float(T0)
        self.lapse_rate = float(lapse_rate)
        self.alpha = float(alpha)

    def density_ratio(self, y: float) -> float:
        """返回 rho(y)/rho0。"""
        base = 1.0 - self.lapse_rate * y / self.T0
        if base <= 0.0:
            return 0.0
        return base ** self.alpha

    def density(self, y: float) -> float:
        """返回高度 y 处的空气密度 rho(y)。"""
        return self.rho0 * self.density_ratio(y)

    def b2_star(self, y: float) -> float:
        """返回修正后的阻力系数 B2*(y) = B2*rho(y)/rho0。"""
        return self.B2 * self.density_ratio(y)

    def acceleration(
        self, x: float, y: float, vx: float, vy: float
    ) -> tuple[float, float]:
        """高度相关的二次阻力加速度（重写基类方法）。"""
        speed = math.hypot(vx, vy)
        drag_factor = self.b2_star(y) / self.mass

        ax = -drag_factor * speed * vx
        ay = -self.g - drag_factor * speed * vy
        return ax, ay


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------
def make_shot(v0: float, theta_deg: float) -> AdiabaticDragProjectile:
    """按题目参数构造一次模拟。"""
    return AdiabaticDragProjectile(
        v0,
        theta_deg,
        mass=10.0,
        B2=4.0e-4,          # B2/m = 4e-5
        rho0=1.225,
        T0=300.0,
        lapse_rate=6.5e-3,
        alpha=2.5,
        g=G,
        dt=DT,
        target=TARGET,
        hit_radius=HIT_RADIUS,
    )


def crossing_error(v0: float, theta_deg: float) -> float:
    """
    返回轨迹在 x=x_t 处的高度与 y_t 之差。

    若炮弹在到达目标水平位置前落地，返回一个很大的负数。
    对固定角度，该函数随 v0 近似单调增加。
    """
    result = make_shot(v0, theta_deg).simulate(stop_on_hit=False)
    y_cross = result.y_at_x(TARGET[0])
    if y_cross is None:
        return -1.0e6
    return y_cross - TARGET[1]


def find_center_speed(theta_deg: float) -> Optional[float]:
    """
    对固定角度，二分求使轨迹经过目标高度 y_t 的初速度。

    这个速度会命中目标中心附近，用作“命中”二分的上界。
    """
    lo, hi = V0_MIN, V0_MAX
    f_lo = crossing_error(lo, theta_deg)
    f_hi = crossing_error(hi, theta_deg)

    if f_lo >= 0.0:
        return lo
    if f_hi <= 0.0:
        return None

    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if crossing_error(mid, theta_deg) >= 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def minimum_speed_for_angle(theta_deg: float) -> float:
    """
    对固定发射角，二分求满足命中判据的最小 v0。

    先用 crossing_error 找到命中目标中心的速度 center；
    再在 [V0_MIN, center] 上对“是否命中”做二分。
    """
    center = find_center_speed(theta_deg)
    if center is None:
        return math.inf

    center_result = make_shot(center, theta_deg).simulate()
    if not center_result.hit:
        # 正常不会发生；保留一个小的数值容错区间。
        for dv in np.linspace(0.001, 1.0, 1000):
            for v in (center + dv, center - dv):
                if V0_MIN <= v <= V0_MAX and make_shot(v, theta_deg).simulate().hit:
                    center = float(v)
                    break
            else:
                continue
            break
        else:
            return math.inf

    if make_shot(V0_MIN, theta_deg).simulate().hit:
        return V0_MIN

    lo, hi = V0_MIN, center
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if make_shot(mid, theta_deg).simulate().hit:
            hi = mid
        else:
            lo = mid
    return hi


def optimize_launch() -> tuple[float, float]:
    """
    先以 1° 粗搜，再在最优点附近以 0.02° 细搜。

    返回 (最佳角度/度, 最小初速度/m/s)。
    若速度相同，保留较小角度。
    """
    best_v = math.inf
    best_theta = math.nan
    tie_tol = 1.0e-5

    def consider(theta_deg: float) -> None:
        nonlocal best_v, best_theta
        v = minimum_speed_for_angle(theta_deg)
        if not math.isfinite(v):
            return
        if v < best_v - tie_tol or (
            abs(v - best_v) <= tie_tol and theta_deg < best_theta
        ):
            best_v = v
            best_theta = theta_deg

    # 粗搜
    coarse = np.arange(THETA_MIN, THETA_MAX + 1.0e-9, 1.0)
    for theta in coarse:
        consider(float(theta))

    if not math.isfinite(best_v):
        raise RuntimeError("在给定范围内没有找到命中方案")

    # 细搜
    fine_lo = max(THETA_MIN, best_theta - 1.5)
    fine_hi = min(THETA_MAX, best_theta + 1.5)
    fine = np.arange(fine_lo, fine_hi + 1.0e-9, 0.02)
    for theta in fine:
        consider(float(theta))

    return best_theta, best_v


def plot_result(result: Trajectory, theta_deg: float, v0: float) -> None:
    """绘制炮弹轨迹和目标位置。"""
    fig, ax = plt.subplots(figsize=(11, 5.2))

    ax.plot(result.x, result.y, color="tab:blue", lw=2.0, label="Projectile trajectory")
    ax.plot([0.0], [0.0], "ko", ms=6, label="Launch point")
    ax.plot(
        [TARGET[0]],
        [TARGET[1]],
        marker="*",
        color="red",
        ms=16,
        linestyle="None",
        label="Target (1000, 100)",
    )
    ax.axhline(0.0, color="0.4", lw=0.8)

    text = (
        rf"$\theta={theta_deg:.2f}^\circ$" "\n"
        rf"$v_0={v0:.3f}\ \mathrm{{m/s}}$" "\n"
        rf"$t={result.flight_time:.2f}\ \mathrm{{s}}$" "\n"
        rf"$h_{{\max}}={result.max_height:.2f}\ \mathrm{{m}}$"
    )
    ax.text(
        0.025,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.annotate(
        "target",
        xy=TARGET,
        xytext=(TARGET[0] - 220.0, TARGET[1] + 45.0),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )

    ax.set_xlim(0.0, TARGET[0] * 1.08)
    ax.set_ylim(0.0, max(result.max_height * 1.15, TARGET[1] * 1.5))
    ax.set_xlabel("Horizontal distance x / m")
    ax.set_ylabel("Height y / m")
    ax.set_title("Projectile trajectory with adiabatic air density")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig("adiabatic_projectile_trajectory.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    best_theta, best_v = optimize_launch()
    final = make_shot(best_v, best_theta).simulate()

    if not final.hit:
        raise RuntimeError("搜索结果未命中目标，请检查数值参数")

    print("=" * 42)
    print(f"最佳发射角度：{best_theta:.2f} °")
    print(f"最小初速度：{best_v:.3f} m/s")
    print(f"飞行时间：{final.flight_time:.3f} s")
    print(f"最高高度：{final.max_height:.3f} m")
    print(f"最近距离：{final.min_distance:.3f} m")
    print("=" * 42)

    # 无阻力理论值，仅作为对照
    xt, yt = TARGET
    r = math.hypot(xt, yt)
    v_vac = math.sqrt(G * (yt + r))
    theta_vac = 0.5 * (math.pi / 2.0 + math.atan2(yt, xt))
    print(
        "无空气阻力理论参考："
        f"theta = {math.degrees(theta_vac):.2f}°, "
        f"v0 = {v_vac:.2f} m/s"
    )

    peak_ratio = make_shot(best_v, best_theta).density_ratio(final.max_height)
    print(f"最高点空气密度比 rho/rho0：{peak_ratio:.4f}")

    plot_result(final, best_theta, best_v)


if __name__ == "__main__":
    main()

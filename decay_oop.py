"""
hw-4 计算物理第二次作业：衰变模型的 Euler 法与 OOP 实现。

分析题回答
----------
1. 为什么 error 恒为负？
   Euler 单步更新因子为 (1 - dt/tau)。
   对任意 dt > 0，有 1 - dt/tau < exp(-dt/tau)，
   所以每一步数值解都小于解析解，error = numerical - exact 恒为负，
   这是系统性误差，不是随机误差。

2. 右图斜率 ≈ 1 说明什么？
   log(max_abs_error) 与 log(dt) 近似为斜率 1 的直线，
   即 max_abs_error 正比于 dt，说明 Euler 法是一阶方法。

3. dt = 2.5 时为什么数值解发散？
   迭代放大因子为 1 - dt/tau。本例 tau = 1，dt = 2.5，
   |1 - dt/tau| = |1 - 2.5| = 1.5 > 1，
   每迭代一步振幅放大 1.5 倍，因此数值解振荡并指数发散。
"""

from pathlib import Path

import numpy as np


class Decay:
    """Radioactive decay model using Euler's method."""

    def __init__(self, N0, tau, dt, t_max):
        self.N0 = N0
        self.tau = tau
        self.dt = dt
        self.t_max = t_max

    # ---------- validated properties ----------

    @property
    def N0(self):
        return self._N0

    @N0.setter
    def N0(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("N0 must be positive")
        self._N0 = value

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("tau must be positive")
        self._tau = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("dt must be positive")
        self._dt = value

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("t_max must be positive")
        self._t_max = value

    # ---------- computed properties (always recomputed) ----------

    @property
    def t(self):
        """Grid t_k = k * dt, aligned with Euler points."""
        n_steps = int(np.floor(self._t_max / self._dt + 1e-12))
        return np.arange(n_steps + 1) * self._dt

    @property
    def numerical(self):
        """Euler numerical solution."""
        t = self.t
        n = np.empty_like(t)
        n[0] = self._N0
        for k in range(1, t.size):
            n[k] = self.euler_step(n[k - 1], self._dt, self._tau)
        return n

    @property
    def exact(self):
        """Analytical solution N0 * exp(-t / tau)."""
        return self._N0 * np.exp(-self.t / self._tau)

    @property
    def error(self):
        """Signed pointwise error: numerical - exact."""
        return self.numerical - self.exact

    @property
    def max_abs_error(self):
        return float(np.max(np.abs(self.error)))

    @property
    def half_life(self):
        return self._tau * np.log(2.0)

    # ---------- factory methods ----------

    @staticmethod
    def euler_step(n, dt, tau):
        """One Euler step: n - n * dt / tau."""
        return n - n * dt / tau

    @classmethod
    def from_half_life(cls, N0, t_half, dt, t_max):
        """Construct from half-life: tau = t_half / ln(2)."""
        if t_half <= 0:
            raise ValueError("t_half must be positive")
        tau = t_half / np.log(2.0)
        return cls(N0, tau, dt, t_max)


# ---------------------- plotting ----------------------

def make_plots(save_path="decay_oop.png"):
    import matplotlib
    matplotlib.use("Agg")  # 无图形界面服务器也能运行

    import matplotlib.pyplot as plt

    # 题目指定参数
    N0 = 100.0
    tau = 1.0
    t_max = 5.0
    dt_main = 0.2

    d_main = Decay(N0, tau, dt=dt_main, t_max=t_max)

    # 中间面板：不同 dt 的带符号误差
    dt_error_values = [0.5, 0.25, 0.2, 0.1, 0.05, 0.025]
    decays_for_error = [
        Decay(N0, tau, dt=dt, t_max=t_max)
        for dt in dt_error_values
    ]

    # 右面板：题目要求的 dt 列表
    dt_conv = np.array([0.5, 0.25, 0.2, 0.1, 0.05, 0.025])
    max_errs = np.array([
        Decay(N0, tau, dt=float(dt), t_max=t_max).max_abs_error
        for dt in dt_conv
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1, ax2, ax3 = axes

    # ---------- 左图：解析解 + Euler 数值解 ----------
    ax1.plot(d_main.t, d_main.exact, "k-", lw=2, label="Analytical")
    ax1.plot(
        d_main.t,
        d_main.numerical,
        "o",
        ms=3.5,
        color="tab:red",
        label=f"Euler, dt={d_main.dt:g}",
    )
    ax1.set_xlabel("t / s")
    ax1.set_ylabel("N (nuclei)")
    ax1.set_title(
        f"Analytical vs Euler\n"
        f"N0={N0:g}, tau={tau:g} s, dt={dt_main:g} s, t_max={t_max:g} s",
        fontsize=10,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---------- 中图：带符号误差 ----------
    for d in decays_for_error:
        ax2.plot(
            d.t,
            d.error,
            marker=".",
            ms=3,
            lw=1,
            label=f"dt={d.dt:g} s",
        )

    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("t / s")
    ax2.set_ylabel(r"$N_{\mathrm{Euler}} - N_{\mathrm{exact}}$ (nuclei)")
    ax2.set_title(
        f"Signed error for different dt\n"
        f"dt={dt_main:g} s: max |error| = {d_main.max_abs_error:.4f}",
        fontsize=10,
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---------- 右图：误差收敛阶 ----------
    ax3.loglog(dt_conv, max_errs, "o-", label="measured max |error|")
    ref = max_errs[0] * (dt_conv / dt_conv[0])
    ax3.loglog(dt_conv, ref, "k--", label="slope 1 reference")

    slope = np.polyfit(np.log(dt_conv), np.log(max_errs), 1)[0]

    ax3.set_xlabel(r"$\Delta t$ / s")
    ax3.set_ylabel(r"$\max |N_{\mathrm{Euler}} - N_{\mathrm{exact}}|$ (nuclei)")
    ax3.set_title(
        f"First-order convergence\n"
        f"fitted slope = {slope:.3f}",
        fontsize=10,
    )
    ax3.grid(True, which="both", ls=":", alpha=0.6)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")
    return slope


if __name__ == "__main__":
    try:
        out_path = Path(__file__).resolve().with_name("decay_oop.png")
    except NameError:
        out_path = Path("decay_oop.png")
    make_plots(out_path)

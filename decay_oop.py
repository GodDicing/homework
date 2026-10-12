"""hw-4: radioactive decay via Euler method, OOP + matplotlib."""

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
        """Euler numerical solution (1.3)-(1.4)."""
        t = self.t
        n = np.empty_like(t)
        n[0] = self._N0
        for k in range(1, t.size):
            n[k] = self.euler_step(n[k - 1], self._dt, self._tau)
        return n

    @property
    def exact(self):
        """Analytical solution (1.2)."""
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
    matplotlib.use("Agg")  # headless server

    import matplotlib.pyplot as plt

    N0 = 100.0
    tau = 2.0
    t_max = 10.0

    # panel 1: analytical + Euler
    d_main = Decay(N0, tau, dt=0.5, t_max=t_max)

    # panel 2: signed error for several dt
    dt_error_values = [0.05, 0.1, 0.25, 0.5, 1.0]
    decays_for_error = [
        Decay(N0, tau, dt=dt, t_max=t_max)
        for dt in dt_error_values
    ]

    # panel 3: convergence of max abs error with dt
    dt_conv = np.array([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
    max_errs = np.array([
        Decay(N0, tau, dt=float(dt), t_max=t_max).max_abs_error
        for dt in dt_conv
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1, ax2, ax3 = axes

    # left panel
    ax1.plot(d_main.t, d_main.exact, "k-", lw=2, label="Analytical")
    ax1.plot(
        d_main.t,
        d_main.numerical,
        "o",
        ms=3.5,
        color="tab:red",
        label=f"Euler, dt={d_main.dt:g}",
    )
    ax1.set_xlabel("t")
    ax1.set_ylabel("N(t)")
    ax1.set_title("Analytical vs Euler")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # middle panel
    for d in decays_for_error:
        ax2.plot(
            d.t,
            d.error,
            marker=".",
            ms=3,
            lw=1,
            label=f"dt={d.dt:g}",
        )
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("t")
    ax2.set_ylabel("numerical - exact (signed error)")
    ax2.set_title("Systematic signed error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # right panel
    ax3.loglog(dt_conv, max_errs, "o-", label="measured max |error|")
    ref = max_errs[0] * (dt_conv / dt_conv[0])
    ax3.loglog(dt_conv, ref, "k--", label="slope 1 reference")

    slope = np.polyfit(np.log(dt_conv), np.log(max_errs), 1)[0]
    ax3.text(
        0.05,
        0.08,
        f"fitted slope = {slope:.2f}",
        transform=ax3.transAxes,
        fontsize=9,
    )
    ax3.set_xlabel(r"$\Delta t$")
    ax3.set_ylabel(r"$\max |N_{\mathrm{Euler}} - N_{\mathrm{exact}}|$")
    ax3.set_title("First-order convergence")
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

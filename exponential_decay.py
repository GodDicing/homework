from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无图形界面服务器也能运行，必须放在 pyplot 之前

import numpy as np
import matplotlib.pyplot as plt


def main():
    # 参数
    N0 = 1.0
    lam = 0.5  # 不要写成 lambda，它是 Python 关键字

    # numpy 离散采样
    t = np.linspace(0, 10, 500)
    N = N0 * np.exp(-lam * t)

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(t, N, label=r'$N(t)=N_0 e^{-\lambda t}$')

    plt.xlabel('t')
    plt.ylabel('N(t)')
    plt.title('Exponential Decay')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    output_path = base_dir / "exponential_decay.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

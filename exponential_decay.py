import numpy as np
import matplotlib.pyplot as plt

# 参数
N0 = 1.0
lam = 0.5

# 离散采样
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

# 保存图片，方便插入课件
plt.savefig('exponential_decay.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
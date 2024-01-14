import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='DengXian')

# 信号的频率
F1 = 30
F2 = 35

# 采样区间
T = 0.02

# 数据长度
N_values = [10, 15, 30, 40, 60, 70, 100]

# 时间向量
t = np.arange(0, 1, T)

# 生成信号
signal1 = np.sin(np.pi * F1 * t)
signal2 = np.sin(np.pi * F2 * t)

# 混合两个信号
mixed_signal = signal1 + signal2

# 对每个数据长度进行处理
for N in N_values:
    # 截取信号
    mixed_signal_N = mixed_signal[:N]

    # 填0扩充至512个点
    mixed_signal_padded = np.pad(mixed_signal_N, (0, 512 - N), 'constant')

    # 进行DFT变换
    dft_transformed = np.fft.fft(mixed_signal_padded)

    # 画出图像
    plt.figure(figsize=(6, 4))
    plt.plot(np.abs(dft_transformed))
    plt.title('DFT处理后的音频, N = {}'.format(N))
    plt.show()

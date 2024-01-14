import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import pywt
import matplotlib
matplotlib.rc("font",family='DengXian')

# 1. 读取wav音频文件
rate, data = wav.read('2.wav')

# 2. 以1024长度对音频分窗处理
window_size = 1024
windows = [data[i:i+window_size] for i in range(0, len(data), window_size)]

# 对每个窗口进行DFT，DCT，DWT处理
dft_windows = [np.fft.fft(window) for window in windows]
dct_windows = [dct(window, norm='ortho') for window in windows]
dwt_windows = [pywt.dwt(window, 'db1') for window in windows]

# 画出原始音频，以及处理后音频的图形
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title('原始音频')
plt.show()

plt.figure(figsize=(10, 4))
for dft_window in dft_windows:
    plt.plot(np.abs(dft_window))
plt.title('DFT处理后的音频')
plt.show()

plt.figure(figsize=(10, 4))
for dct_window in dct_windows:
    plt.plot(dct_window)
plt.title('DCT处理后的音频')
plt.show()

plt.figure(figsize=(10, 4))
for dwt_window in dwt_windows:
    plt.plot(dwt_window[0])
plt.title('DWT处理后的音频')
plt.show()

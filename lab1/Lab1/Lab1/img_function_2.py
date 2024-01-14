import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import matplotlib
matplotlib.rc("font",family='DengXian')

# 读取图像
img = cv2.imread('1.bmp', cv2.IMREAD_GRAYSCALE)

# 选择一个图像块
block = img[0:500, 0:500]

# 1. 使用一维DFT变换
start_time = time.time()
dft = np.fft.fft2(block)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(np.abs(dft_shift))
phase_spectrum = np.angle(dft_shift)
print("一维DFT变换时间：", time.time() - start_time)

plt.figure(figsize=(12, 12))

plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('幅度图'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('相位图'), plt.xticks([]), plt.yticks([])

plt.show()

# 2. 使用FFT变换
start_time = time.time()
fft = np.fft.fft2(block)
fft_shift = np.fft.fftshift(fft)
print("FFT变换时间：", time.time() - start_time)


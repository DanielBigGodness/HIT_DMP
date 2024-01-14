import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 读取图像
img = cv2.imread('1.bmp', cv2.IMREAD_GRAYSCALE)

# 选择一个8x8的图像块
block = img[0:250, 0:250]

# 进行DCT变换
dct_transformed = dct(dct(block.T, norm='ortho').T, norm='ortho')

# 保留左上角k个系数
k = 10
dct_transformed_k = dct_transformed.copy()
dct_transformed_k[k:, :] = 0
dct_transformed_k[:, k:] = 0

# 进行逆DCT变换
idct_transformed = idct(idct(dct_transformed_k.T, norm='ortho').T, norm='ortho')

# 计算PSNR和SSIM
psnr = peak_signal_noise_ratio(block, idct_transformed)
ssim = structural_similarity(block, idct_transformed)

print("PSNR: ", psnr)
print("SSIM: ", ssim)

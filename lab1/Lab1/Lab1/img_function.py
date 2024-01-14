from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from random import *

# 要求1：获取任意一点的像素值
def getPixel(x,y):
    im=Image.open('1.bmp')#文件的路径
    im2=im.convert("RGB")
    print(im2.mode)
    print(im2.getpixel((x,y)))#（0，0）表示像素点的坐标

# 要求2： 将任意一行和一列的像素值在窗口显示
def drawRow(row):
    pixels = list(img.getdata())
    width, height = img.size
    print(width,height)
    row_pixels = pixels[row*width:(row+1)*width]
    plt.plot(row_pixels)
    plt.show()

def drawCol(col):
    pixels = list(img.getdata())
    width, height = img.size
    col_pixels = [pixels[i*width+col] for i in range(height)]
    plt.plot(col_pixels)
    plt.show()


# 要求3 ：统计图像的像素直方图
def getHist(image_path: str):
    # 一维直方图（单通道直方图）
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)
    color = ('blue', 'green', 'red')
    # 使用plt内置函数直接绘制
    plt.hist(img.ravel(), 20, [0, 256])
    plt.show()
    # 一维像素直方图，也即是单通道直方图
    for i, color in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        #print(hist)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()

# 要求3 ： 计算图像的信息熵
def getEntropy():
    # img = cv2.imread('20201210_3.bmp',0)
    # img = np.zeros([16,16]).astype(np.uint8)

    a = [i for i in range(256)]
    img = np.array(a).astype(np.uint8).reshape(16, 16)

    hist_cv = cv.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数
    P = hist_cv / (len(img) * len(img[0]))  # 概率
    E = np.sum([p * np.log2(1 / p) for p in P])
    print("一维熵：",E)  # 熵

# 要求4 ： 能将图像分成任意块大小，并置乱块的位置并显示（类似马赛克
def PermutationFun(inputImage, blockwidth, blockheight, sed):
    seed(sed)
    width, height = inputImage.size
    xblock = width // blockwidth
    yblock = height // blockheight
    regions = []

    for i in range(0, yblock * blockheight, blockheight):
        for j in range(0, xblock * blockwidth, blockwidth):
            region = inputImage.crop((j, i, j + blockwidth, i + blockheight))
            regions.append(region)
    shuffle(regions)
    outputImage = Image.new('RGB', (width, height))
    idx = 0
    for i in range(0, yblock * blockheight, blockheight):
        for j in range(0, xblock * blockwidth, blockwidth):
            outputImage.paste(regions[idx], (j, i))
            idx += 1

    plt.imshow(outputImage)
    plt.show()


# 要求5 ： 能够截图任意一个区域并存成一幅图像
def read_bmp(filename):
    with open(filename, 'rb') as f:
        return bytearray(f.read())

def write_bmp(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

def crop_bmp(input_filename, output_filename, left, top, right, bottom):
    data = read_bmp(input_filename)
    file_header = data[:14]
    info_header = data[14:54]
    pixels = data[54:]
    width = int.from_bytes(info_header[4:8], byteorder='little')
    height = int.from_bytes(info_header[8:12], byteorder='little')
    row_size = (width * 3 + 3) & ~3
    pixel_array_offset = int.from_bytes(file_header[10:14], byteorder='little')
    cropped_pixels = bytearray()
    for y in range(top, bottom):
        start = pixel_array_offset + y * row_size + left * 3
        end = start + (right - left) * 3
        cropped_pixels.extend(pixels[start:end])
    new_width = right - left
    new_height = bottom - top
    new_row_size = (new_width * 3 + 3) & ~3
    new_pixel_array_size = new_row_size * new_height
    new_file_size = 14 + 40 + new_pixel_array_size
    file_header[2:6] = new_file_size.to_bytes(4, byteorder='little')
    info_header[4:8] = new_width.to_bytes(4, byteorder='little')
    info_header[8:12] = new_height.to_bytes(4, byteorder='little')
    info_header[20:24] = new_pixel_array_size.to_bytes(4, byteorder='little')
    new_data = file_header + info_header + cropped_pixels
    write_bmp(output_filename, new_data)

def crop2(a,b,c,d):
    from PIL import Image
    img = Image.open('1.bmp')
    width, height = img.size

    # 前两个坐标点是左上角坐标
    # 后两个坐标点是右下角坐标
    # width在前， height在后
    box = (a, b, c, d)

    region = img.crop(box)
    region.save('crop.bmp')


if __name__ == '__main__':
    image = '1.bmp'
    img = Image.open('1.bmp')
    getPixel(10,10)       #要求一
    drawRow(10)    #要求二
    drawCol(10)
    getHist(image)        #要求三
    getEntropy()
    PermutationFun(img,10,10,12)    #要求四
    #crop_bmp(image,"crop.bmp", 0,0,500,500)       #要求五
    crop2(10,10,400,400)



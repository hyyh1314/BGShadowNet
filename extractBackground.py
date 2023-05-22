#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn import mixture
import cv2
import os

def get_average_color(x):
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return np.array([np.mean(b), np.mean(g), np.mean(r)])

def gaus_kernel(winsize, gsigma):
    r = int(winsize/2)
    c = r
    kernel = np.zeros((winsize, winsize))
    sigma1 = 2*gsigma*gsigma
    for i in range(-r, r+1):
        for j in range(-c, c+1):
            kernel[i+r][j+c] = np.exp(-float(float((i*i+j*j))/sigma1))
    return kernel

def Bilater(image,I_2,gsigma,ssigma,winsize):
    r = int(winsize/2)
    c = r
    image1 = np.pad(image, ((r, c),(r, c)), constant_values=0)
    I_2 = np.pad(I_2, ((r, c),(r, c),(0,0)), constant_values=0)
    image = image1
    row, col = image.shape    
    sigma2 = 2*ssigma*ssigma
    gkernel = gaus_kernel(winsize, gsigma)
    kernel = np.zeros((winsize, winsize))
    bilater_image = np.zeros((row, col,3))
    bilater_image_2 = np.zeros((row, col,3))

    for i in range(r,row-r):
        for j in range(c,col-c):
            skernel = np.zeros((winsize, winsize))
            skernel_1 = np.zeros((winsize, winsize))
            skernel=np.exp(-np.power((image[i,j]-image[i-r:i+r+1,j-c:j+c+1]),2)/sigma2)
            kernel = skernel*gkernel
            sum_kernel = sum(sum(kernel))
            kernel = kernel/sum_kernel
            temp = np.empty(3)
            for channel in range(3):
                bilater_image_2[i][j][channel] = np.sum(I_2[i-r:i+r+1,j-c:j+c+1,channel]*kernel)  
    return bilater_image_2[r:-r,c:-c,:]

patch_size = 32
gsigma=10
ssigma=3
winsize=31
r = int(winsize/2)
c = r

img_path = './dataset/Jung/test/gt/'
paths = os.listdir(img_path)
paths.sort()

len(paths)
for i in range(0,len(paths)):
    path = paths[i]
    x = cv2.imread(img_path+path)
    I1 = cv2.imread(img_path+path, cv2.IMREAD_GRAYSCALE)#读取灰度图片，用来计算双边滤波的kernel
    I1  = np.asarray(I1 , dtype = float)
    h, w, c = x.shape
    background_img = np.empty((h,w,c),dtype=np.uint8)
    for sub_h in range(0,h,patch_size):
        sub_h_end = min(sub_h+patch_size,h)
        for sub_w in range(0,w,patch_size):
            sub_w_end = min(sub_w+patch_size,w)
            patch = x[sub_h:sub_h_end,sub_w:sub_w_end,:]
            patch_h,patch_w,patch_c = patch.shape
            patch = patch.flatten().reshape(patch_h*patch_w, patch_c)
            gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
            gmm.fit(patch)
            cls = gmm.predict(patch.flatten().reshape(patch_h*patch_w,patch_c))
            cls0_colors = patch[cls == 0]
            cls1_colors = patch[cls == 1]
            cls0_avg_color = get_average_color(cls0_colors)
            cls1_avg_color = get_average_color(cls1_colors)
            if np.sum(cls0_avg_color)>=np.sum(cls1_avg_color) or np.isnan(cls1_avg_color).any():
                background_img[sub_h:sub_h_end,sub_w:sub_w_end,:] = cls0_avg_color
            else:
                background_img[sub_h:sub_h_end,sub_w:sub_w_end,:] = cls1_avg_color
    bilater_image_2 = Bilater(I1,background_img,gsigma,ssigma,winsize)
    cv2.imwrite('./dataset/Jung/test/back_gt/{:s}'.format(path), bilater_image_2)
    print(path)







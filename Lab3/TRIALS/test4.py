import cv2 as cv
from sift import sift as sift_corresp
import numpy as np
import math as m 
from PIL import Image
import random

img1  = np.array(Image.open('./img1.png'))  
img2  = np.array(Image.open('./img2.png'))
img3  = np.array(Image.open('./img3.png'))

threshold = 0.98
points = 4

corresp1, corresp2 = sift_corresp(img1,img2)
all_points = corresp2.shape[0]


def finding_homo(all_points ) :
       iterations = 0
       homo_points = 4
       max_iterations = 15

       A = np.zeros((8, 9))
       flag = 0

       while iterations < max_iterations and flag == 0:
        random_index = np.random.choice(all_points , size = 4 , replace= False)
        for i in range(homo_points):
              all_points = corresp2.shape[0]
              xi , yi = corresp1[random_index[i]]
              xi_ , yi_ = corresp2[random_index[i]]
              A[2*i] = np.array([-xi , -yi , -1 , 0 , 0 , 0 , xi*xi_ , xi_*yi , xi_ ])
              A[(2*i)+1] = np.array([0 , 0 , 0 , -xi , -yi , -1 , xi*yi_ , yi*yi_ , yi_])
        [U, S, Vt] = np.linalg.svd(A)
        H = np.reshape(Vt[-1], (3,3))
        remaining_points = [point for point in range(all_points) if point not in random_index]
      
        inliers = 0
        for i in remaining_points:
              homo_corresp1 = corresp1[i]
              homo_corresp1  = np.array([homo_corresp1[0] ,  homo_corresp1[1] , 1])
              homo_corresp2 = H @ homo_corresp1
              x_h , y_h = float(homo_corresp2[0] / homo_corresp2[2]), float(homo_corresp2[1] / homo_corresp2[2])
              x_s , y_s = corresp2[i][0] , corresp2[i][1]
              err = np.sqrt((x_s - x_h)**2 + (y_s - y_h)**2)
              print(err)
              E = 5
              
              if err < E:
                     inliers += 1

        print('inliers',inliers)
        iterations += 1
        print('iterations',iterations)
        if ((inliers/ (all_points-4))> threshold):
              flag = 1
       
       return H

H21 = finding_homo(all_points)
print(H21)


def bilinear_interpolation(input, x, y):
    x_ = m.floor(x)
    y_ = m.floor(y)
    a = x - x_
    b = y - y_
    h, w = input.shape

    if 0 <= x_ < h - 1 and 0 <= y_ < w - 1:
        It_1 = (1 - a) * (1 - b) * input[x_, y_]
        It_2 = (1 - a) * b * input[x_, y_ + 1]
        It_3 = a * b * input[x_ + 1, y_ + 1]
        It_4 = a * (1 - b) * input[x_ + 1, y_]

        return It_1 + It_2 + It_3 + It_4
    else:
        return 255
    

'''

EE5175 - Lab 
Date : 03/02/2024
command to run : python3 assignment_3.py

'''


import cv2 as cv
from sift import sift as sift_corresp
import numpy as np
import math as m 
from PIL import Image
import random
import imageio

threshold = 0.7
points = 5

'''
Function to find Homography matrix :
input : 
       two images whose homography we want to find 

RANSAC Algorithm implemented to find good homography matrix 

output : 
       good homography matrix (found using RANSAC)

'''
def homography_mat(image1 , image2 ) :
       iterations = 0
       homo_points = 4
       max_iterations = 30
       corresp1, corresp2 = sift_corresp(image1,image2)
       all_points = corresp2.shape[0]

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

'''
Bilinear Interpolation Function:
    input: 
        - input: Coordinates (x, y) for interpolation
        - x: x-coordinate
        - y: y-coordinate
    output: 
        - Returns the average intensity value obtained through bilinear interpolation of the four neighboring points.
'''

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
    

'''
Average Intensity Calculation:
    input:
        - V: List of intensity values
    output:
        - Returns the average intensity value .
'''

def avg_intensity(V):
    valid_values = [v for v in V if 0 < v <= 255]
    if len(valid_values) == 0:
        return 0
    else:
        return np.mean(valid_values)
    
'''
Image Stitching Function:
    input:
        - 3 images to be stitched
        - canvas_size: Size of the canvas for stitching
        - offset: Offset values for aligning images
    output:
        - Returns the stitched image using homography matrices and bilinear interpolation.
'''
    
def stitch(image1, image2, image3, canvas_size=(500, 1500, 3), offset=(50, 300)):
          
     H21 = homography_mat(image2 , image1)
     H23 = homography_mat(image2 , image3)
     canvas = np.zeros(canvas_size, dtype=np.uint8)

     h , w = canvas.shape[:2]
     for i in range(h):
          for j in range(w):
               x2 = i - offset[0]  # offset
               y2 = j - offset[1]
               tmp = np.matmul( H21 , np.array([x2 , y2, 1]))
               x1 , y1 = tmp[0]/tmp[2] , tmp[1]/tmp[2]
               tmp = np.matmul( H23 , np.array([x2 , y2 ,1]))
               x3 , y3 = tmp[0]/tmp[2] , tmp[1]/tmp[2]
               v1 = bilinear_interpolation(image1 , x1 , y1 )
               v2 = bilinear_interpolation(image2 , x2 , y2 )
               v3 = bilinear_interpolation(image3 , x3 , y3 )
               V = [v1 , v2 , v3]
               canvas[i , j] = avg_intensity(V)
     return canvas
  

if __name__ == "__main__":
    # img1  = np.array(Image.open('./img1.png'))  
    # img2  = np.array(Image.open('./img2.png'))
    # img3  = np.array(Image.open('./img3.png'))

    my_img1 =  cv.resize(cv.imread('./rp1.jpg', cv.IMREAD_GRAYSCALE), (0, 0), fx=0.5, fy=0.5)
    my_img2 =  cv.resize(cv.imread('./rp3.jpg', cv.IMREAD_GRAYSCALE), (0, 0), fx=0.5, fy=0.5)
    my_img3 =  cv.resize(cv.imread('./rp2.jpg', cv.IMREAD_GRAYSCALE), (0, 0), fx=0.5, fy=0.5)
    
    # s1 = stitch(img1 , img2 , img3 , canvas_size=(800, 1500, 3), offset=(200, 300))
    s2 = stitch(my_img1 , my_img2 , my_img3 , canvas_size=(6500, 6500, 3), offset=(1500, 2000))

    # imageio.imwrite('stitch1.jpg', s1)
    imageio.imwrite('stitch2__.jpg', s2)


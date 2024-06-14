import cv2 as cv
from sift import sift as sift_corresp
import numpy as np
import math as m 
from PIL import Image
import random
import sys

# with open('output.txt', 'w') as f:
#      sys.stdout = f

img1  = np.array(Image.open('./img1.png'))  
img2  = np.array(Image.open('./img2.png'))
img3  = np.array(Image.open('./img3.png'))

[corresp1, corresp2] = sift_corresp(img1,img2)
# print(corresp1.shape)
all_points = corresp2.shape[0]
# random_index = random.randint(0 , all_points-1)

# param: no. of points
threshold = 0.9 
points = 4
selected_points = []
flag = 0
iter = 0
while flag == 0:
    iter +=1
    print(iter)
    A = np.zeros((8, 9))
    # random_index = random.randint(0 , all_points-1)
    
    for i in range(points):
        random_index = random.randint(0 , all_points-1)
        # print('random_index',random_index)
        print(f"point {i+1} : " , random_index)
        selected_points.append(random_index)
        xi , yi = corresp1[random_index]
        xi_ , yi_ = corresp2[random_index]
        A[2*i] = [-xi , -yi , 1 , 0 , 0 , 0 , xi*xi_ , xi_*yi , xi_ ]
        A[(2*i)+1] = [0 , 0 , 0 , -xi , -yi , -1 , xi*yi , yi*yi_ , yi_]

    [U, S, Vt] = np.linalg.svd(A)
    H = np.reshape(Vt[-1], (3,3))
            
    # print(selected_points)
    remaining_points = [point for point in range(all_points) if point not in selected_points]
    # print(remaining_points)
    inliers = 0
    '''  RANSAC  '''
    for i in range(len(remaining_points)):
        # print(remaining_points[i])
        index = remaining_points[i]
        homo_corresp1 = corresp1[index]
        homo_corresp1 = (np.append(homo_corresp1, 1)).reshape(3,1)
        #  print(i)
        # print(homo_corresp1.shape)
        homo_corresp2 = np.dot( H , homo_corresp1)
        # print(homo_corresp2.shape)

        # print(homo_corresp2)
        # homo_corresp = [homo_corresp1 , homo_corresp2]
        Ph = np.array([[homo_corresp2[0] / homo_corresp2[2] ], [homo_corresp2[1] / homo_corresp2[2]]])
        # print(Ph)
        # print(xi_h , yi_h)
        # print(homo_corresp)
    #         ''' finding sift corresp'''
        # [sift_corresp1 , sift_corresp2] = sift_corresp(img1,img2)
        Ps = np.array([[corresp2[index][0] ], [corresp2[index][1]]])
    #         # print(xi_s , yi_s)
        print(Ps.shape)
        err = np.linalg.norm(Ps - Ph)
    #    E = 5
        print(err)
        E = 5
    
        if err < E:
            inliers += 1
    # print(inliers)
    print('__________')
    print('__________')

    if (inliers/ (all_points-points))> threshold:
        flag = 1

# print(H)
# print(inliers)
    
    # print(f'E{i} is '  , err )
# print(sift_corresp1)
# print(sift_corresp1 , sift_corresp2)

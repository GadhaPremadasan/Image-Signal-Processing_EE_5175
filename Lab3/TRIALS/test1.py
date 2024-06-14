# import cv2 as cv
# from sift import sift as sift_corresp
# import numpy as np
# import math as m 
# from PIL import Image
# import random

# img1  = np.array(Image.open('./img1.png'))  
# img2  = np.array(Image.open('./img2.png'))
# img3  = np.array(Image.open('./img3.png'))

# [corresp1, corresp2] = sift_corresp(img1,img2)
# # print(corresp1.shape)
# all_points = corresp2.shape[0]
# random_index = random.randint(0 , all_points-1)

# # param: no. of points 
# points = 4

# # xi , yi = corresp1[random_index]
# # xi_ , yi_ = corresp2[random_index]
# # print(xi_)
# # Xi = np.array([xi , yi , 1])
# # Xi_ = np.array([xi_ , yi_ , 1])
# # Xi = Xi.reshape(3,1)
# # Xi_ = Xi_.reshape(3, 1)



# A = np.zeros((8, 9))
# for i in range(points):
#     random_index = random.randint(0 , all_points-1)
#     xi , yi = corresp1[random_index]
#     xi_ , yi_ = corresp2[random_index]
#     A[2*i] = [-xi , -yi , 1 , 0 , 0 , 0 , xi*xi_ , xi_*yi , xi_ ]
#     A[(2*i)+1] = [0 , 0 , 0 , -xi , -yi , -1 , xi*yi , yi*yi_ , yi_]
#     xi , yi = corresp1[random_index]
#     xi_ , yi_ = corresp2[random_index]

# # [h11 , h12 , h13 , h21 , h22 , h23 , h31 , h32 , h33] = H
#     # print(xi_)
# # B = np.round(A , 2)
# # A[:] = np.round(A, decimals=2)
# # print(B)
#     # Xi = np.array([xi , yi , 1])
#     # Xi_ = np.array([xi_ , yi_ , 1])
#     # Xi = Xi.reshape(3,1)
#     # Xi_ = Xi_.reshape(3, 1)

# # print(A)
# # Z = np.zeros((8 , 1 ))
# # H = np.linalg.solve(Z , np.linalg.inv(A))
# [U, S, Vt] = np.linalg.svd(A)
# # print(Vt)
# H21 = np.reshape(Vt[-1], (3,3))
# print(H21)

# # [h11 , h12 , h13 , h21 , h22 , h23 , h31 , h32 , h33] = H
#     # print(xi_)
# # B = np.round(A , 2)
# # A[:] = np.round(A, decimals=2)
# # print(B)
#     # Xi = np.array([xi , yi , 1])
#     # Xi_ = np.array([xi_ , yi_ , 1])
#     # Xi = Xi.reshape(3,1)
#     # Xi_ = Xi_.reshape(3, 1)



# # print(Xi.shape)
# # print(xi , yi)
# # print(xi_ , yi_)

# # Xi = corresp1[random_index]
# # Xi_ = corresp2[random_index]
# # print(type(Xi))
# # print(Xi)
# # print(np.linalg.inv(Xi))
# # H = np.linalg.solve(Xi_ , np.linalg.inv(Xi))
# # print(H)
# ''' now we found H21'''

# ''' similarly for H23 '''

# [corresp2, corresp3] = sift_corresp(img2,img3)


# selected_points = []

# A = np.zeros((8, 9))
# for i in range(points):
#     random_index = random.randint(0 , all_points-1)
#     print(f"point {i+1} : " , random_index)
#     selected_points.append(random_index)
#     xi , yi = corresp1[random_index]
#     xi_ , yi_ = corresp2[random_index]
#     A[2*i] = [-xi , -yi , 1 , 0 , 0 , 0 , xi*xi_ , xi_*yi , xi_ ]
#     A[(2*i)+1] = [0 , 0 , 0 , -xi , -yi , -1 , xi*yi , yi*yi_ , yi_]
    
# print(selected_points)
# # [h11 , h12 , h13 , h21 , h22 , h23 , h31 , h32 , h33] = H
#     # print(xi_)
# # B = np.round(A , 2)
# # A[:] = np.round(A, decimals=2)
# # print(B)
#     # Xi = np.array([xi , yi , 1])
#     # Xi_ = np.array([xi_ , yi_ , 1])
#     # Xi = Xi.reshape(3,1)
#     # Xi_ = Xi_.reshape(3, 1)

# # print(A)
# # Z = np.zeros((8 , 1 ))
# # H = np.linalg.solve(Z , np.linalg.inv(A))
    
    
# [U, S, Vt] = np.linalg.svd(A)
# # print(Vt)
# H21 = np.reshape(Vt[-1], (3,3))
# # print(H21)


# # def finding_homo(image_1 , images_2) :
# #  [corresp1, corresp2] = sift_corresp(image_1,images_2)
# #  A = np.zeros((8, 9))
# #  for i in range(points):
# #         random_index = random.randint(0 , all_points-1)
# #         xi , yi = corresp1[random_index]
# #         xi_ , yi_ = corresp2[random_index]
# #         A[2*i] = [-xi , -yi , 1 , 0 , 0 , 0 , xi*xi_ , xi_*yi , xi_ ]
# #         A[(2*i)+1] = [0 , 0 , 0 , -xi , -yi , -1 , xi*yi , yi*yi_ , yi_]
# #  [U, S, Vt] = np.linalg.svd(A)
# #  H = np.reshape(Vt[-1], (3,3))
# #  return H

# # H23 = finding_homo(img2 , img3)
# # print(H23)


# ''' RANSAC '''
# remaining_points = [point for point in all_points if point not in selected_points]

import cv2 as cv
from sift import sift as sift_corresp
import numpy as np
import math as m 
from PIL import Image
import random

img1  = np.array(Image.open('./img1.png'))  
img2  = np.array(Image.open('./img2.png'))
img3  = np.array(Image.open('./img3.png'))

[corresp1, corresp2] = sift_corresp(img1,img2)
# print(corresp1.shape)
all_points = corresp2.shape[0]
random_index = random.randint(0 , all_points-1)

# param: no. of points 
points = 4
selected_points = []

A = np.zeros((8, 9))
for i in range(points):
    random_index = random.randint(0 , all_points-1)
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





for i in range(len(remaining_points)):
    homo_corresp1 = corresp1[remaining_points[i]]
    homo_corresp1 = np.append(homo_corresp1, 1)
    homo_corresp2 = np.matmul(H, homo_corresp1)

    # Extract coordinates from homogeneous coordinates
    homo_corresp1 = homo_corresp1[:2] / homo_corresp1[2]
    homo_corresp2 = homo_corresp2[:2] / homo_corresp2[2]

    # Finding sift corresp
    sift_corresp1, sift_corresp2 = sift_corresp(img1, img2)

    # Calculate Euclidean distance
    distance = np.linalg.norm(homo_corresp2 - sift_corresp2[remaining_points[i]])
    print(f"Euclidean distance: {distance}")

    # You can store or use the distance as needed

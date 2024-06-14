import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

image = np.array(Image.open('Globe.png').convert("L"))
h,w = image.shape[:2]
N = h
# print(h,w)

A = 2.0
B = (N*N)/(2*np.log(200))
sigma_mat = np.zeros((N,N))
# print(sigma_mat)
# filling values in sigma matrix
largest_k = 0
for m in range(N):
   for n in range(N):
    exp_term = np.exp(-((m - (N/2))**2 + (n - (N/2))**2) / B )
    sigma1 = A*exp_term 
    
    sigma_mat[m,n] = sigma1
    Sum = 0
    for i in range(m):
        # Traverse in column of that row
        for j in range(n):
 
            # Add element in variable sum
            sigma = sigma_mat[i][j]
            k = math.ceil(6 * sigma) + 1
            k_mid = k // 2
            kernel = np.zeros((k,k) , np.float32)
            for y in range(-k_mid, k_mid+1):  
             for x in range(-k_mid , k_mid+1):    
                normal = 1 / ( 2 * np.pi * sigma**2 )
                exp_term = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
                kernel [ y + k_mid , x + k_mid ] = normal * exp_term


# print(Sum)
# ''' now padding is something to be done only once. and it has to be done for the largest sigma , we cant keep padding for different values of sigma :)
# so in next is how to find this max length to which k can go.'''
#     if k > largest_k:
#            largest_k = k
# largest_mid = largest_k//2
# output_image = np.zeros((h, w), dtype=np.float32)
# output_image = np.pad(image,(int(( largest_k - 1 )/2), int(( largest_k - 1 )/2)), 'constant')
# print(len(sigma_mat))
# for i in range(len(sigma_mat)):
#    for j in range(len(sigma_mat)):
            
#         sigma = sigma_mat[i,j]
#    print(sigma)
# sigma_sum = np.sum(sigma_mat)
# print("Sum of all sigma values:", sigma_sum)

    # print(np.sum(sigma))
        # k = math.ceil(6 * sigma) + 1
        # k_mid = k // 2
        # print(k_mid)
        # kernel = np.zeros((k,k) , np.float32)
        # for y in range(-k_mid, k_mid+1):  
        #  for x in range(-k_mid , k_mid+1):    
        #     normal = 1 / ( 2 * np.pi * sigma**2 )
        #     exp_term = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
        #     kernel [ y + k_mid , x + k_mid ] = normal * exp_term
    
# temp_out = np.zeros((h,w))
# for i in range(h):
#    for j in range(w):
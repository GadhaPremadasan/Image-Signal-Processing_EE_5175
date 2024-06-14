import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

image = np.array(Image.open('Globe.png').convert("L"))
h,w = image.shape[:2]
N = h
print(h,w)

A = 2.0
B = (N*N)/(2*np.log(200))
sigma_mat = np.zeros((N,N))
# print(sigma_mat)
# filling values in sigma matrix
largest_k = 0

for m in range(N):
    for n in range(N):
     exp_term = np.exp(-((m - (N/2))**2 + (n - (N/2))**2) / B )
     sigma = A*exp_term 
     k = math.ceil(6 * sigma) + 1
     kernel = np.zeros((k,k) , np.float32)
     print(kernel.shape)
    # print(k)
     k_mid = k // 2
     for y in range( -k_mid , k_mid+1 ):
      for x in range( -k_mid , k_mid+1 ):
        normal = 1 / ( 2 * np.pi * sigma**2 )
        exp_term1 = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
        G = normal * exp_term1
    # print(y+k_mid)
        kernel [ y + k_mid , x + k_mid ] = normal * exp_term1
    kernel = kernel/np.sum(kernel)
# print(kernel)
#      sigma_mat[m,n] = sigma
# # ''' now padding is something to be done only once. and it has to be done for the largest sigma , we cant keep padding for different values of sigma :)
# # so in next is how to find this max length to which k can go.'''
#      if k > largest_k:
#             largest_k = k
# # print(largest_k)
# # ''' yay! we found out largest k. now padding ''' 
# output_image = np.zeros((h, w), dtype=np.float32)
# output_image = np.pad(image,(int(( largest_k - 1 )/2), int(( largest_k - 1 )/2)), 'constant')
# # print(output_image.shape)

# #''' padding done successfully
# # now for each pixel , kernel is different.'''
# temp_out = np.zeros((h,w))
# for y in range(-k_mid , k_mid+1):
#    for x in range(-k_mid , k_mid+1 ):
#     normal = 1 / ( 2 * np.pi * sigma**2 )
#     exp_term1 = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
#     kernel [ y + k_mid , x + k_mid ] = normal * exp_term1
    
 
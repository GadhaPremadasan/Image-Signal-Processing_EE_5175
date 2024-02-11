'''

EE5175 - Lab 1 
Date : 03/02/2024
command to run : assignment_1.py

'''


# import necessary libraries

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.image import imread 
import math as m

image1 = imread('lena_translate.png')  # 

tx = 3.75
ty = 4.3

h, w = image1.shape[:2]
target = np.zeros_like(image1) # Initialising target image array with all zero array with same shape as the source image
 
'''

bilinear interpolation function
Input: Source image, tx and ty coordinates
Output: returns the target image

'''
 
def bilinear_interpolation(input,tx,ty):
 for i in range(h):
    for j in range(w):
        x = i - tx
        y = j - ty

        x_ = m.floor(x)
        y_ = m.floor(y)
        a = x - x_
        b = y - y_

        if x<0 or x>=h-1 or y<0 or y>w-1:
            target[i,j] = 0
        else:

            It_1 = (1 - a) * (1 - b) * input[x_, y_]
            It_2 = (1 - a) * b * input[x_, y_ + 1]
            It_3 = a * b * input[x_ + 1, y_ + 1]
            It_4= a * (1 - b) * input[x_ + 1, y_]

            target[i,j] = It_1 + It_2 + It_3 + It_4
 
 return target
      
# target = bilinear_interpolation(image1,tx,ty)
# print(target)
            
output1 = bilinear_interpolation (image1, tx, ty) #
fig = plt.figure()
row = 1
column = 2
fig.add_subplot(row,column , 1)
plt.imshow(image1 ,cmap = 'gray')
plt.title('input image')

fig.add_subplot(row,column , 2)
plt.imshow(target,cmap = 'gray')
plt.title('translated image')

plt.show()



'''
Question 2

'''


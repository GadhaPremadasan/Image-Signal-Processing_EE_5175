'''

EE5175 - Lab 2 
Date : 15/02/2024
Command to run : python3 assignment_2.py

'''

from PIL import Image
import numpy as np
import math as m 
import matplotlib.pyplot as plt

img1  = np.array(Image.open('./IMG1.png'))  
img2  = np.array(Image.open('./IMG2.png'))

A = np.array ([
     
    [29 , 124 , 1 , 0] , 
    [124 , -29 , 0 , 1] ,     # This is the matrix created to solve for the values        
    [157 , 372 , 1 , 0] ,     # of rotation (sin(theta) , cos(theta)), translation(tx and ty)
    [372 , -157 , 0 , 1]

])

B = np.array([93 , 248 , 328 , 399])  

x = np.linalg.solve(A , B)

a , b , c , d = x[0] , x[1] , x[2] , x[3]   # values of sin , cos , tx and ty 
                                            # given to a , b , c and d respectively

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


# Here onwards we are trying to find source coordinate x and y 
# from the matrix C . 
    
def occlusion(cos , sin , tx , ty , input1 , input2 ):
    
    C = np.array([
    [cos , sin , tx] ,                  # matrix that has both rotation and translation                                  
    [-sin , cos , ty] ,                 # i.e  [R | T] created by using the values we found earlier
    [0 , 0 , 1]
  ])
    inv_mat = np.linalg.inv(C)
    h , w = input2.shape

    target = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            
            image_mat = np.array([i  , j  , 1])
            target_mat = image_mat.reshape((3,1))
            source_mat = np.matmul(inv_mat , target_mat) 
            x = source_mat[0, 0] 
            y = source_mat[1 , 0] 
            target[i, j] = bilinear_interpolation (input1 , x , y )


    D = img2 - target
    return D


if __name__ == "__main__":

    img1  = np.array(Image.open('./IMG1.png'))  
    img2  = np.array(Image.open('./IMG2.png'))
    output = occlusion(a , b, c , d , img1 , img2)


plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title('Image 1')

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title('Image 2')

plt.subplot(1, 3, 3)
plt.imshow(output, cmap='gray')  
plt.title('Output Image')

plt.savefig('output.png')
plt.imsave('occlusion.png', output , cmap='gray')
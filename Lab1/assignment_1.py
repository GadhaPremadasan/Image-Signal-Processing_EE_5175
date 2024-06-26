'''

EE5175 - Lab 1 
Date : 03/02/2024
command to run : python3 assignment_1.py

'''
#  import necessary libraries

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math as m

'''
translation function :

input : 1. image to be translated
        2. values by which the image 
           should be translated in x and y axis

output : translated matrix with pixels
        filled with bilinearly interpolated values

'''
def translation(input, tx, ty):                   
    h, w = input.shape
    
    target = np.zeros_like(input)
    translation_mat = np.array([[1, 0, tx],
                                [0, 1, ty],
                                [0, 0, 1]])

    inv_mat = np.linalg.inv(translation_mat)

    for i in range(h):
        for j in range(w):
            image_mat = np.float32([i, j, 1])
            target_mat = image_mat.reshape((3, 1))

            source_mat = np.matmul(inv_mat, target_mat)
            x, y = source_mat[0], source_mat[1]

            target[i, j] = bilinear_interpolation(input, x, y)

    return target
'''
rotation function :

input : 1. image to be rotated
        2. theta value by which the image 
           must be rotated

output : rotated matrix with pixels
        filled with bilinearly interpolated values

image is rotated w.r.t center of the image 

'''
def rotation(input , theta):
    h, w = input.shape
    h_mid , w_mid = h // 2 , w // 2
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))
    target = np.zeros_like(input)

    rotation_mat = np.array([[cos , sin , 0],
                            [-sin , cos , 0],
                            [0 , 0 , 1 ]])
    inv_mat = np.linalg.inv(rotation_mat)

    for i in range(h):
        for j in range(w):
            
            image_mat = np.array([i - h_mid , j - w_mid , 1])
            target_mat = image_mat.reshape((3,1))
            source_mat = np.matmul(inv_mat , target_mat)
            x = source_mat[0] + h_mid
            y = source_mat[1] + w_mid
            target[i, j] = bilinear_interpolation (input , x , y )
    return target
'''
scaling function :

input : 1. image to be scaled
        2. values by which the x and y axis 
           must be scaled

output : scaled matrix with pixels
         filled with bilinearly interpolated values

image is scaled w.r.t center of the image 

'''
def scaling(input , x_scale , y_scale) :

    h, w = input.shape
    h_mid , w_mid = h // 2 , w // 2
    target = np.zeros((450,450))
    ht, wt = target.shape
    ht_mid , wt_mid  = ht //2 , wt // 2

    scaling_mat = np.array([[x_scale, 0, 0],
                            [0, y_scale, 0],
                            [0, 0, 1]])
    inv_mat = np.linalg.inv(scaling_mat)

    for i in range(ht):
        for j in range(wt):
            image_mat = np.array([i - ht_mid, j - wt_mid, 1])  
            target_mat = image_mat.reshape((3, 1))
            source_mat = np.matmul(inv_mat, target_mat)
            x, y = source_mat[0] + h_mid, source_mat[1] + w_mid
            target[i, j] = bilinear_interpolation (input , x , y )
    return target

'''
bilinear interpolaton function :

input : 1. image to be rotated
        2. x and y axis values of source matrix

output : gives bilinearly interpolated values to each pixels


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
function to show and save the images 

'''
def show_image ( input , output , output_name ):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(input, cmap='gray')
    plt.title('Source Image')
 
    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('Output Image')

    plt.savefig(f'./{output_name}.png')

if __name__ == "__main__":
    image_1 = np.array(Image.open('lena_translate.png'))
    image_2 = np.array(Image.open('pisa_rotate.png'))
    image_3 = np.array(Image.open('cells_scale.png' ))


    translated_image = translation(image_1, 3.75, 4.3)
    rotated_image = rotation(image_2 , -4)
    scaled_image = scaling(image_3 , 0.8 , 1.3)

    show_image(image_1,translated_image, 'translated_image')
    show_image(image_2, rotated_image, 'rotated_image')
    show_image(image_3, scaled_image, 'scaled_image')+

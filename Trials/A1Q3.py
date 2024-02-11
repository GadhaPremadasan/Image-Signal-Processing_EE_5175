# trial using function
# yayy this is working code 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math as m


# a = 0.8
# b = 1.3

# rotation_mat = np.array([[a, 0, 0],
#                         [0, b, 0],
#                         [0, 0, 1]])
# inv_mat = np.linalg.inv(rotation_mat)
           
def bilinear_interpolation(input, x, y):
    x_ = m.floor(x)
    y_ = m.floor(y)
    a = x - x_
    b = y - y_

    if 0 <= x_ < h - 1 and 0 <= y_ < w - 1:
        It_1 = (1 - a) * (1 - b) * input[x_, y_]
        It_2 = (1 - a) * b * input[x_, y_ + 1]
        It_3 = a * b * input[x_ + 1, y_ + 1]
        It_4 = a * (1 - b) * input[x_ + 1, y_]

        return It_1 + It_2 + It_3 + It_4
    else:
        return 0
    

def translation(input , tx, ty):
    translation_mat = np.array([[1, 0, tx],
                                [0, 1, ty],
                                [0, 0, 1]])

    inv_mat = np.linalg.inv(translation_mat)

    for i in range(h):
        for j in range(w):
            image_mat = np.float32([i, j, 1])
            target_mat = image_mat.reshape((3, 1))

            source_mat = np.matmul(inv_mat, target_mat)
            x, y = source_mat[0] , source_mat[1]

            target[i, j] = bilinear_interpolation(input, x, y)

    return target

def rotation(input , theta):
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))
    # target_mat = np.zeros_like(image)
    target = np.zeros_like(input)

    # image_mat = np.array([i , j , 1])
    # source_mat = image_mat.reshape((3,1))
    rotation_mat = np.array([[cos , sin , 0],
                            [-sin , cos , 0],
                            [0 , 0 , 1 ]])
    inv_mat = np.linalg.inv(rotation_mat)

    for i in range(h):
        for j in range(w):
            image_mat = np.array([i - h_mid , j - w_mid , 1])
            target_mat = image_mat.reshape((3,1))
            source_mat = np.matmul(inv_mat , target_mat)
            x = source_mat[0,0] + h_mid
            y = source_mat[1 , 0] + w_mid
            target[i, j] = bilinear_interpolation (input , x , y )
    return target

def scaling(input , x_scale , y_scale) :
    ht, wt = target.shape
    

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

if __name__ == "main":

    
    image_1 = cv.imread('lena_translate.png')
    image_2 = cv.imread('pisa_rotate.png')
    image_3 = cv.imread('cells_scale.png')

    target = np.zeros_like(image_1)
    h, w = image_1.shape[:2]
    h_mid, w_mid = (h // 2, w // 2)
    # Calling all the Fuctions

    translated_image = translation(image_1 , 3.75 , 4.3 )
    # rotated_image = rotation(image_rotation , '-4')
    # scaled_image = scaling(image_scaling, '0.8' , '1.3' )


    # Now Plotting plots of all the Transformations here

    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(image_1,cmap = 'gray')
    plt.title('Source Image')
    plt.subplot(1,2,2)
    plt.imshow(translated_image,cmap='gray')
    plt.title('Translated Image')
    # plt.close()
    plt.show()




    # image1 = cv.imread('./cells_scale.png')
    # h, w = image1.shape[:2]
    # h_mid, w_mid = (h // 2, w // 2)

    # target = np.zeros((450, 450), dtype=np.uint8)  
    # ht, wt = target.shape
    # ht_mid , wt_mid = ht //2 , wt // 2


    # fig = plt.figure()
    
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(image1, cmap='gray')  
    # plt.title('input image')

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(rotated_image, cmap='gray')
    # plt.title('rotated image')
    
    # # plt.savefig('./image.png') 
    # plt.show()


    # fig = plt.figure()
    
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(image1, cmap='gray')  
    # plt.title('input image')

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(scaled_image, cmap='gray')
    # plt.title('scaled image')
    
    # # plt.savefig('./image.png') 
    # plt.show()

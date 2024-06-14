from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math as m
import cv2 as cv
import imageio
import sys

sigma = 5
image = np.array(Image.open('Mandrill.png').convert("L"))


h_k = w_k = m.ceil(6*sigma)+1
#  m = n = (6*sigma)+1
hk_mid = h_k // 2 
wk_mid = w_k // 2
# print(hk_mid  ,  wk_mid)
#  m_mid = round(m / 2 )
#  n_mid = round(n / 2)

#  print(  hk_mid   ,   wk_mid   ,   m_mid   ,  n_mid   )
gaussian_filter = np.zeros((h_k,w_k) , np.float32)

for y in range(-hk_mid , hk_mid+1):
 for x in range(-wk_mid , wk_mid+1 ):
    normal = 1 / ( 2 * np.pi * sigma**2 )
    exp_term = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
    gaussian_filter [ y + hk_mid , x + wk_mid ] = normal * exp_term
    
gaussian_filter = gaussian_filter/np.sum(gaussian_filter)
print(np.sum(gaussian_filter))
# gaussian_filter = np.flipud(np.fliplr(gaussian_filter))
# padding 
ph = ( h_k - 1 )/2
pw = ( w_k - 1 )/2

#  return gaussian_filter

# def finding_convolution(image , ):
h_c, w_c = image.shape[:2]
output_image = np.zeros((h_c, w_c), dtype=np.float32)

print(image.shape)
# output_image = np.zeros((h_c + 2*( h_k - 1 )) ,(w_c + 2*( w_k - 1 ) ))
# output_image = np.zeros((( h_c +  h_k - 1 ), ( w_c + w_k - 1 )))
output_image = np.pad(image,(int(( h_k - 1 )/2), int(( w_k - 1 )/2)), 'constant')
print(output_image.shape)
temp_out = np.zeros(output_image.shape, dtype=output_image.dtype)

y_strides = h_c - h_k + 1 
x_strides = w_c - w_k + 1 
# print(x_strides)
for i in range(y_strides):
  for j in range(x_strides):
    roi = output_image[i:i+h_k,j:j+w_k]
    # temp_out[i,j] = np.sum(np.multiply(roi , gaussian_filter))
    temp_out[i+int((h_k-1)/2), j+int((w_k-1)/2)] = np.sum(np.multiply(roi, gaussian_filter))

print(roi.shape)



# for i in range (hk_mid , output_image.shape[0] - hk_mid):
#   for j in range (wk_mid , output_image.shape[1] - wk_mid):
#     roi = output_image[i - hk_mid  :  i - h_k+1 , j-wk_mid : j - w_k+1]
#     roi = roi.flatten()*gaussian_filter.flatten()
#     temp_out[i][j] = roi.sum()

# h_end = -hk_mid 
# w_end = -wk_mid

# if( hk_mid == 0 ):
#   temp_out [hk_mid : , wk_mid : w_end]
# if( wk_mid == 0 ):
#   temp_out [hk_mid : h_end, wk_mid : ]

# temp_out[hk_mid : h_end , wk_mid : w_end] 


# if hk_mid == 0:
#     temp_out = temp_out[hk_mid:, :]
# if wk_mid == 0:
#     temp_out = temp_out[:, wk_mid:]
# else:
#     temp_out = temp_out[hk_mid:-hk_mid, wk_mid:-wk_mid]


imageio.imwrite('tempout1.jpg', temp_out)


# cv.imshow('Output Image', temp_out)
# cv.waitKey(0)
# cv.destroyAllWindows()


# print (range(len( output_image[0])))

# for i in range(h_c):
#   for j in range(w_c):  
#     canvas = np.zeros[ i , j ]



# for i in y_strides:
#   for j in x_strides :





# if __name__ == "__main__":
#   g = gaussian_filter(1)
#   print(g.shape)
    # image_1 = np.array(Image.open('Mandrill.png'))
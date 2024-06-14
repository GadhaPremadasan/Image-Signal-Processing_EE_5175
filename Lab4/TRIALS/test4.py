import numpy as np
import math as m
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import imageio

image = np.array(Image.open('Mandrill.png').convert("L"))
h,w = image.shape[:2]

sigma = 1.6
k = m.ceil(6*sigma)+1
k_mid = k//2
gaussian_filter = np.zeros((k,k) , np.float32)

for y in range(-k_mid , k_mid+1):
 for x in range(-k_mid , k_mid+1 ):
    normal = 1 / ( 2 * np.pi * sigma**2 )
    exp_term = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
    gaussian_filter [ y + k_mid , x + k_mid ] = normal * exp_term
    
gaussian_filter = gaussian_filter/np.sum(gaussian_filter)

# padding
output_image = np.zeros((h, w), dtype=np.float32)
# print(output_image.shape)
output_image = np.pad(image,(int(( k - 1 )/2), int(( k - 1 )/2)), 'constant')
print(output_image.shape)
# plt.imshow(output_image,'gray')
hp , wp = output_image.shape

temp_out = np.zeros((h,w))
# plt.imshow(temp_out)

for x in range(k_mid , h + k_mid):
    for y in range(k_mid , w + k_mid):
        op = 0
        for i in range(k):
            for j in range(k):
                op+= gaussian_filter[i,j]*output_image[x-k_mid+i , y - k_mid+j]
        temp_out[x-k_mid , y - k_mid] = op
# Convert floating-point image to uint8 for JPEG saving
temp_out_normalized = ((temp_out - temp_out.min()) / (temp_out.max() - temp_out.min()) * 255).astype(np.uint8)

# Save the normalized image as JPEG
imageio.imwrite('tempout1.jpg', temp_out_normalized)

# imageio.imwrite('tempout1.jpg', temp_out)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,15), constrained_layout=True)
# ax1.imshow(image,'gray')                #displaying gray scale image
# ax1.title.set_text("Original Image")  #setting title to the figure
# ax2.imshow(temp_out,'gray')       
# ax2.title.set_text("Blurred with sigma=1.6")


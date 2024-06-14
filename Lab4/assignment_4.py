'''

EE5175 - Lab 
Date : 07/03/2024
command to run : python3 assignment_4.py

'''

# necessary imports
import numpy as np
import math as m
import matplotlib.pyplot as plt
from PIL import Image

'''

Function to Create Gaussian Filter Kernel:
Input:
       sigma - standard deviation parameter for the Gaussian function

Output:
       2D Gaussian filter kernel of size determined by the provided sigma value

'''
def make_kernel(sigma):
  k = m.ceil(6*sigma)+1
  k_mid = k//2
  gaussian_filter = np.zeros((k,k) , np.float32)
  for y in range(-k_mid , k_mid+1):
   for x in range(-k_mid , k_mid+1 ):
    normal = 1 / ( 2 * np.pi * sigma**2 )
    exp_term = np.exp(-( x**2 + y**2 ) / (2 * sigma**2))
    gaussian_filter [ y + k_mid , x + k_mid ] = normal * exp_term
    
  gaussian_filter = gaussian_filter/np.sum(gaussian_filter)   # to normalise the gaussian filter
  return gaussian_filter

'''

Function for padding the input image :
Input:
       sigma - standard deviation parameter for determining padding size
       image - input image to be padded

Output:
       Padded image with additional border pixels based on the provided sigma value
'''

def pad(sigma , image):

 h,w = image.shape[:2]
 k = m.ceil(6*sigma)+1
 output_image = np.zeros((h, w), dtype=np.float32)
 output_image = np.pad(image,(int(( k - 1 )/2), int(( k - 1 )/2)), 'constant')
 plt.imshow(output_image,'gray')
 return output_image


'''

Function for Gaussian Blur:
Input:
       sigma - standard deviation parameter for the Gaussian filter
       image - input image to undergo Gaussian blur

Output:
       Resultant image after applying Gaussian blur with the specified sigma

'''
def gaussian_blur(sigma , image):
 h,w = image.shape[:2]
 temp_out = np.zeros((h,w))
 k = m.ceil(6*sigma)+1
 k_mid = k//2
 op = 0

 if (sigma!=0):
  gaussian_filter = make_kernel(sigma)
  output_image = pad(sigma , image)
  for x in range(k_mid , h + k_mid):
   for y in range(k_mid , w + k_mid):
    for i in range(k):
     for j in range(k):
      op+= gaussian_filter[i,j]*output_image[x-k_mid+i , y - k_mid+j]
    temp_out[x-k_mid , y - k_mid] = op
 else :
  temp_out = image
 return temp_out


if __name__ == "__main__":
 
 #  loading the input image and listing the sigma values
 image = np.array(Image.open('Mandrill.png').convert("L"))
 sigma_values = [1.6, 1.2, 1.0, 0.6, 0.3 ,0.0]

 fig, axs = plt.subplots(2, 3, figsize=(11, 8), constrained_layout=True)

 axs = axs.flatten()

 # Loop through sigma values and display blurred images
 for i, sigma in enumerate(sigma_values):   
  result = gaussian_blur(sigma, image)
  axs[i].imshow(result, 'gray')
  axs[i].set_title(f"Blurred (Sigma={sigma})")

 # Show the plot
 plt.show()



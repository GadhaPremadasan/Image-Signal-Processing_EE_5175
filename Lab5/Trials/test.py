import numpy as np
import math as m
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

# Read the image
image = np.array(Image.open('Nautilus.png').convert("L"))
h, w = image.shape[:2]
N = h

# Define constants A and B
A = 2.0

# Create empty array to store the blurred image
temp_out = np.zeros((h, w))

# Calculate Gaussian blur for each pixel
for x in range(h):
    for y in range(w):
        # Calculate sigma for current pixel
        B = (N * N) / (2 * np.log(200))
        exp_term = np.exp(-((x - (N / 2))**2 + (y - (N / 2))**2) / B)
        sigma = A * exp_term
# print(sigma)
        # Calculate kernel size
        # k = m.ceil(6 * sigma) + 1
        # k_mid = k // 2

        # # Calculate Gaussian filter
        # kernel = np.zeros((k, k), np.float32)
        # for i in range(-k_mid, k_mid + 1):
        #     for j in range(-k_mid, k_mid + 1):
        #         normal = 1 / (2 * np.pi * sigma)
        #         exp_term = np.exp(-((i**2 + j**2) / (2 * sigma**2)))
        #         J = normal * exp_term
        #         # Use relative indices to access elements of the filter
        #         kernel[i + k_mid, j + k_mid] = J
        # kernel /= np.sum(kernel)

        # Pad the image
        # output_image = np.zeros((h, w), dtype=np.float32)
        # output_image = np.pad(image,(int(( k - 1 )/2), int(( k - 1 )/2)), 'constant')
        # print(output_image.shape)
#         # Convolve the Gaussian filter with the padded image
#         for i in range(k):
#             for j in range(k):
#                 temp_out[x, y] += kernel[i, j] * padded_image[x + i, y + j]

# # Display the result
# plt.imshow(temp_out, 'gray')
# plt.title("Blurred with given sigma")
# plt.show()

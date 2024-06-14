import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def get_kernel_size(sigma):
    if isinstance(sigma, np.ndarray):  # Check if sigma is a numpy array (matrix)
        k_largest = 0
        sigma_with_largest_k = 0
        for i in range(len(sigma)):
            for j in range(len(sigma)):
                k = np.ceil(6 * sigma[i, j]) + 1
                if k > k_largest:
                    k_largest = k
                    sigma_with_largest_k = sigma[i, j]
        return sigma_with_largest_k, k_largest.astype(int)

    else:  # If sigma is a single value
        k = np.ceil(6 * sigma)
        return int(k)


def pad(k, image):
    h, w = image.shape[:2]
    output_image = np.zeros((h, w), dtype=np.float32)
    output_image = np.pad(image, (int((k - 1) / 2), int((k - 1) / 2)), 'constant')
    # print(output_image.shape)
    plt.imshow(output_image, 'gray')
    hp, wp = output_image.shape
    return output_image


def make_kernel(k, sigma):
    kmid = k // 2
    kernel = np.zeros((k, k), np.float32)
    if k == 1:
        return np.ones((k, k))
    for x in range(k):
        for y in range(k):
            normal = 1 / (2 * np.pi * sigma ** 2)
            exp_term = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[x, y] = normal * exp_term
    kernel = kernel / np.sum(kernel)

    return kernel


def make_sigma_matrix(image):
    h, w = image.shape[:2]
    N = h
    A = 2.0
    B = (N * N) / (2 * np.log(200))
    sigma_mat = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            exp_term = np.exp(-((m - (N / 2)) ** 2 + (n - (N / 2)) ** 2) / B)
            sigma = A * exp_term
            sigma_mat[m, n] = sigma
    return sigma_mat


def space_invariant(sigma, image):
    h, w = image.shape[:2]
    temp_out = np.zeros((h, w))
    k = np.ceil(6 * sigma) + 1
    k_mid = k // 2
    if sigma != 0:
        gaussian_filter = make_kernel(k, sigma)
        output_image = pad(sigma, image)
        for x in range(k_mid, h + k_mid):
            for y in range(k_mid, w + k_mid):
                op = 0
                for i in range(k):
                    for j in range(k):
                        op += gaussian_filter[i, j] * output_image[x - k_mid + i, y - k_mid + j]
                temp_out[x - k_mid, y - k_mid] = op
    else:
        temp_out = image
    return temp_out


def space_variant(sigma_matrix, image):
    h, w = image.shape[:2]
    k_largest = 0
    for i in range(len(sigma_matrix)):
        for j in range(len(sigma_matrix)):
            k = np.ceil(6 * sigma_matrix[i, j]) + 1
            k = k.astype(int)
            if k > k_largest:
                k_largest = k
                sigma_with_largest_k = sigma_matrix[i, j]
    k_mid = k_largest // 2
    output_image = pad(sigma_with_largest_k, image)
    h_out, w_out = output_image.shape[:2]

    # No need to pad the image here, as it will be handled within the convolution loop
    # output_image = pad(sigma_with_largest_k, image)
    # h_out, w_out = h, w  # No change in image size due to padding
    temp_out = np.zeros((h_out, w_out))
    for i in range(k_mid, h_out + k_mid):
        for j in range(k_mid, w_out + k_mid):
            sigma = sigma_matrix[i - k_mid, j - k_mid]  # Adjust index to access sigma_matrix correctly
            l= get_kernel_size(sigma)
            l_mid = l // 2
            img_patch = image[i - l_mid:i + l_mid + 1, j - l_mid:j + l_mid + 1]  # Extract image patch
            kernel = make_kernel(l, sigma)
            temp_out[i - l_mid:i + l_mid + 1, j - l_mid:j + l_mid + 1] += img_patch * kernel

    return temp_out


image1 = np.array(Image.open('Globe.png').convert("L"))
sigma_matrix = make_sigma_matrix(image1)
svblur_img = space_variant(image1, sigma_matrix)

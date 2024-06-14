import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

folder_path = '/home/gadha/Desktop/Courses/ISP_2024/Lab8/result_1' # put your own folder path
os.makedirs(folder_path)

'''
    Function to find the optimal threshold using Otsu's method.

    Parameters:
    - input_image: Input grayscale image.

    Returns:
    - best_threshold: threshold that gives least within class var
'''

def find_best_t(input_image):
    threshold_range = range(np.max(input_image) + 1) #range of the threshold values found(255)
    F = np.bincount(input_image.flatten()) # returns an array that gives the number of pixels in each pixel value 
    N = len(F) 
    min_var_w = float('inf') # initialising to largest value
    best_threshold = 0

    for t in threshold_range:
        N1 = np.sum(F[:t]) # number of pixels that are within the threshold t (set 1) 
        N2 = np.sum(F[t:]) # all the other pixels(set 2)
        if N1 == 0 or N2 == 0:
            continue

        u_1 = sum(i * F[i] for i in range(t)) / N1 # mean of set 1
        u_2 = sum(j * F[j] for j in range(t, len(F))) / N2 # mean of set 2

        var_1 = sum(((i - u_1) ** 2) * F[i] for i in range(t)) / N1 # variance of set 1
        var_2 = sum(((j - u_2) ** 2) * F[j] for j in range(t, len(F))) / N2 # variance of set 2

        var_w = (N1 * var_1 + N2 * var_2) / N # within-clas variance . It should be smallest. 
                                              #  Accordingly between class variance increases

        if var_w < min_var_w:
            min_var_w = var_w
            best_threshold = t

    return best_threshold # returns the threshold that gives the least within-class variance

'''
    Function to perform Otsu's thresholding on an input image.

    Parameters:
    - input_image: Input grayscale image.

    Returns:
    - thresholded_img: Thresholded image.
'''

def otsu_threshold(input_image):

    threshold = find_best_t(input_image)
    thresholded_img = np.zeros_like(input_image)
    thresholded_img[input_image >= threshold] = 255
    return thresholded_img

if __name__ == '__main__':

    img1 = np.array(Image.open("palmleaf1.png").convert("L"))
    img2 = np.array(Image.open("palmleaf2.png").convert("L"))

    thresholded_img1 = otsu_threshold(img1)
    thresholded_img2 = otsu_threshold(img2)

    palmleaf1_result_path = os.path.join(folder_path, "palmleaf1_result.png")
    palmleaf2_result_path = os.path.join(folder_path, "palmleaf2_result.png")

    Image.fromarray(thresholded_img1).save(palmleaf1_result_path)
    Image.fromarray(thresholded_img2).save(palmleaf2_result_path)


    fig, axes = plt.subplots(2, 2, figsize=(7,7))

    axes[0,0].imshow(img1, cmap='gray')
    axes[0,0].set_title('Palm Leaf 1')

    axes[0,1].imshow(thresholded_img1, cmap='gray')
    axes[0,1].set_title('Thresholded Image 1')

    axes[1,0].imshow(img2, cmap='gray')
    axes[1,0].set_title('Palm Leaf 2')

    axes[1,1].imshow(thresholded_img2, cmap='gray')
    axes[1,1].set_title('Thresholded Image 2')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
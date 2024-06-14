# OTSU
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#  u of N1 samples:
# F(i) is intensity 

def find_best_b(input_image):
    var_b_values = []
    threshold_range = range(np.max(input_image)+1)
    F = np.bincount(input_image.flatten())
    N = len(F)
    min_var_w = float('inf')
    best_threshold = 0
    # u_t = np.mean(input_image)

        

    for t in threshold_range:
        print('-----------------------')
        print('threshold = ' , t)
        N1 = np.sum(F[:t])
        N2 = np.sum(F[t:])
        if N1 == 0 or N2 == 0:
            continue

        u_t = np.mean(input_image)
        var_t = np.sum(((t - u_t) ** 2) * F[t]) / N
        # var_1 = 0 
        # var_2 = 0
        # u_1 = 0
        # u_2 = 0

        for i in range(t):
            u_1 = np.sum(i * F[i]) / N1
            var_1 = np.sum(((i - u_1) ** 2) * F[i]) / N2
        for j in range(t+1,N):
            u_2 = np.sum(j * F[j]) / N2
            var_2 = np.sum(((j - u_2) ** 2) * F[j]) / N2

        var_w = (var_1 * N1 + var_2 * N2) / N
        print(var_w)
        # var_b = ((((u_1 - u_t)**2)*N1)/N) + ((((u_2 - u_t)**2)*N2)/N)
        # # total mean and var 
        # if var_b > max:
        #     min_var_w = var_w
        #     best_threshold_1 = t 


        if var_w < min_var_w:
            min_var_w = var_w
            best_threshold = t 
        

    return best_threshold

# thresholded image
# def thresh_img(input):
#     thresholded_img = np.zeros(input.shape)
#     threshold = find_best_b(input)
#     m,n = input.shape[:2]
#     print(m,n)
#     for i in range(m):
#         for j in range(n):
#             if input[i,j] < threshold:
#                 thresholded_img[i,j] = 0
#             else :
#                 thresholded_img[i,j] = 1
#     return thresholded_img
def thresh_img(input_image):
    threshold = find_best_b(input_image)
    thresholded_img = np.zeros_like(input_image)
    thresholded_img[input_image >= threshold] = 255
    return thresholded_img

if __name__ == '__main__':
    img1 = np.array(Image.open("palmleaf1.png").convert("L"))
    max = np.max(img1)
    print(type(img1.flatten().shape))

    img2 = np.array(Image.open("palmleaf2.png"))

    varss_1= thresh_img(img1)
    plt.imshow(varss_1,cmap='gray')    
    plt.show()

    # find t such that it maximizes between class threshold 



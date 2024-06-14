# OTSU
import numpy as np
from PIL import Image
#  u of N1 samples:
# F(i) is intensity 

def find_best_b(input_image):
    var_b_values = []
    threshold_range = range(np.max(input_image)+1)
    for t in threshold_range:
        F = np.bincount(input_image.flatten())
        img_size = np.prod(input_image.flatten().shape)
        print("tot size",img_size)
        vals_in_N1 = [x for x in input_image.flatten() if x < t]
        N1 = len(vals_in_N1)
        print("N1 = " , N1)
        for i in range(N1):
            u_1 = np.sum((i*F[i])/N1)
            var_1 = np.sum((((i-u_1)**2)*F[i])/N1)
        
        vals_in_N2 = [x for x in input_image.flatten() if x >= t]
        max_intensity = np.max([x for x in input_image.flatten() if x >= t])
        N2 = len(vals_in_N2)
        print("max_intensity" , max_intensity)
        print("N2 = " , N2)

        for i in range(t+1,np.max(input_image)+1):
            u_2 = np.sum((i*F[i])/N2)
            var_2 = np.sum((((i-u_2)**2)*F[i])/N2)
        
        N = N1 + N2

        # total mean and var 
        u_t = np.mean(input_image)

    for t in threshold_range:
        for i in range(t):
            var_t = np.sum(((i - u_t) ** 2) * F[i] for i in range(t)) / N

    # within class variance 
    var_w = (var_1 * N1 + var_2 * N2) / (N1 + N2)
    
    #  between class:
    var_b = ((((u_1 - u_t)**2)*N1)/N) + ((((u_2 - u_t)**2)*N2)/N)
    var_b_values.append(var_b)
    max_var_b = max(var_b_values)

    return max_var_b

if __name__ == '__main__':
    img1 = np.array(Image.open("palmleaf1.png"))
    max = np.max(img1)
    print(type(img1.flatten().shape))

    img2 = np.array(Image.open("palmleaf2.png"))

    varss = find_best_b(img1)
    print(varss)
    

    # find t such that it maximizes between class threshold 


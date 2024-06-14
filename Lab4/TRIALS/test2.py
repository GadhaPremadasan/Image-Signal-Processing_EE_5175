from PIL import Image
import numpy as np
import imageio

sigma = 5
image = np.array(Image.open('Mandrill.png').convert("L"))

h_k = w_k = np.ceil(6 * sigma) + 1
hk_mid = h_k // 2
wk_mid = w_k // 2

# Correcting data type for gaussian_filter
gaussian_filter = np.zeros((int(h_k), int(w_k)), dtype=np.float32)

for y in range(int(-hk_mid), int(hk_mid) + 1):
 for x in range(int(-wk_mid), int(wk_mid) + 1):
        normal = 1 / (2 * np.pi * sigma**2)
        exp_term = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_filter[int(y + hk_mid), int(x + wk_mid)] = normal * exp_term

gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

# Correct padding calculation and casting to integer
ph = int(hk_mid)
pw = int(wk_mid)

h_c, w_c = image.shape[:2]
output_image = np.zeros((h_c + 2 * ph, w_c + 2 * pw), dtype=np.float32)

output_image[ph:h_c + ph, pw:w_c + pw] = image

temp_out = np.zeros_like(output_image, dtype=np.float32)

y_strides = h_c
x_strides = w_c

for i in range(y_strides):
    for j in range(x_strides):
        roi = output_image[int(i):int(i + h_k), int(j):int(j + w_k)]

        temp_out[i, j] = np.sum(np.multiply(roi, gaussian_filter))

# Crop the result to the original image size
result = temp_out[ph:h_c + ph, pw:w_c + pw]
result = result.astype(np.uint8)
imageio.imwrite('tempout1.jpg', result)

# imageio.imwrite('tempout1.jpg', result)

import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load stack.mat file
mat1 = loadmat('stack.mat')
N = int(mat1['numframes']) 
frames = {}

# Store the images in a dictionary for easier access later
for key in mat1.keys():
   if key[:5] == 'frame':
       frames[int(key[5:])] = mat1[key]

d = 50.50
q_values = [0, 1, 2]
lx = np.array([[0,  0, 0],[1, -2, 1],[0,  0, 0]])
ly = np.array([[0,  1, 0],[0, -2, 0],[0,  1, 0]])

def conv(img, kernel):
   width, height = img.shape
   k_size = len(kernel)
   t = k_size // 2 
   filtered_image = np.zeros((width - 2 * t, height - 2 * t))
   for i in range(t, width - t):
       for j in range(t, height - t):
           patch = img[i - t:i + t + 1, j - t:j + t + 1]
           filtered_image[i - t, j - t] = np.sum(patch * kernel)
   return filtered_image

fig = plt.figure(figsize=(15, 10))
axes = []
depths = {}
Maxs = {}
qs = {}

for l, q in enumerate(q_values):
   width, height = frames[1].shape
   stacked = np.zeros((width, height, N))

   for i in range(N):
       img = frames[i + 1]
       k = 2 * q + 1
       kernel = np.ones((k, k))
       p = q + 1
       img_padded = np.zeros((width + 2 * p, height + 2 * p))
       img_padded[p:-p, p:-p] = img

       I_xx = conv(img_padded, lx)
       I_yy = conv(img_padded, ly)
       ML = np.abs(I_xx) + np.abs(I_yy)
       SML = conv(ML, kernel)
       stacked[:, :, i] = SML

   max_sharpness_frames = np.argmax(stacked, axis=2)

   depth = max_sharpness_frames * d
   depths[l] = depth
   Maxs[l] = max_sharpness_frames
   sharp_img = np.zeros((width, height))

   for i in range(width):
       for j in range(height):
           frame = max_sharpness_frames[i, j] + 1
           sharp_img[i, j] = frames[frame][i, j]

   qs[l] = sharp_img
   axes.append(fig.add_subplot(1, 3, l + 1))
   subplot_title = ("q: value: " + str(q))
   axes[-1].set_title(subplot_title)
   plt.imshow(sharp_img, 'gray')

plt.show()


for l, q in enumerate(q_values):
   width, height = frames[1].shape
   X, Y = np.meshgrid(np.arange(height) + 1, np.arange(width) + 1)
   fig = plt.figure(figsize=(10, 5))
   ax = fig.add_subplot(111, projection='3d')
   surf = ax.plot_surface(X, Y, depths[l], cmap='viridis', linewidth=0, antialiased=False)  # Set the colormap to 'viridis'

   title = rf"Depth Map for q={q}"
   plt.title(title)
   fig.colorbar(surf)
   plt.show()


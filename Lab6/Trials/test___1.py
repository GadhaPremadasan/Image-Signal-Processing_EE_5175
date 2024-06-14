import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load mat file
mat_content = sio.loadmat('stack.mat')
num_of_frames = int(mat_content['numframes'])
frames = {int(key[5:]): mat_content[key] for key in mat_content.keys() if key.startswith('frame')}

# Define convolution kernels
lx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
ly = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])

# Function to compute modified Laplacian
def calculate_ml(image):
    fxx = cv2.filter2D(image, -1, lx)
    fyy = cv2.filter2D(image, -1, ly)
    ml = np.abs(fxx) + np.abs(fyy)
    return ml

# Function to calculate Sum Modified Laplacian (SML)
def calculate_sml(image, q):
    ml = calculate_ml(image)
    sml = cv2.blur(ml, (q*2+1, q*2+1))
    return sml

# Function to pad image
def pad_image(image, q):
    if q == 0:
        return image
    else:
        p = q
        return cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=0)

# Parameters
d = 50.50
q_values = [0, 1, 2]

# Iterate over q values
fig, axes = plt.subplots(1, len(q_values), figsize=(15, 5))
for i, q in enumerate(q_values):
    sharp_image = np.zeros_like(frames[1])
    stacked = np.zeros((frames[1].shape[0], frames[1].shape[1], num_of_frames))
    
    # Process each frame
    for frame_number in range(1, num_of_frames + 1):
        frame_key = "frame{:03d}".format(frame_number)
        image = pad_image(frames[frame_number], q)
        sml = calculate_sml(image, q)
        stacked[:, :, frame_number - 1] = sml
    
    # Find the sharpest frame index
    max_sharpness_frame_index = np.argmax(stacked, axis=2)
    
    # Generate sharp image
    for x in range(sharp_image.shape[0]):
        for y in range(sharp_image.shape[1]):
            frame_index = max_sharpness_frame_index[x, y]
            sharp_image[x, y] = frames[frame_index + 1][x, y]
    
    # Display sharp image
    axes[i].imshow(sharp_image, cmap='gray')
    axes[i].set_title(f'Sharp Image (q={q})')
    axes[i].axis('off')

plt.show()

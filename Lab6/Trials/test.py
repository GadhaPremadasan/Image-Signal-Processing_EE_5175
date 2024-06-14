import scipy.io as sio
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

mat_content = sio.loadmat('stack.mat')
# print(mat_content)
num_of_images = int(mat_content['numframes'])
# print(num_of_images)
x = mat_content["frame100"]
frame_key = "frame001"
num_of_frames = int(mat_content['numframes'])
# print(type(mat_content[frame_key]))


def calculate_ml(image):

    dxx = np.array([[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]])
    
    dyy = np.array([[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]])
    der_x = 0
    der_y = 0
    dim = image.shape[0]
    # print(dim)
    fxx =  np.zeros((dim,dim))
    fyy =  np.zeros((dim,dim))
    ML = np.zeros((dim,dim))
    # print(k_mid)
    for x in range(1 , dim + 1):
        for y in range(1 , dim + 1):
            der_x  = 0
            der_y = 0
            for i in range(3):
                for j in range(3):
                 if 0 <= x - 1 + i < image.shape[0] and 0 <= y - 1 + j < image.shape[1]:
                    der_x+= dxx[i,j]*image[x-1+i , y - 1+j]
                    # double der of y
                    der_y+= dyy[i,j]*image[x-1+i , y - 1+j]
                    # print(der_x)
            fxx[x-1 , y - 1] = der_x
            fyy[x-1 , y - 1] = der_y
            ML[x-1 , y - 1] = np.sum((np.abs(fxx[[x-1 , y - 1]]))+np.abs(fyy[[x-1 , y - 1]]))
    return ML 

def calculate_sml(image , q):
    ML = calculate_ml(image)
    dim = image.shape[0]
    SML = np.zeros((dim,dim))
    if q == 0:
     for x in range( dim):
        for y in range( dim):
            SML[x,y] = ML[x,y]
    else:
        for x in range( q , dim + q):
            for y in range( q , dim + q):
                for i in range(x-q,x+q):
                    for j in range(y-q,y+q):
                        if 0 <= x - 1 + i < image.shape[0] and 0 <= y - 1 + j < image.shape[1]:
                            SML[x,y] += ML[i,j]
    return SML

def pad(input_image, q ):
    padded_image = np.zeros((input_image.shape[0],input_image.shape[1]))
    if q == 0:
        padded_image = input_image
    else:
        p = (((2*q)+1) - 1)/2
        padded_image = np.pad(input_image,(int(p),int(p)))
    return padded_image

q = 0

if (q == 0):
    stacked = np.zeros((115,115,101))
elif (q==1):
   stacked = np.zeros((117,117,101))
else:
   stacked = np.zeros((119,119,101))


for i in range(1,num_of_frames+1):
    frame_number = i
    frame_key = "frame{:03d}".format(frame_number)
    
    image = pad(mat_content[frame_key],0)
    stacked[:,:,i] = calculate_sml(image,0)
    print(f'sml of {i} th frame calculated')
print("stack created")


d = 50.50
fig=plt.figure(figsize=(15,10))
axes=[]

frames = {}
qs = {}
max_sharp_frames = np.argmax(stacked,axis=2)
sharp_img =np.zeros((117, 117))
for i in range(stacked.shape[0]):
    for j in range(stacked.shape[1]):
        sharp_img[i, j] = stacked[i, j,max_sharp_frames[i, j]]


# depth = max_sharp_frames*d
# # depths[q]=depth
# # Maxs[q]=max_sharp_frames
# sharp_img =np.zeros((117, 117))
# for key in mat_content.keys():
#     if key[:5] == 'frame':
#         frames[int(key[5:])] = mat_content[key]
# for i in range(stacked.shape[0]):
#     for j in range(stacked.shape[1]):
#         frame = max_sharp_frames[i, j]+1
#         # frame_k = "frame{:03d}".format(frame)
#         sharp_img[i, j] = frames[frame][i,j]

# qs[q]=sharp_img
axes.append(fig.add_subplot(1,3,q+1))
subplot_title=("q:value: "+str(q))
axes[-1].set_title(subplot_title)
plt.imshow(sharp_img,'gray')

plt.show()




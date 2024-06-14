import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

def calculate_gaussian(V_p, V_q ,sigma = 1.0):
    diff = np.linalg.norm(V_p - V_q)
    print(diff)
    return np.exp(-diff**2 / (2 * sigma**2))

def nlm( image , W , W_sim , sigma):
    pad_size = W + W_sim
    padded_image=np.pad(image,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values=0)
    filtered_image = np.zeros_like(image)
    for channel in range(3):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                weights = []
                img_patch = []
                m,n=i+pad_size,j+pad_size
                p=padded_image[m-W_sim:m+W_sim+1,n-W_sim:n+W_sim+1,channel]
                p=p.flatten()
                for x in range(m - W , m + W + 1):
                    for y in range(n - W , n + W + 1):
                        q = padded_image[x-W_sim:x+W_sim+1,y-W_sim:y+W_sim+1,channel]
                        q = q.flatten()
                        index = len(q)//2
                        q_ = q[index]
                        gaussian = calculate_gaussian(p,q,sigma)
                        weights.append(gaussian)
                        img_patch.append(q_)
                weights = weights/np.sum(weights)
                filtered_image[i,j,channel] = np.sum(np.dot(weights,img_patch))
    return filtered_image

def PSNR(f , f_hat):
    total_pixels = f.size // 3  
    MSE = np.sum((f - f_hat) ** 2) / (total_pixels * 3)
    PSNR = 10 * np.log10(1 / (MSE + 1e-10))  # Added small value to prevent division by zero
    return PSNR



if __name__ == "__main__":
    # Read the RGB image
    krishna_g = cv.imread("/home/gadha/Desktop/Courses/ISP_2024/lab_10/krishna_0_001.png")
    krishna_f = cv.imread("/home/gadha/Desktop/Courses/ISP_2024/lab_10/krishna.png")

    # Convert the image to floating point format and normalize intensity range to [0, 1]
    g = krishna_g.astype(float) / 255.0
    f = krishna_f.astype(float) / 255.0
    # print(krishna_f.size)
    # params
    W_sim = 3
    # W = 5

    win_half = W_sim // 2
    win_half

    # sigma = 1.0
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    W1=3
    W2=5
    Filter_images1=[]
    Filter_images2=[]
    Psnr1=[]
    Psnr2=[]
    for sigma in sigma_values:   
        filter_img1=nlm(g,W1,3,sigma)
        filter_img2=nlm(g,W2,3,sigma)
        Filter_images1.append(filter_img1)
        Filter_images2.append(filter_img2)
        psnr1=PSNR(f,filter_img1)
        psnr2=PSNR(f,filter_img2)
        Psnr1.append(psnr1)
        Psnr2.append(psnr2)
    baseline_psnr=PSNR(f,g)
    baseline_plot=np.ones(5)
    baseline_plot=baseline_plot*baseline_psnr
    plt.plot(np.arange(.1,.6,.1),Psnr1,label='W=3 & Wsim=3')
    plt.plot(np.arange(.1,.6,.1),Psnr2,label='W=5 & Wsim=3')
    plt.plot(np.arange(.1,.6,.1),baseline_plot,label='baseline')
    plt.legend()

    # res = nlm(krishna_g,W = 3, W_sim = 3 , sigma = 1.0)
    # # print(res)
    # plt.imshow(res)



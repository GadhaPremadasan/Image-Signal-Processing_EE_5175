import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import cv2 as cv

#  Functions:

def dft(input):
    m,n = input.shape
    r_dft = fft(input,axis=0)
    c_dft = fft(r_dft,axis=1) # taking col dft of row dft that we found
    dft_image = np.fft.fftshift(c_dft) #shifted
    mag = np.abs(dft_image)
    # print(mag.shape)
    phase = np.zeros((m,n))
    phase[mag!=0] = dft_image[mag!=0]/mag[mag!=0] #finding phase
    # print(phase[mag!=0])
    # print(r_dft)
    return mag,phase

def idft(mag,phase): # then whatever mag and phase is coming we can create image accordingly
   
    dft = mag*phase
    m,n = dft.shape
    dft = np.fft.fftshift(dft)
    r_dft = fft(dft,axis=1)
    idft_img = fft(r_dft,axis=0)
    mag = np.abs(idft_img)
    mag=mag[::-1, ::-1]
    mag=mag/(np.sqrt(m*n))
    return mag



if __name__ == "__main__":
    img1 = cv.imread('fourier.png', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('fourier_transform.png', cv.IMREAD_GRAYSCALE)
    dft_img1=dft(img1)
    dft_img2=dft(img2)

    swapped_1 = idft(dft_img1[0],dft_img2[1])
    swapped_2= idft(dft_img2[0],dft_img1[1])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    # original images
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title("Fourier Image")
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title("Fourier Transform Image")

    # DFT's
    axes[0, 2].imshow(np.log10(np.abs(dft_img1[0])), cmap='gray')
    axes[0, 2].set_title("DFT of Fourier Image")
    axes[0, 3].imshow(np.log10(np.abs(dft_img2[0])), cmap='gray')
    axes[0, 3].set_title("DFT of Fourier Transform Image")

    axes[1, 0].imshow(swapped_1, cmap='gray')
    axes[1, 0].set_title("Mag of Img1 and Phase of Img2")
    axes[1, 1].imshow(swapped_2, cmap='gray')
    axes[1, 1].set_title("Mag of Img2 and Phase of Img1")

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.show()

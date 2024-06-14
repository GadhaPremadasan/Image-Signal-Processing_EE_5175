from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5

def k_means_a(input , given_means , k , iterations):
    clusters = [[] for _ in range(k)]
    b = input.reshape(-1, 3)
    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
       
        for vector in b:
            distances = [euclidean_distance(vector, centroid) for centroid in given_means]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(vector)

        for i in range(2):
            if clusters[i]:
                given_means[i] =np.mean(clusters[i],axis=0)
		
    output_image = np.zeros_like(b)

    for	i, cluster in enumerate(clusters):
        for pixel in cluster:
            index = np.where((b == pixel).all(axis=1))
            output_image[index] = given_means[i]

    # Reshape the output image to match the input image shape
    output_image = output_image.reshape(input.shape)
    
        # output = np.zeros_like(b)
        # for i,cluster in enumerate(clusters):
        #     fill = given_means[i]
        #     for pixel in cluster:
        #         for index, p in enumerate(b):
        #             if np.all(p == pixel):
                    #  output[index] = fill
    return output_image

# def fill_pixels( input, given_means):
#     output = np.zeros(input)
#     for i,cluster in enumerate(clusters):
#         fill = given_means[i]

if __name__ == '__main__':
    img1 = np.array(Image.open('car.png'))
    img2 = np.array(Image.open('flower.png'))

    given_means = np.array([[255, 0, 0] , 
                           [0, 0, 0] ,
                           [255, 255, 255]])
    print(k_means_a(img1,given_means ,k=3,iterations=5))
    output_image = k_means_a(img1, given_means, k=3,iterations=5)
    
    print(output_image)

    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

    
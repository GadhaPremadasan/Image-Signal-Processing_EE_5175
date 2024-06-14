from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img1 = np.array(Image.open('car.png'))
# print(type(img1))
# print(img1.shape)
b = img1.reshape(-1, 3)
for triplet in b:
    pass
# print(triplet)

c1 = np.array([255, 0, 0])
c2 = np.array([0, 0, 0])
c3 = np.array([255, 255, 255])

def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5

k = 3

def clustering(input,k):
    clusters = [[] for _ in range(k)]
    
    

    for vector in input:
        distances = [euclidean_distance(vector, centroid) for centroid in [c1, c2, c3]]
        closest_cluster = np.argmin(distances)
        clusters[closest_cluster].append(vector)
    
    return clusters
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img1 = np.array(Image.open('car.png'))
# print(type(img1))
# print(img1.shape)
b = img1.reshape(-1, 3)
for triplet in b:
    pass
# print(triplet)

c1 = np.array([255, 0, 0])
c2 = np.array([0, 0, 0])
c3 = np.array([255, 255, 255])

def euclidean_distance(vx, vy):
    return sum((y-x)**2 for x, y in zip(vx, vy)) ** 0.5

k = 3

def clustering(input,k):
    clusters = [[] for _ in range(k)]
    
    

    for vector in input:
        distances = [euclidean_distance(vector, centroid) for centroid in [c1, c2, c3]]
        closest_cluster = np.argmin(distances)
        clusters[closest_cluster].append(vector)
    
    return clusters

def replace_with_mean(image_data, clusters, centroids):
    new_image_data = np.copy(image_data)
    for i, cluster in enumerate(clusters):
        for pixel_idx in range(len(image_data)):
            closest_cluster = None
            for j, c in enumerate(clusters):
                if any(np.all(image_data[pixel_idx] == p) for p in c):
                    closest_cluster = j
                    break
            if closest_cluster is not None:
                new_image_data[pixel_idx] = centroids[closest_cluster]
    return new_image_data


if __name__ == '__main__':
    img1 = np.array(Image.open('car.png'))
    c1 = np.array([255, 0, 0])
    c2 = np.array([0, 0, 0])
    c3 = np.array([255, 255, 255])
    centroids = [c1, c2, c3]
    # print(type(img1))
    # print(img1.shape)
    b = img1.reshape(-1, 3)
    vals = clustering(b,3)
    print(len(vals))
    img_result = replace_with_mean(b,vals,centroids)
    # Reshape new pixel array to image shape
    new_img = img_result.reshape(img1.shape)
    
    # Create PIL image and display/save it
    new_image = Image.fromarray(np.uint8(new_img))
    # new_image.show()
    plt.imshow(img_result)
    plt.show()
        


if __name__ == '__main__':
    img1 = np.array(Image.open('car.png'))
    c1 = np.array([255, 0, 0])
    c2 = np.array([0, 0, 0])
    c3 = np.array([255, 255, 255])
    centroids = [c1, c2, c3]
    # print(type(img1))
    # print(img1.shape)
    b = img1.reshape(-1, 3)
    vals = clustering(b,3)
    print(len(vals))
    img_result = replace_with_mean(b,vals,centroids)
    # Reshape new pixel array to image shape
    new_img = img_result.reshape(img1.shape)
    
    # Create PIL image and display/save it
    new_image = Image.fromarray(np.uint8(new_img))
    # new_image.show()
    plt.imshow(img_result)
    plt.show()
        

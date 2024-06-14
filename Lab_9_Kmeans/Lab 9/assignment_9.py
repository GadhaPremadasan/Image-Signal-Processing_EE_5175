from PIL import Image
import numpy as np

def euclidean_distance(vx, vy):
    """
    Calculates the Euclidean distance between two vectors.

    Parameters:
    - vx: First vector.
    - vy: Second vector.

    Returns:
    - distance: Euclidean distance between the two vectors.
    """

    return np.linalg.norm(vx - vy)

def k_means_clustering(input_image, initial_means, k, iterations=5):

    """
    Perform K-means clustering on an input image.

    Parameters:
    - input_image: Input image array.
    - initial_means: Initial cluster means.
    - k: Number of clusters.
    - iterations: Number of iterations for clustering.

    Returns:
    - output_image: Clustered output image.
    """

    pixels = np.array(input_image).reshape((-1, 3))
    
    cluster_means = np.array(initial_means) # Centroids given in Q(a)
    
    for _ in range(iterations):
        clusters = [[] for _ in range(k)] # list of list
        
        # pixel values goes inside 0,1,2
        for pixel in pixels:
            distances = [euclidean_distance(pixel, mean) for mean in cluster_means]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(pixel)
        
        # Getting Centroid Means from all datapoints --> Updating 
        for i in range(k):
            if clusters[i]:
                cluster_means[i] = np.mean(clusters[i], axis=0)
    
    output_image = np.zeros_like(pixels)
    for i, cluster in enumerate(clusters):
        for pixel in cluster:
            index = np.where((pixels == pixel).all(axis=1))
            output_image[index] = cluster_means[i]
    
    output_image = output_image.reshape(input_image.shape)
    
    return output_image

def k_means_b(input_image, iterations):
   
   """
    Perform K-means clustering using random initialization of centroids.

    Parameters:
    - input_image: Input image array.
    - iterations: Number of iterations for clustering.

    Returns:
    - min_u_clusters: Clusters with minimum cost.
    - max_u_clusters: Clusters with maximum cost.
    - min_u: Centroids corresponding to minimum cost.
    - max_u: Centroids corresponding to maximum cost.
    """
   
   min_cost = float('inf')
   max_cost = float('-inf')
   k = 3
   min_u_clusters = []
   max_u_clusters = []
   

   for _ in range(iterations):
       
       cost = 0
       random_centroids = np.random.randint(0, 256, size=(3, 3))
       cluster_means = np.array(random_centroids)

       pixels = np.array(input_image).reshape((-1, 3))
       clusters = [[] for _ in range(k)]
       
       # Assign each pixel to the nearest cluster mean
       for pixel in pixels:
           distances = [euclidean_distance(pixel, mean) for mean in random_centroids]
           closest_cluster = np.argmin(distances)
           clusters[closest_cluster].append(pixel)

       for i in range(k):
         if clusters[i]:
            cluster_means[i] = np.mean(clusters[i], axis=0)


       # Calculate cost
       cost = sum(euclidean_distance(pixel, cluster_means[closest_cluster])
                   for closest_cluster, cluster_pixels in enumerate(clusters)
                   for pixel in cluster_pixels)
               
       if cost < min_cost:
           min_cost = cost
           min_u = cluster_means.copy()  # Store the current centroids
           min_u_clusters = clusters.copy()

       if cost > max_cost:
           max_cost = cost
           max_u = cluster_means.copy()  # Store the current centroids
           max_u_clusters = clusters.copy()

   return min_u_clusters, max_u_clusters, min_u, max_u

def final_max(input_image, N):
   
    """
    Finalize clustering with centroids corresponding to maximum cost.

    Parameters:
    - input_image: Input image array.
    - N: Number of iterations for clustering.

    Returns:
    - output_image: Clustered output image.
    """

    min_cluster, max_cluster, minu, maxu = k_means_b(input_image, N)
    pixels = np.array(input_image).reshape((-1, 3))
    cluster_means = np.array(maxu)

    # Create the output image by replacing each pixel with its corresponding cluster mean
    output_image = np.zeros_like(pixels)
    for i, cluster in enumerate(max_cluster):
        for pixel in cluster:
            index = np.where((pixels == pixel).all(axis=1))
            output_image[index] = cluster_means[i]

    # Reshape the output image to match the input image shape
    output_image = output_image.reshape(input_image.shape)

    return output_image

def final_min(input_image, N):
   
   """
    Finalize clustering with centroids corresponding to minimum cost.

    Parameters:
    - input_image: Input image array.
    - k: Number of clusters.
    - N: Number of iterations for clustering.

    Returns:
    - output_image: Clustered output image.

    """
   
   min_cluster, max_cluster, minu, maxu = k_means_b(input_image, N)
   pixels = np.array(input_image).reshape((-1, 3))
   cluster_means = np.array(minu)

   # Create the output image by replacing each pixel with its corresponding cluster mean
   output_image = np.zeros_like(pixels)
   for i, cluster in enumerate(min_cluster):
      for pixel in cluster:
         index = np.where((pixels == pixel).all(axis=1))
         output_image[index] = cluster_means[i]
   
   # Reshape the output image to match the input image shape
   output_image = output_image.reshape(input_image.shape)
   
   return output_image


if __name__ == '__main__':


    car_image = np.array(Image.open('car.png'))
    flower_image = np.array(Image.open('flower.png'))

    initial_means = [[255, 0, 0], [0, 0, 0], [255, 255, 255]]
    
    clustered_car = k_means_clustering(car_image, initial_means, k=3, iterations=5)
    clustered_flower = k_means_clustering(flower_image, initial_means, k=3, iterations=5)

    k = 3
    N = 30

    result_car_max = final_max(car_image, N)
    result_flower_max = final_max(flower_image, N)

    result_car_min = final_min(car_image, N)
    result_flower_min = final_min(flower_image, N)


    
    Image.fromarray(clustered_car.astype(np.uint8)).save("clustered_car.png")
    Image.fromarray(clustered_flower.astype(np.uint8)).save("clustered_flower.png")
    
    Image.fromarray(result_car_max.astype(np.uint8)).save("car_max.png")
    Image.fromarray(result_flower_max.astype(np.uint8)).save("flower_max.png")

    Image.fromarray(result_car_min.astype(np.uint8)).save("car_min.png")
    Image.fromarray(result_flower_min.astype(np.uint8)).save("flower_min.png")


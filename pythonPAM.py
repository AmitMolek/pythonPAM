import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_distance_matrix(dataset):
    dist_arr = pdist(dataset)
    dist_mat = squareform(dist_arr)

    return dist_mat


def select_randomly(dataset, k):
    return np.random.choice(dataset.shape[0], k, replace=False)


def calculateTotalCost(dataset, medoids, distance_matrix):
    totalCose = 0
    config = np.empty([np.size(dataset, 0)], dtype=float)

    # Distance to all medoids
    print(distance_matrix[0][medoids])
    # Index of the closest medoid
    print(np.argmin(distance_matrix[0][medoids]))
    # Closest medoid
    print(medoids[np.argmin(distance_matrix[0][medoids])])
    # Distance to closest medoid
    print(distance_matrix[0][medoids[np.argmin(distance_matrix[0][medoids])]])
        


dataset = np.array([[1,1], [2,2], [3,3]], dtype=float)
distance_matrix = calculate_distance_matrix(dataset)
print(distance_matrix)
medoids = select_randomly(dataset, 2)
calculateTotalCost(dataset, medoids, distance_matrix)
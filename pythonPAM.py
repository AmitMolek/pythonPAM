import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_distance_matrix(dataset):
    dist_arr = pdist(dataset)
    dist_mat = squareform(dist_arr)

    return dist_mat


def select_randomly(dataset, k):
    return np.random.choice(dataset.shape[0], k, replace=False)


def calculateTotalCost(dataset, medoids, distance_matrix):
    config = np.empty([np.size(dataset, 0)], dtype=float)

    # This block of code is going to 'calculate' for each point what is 
    # the closest medoid, using the distance matrix

    # Taking the colums of the medoids from the distance_matrix
    # Meaning now we have only the colums of the medoids for every point
    medoids_distances = distance_matrix[:][medoids]
    # For every colum we select the minimum distance
    arr_min_indexes = np.argmin(medoids_distances, axis=0)
    # Selecting the best distances (we had indexes of them, now we are taking the actual distance)
    best_distance_from_points_to_medoids = np.choose(arr_min_indexes, medoids_distances)
    # Sum the distances to know what this config cost is
    totalCost = np.sum(best_distance_from_points_to_medoids)

    return [totalCost, arr_min_indexes]
        

def pam(dataset, k):
    distance_matrix = calculate_distance_matrix(dataset)
    #print(distance_matrix)
    medoids = select_randomly(dataset, k)

    best_cost, best_config = calculateTotalCost(dataset, medoids, distance_matrix)

    dataset_size = np.size(dataset, axis=0)

    is_config_not_changed = False

    while not is_config_not_changed:

        is_config_not_changed = True

        for medoid_point in range(np.size(medoids)):
            for data_point in range(dataset_size):
                medoids_swapped = np.copy(medoids)
                medoids_swapped[medoid_point] = data_point
                
                current_cost, current_config = calculateTotalCost(dataset, medoids_swapped, distance_matrix)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_config = current_config
                    is_config_not_changed = False

                    medoids = np.copy(medoids_swapped)


    return [medoids, best_config, best_cost]


np.set_printoptions(precision=2)
dataset = np.array([[1,1], [2,1], [0,1], [1,2], [1,0], [5,5], [6,5], [4,5], [5,6], [5,4]], dtype=float)
#dataset = np.array([[1,1], [2,1], [0,1], [1,2], [1,0]], dtype=float)
k = 2

medoids, config, cost = pam(dataset, k)

print("Config Cost =", cost)
print("Medoids:")
print(dataset[medoids])
print("Dataset Labeling: ")
print(np.concatenate((dataset, dataset[medoids[config]]), axis=1))
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from xlwt import Workbook

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
    labels = np.argmin(medoids_distances, axis=0)
    # Selecting the best distances (we had indexes of them, now we are taking the actual distance)
    best_distance_from_points_to_medoids = np.choose(labels, medoids_distances)
    # Sum the distances to know what this config cost is
    totalCost = np.sum(best_distance_from_points_to_medoids)

    return [totalCost, labels]
        

def pam(dataset, k):
    distance_matrix = calculate_distance_matrix(dataset)
    #print(distance_matrix)
    medoids = select_randomly(dataset, k)

    best_cost, best_config = calculateTotalCost(dataset, medoids, distance_matrix)

    dataset_size = np.size(dataset, axis=0)


    is_cost_decreased = True
    while is_cost_decreased:

        is_cost_decreased = False

        for medoid_point in range(np.size(medoids)):
            for data_point in range(dataset_size):
                if medoids[medoid_point] is data_point:
                    continue

                medoids_swapped = np.copy(medoids)
                medoids_swapped[medoid_point] = data_point
                
                current_cost, current_config = calculateTotalCost(dataset, medoids_swapped, distance_matrix)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_config = current_config
                    is_cost_decreased = True

                    medoids = np.copy(medoids_swapped)


    return [medoids, best_config, best_cost]



def runPam():
    sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
    trimmed_sample = sample[:100]
    dataset = np.asarray(trimmed_sample, dtype=float)

    print("Dataset size =", len(trimmed_sample))

    for k in range(1,11):
        print("----- k =", k, "-----")

        pam_start_time = time.time()

        medoids, config, cost = pam(dataset, k)

        pam_end_time = time.time()

        elapsedRunTime = (pam_end_time - pam_start_time) * 1000

        #print("Config Cost =", cost)
        #print("Medoids:")
        #print(dataset[medoids])
        #print("Dataset Labeling: ")
        #print(np.concatenate((dataset, dataset[medoids[config]]), axis=1))

        print("Our Pam run for {:.3f}".format(elapsedRunTime))
        print()

        initial_medoids = select_randomly(np.asarray(trimmed_sample, dtype=float), k)

        pycluster_start_time = time.time()

        kmedoids_instance = kmedoids(sample, initial_medoids, ccore=False)

        kmedoids_instance.process()

        pycluster_end_time = time.time()

        elapsedPyclusterRunTime = (pycluster_end_time - pycluster_start_time) * 1000

        print("Pycluster's Pam (CCore=False) run for {:.3f}".format(elapsedPyclusterRunTime))

        pycluster_start_time_ccore_on = time.time()

        kmedoids_instance = kmedoids(sample, initial_medoids, ccore=True)

        kmedoids_instance.process()

        pycluster_end_time_ccore_on = time.time()

        elapsedPyclusterCCoreRunTime = (pycluster_end_time_ccore_on - pycluster_start_time_ccore_on) * 1000

        print("Pycluster's Pam (CCore=True) run for {:.3f}".format(elapsedPyclusterCCoreRunTime))


def runExample():
    k = 2
    dataset = np.array([[0,0], [0,1], [1,0], [5,5], [6,5], [5,6]], dtype=float)
    medoids, config, cost = pam(dataset, k)

    print("Labeling:", config)

    print("Config Cost =", cost)
    print("Medoids:")
    print(dataset[medoids])
    print("Dataset Labeling: ")
    print(np.concatenate((dataset, dataset[medoids[config]]), axis=1))

np.set_printoptions(precision=2)
#runPam()
runExample()
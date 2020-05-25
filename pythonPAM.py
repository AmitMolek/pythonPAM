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


def getRandomDataset(size):
    arr = np.array([np.random.rand(size)])
    return arr.T


def runPam():
    #dataset = np.array([[1,1], [2,1], [0,1], [1,2], [1,0], [5,5], [6,5], [4,5], [5,6], [5,4]], dtype=float)
    #dataset = getRandomDataset(1000)

    sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
    trimmed_sample = sample[:100]
    dataset = np.asarray(trimmed_sample, dtype=float)

    #distance_matrix = calculate_distance_matrix(dataset)
    #calculateTotalCost(dataset, np.array([[1], [2]]), distance_matrix)

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

        pycluster_start_time = time.time()
        initial_medoids = select_randomly(np.asarray(trimmed_sample, dtype=float), k)

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


def runPamExcel():
    wb = Workbook()

    excelSheet = wb.add_sheet('Sheet_1')

    sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
    trimmed_sample = sample[:400]
    dataset = np.asarray(trimmed_sample, dtype=float)

    excelSheet.write(0, 0, 'Dataset Size=')
    excelSheet.write(0, 1, len(trimmed_sample))

    min_k = 1
    max_k = 10

    run_t_times = 3

    start_line = 3

    excelSheet.write(start_line - 1, 0, 'k')

    for k in range(min_k, max_k + 1):
        excelSheet.write(start_line - 1, k, k)

    for i in range(run_t_times):
        excelSheet.write(start_line + i, 0, 'Our')

        for k in range(min_k, max_k + 1):
            pam_start_time = time.time()
            medoids, config, cost = pam(dataset, k)
            pam_end_time = time.time()
            elapsedRunTime = (pam_end_time - pam_start_time) * 1000

            excelSheet.write(start_line + i, k, elapsedRunTime)

    for i in range(run_t_times):
        excelSheet.write(15 + start_line + i, 0, 'Pycluster CCORE off')

        for k in range(min_k, max_k + 1):
            pycluster_start_time = time.time()
            initial_medoids = select_randomly(np.asarray(trimmed_sample, dtype=float), k)
            kmedoids_instance = kmedoids(sample, initial_medoids, ccore=False)
            kmedoids_instance.process()
            pycluster_end_time = time.time()
            elapsedPyclusterRunTime = (pycluster_end_time - pycluster_start_time) * 1000

            excelSheet.write(15 + start_line + i, k, elapsedPyclusterRunTime)

    for i in range(run_t_times):
        excelSheet.write(30 + start_line + i, 0, 'Pycluster CCORE on')

        for k in range(min_k, max_k + 1):
            pycluster_start_time_ccore_on = time.time()
            kmedoids_instance = kmedoids(sample, initial_medoids, ccore=True)
            kmedoids_instance.process()
            pycluster_end_time_ccore_on = time.time()
            elapsedPyclusterCCoreRunTime = (pycluster_end_time_ccore_on - pycluster_start_time_ccore_on) * 1000

            excelSheet.write(30 + start_line + i, k, elapsedPyclusterCCoreRunTime)

    wb.save('_example.xls') 


np.set_printoptions(precision=2)
#runPamExcel()
runPam()
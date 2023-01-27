import lpq
import scann
import h5py
import numpy as np
import requests
import time
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")


def download_glove_angular_100():
    with tempfile.TemporaryDirectory() as tmp:
        response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
        loc = os.path.join(tmp, "glove_100.hdf5")
        with open(loc, "wb") as f:
            f.write(response.content)

        glove_100 = h5py.File(loc, "r")

    training_data = glove_100["train"]
    testing_data = glove_100["test"]
    true_neighbors = glove_100["neighbors"]

    return training_data, testing_data, true_neighbors


def quantize_datasets(training_set, testing_set):
    quantizer = lpq.LowPrecisionQuantizer()
    print(f"Invoking a Low Precision Quantizer with bit width = {quantizer.bit_width}")

    np_training_set = np.array(training_set[:])
    np_testing_set = np.array(testing_set[:])

    quantized_training = quantizer.quantize_vectors(vectors=np_training_set)
    quantized_testing = quantizer.quantize_vectors(vectors=np_testing_set)

    return quantized_training, quantized_testing


def create_scann_index(dataset):
    # configure ScaNN as a tree - asymmetric hash hybrid with reordering
    # anisotropic quantization as described in the paper;

    normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

    index = (
        scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
        .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(100)
        .build()
    )
    return index


def compute_recall(computed_neighbors, true_neighbors):
    print(f"true_neighbors shape = {true_neighbors.shape}")
    print(f"computed neigbors shape = {computed_neighbors.shape}")

    total = 0
    for ground_truth_row, row in zip(true_neighbors, computed_neighbors):
        total += np.intersect1d(ground_truth_row, row).shape[0]
    return total / true_neighbors.size


def train_and_evaluate(training_data, testing_data, true_neighbors):
    scann_index = create_scann_index(dataset=training_data)

    start_time = time.time()
    computed_neighbors, distances = scann_index.search_batched(testing_data)
    end_time = time.time()

    # we are given top 100 neighbors in the ground truth, so select top 10
    recall = compute_recall(
        computed_neighbors=computed_neighbors, true_neighbors=true_neighborts[:, :10]
    )
    print("Recall:", recall)
    print("Time:", end_time - start_time)


if __name__ == "__main__":
    training_data, testing_data, true_neighborts = download_glove_angular_100()

    quantized_train, quantized_test = quantize_datasets(
        training_set=training_data, testing_set=testing_data
    )

    train_and_evaluate(
        training_data=quantized_train,
        testing_data=testing_data,
        true_neighbors=true_neighborts,
    )

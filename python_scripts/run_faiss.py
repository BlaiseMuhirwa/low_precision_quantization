
import lpq
import os 
import h5py 
import numpy as np
import pandas as pd 

BASE_PATH = os.getcwd()

QUANTIZED_GLOVE_TRAINING_PATH = BASE_PATH + "glove-25-angular-quantized-training.hdf5"
QUANTIZED_GLOVE_TESTING_PATH = BASE_PATH + "glove-25-angular-quantized-testing.hdf5"
GLOVE_DATASET_SIZE = 1183514


def read_glove_dataset(dimensions=25):
    """"
    Reads the Glove embeddings dataset with dimensions either 25 or 100
    """
    path = f"/Users/blaisethirdai/Downloads/glove-{dimensions}-angular.hdf5"
    keys = ("distances", "neighbors", "train", "test")
    with h5py.File(path, 'r') as file:
        training_set = np.array(file['train'][:])
        testing_set = np.array(file['test'][:])

    return training_set, testing_set


def main():
    quantizer = lpq.LowPrecisionQuantizer()
    training_set, testing_set = read_glove_dataset(dimensions=25)

    quantized_training_set = quantizer.quantize_vectors(vectors=training_set)
    quantized_testing_set = quantizer.quantize_vectors(vectors=testing_set)

    print(len(quantized_training_set))

    with h5py.File(QUANTIZED_GLOVE_TRAINING_PATH, "w") as hf:
        hf.create_dataset("training", data=np.array(quantized_training_set))
    # with h5py.File(QUANTIZED_GLOVE_DATASET_PATH, "w") as file:
    #     dataset = file.create_dataset("training", (GLOVE_DATASET_SIZE, 25), dtype="i8")
    #     dataset[0:GLOVE_DATASET_SIZE, 0:25] = quantized_training_set[0:GLOVE_DATASET_SIZE]


if __name__=="__main__":
    quantizer = lpq.LowPrecisionQuantizer()
    print(f"quantizer bit width = {quantizer.bit_width}")
    main()
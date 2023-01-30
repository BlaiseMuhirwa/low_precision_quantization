import mlflow
import faiss
import argparse
import time 
import numpy as np
from utils import get_ann_benchmark_dataset, set_tracking_uri, log_mlflow_run

DATASETS = {"glove-100-angular.hdf5", "gist-960-euclidean.hdf5"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", required=True, help="MLflow URI")

    args = parser.parse_args()
    mlflow_uri = args.mlflow_uri

    set_tracking_uri(uri=mlflow_uri)

    for dataset in DATASETS:
        training_set, queries, true_neighbors, _ = get_ann_benchmark_dataset(
            dataset_name=dataset
        )
        dimension = int(dataset.split("-")[1])
        start = time.time()
        index = faiss.IndexFlatIP(dimension)
        print(f"index.is_trained = {index.is_trained}")

        index.add(np.array(training_set))

        end = time.time()
        print(f"took {end - start} seconds")

        exit()


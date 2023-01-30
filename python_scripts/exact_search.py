import mlflow
import faiss
import argparse
import time
import numpy as np
from utils import (
    get_ann_benchmark_dataset,
    set_tracking_uri,
    log_mlflow_run,
    quantize_dataset,
    compute_recall,
)

DATASETS = {"glove-100-angular"}


def create_faiss_index(distance_type, dim=None):
    """
    distance_type will just be between inner product and euclidean
    """
    if distance_type == "angular":
        assert dim != None
        return faiss.IndexFlatIP(dim)
    elif distance_type == "euclidean":
        assert dim != None
        return faiss.IndexFlatL2(dim)

    return None


def train_and_evaluate(top_k=100):

    mlflow.set_experiment("Low Precision Quantization")
    for dataset in DATASETS:
        training_set, queries, true_neighbors, _ = get_ann_benchmark_dataset(
            dataset_name=dataset
        )
        dataset_context = dataset.split("-")

        quantized_training_set = quantize_dataset(dataset=training_set)
        distance_type = dataset_context[2]

        if distance_type == "angular":
            # For angular we need to normalize.
            training_set = (
                training_set / np.linalg.norm(training_set, axis=1)[:, np.newaxis]
            )

        dimension = int(dataset_context[1])
        index = create_faiss_index(distance_type=distance_type, dim=dimension)

        start = time.time()
        index.train(training_set)
        index.add(training_set)
        end = time.time()
        indexing_time = end - start

        start = time.time()
        _, computed_neighbors = index.search(np.array(queries), top_k)
        end = time.time()
        querying_time = end - start
        recall = compute_recall(
            computed_neighbors=computed_neighbors, true_neighbors=true_neighbors
        )

        log_mlflow_run(
            dataset=dataset,
            algorithm="faiss-exact-search",
            querying_time=querying_time,
            num_test_queries=queries.size,
            recall=recall,
            indexing_time=indexing_time,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", required=True, help="MLflow URI")

    args = parser.parse_args()
    mlflow_uri = args.mlflow_uri

    set_tracking_uri(uri=mlflow_uri)

    train_and_evaluate(top_k=100)

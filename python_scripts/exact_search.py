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
    setup_mlflow_auth,
)

DATASETS = [
    "glove-100-angular",
    "sift-128-euclidean",
    "gist-960-euclidean",
    "glove-25-angular",
    "deep-image-96-angular",
]


def create_faiss_index(distance_metric, dim=None):
    """
    distance_type will just be between inner product and euclidean
    """
    assert dim != None
    if distance_metric == "angular":
        print("creating IndexFlatIP index")
        index = faiss.IndexFlatIP(dim)
    elif distance_metric == "euclidean":
        print("creating IndexFlatL2 index")
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Invalid distance metric: {distance_metric}")

    return index


def train_and_evaluate(
    dataset, quantize=False, index_needs_training=False, top_k=100, test_run=True
):

    training_set, queries, true_neighbors, _ = get_ann_benchmark_dataset(
        dataset_name=dataset
    )
    dataset_context = dataset.split("-")
    if quantize:
        training_set = quantize_dataset(dataset=training_set)
        queries = quantize_dataset(dataset=queries)

    if dataset == "deep-image-96-angular":
        distance_metric = dataset_context[3]
        dimension = int(dataset_context[2])
    else:
        distance_metric = dataset_context[2]
        dimension = int(dataset_context[1])

    if distance_metric == "angular":
        # For angular we need to normalize.
        training_set = (
            training_set / np.linalg.norm(training_set, axis=1)[:, np.newaxis]
        )

    index = create_faiss_index(distance_metric=distance_metric, dim=dimension)

    start = time.time()
    if index_needs_training:
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

    print(f"dataset: {dataset}")
    print(f"recall@{top_k}: {recall}")
    print(f"indexing time: {indexing_time} secs")
    print(f"querying time: {querying_time} secs\n")

    if not test_run:
        run_name = "lpq-" + dataset if quantize else dataset
        log_mlflow_run(
            dataset=dataset,
            run_name=run_name,
            algorithm="faiss-exact-search",
            querying_time=querying_time,
            num_training_queries=index.ntotal,
            num_test_queries=queries.size,
            recall=recall,
            indexing_time=indexing_time,
        )


def run_experiment(dataset_name, mlflow_uri, quantize):
    # set_tracking_uri(uri=mlflow_uri)

    # mlflow.set_experiment("Low Precision Quantization")

    train_and_evaluate(
        dataset=dataset_name,
        quantize=quantize,
        index_needs_training=True,
        test_run=True,
    )

    # mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", required=True, help="MLflow URI")
    parser.add_argument("--username", required=True, help="MLflow username")
    parser.add_argument("--password", required=True, help="MLflow password")

    args = parser.parse_args()
    mlflow_uri = args.mlflow_uri
    mlflow_username = args.username
    mlflow_password = args.password

    setup_mlflow_auth(username=mlflow_username, password=mlflow_password)
    for dataset in ["glove-100-angular"]:
        run_experiment(dataset_name=dataset, mlflow_uri=mlflow_uri, quantize=False)
        run_experiment(dataset_name=dataset, mlflow_uri=mlflow_uri, quantize=True)

import time
import numpy as np
import argparse
import mlflow
from lpq.quantizer import LowPrecisionQuantizer
from lpq.index import ExactSearchIndex, ExactSearchIndexF
from utils import (
    get_ann_benchmark_dataset,
    compute_recall,
    log_mlflow_run,
    setup_mlflow_auth,
    set_tracking_uri,
    get_principal_components,
)

"""
Current Recalls with Affine Quantization:
    - sift: 86.6% -> 98% 
    - gist: 0.00518
    - mnist: 99.9609% -> 100% 
    - glove-100: 37.71%
    - nytimes: 70.56% -> 93% 
Keys Questions & Observations:
    - To what extent are the dimensions dependent on each other 
        in the GIST dataset? 
    - Seems like good/bad performance is not correlated with 
        the distance metric. 
    - For GIST, 12% of the principal components explain 85% of the variance. 
    - For SIFT, 29% of the principal components explain 85% of the variance. 
    - For NYTimes, 80% of the principal components explain 85% of the variance. 
    - For Glove, also 80% of the principal components explain 85% of the variance. 
"""

DATASETS = [
    "sift-128-euclidean",
    # "gist-960-euclidean",
    # "mnist-784-euclidean",
    # "glove-25-angular",
    # "glove-100-angular",
    # "nytimes-256-angular",
    # "lastfm-64-dot",
    # "deep-image-96-angular",
]


def get_exact_search_index(metric, quantize):
    """
    distance_type will just be between inner product and euclidean
    """
    assert metric == "angular" or metric == "euclidean"
    if quantize:
        idx = ExactSearchIndex(metric)
    else:
        # This searches for vectors in the floating point domain
        idx = ExactSearchIndexF(metric)

    return idx


def train_and_eval(
    idx,
    dataset_name,
    train_set,
    queries,
    true_neighbors,
    metric,
    top_k=100,
    quantize=False,
    test_run=True,
):
    if metric == "angular":
        train_set = train_set / np.linalg.norm(train_set, axis=1)[:, np.newaxis]

    if quantize:
        print("Invoking Quantizer...")
        quantizer_ = LowPrecisionQuantizer()
        train_set = quantizer_.quantize_vectors(vectors=train_set)
        queries = quantizer_.quantize_vectors(vectors=queries)
        print("Finished Quantizing")

    print(f"[EXPERIMENT]: {dataset_name}")
    start = time.time()
    idx.add(train_set)
    end = time.time()
    indexing_time = end - start
    print(f"Indexing time: {indexing_time} secs")

    start = time.time()
    distances, computed_neighbors = idx.search(queries, top_k)
    end = time.time()
    querying_time = end - start
    print(f"Querying time: {querying_time} secs")

    distances, computed_neighbors = np.array(distances), np.array(computed_neighbors)

    recall = compute_recall(
        computed_neighbors=computed_neighbors, true_neighbors=true_neighbors
    )
    print(f"recall@{top_k}: {recall} -- dataset: {dataset_name} \n")

    if not test_run:
        run_name = "affine_quantizer-" + dataset if quantize else dataset
        log_mlflow_run(
            dataset=dataset,
            run_name=run_name,
            algorithm="exact-search",
            querying_time=querying_time,
            num_training_queries=len(train_set),
            num_test_queries=len(queries),
            recall=recall,
            indexing_time=indexing_time,
        )


def run_experiment(
    index,
    dataset_name,
    train_set,
    queries,
    true_neighbors,
    distance_metric,
    quantize,
):
    # set_tracking_uri(uri=mlflow_uri)

    # mlflow.set_experiment("Low Precision Quantization")

    train_and_eval(
        idx=index,
        dataset_name=dataset_name,
        train_set=train_set,
        queries=queries,
        true_neighbors=true_neighbors,
        metric=distance_metric,
        quantize=quantize,
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

    for dataset in DATASETS:
        dataset_context = dataset.split("-")
        distance_metric = dataset_context[-1]

        train_set, queries, true_neighbors, _ = get_ann_benchmark_dataset(
            dataset_name=dataset
        )

        idx = get_exact_search_index(metric=distance_metric, quantize=True)
        run_experiment(
            index=idx,
            dataset_name=dataset,
            train_set=train_set,
            queries=queries,
            true_neighbors=true_neighbors,
            distance_metric=distance_metric,
            quantize=True,
        )

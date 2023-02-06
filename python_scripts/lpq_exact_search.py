import faiss
import time
import numpy as np
from lpq import quantizer, index
from utils import (
    get_ann_benchmark_dataset,
    compute_recall,
)

DATASETS = [
    "sift-128-euclidean",
    "gist-960-euclidean",
    "mnist-784-euclidean",
    "glove-25-angular",
    "glove-100-angular",
    "nytimes-256-angular",
    "lastfm-64-dot",
    "deep-image-96-angular",
]


def get_exact_search_index(metric, quantize):
    assert metric == "angular" or metric == "euclidean"
    if quantize:
        idx = index.ExactSearchIndexInt8(metric)
    else:
        idx = index.ExactSearchIndexF(metric)
    return idx


def get_dataset(dataset_name):
    a, b, c, d = get_ann_benchmark_dataset(dataset_name=dataset_name)
    return a, b, c, d


def train_and_eval(
    idx,
    dataset_name,
    train_set,
    queries,
    true_neighbors,
    metric,
    top_k=100,
    quantize=False,
):
    if metric == "angular":
        train_set = train_set / np.linalg.norm(train_set, axis=1)[:, np.newaxis]

    if quantize:
        quantizer_ = quantizer.LowPrecisionQuantizerInt8()
        train_set = quantizer_.quantize_vectors(vectors=train_set)
        queries = quantizer_.quantize_vectors(vectors=queries)

    print(f"[EXPERIMENT]: {dataset_name}")
    start = time.time()
    idx.add(train_set)
    end = time.time()
    print(f"Indexing time: {end - start} secs")

    start = time.time()
    distances, computed_neighbors = idx.search(queries, top_k)
    end = time.time()
    print(f"Search time: {end - start} secs")

    distances, computed_neighbors = np.array(distances), np.array(computed_neighbors)

    recall = compute_recall(
        computed_neighbors=computed_neighbors, true_neighbors=true_neighbors
    )
    print(f"recall@{top_k}: {recall} -- dataset: {dataset_name} \n")


if __name__ == "__main__":

    for dataset in DATASETS:
        dataset_context = dataset.split("-")
        distance_metric = dataset_context[-1]

        train_set, queries, true_neighbors, _ = get_dataset(dataset_name=dataset)

        idx = get_exact_search_index(metric=distance_metric, quantize=True)
        train_and_eval(
            idx=idx,
            dataset_name=dataset,
            train_set=train_set,
            queries=queries,
            true_neighbors=true_neighbors,
            metric=distance_metric,
            quantize=True,
        )

import lpq
import mlflow
import socket, platform, psutil, os
import tempfile
import h5py
import numpy as np
import requests


machine_info = {
    "load_before_experiment": os.getloadavg()[2],
    "platform": platform.platform(),
    "platform_version": platform.version(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "hostname": socket.gethostname(),
    "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
    "num_cores": psutil.cpu_count(logical=True),
}


def _log_machine_info():
    for key, val in machine_info.items():
        mlflow.log_param(key, val)


def set_tracking_uri(uri):
    mlflow.set_tracking_uri(uri=uri)


def log_mlflow_run(
    dataset, algorithm, querying_time, num_test_queries, recall, indexing_time=None
):

    if not mlflow.is_tracking_uri_set():
        raise ValueError("mlflow tracking uri must first be set.")

    with mlflow.start_run(tags={"dataset": dataset, "algorithm": algorithm}):
        _log_machine_info()
        mlflow.log_param("indexing_time", indexing_time)
        mlflow.log_param("querying_time", querying_time)
        mlflow.log_param("num_queries", num_test_queries)
        mlflow.log_param("queries_per_second", num_test_queries / querying_time)
        mlflow.log_param("recall", recall)


def get_ann_benchmark_dataset(dataset_name):
    base_uri = "http://ann-benchmarks.com"
    dataset_uri = f"{base_uri}/{dataset_name}.hdf5"

    with tempfile.TemporaryDirectory() as tmp:
        response = requests.get(dataset_uri)
        loc = os.path.join(tmp, dataset_name)

        with open(loc, "wb") as f:
            f.write(response.content)
        data = h5py.File(loc, "r")

    training_set = data["train"]
    queries = data["test"]
    true_neighbors = data["neighbors"]
    distances = data["distances"]

    return (
        np.array(training_set),
        np.array(queries),
        np.array(true_neighbors),
        np.array(distances),
    )


def quantize_dataset(dataset):
    quantizer = lpq.LowPrecisionQuantizerInt8()
    print(f"Invoking a low precision quantizer with bit width = {quantizer.bit_width}")

    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset[:])

    quantized_dataset = quantizer.quantize_vectors(vectors=dataset)
    return np.array(quantized_dataset, dtype=np.float32)


def compute_recall(computed_neighbors, true_neighbors):
    total = 0
    for ground_truth_row, row in zip(true_neighbors, computed_neighbors):
        total += np.intersect1d(ground_truth_row, row).shape[0]

    return total / true_neighbors.size

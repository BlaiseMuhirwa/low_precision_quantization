import mlflow
import socket, platform, psutil, os
import tempfile 
import h5py
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
    """Starts and finishes an mlflow run for magsearch, logging all
    necessary information."""

    if not mlflow.is_tracking_uri_set():
        raise ValueError("mlflow tracking uri must first be set.")

    mlflow.set_experiment("Low Precision Quantization")
    with mlflow.start_run(tags={"dataset": dataset, "algorithm": algorithm}):
        _log_machine_info()
        mlflow.log_param("indexing_time", indexing_time)
        mlflow.log_param("querying_time", querying_time)
        mlflow.log_param("num_queries", num_test_queries)
        mlflow.log_param("queries_per_second", num_test_queries / querying_time)
        mlflow.log_param("recall", recall)


def get_ann_benchmark_dataset(dataset_name):
    base_uri = "http://ann-benchmarks.com"
    dataset_uri = f"{base_uri}/{dataset_name}"

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

    return training_set, queries, true_neighbors, distances


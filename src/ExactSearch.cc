
#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <src/DistanceMetrics.h>
#include <src/ExactSearch.h>
#include <tuple>
#include <utility>

namespace lpq::index {

template <typename PRECISION_TYPE>
void ExactSearchIndex<PRECISION_TYPE>::addDataset(
    const std::vector<std::vector<PRECISION_TYPE>> &dataset) {
  assert(_index.size() == 0);
  for (uint32_t index = 0; index < dataset.size(); index++) {
    auto current_item = std::make_pair(std::move(dataset[index]), index);
    _index.push_back(std::move(current_item));
  }
}

template <typename PRECISION_TYPE>
std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<uint32_t>>>
ExactSearchIndex<PRECISION_TYPE>::search(
    const std::vector<std::vector<PRECISION_TYPE>> &queries, uint32_t top_k) {

  std::vector<std::vector<float>> distances(queries.size());
  std::vector<std::vector<uint32_t>> ids(queries.size());

#pragma omp parallel for default(none) shared(distances, ids, queries, top_k)
  for (uint32_t index = 0; index < queries.size(); index++) {

    auto [top_k_distances, top_k_ids] = getTopKClosestVectors(
        /* query_vector=*/queries[index], /* top_k=*/top_k);

    distances[index] = std::move(top_k_distances);
    ids[index] = std::move(top_k_ids);
  }
  return {distances, ids};
}

template <typename PRECISION_TYPE>
std::tuple<std::vector<float>, std::vector<uint32_t>>
ExactSearchIndex<PRECISION_TYPE>::getTopKClosestVectors(
    const std::vector<PRECISION_TYPE> &query_vector, uint32_t top_k) {

  /**
   * We use a priority queue. Every element in the queue
   * consists of a pair of the distance and the actual
   * vector in the index. We use a max heap for the Euclidean distance
   * metric instead of min heap since the invariant here is that if any
   * vector is not in the top k closest vectors, it will be removed
   * from the heap at some point. But this means we have to sort the
   * resulting output since the top will contain the k^th closest vector.
   * The final sorting step takes O(klogk). So, the bottleneck is the
   * pre-processing step which takes O(n logk);
   */
  std::vector<std::pair<float, uint32_t>> top_k_results;
  if (_distance_metric == "angular") {
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        heap;
    for (uint32_t vec_index = 0; vec_index < _index.size(); vec_index++) {
      auto distance = lpq::index::computeDistance(
          /* first_vector = */ query_vector,
          /* second_vector = */ _index[vec_index].first,
          /* metric = */ _distance_metric);

      auto current_pair = std::make_pair(distance, _index[vec_index].second);
      heap.push(current_pair);

      if (heap.size() > top_k) {
        heap.pop();
      }
      while (!heap.empty()) {
        auto pair = heap.top();
        top_k_results.emplace_back(std::move(pair));
        heap.pop();
      }
    }
  } else if (_distance_metric == "euclidean") {
    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (uint32_t vec_index = 0; vec_index < _index.size(); vec_index++) {
      auto distance = lpq::index::computeDistance(
          /* first_vector = */ query_vector,
          /* second_vector = */ _index[vec_index].first,
          /* metric = */ _distance_metric);

      auto current_pair = std::make_pair(distance, _index[vec_index].second);
      heap.push(current_pair);

      if (heap.size() > top_k) {
        heap.pop();
      }

      while (!heap.empty()) {
        auto pair = heap.top();
        top_k_results.emplace_back(std::move(pair));
        heap.pop();
      }
    }
  }

  if (_distance_metric == "angular") {
    std::sort(top_k_results.begin(), top_k_results.end(),
              std::greater<std::pair<float, uint32_t>>());
  } else if (_distance_metric == "euclidean") {
    std::sort(top_k_results.begin(), top_k_results.end());
  }
  std::vector<float> distances(top_k);
  std::vector<uint32_t> ids(top_k);

  for (uint32_t i = 0; i < top_k; i++) {
    distances[i] = top_k_results[i].first;
    ids[i] = top_k_results[i].second;
  }

  return {std::move(distances), std::move(ids)};
}

// Floating point based index used for baseline comparision
template class ExactSearchIndex<float>;

// Lower precision based indices constructed after quantization
template class ExactSearchIndex<int_least8_t>;
template class ExactSearchIndex<int_least16_t>;

} // namespace lpq::index

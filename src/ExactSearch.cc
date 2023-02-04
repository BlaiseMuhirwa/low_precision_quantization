
#include <cstdint>
#include <queue>
#include <src/DistanceMetrics.h>
#include <src/ExactSearch.h>
#include <utility>

namespace lpq::index {

template <typename PRECISION_TYPE>
void ExactSearchIndex<PRECISION_TYPE>::addDataset(
    const std::vector<std::vector<PRECISION_TYPE>> &dataset) {
  assert(_index.size() == 0);
  for (const auto &vector : dataset) {
    _index.push_back(std::move(vector));
  }
}

template <typename PRECISION_TYPE>
std::vector<std::vector<std::pair<float, std::vector<PRECISION_TYPE>>>>
ExactSearchIndex<PRECISION_TYPE>::search(
    const std::vector<std::vector<PRECISION_TYPE>> &queries, uint32_t top_k) {
  std::vector<TopKType> output;
  output.reserve(queries.size());

#pragma omp parallel for default(none) shared(output, queries, top_k)
  for (uint32_t index = 0; index < queries.size(); index++) {

    TopKType top_k_vectors = getTopKClosestVectors(
        /* query_vector=*/queries[index], /* top_k=*/top_k);

    output[index] = std::move(top_k_vectors);
  }

  return output;
}

template <typename PRECISION_TYPE>
std::vector<std::pair<float, std::vector<PRECISION_TYPE>>>
ExactSearchIndex<PRECISION_TYPE>::getTopKClosestVectors(
    const std::vector<PRECISION_TYPE> &query_vector, uint32_t top_k) {

  /**
   * We use a priority queue. Every element in the queue
   * consists of a pair of the distance and the actual
   * vector in the index. We use a max heap instead of min heap
   * since the invariant here is that if any vector is not in the
   * top k closest vectors, it will be removed from the heap at some
   * point. But this means we have to sort the resulting output since
   * the top will contain the k^th closest vector. The final sorting
   * step takes O(klogk). So, the bottleneck is the pre-processing step
   * which takes O(n logk);
   */

  std::priority_queue<MaxHeapElementType> max_heap;
  for (uint32_t vec_index = 0; vec_index < _index.size(); vec_index++) {
    auto distance =
        lpq::index::computeDistance(/* first_vector = */ query_vector,
                                    /* second_vector = */ _index[vec_index],
                                    /* metric = */ _distance_metric);

    auto current_pair = std::make_pair(distance, _index[vec_index]);
    max_heap.push(current_pair);

    if (max_heap.size() > top_k) {
      max_heap.pop();
    }
  }
  std::vector<std::pair<float, std::vector<PRECISION_TYPE>>> output;
  while (!max_heap.empty()) {
    auto pair = max_heap.top();
    output.emplace_back(std::move(pair));
    max_heap.pop();
  }
  std::sort(output.begin(), output.end());
  return output;
}

// Floating point based index used for baseline comparision
template class ExactSearchIndex<float>;

// Lower precision based indices constructed after quantization
template class ExactSearchIndex<int_least8_t>;
template class ExactSearchIndex<int_least16_t>;

} // namespace lpq::index
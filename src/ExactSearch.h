#pragma once

#include <memory.h>
#include <string>
#include <vector>

namespace lpq::index {

/**
 * Exact search index stores all the vectors and performs
 * exhaustive search to compute distances
 **/

template <typename PRECISION_TYPE>
class ExactSearchIndex
    : public std::enable_shared_from_this<ExactSearchIndex<PRECISION_TYPE>> {

  using MaxHeapElementType = std::pair<float, std::vector<PRECISION_TYPE>>;
  using TopKType = std::vector<std::pair<float, std::vector<PRECISION_TYPE>>>;

public:
  explicit ExactSearchIndex(const std::string &distance_metric)
      : _distance_metric(distance_metric) {}

  void addDataset(const std::vector<std::vector<PRECISION_TYPE>> &dataset);

  /**
   * Returns a vector of the same size as the size of the input `queries`
   * vector.
   * Suppose we have n queries where each query is a vector of a fixed
   * dimension d.
   * Then, the output will be a vector of size n, and each element in
   * the output vector will also be a vector (of size k) of vectors, each
   * of which has dim d.
   */
  std::vector<std::vector<std::pair<float, std::vector<PRECISION_TYPE>>>>
  search(const std::vector<std::vector<PRECISION_TYPE>> &queries,
         uint32_t top_k);

private:
  /**
   * Computes the distance between the input query vector
   * and every other query in the index, and returns the top_k
   * closest queries based on the given distance metric and
   * the corresponding distances in a sorted order.
   */
  std::vector<std::pair<float, std::vector<PRECISION_TYPE>>>
  getTopKClosestVectors(const std::vector<PRECISION_TYPE> &query_vector,
                        uint32_t top_k);

  std::string _distance_metric;
  std::vector<std::vector<PRECISION_TYPE>> _index;
};

} // namespace lpq::index
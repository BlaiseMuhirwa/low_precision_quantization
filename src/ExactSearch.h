#pragma once

#include <memory.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lpq::index {

/**
 * Exact search index stores all the vectors and performs
 * exhaustive search to compute distances
 **/

template <typename PRECISION_TYPE> class ExactSearchIndex {

public:
  explicit ExactSearchIndex(const std::string &distance_metric)
      : _distance_metric(distance_metric) {}

  /**
   * Adds every vector to the index. Every vector in the dataset
   * is assigned a unique ID. This assignment is sequential so
   * that the first vector has ID 0 and so on.
   **/
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
  std::tuple<std::vector<std::vector<float>>,
             std::vector<std::vector<uint32_t>>>
  search(const std::vector<std::vector<PRECISION_TYPE>> &queries,
         uint32_t top_k);

private:
  /**
   * Computes the distance between the input query vector
   * and every other query in the index, and returns the top_k
   * closest queries based on the given distance metric and
   * the corresponding distances in a sorted order. The output
   * is returned as a tuple of the top k distances and top k
   * vector ids.
   */
  std::tuple<std::vector<float>, std::vector<uint32_t>>
  getTopKClosestVectors(const std::vector<PRECISION_TYPE> &query_vector,
                        uint32_t top_k);

  std::string _distance_metric;
  std::vector<std::pair<std::vector<PRECISION_TYPE>, uint32_t>> _index;
};

} // namespace lpq::index
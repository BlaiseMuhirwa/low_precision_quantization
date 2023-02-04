#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace lpq::index {

template <typename PRECISION_TYPE>
static float norm(const std::vector<PRECISION_TYPE> &vector) {
  float sum = 0;
  std::for_each(
      vector.begin(), vector.end(),
      [&sum](const PRECISION_TYPE &element) { sum += element * element; });
  return sum;
}

template <typename PRECISION_TYPE>
static float
euclideanDistance(const std::vector<PRECISION_TYPE> &first_vector,
                  const std::vector<PRECISION_TYPE> &second_vector) {
  float distance = 0.0;
  for (uint32_t i = 0; i < first_vector.size(); i++) {
    distance += (first_vector[i] - second_vector[i]) *
                (first_vector[i] - second_vector[i]);
  }
  return distance;
}

/**
 * This is equivalent to inner product scaled by the product of the
 * norms of the input vectors
 */
template <typename PRECISION_TYPE>
static float angularDistance(const std::vector<PRECISION_TYPE> &first_vector,
                             const std::vector<PRECISION_TYPE> &second_vector) {
  float distance = 0.0;
  for (uint32_t i = 0; i < first_vector.size(); i++) {
    distance += first_vector[i] * second_vector[i];
  }
  auto norm_first_vector = norm(first_vector);
  auto norm_second_vector = norm(second_vector);

  distance /= (norm_first_vector * norm_second_vector);
  return distance;
}

template <typename PRECISION_TYPE>
static float computeDistance(const std::vector<PRECISION_TYPE> &first_vector,
                             const std::vector<PRECISION_TYPE> &second_vector,
                             const std::string metric) {
  if (first_vector.size() != second_vector.size()) {
    throw std::invalid_argument("Input vectors must be of the same size in "
                                "order to compute the euclidean distance.");
  }
  std::string metric_to_lower = metric;
  std::for_each(metric.begin(), metric.end(),
                [&](char c) { c = std::tolower(c); });

  if (metric_to_lower == "euclidean") {
    return euclideanDistance(first_vector, second_vector);
  }
  if (metric_to_lower == "angular") {
    return angularDistance(first_vector, second_vector);
  }

  throw std::invalid_argument("Invalid metric distance. Supported metric "
                              "include 'euclidean' and 'angular'");
}
} // namespace lpq::index
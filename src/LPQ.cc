#include "LPQ.h"
#include <cmath>
#include <iostream>
#include <math.h>
#include <numeric>
#include <algorithm>

namespace lpq {

template <typename PRECISION_TYPE>
LowPrecisionQuantizer<PRECISION_TYPE>::LowPrecisionQuantizer()
    : _bit_width(8 * sizeof(PRECISION_TYPE)) {}

template <typename PRECISION_TYPE>
std::vector<std::vector<PRECISION_TYPE>>
LowPrecisionQuantizer<PRECISION_TYPE>::quantizeVectors(
    const std::vector<std::vector<float>> &vectors) {

  if (vectors.size() == 0) {
    return {};
  }
  std::vector<std::vector<PRECISION_TYPE>> quantized_vectors;

// #pragma omp parallel for default(none) shared(vectors, quantized_vectors)
  for (const auto &vector : vectors) {
    std::vector<PRECISION_TYPE> quantized_vector;
    auto [mean, stddev] = computeVectorWiseStatistics(vector);

    for (uint32_t vec_index = 0; vec_index < vector.size(); vec_index++) {
      PRECISION_TYPE quantized_value =
          quantize(vector[vec_index], mean, stddev);
      quantized_vector.push_back(quantized_value);
    }

    quantized_vectors.emplace_back(std::move(quantized_vector));
  }
  return quantized_vectors;
}

template <typename PRECISION_TYPE>
std::tuple<float, float>
lpq::LowPrecisionQuantizer<PRECISION_TYPE>::computeVectorWiseStatistics(
    const std::vector<float> &vector) {
  if (vector.size() == 0) {
    return {};
  }
  auto const count = static_cast<float>(vector.size());

  float mean = std::accumulate(vector.begin(), vector.end(), 0.000000) / count;
  float variance = 0.000000;
  std::for_each(vector.begin(), vector.end(), [&variance, &mean](float item) {
    variance += (item - mean) * (item - mean);
  });
  variance /= (vector.size() - 1);
  float stddev = sqrt(variance);
  return std::make_tuple(mean, stddev);
}

template <typename PRECISION_TYPE>
PRECISION_TYPE
lpq::LowPrecisionQuantizer<PRECISION_TYPE>::quantize(float value, float mean,
                                                     float standard_deviation) {
  auto sb = mean - standard_deviation;
  auto se = mean + standard_deviation;

  if (value < sb) {
    auto val = -((1 << (_bit_width - 1)) - 1);
    return val;
  } else if (value > se) {
    auto val = (1 << (_bit_width - 1)) - 1;
    return val;
  }

  auto val =
      floor((1 << (_bit_width - 1)) * ((value - mean) / standard_deviation));
  PRECISION_TYPE largest_int = (1 << (_bit_width - 1)) - 1;
  PRECISION_TYPE smallest_int = -(largest_int + 1);

  if (val > largest_int) {
    return largest_int;
  } else if (val < smallest_int) {
    return smallest_int;
  }
  return val;
}

// Template specialization for int8 and int16 types
template class LowPrecisionQuantizer<int8_t>;
template class LowPrecisionQuantizer<int16_t>;

} // namespace lpq
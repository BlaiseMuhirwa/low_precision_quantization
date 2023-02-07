#include "LPQ.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <numeric>

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

  auto dataset_statistics = getDatasetStatistics(/* dataset = */ vectors);
  assert(dataset_statistics.size() == vectors[0].size());

#pragma omp parallel for default(none)                                         \
    shared(vectors, quantized_vectors, dataset_statistics)
  for (uint32_t vec_index = 0; vec_index < vectors.size(); vec_index++) {

    std::vector<PRECISION_TYPE> quantized_vector;
    for (uint32_t dim_index = 0; dim_index < vectors[0].size(); dim_index++) {
      auto [mean, stdev] = dataset_statistics[dim_index];
      PRECISION_TYPE quantized_value =
          quantize(/* value = */ vectors[vec_index][dim_index],
                   /* mean = */ mean, /* standard_deviation = */ stdev);
      quantized_vector.emplace_back(quantized_value);
    }
    quantized_vectors.emplace_back(std::move(quantized_vector));
  }
  return quantized_vectors;
}

std::vector<std::tuple<float, float>>
getDatasetStatistics(const std::vector<std::vector<float>> &dataset) {
  if (dataset.size() == 0) {
    return {};
  }
  auto output_dimension = dataset[0].size();
  auto dataset_size = dataset.size();
  std::vector<std::tuple<float, float>> output(output_dimension);

#pragma omp parallel for default(none)                                         \
    shared(dataset, output_dimension, dataset_size)
  for (uint32_t dim_index = 0; dim_index < output_dimension; dim_index++) {
    float mean = 0.0000000;
    float variance = 0.0000000;
    for (uint32_t row_index = 0; row_index < dataset_size; row_index++) {
      mean += dataset[row_index][dim_index];
    }
    mean /= dataset_size;
    for (uint32_t row_index = 0; row_index < dataset_size; row_index++) {
      variance += (dataset[row_index][dim_index] - mean) *
                  (dataset[row_index][dim_index] - mean);
    }
    variance /= (dataset_size - 1);
    auto current_statistics = std::make_tuple(mean, sqrt(variance));
    output[dim_index] = current_statistics;
  }
  return output;
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
template class LowPrecisionQuantizer<int_least8_t>;
template class LowPrecisionQuantizer<int_least16_t>;

} // namespace lpq
#include "LPQ.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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

  std::vector<std::vector<PRECISION_TYPE>> quantized_vectors(vectors.size());

  auto min_max_values = getMinMaxValues(/* dataset = */ vectors);
  assert(min_max_values.size() == vectors[0].size());

  std::vector<std::tuple<float, PRECISION_TYPE>> quantization_parameters;
  for (auto [min, max] : min_max_values) {
    quantization_parameters.emplace_back(
        getQuantizationParams(/* min = */ min, /* max = */ max));
  }

#pragma omp parallel for default(none)                                         \
    shared(vectors, quantization_parameters, quantized_vectors)
  for (uint32_t vec_index = 0; vec_index < vectors.size(); vec_index++) {

    std::vector<PRECISION_TYPE> quantized_vector(vectors[vec_index].size());
#pragma omp critical
    {
      for (uint32_t dim_index = 0; dim_index < vectors[0].size(); dim_index++) {
        auto [scale, zero_point] = quantization_parameters[dim_index];

        PRECISION_TYPE quantized_value =
            affine_quantize(/* value = */ vectors[vec_index][dim_index],
                            /* scale = */ scale, /* zero_point = */ zero_point);
        quantized_vector[dim_index] = quantized_value;
      }
    }
    quantized_vectors[vec_index] = std::move(quantized_vector);
  }
  return quantized_vectors;
}

template <typename PRECISION_TYPE>
std::vector<std::tuple<float, float>>
LowPrecisionQuantizer<PRECISION_TYPE>::getDatasetStatistics(
    const std::vector<std::vector<float>> &dataset) {
  if (dataset.size() == 0) {
    return {};
  }
  auto output_dimension = dataset[0].size();
  auto dataset_size = dataset.size();
  std::vector<std::tuple<float, float>> output(output_dimension);

#pragma omp parallel for default(none)                                         \
    shared(dataset, output_dimension, dataset_size, output)
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
std::tuple<float, PRECISION_TYPE>
LowPrecisionQuantizer<PRECISION_TYPE>::getQuantizationParams(float min,
                                                             float max) {
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  PRECISION_TYPE qmin = (_bit_width == 8) ? -128 : -32768;
  PRECISION_TYPE qmax = (_bit_width == 8) ? 127 : 32767;

  const double scale = (max - min) / (qmax - qmin);

  auto zero_point = qmin - std::round(min / scale);
  if (zero_point < qmin) {
    return {scale, qmin};
  }
  if (zero_point > qmax) {
    return {scale, qmax};
  }
  return {scale, static_cast<PRECISION_TYPE>(zero_point)};
}

template <typename PRECISION_TYPE>
std::vector<std::tuple<float, float>>
LowPrecisionQuantizer<PRECISION_TYPE>::getMinMaxValues(
    const std::vector<std::vector<float>> &dataset) {
  if (dataset.size() == 0) {
    return {};
  }
  std::vector<std::tuple<float, float>> output(dataset[0].size());

#pragma omp parallel for default(none) shared(output, dataset)
  for (uint32_t dim_index = 0; dim_index < dataset[0].size(); dim_index++) {
    float min = dataset[0][dim_index];
    float max = dataset[0][dim_index];
#pragma omp critical
    {
      for (uint32_t row_index = 0; row_index < dataset.size(); row_index++) {
        min = std::min(min, dataset[row_index][dim_index]);
        max = std::max(max, dataset[row_index][dim_index]);
      }
    }
    output[dim_index] = std::make_tuple(min, max);
  }
  return output;
} // namespace lpq

template <typename PRECISION_TYPE>
PRECISION_TYPE lpq::LowPrecisionQuantizer<PRECISION_TYPE>::affine_quantize(
    float value, float scale, PRECISION_TYPE zero_point) {
  const auto transformed_value = zero_point + std::round(value / scale);

  const auto clamped_value =
      std::max(-128.f, std::min(127.f, transformed_value));
  return static_cast<PRECISION_TYPE>(clamped_value);
}

template <typename PRECISION_TYPE>
PRECISION_TYPE lpq::LowPrecisionQuantizer<PRECISION_TYPE>::lpq_quantize(
    float value, float mean, float standard_deviation) {
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
template class LowPrecisionQuantizer<uint8_t>;

} // namespace lpq
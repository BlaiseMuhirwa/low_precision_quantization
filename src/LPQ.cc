#include "LPQ.h"
#include <cmath>
#include <math.h>
#include <numeric>
// #include <omp.h>

namespace lpq {


template <typename PRECISION_TYPE>
LowPrecisionQuantizer<PRECISION_TYPE>::LowPrecisionQuantizer(uint32_t bit_width)
    : _bit_width(bit_width) {}

template <typename PRECISION_TYPE>
std::vector<std::vector<PRECISION_TYPE>> LowPrecisionQuantizer<PRECISION_TYPE>::quantizeVectors(
    const std::vector<std::vector<float>> &vectors) {

  if (vectors.size() == 0) {
    return {};
  }
  uint32_t vector_size = vectors.size();
  std::vector<std::vector<PRECISION_TYPE>> quantized_vectors(vector_size);

#pragma omp parallel default none shared(vectors, quantized_vectors)
  for (const auto &vector : vectors) {
#pragma omp critical
    auto [mean, variance] = computeVectorWiseStatistics(vector);
    std::vector<PRECISION_TYPE> quantized_vector(vector.size());
    for (const auto &value : vector) {
      quantized_vector.push_back(quantize(value, mean, variance));
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

  auto mean = std::accumulate(vector.begin(), vector.end(), 0) / count;
  auto variance = 0;
  std::for_each(vector.begin(), vector.end(), [&variance, &mean](float item)
  {
    variance += (item - mean) * (item - mean);
  });
  variance = sqrt(variance);
  return std::make_tuple(mean, variance);
}

template <typename PRECISION_TYPE>
PRECISION_TYPE lpq::LowPrecisionQuantizer<PRECISION_TYPE>::quantize(float value, float mean,
                                                float standard_deviation) {
  auto sb = mean - standard_deviation;
  auto se = mean + standard_deviation;

  if (value < mean) {
    return -pow(2, _bit_width - 1);
  } else if (value > mean) {
    return pow(2, _bit_width - 1);
  }
  return floor(pow(2, _bit_width) * ((value - mean) / (se - sb)));
}

// Template specialization for int8 and int16 types
template class LowPrecisionQuantizer<int8_t>;
template class LowPrecisionQuantizer<int16_t>;

} // namespace lpq
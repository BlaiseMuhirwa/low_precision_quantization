#pragma once

#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace lpq {

template <typename PRECISION_TYPE>

class LowPrecisionQuantizer {
public:
  LowPrecisionQuantizer();

  std::vector<std::vector<PRECISION_TYPE>>
  quantizeVectors(const std::vector<std::vector<float>> &vectors);

  constexpr uint32_t getBitWidth() const { return _bit_width; }

private:
  /**
   * Returns a vector of tuples corresponding to the mean and the standard
   * deviation for every dimension.
   * For instance, if X has (nxd) dimensions, we return a vector of
   * size d, where every entry corresponds to the (mean, std) per
   * dimension
   **/
  std::vector<std::tuple<float, float>>
  getDatasetStatistics(const std::vector<std::vector<float>> &dataset);

  // Alternative quantization strategy
  PRECISION_TYPE quantize_simple(float value, float mean,
                                 float standard_deviation);
  uint32_t _bit_width;
};
} // namespace lpq

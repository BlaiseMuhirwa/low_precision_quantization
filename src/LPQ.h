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
  std::tuple<float, float>
  computeVectorWiseStatistics(const std::vector<float> &vector);
  PRECISION_TYPE quantize(float value, float mean, float standard_deviation);
  uint32_t _bit_width;
};
} // namespace lpq

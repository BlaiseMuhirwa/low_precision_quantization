#pragma once

#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include <cereal/access.hpp>

namespace lpq {
template <typename PRECISION_TYPE> 
class LowPrecisionQuantizer {
  public:
    explicit LowPrecisionQuantizer(uint32_t bit_width);

    std::vector<std::vector<PRECISION_TYPE>>
    quantizeVectors(const std::vector<std::vector<float>> &vectors);

  private:
    std::tuple<float, float>
    computeVectorWiseStatistics(const std::vector<float> &vector);
    PRECISION_TYPE quantize(float value, float mean, float standard_deviation);
    uint32_t _bit_width;
};
} // namespace lpq

#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace lpq {

class NaiveQuantizer {
public:
  NaiveQuantizer();

  std::vector<std::vector<int_least8_t>>
  quantizeVectors(const std::vector<std::vector<float>> &vectors);
};
} // namespace lpq

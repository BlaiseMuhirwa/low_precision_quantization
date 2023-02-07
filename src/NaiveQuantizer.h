#pragma once

#include <vector>

namespace lpq {

template <typename PRECISION_TYPE> class NaiveQuantizer {
public:
  LowPrecisionQuantizer() : _bit_width(8 * sizeof(PRECISION_TYPE)) {}

  std::vector<std::vector<PRECISION_TYPE>>
  quantizeVectors(const std::vector<std::vector<float>> &vectors);

  constexpr uint32_t getBitWidth() const { return _bit_width; }

private:
  uint32_t _bit_width;
};
} // namespace lpq

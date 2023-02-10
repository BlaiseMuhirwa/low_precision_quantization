
#include "NaiveQuantizer.h"
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace lpq {

NaiveQuantizer::NaiveQuantizer() {}

std::vector<std::vector<int_least8_t>> NaiveQuantizer::quantizeVectors(
    const std::vector<std::vector<float>> &vectors) {
  if (vectors.size() == 0) {
    return {};
  }
  std::vector<std::vector<int_least8_t>> output;

  for (const auto &vector : vectors) {
    std::vector<int_least8_t> current_vector;
    for (const auto &val : vector) {
      int_least8_t quantized_value = static_cast<int_least8_t>(std::round(val));
      current_vector.push_back(quantized_value);
    }
    output.emplace_back(std::move(current_vector));
  }
  return output;
}

} // namespace lpq

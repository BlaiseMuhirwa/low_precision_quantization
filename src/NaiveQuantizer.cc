
#include "NaiveQuantizer.h"
#include <vector>

namespace lpq {

template <typename PRECISION_TYPE>
std::vector<std::vector<PRECISION_TYPE>>
NaiveQuantizer<PRECISION_TYPE>::quantizeVectors(
    const std::vector<std::vector<float>> &vectors) {
  if (vectors.size() == 0) {
    return {};
  }
  std::vector<std::vector<PRECISION_TYPE>> output;

  for (const auto &vector : vectors) {
    std::vector<PRECISION_TYPE> current_vector;
    for (const auto &val : vector) {
      PRECISION_TYPE quantized_value = val;
      current_vector.push_back(quantized_value);
    }
    output.emplace_back(std::move(current_vector));
  }
  return output;
}

template class NaiveQuantizer<int_least8_t>;
template class NaiveQuantizer<int_least16_t>;

} // namespace lpq

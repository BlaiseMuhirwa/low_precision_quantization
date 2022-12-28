#include <numeric>
#include <vector>
#include <tuple>

namespace lpq {
template <typename TYPE> 
class LowPrecisionQuantizer {

  explicit LowPrecisionQuantizer(uint32_t bit_width);

  std::vector<std::vector<TYPE> > quantizeVectors(const std::vector<std::vector<float>> &vectors);

private:
  std::tuple<float, float>
  computeVectorWiseStatistics(const std::vector<float> &vector);
  TYPE quantize(float value, float mean, float standard_deviation);
  uint32_t _bit_width;
};
} // namespace lpq

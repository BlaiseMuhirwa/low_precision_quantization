#pragma once

#include <cstdint>
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
   * A low-precision quantization sub-routine as defined in the original
   * paper: https://arxiv.org/pdf/2110.08919.pdf
   **/

  PRECISION_TYPE
  lpq_quantize(float value, float mean, float standard_deviation);

  /**
   * Returns a vector of tuples corresponding to the mean and the standard
   * deviation for every dimension.
   * For instance, if X has (nxd) dimensions, we return a vector of
   * size d, where every entry corresponds to the (mean, std) per
   * dimension. This is needed for the LPQ quantizer.
   **/
  std::vector<std::tuple<float, float>>
  getDatasetStatistics(const std::vector<std::vector<float>> &dataset);

  /**
   * Returns the max and min values which define a range
   * needed for Affine Quantization.
   **/
  std::vector<std::tuple<float, float>>
  getMinMaxValues(const std::vector<std::vector<float>> &dataset);

  /**
   * Returns the quantization parameters for Affine Quantization.
   * The two parameters are: scale and the zero point.
   **/
  std::tuple<float, PRECISION_TYPE> getQuantizationParams(float min, float max);

  /**
   * Alternative quantization strategy. This quantization is based on
   * an affine transformation of the original input value.
   * For more details, check out how it's derived here:
   * https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/docs/intro_to_quantization.html
   **/

  PRECISION_TYPE affine_quantize(float value, float scale,
                                 PRECISION_TYPE zero_point);
  uint32_t _bit_width;
};
} // namespace lpq

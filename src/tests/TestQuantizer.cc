
#include "../LPQ.h"
#include <algorithm>
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

using lpq::LowPrecisionQuantizer;

constexpr uint32_t NUM_VECTORS = 10;
constexpr uint32_t VECTOR_DIMENSION = 50;
constexpr float GAUSSIAN_DIST_MEAN = 20.0;
constexpr float GAUSSIAN_DIST_STDEV = 3.0;

// It turns out you can't compute the max using
// std::numeric_limist<int8_t>::max()
constexpr auto MAX_INT8 = (1 << 7) - 1;
constexpr auto MIN_INT8 = -(1 << 7);

/**
 * Constructs some test vectors where each vector
 * entry is drawn from a gaussian distribution with
 * the predefined parameters.
 */
std::vector<std::vector<float>> getTestingVectors() {

  std::default_random_engine generator;
  std::normal_distribution<float> distribution(GAUSSIAN_DIST_MEAN,
                                               GAUSSIAN_DIST_STDEV);

  std::vector<std::vector<float>> output;
  output.reserve(NUM_VECTORS);

  for (uint32_t index = 0; index < NUM_VECTORS; index++) {
    std::vector<float> current_vector;
    current_vector.reserve(VECTOR_DIMENSION);
    for (uint32_t i = 0; i < VECTOR_DIMENSION; i++) {
      float random_number = distribution(generator);

      current_vector.push_back(random_number);
    }
    output.push_back(std::move(current_vector));
  }
  return output;
}

/**
 * Computes the statistics for each vector in the input
 * The statistics include just the mean and the standard
 * deviation
 */
std::tuple<float, float> getMeanAndStDev(const std::vector<float> &vector) {
  if (vector.size() == 0) {
    return {};
  }
  auto const count = static_cast<float>(vector.size());

  float mean = std::accumulate(vector.begin(), vector.end(), 0.000000) / count;
  float variance = 0.000000;
  std::for_each(vector.begin(), vector.end(), [&variance, &mean](float item) {
    variance += (item - mean) * (item - mean);
  });
  variance /= (vector.size() - 1);
  float stddev = sqrt(variance);
  return std::make_tuple(mean, stddev);
}

TEST(LPQTest, TestCorrectQuantizedValuesInt8) {
  auto testing_vectors = getTestingVectors();

  std::unique_ptr<LowPrecisionQuantizer<int8_t>> quantizer =
      std::make_unique<LowPrecisionQuantizer<int8_t>>();

  for (const auto &vector : testing_vectors) {
    auto quantized_vector = quantizer->quantizeVectors({vector})[0];

    auto [mean, stdev] = getMeanAndStDev(vector);
    auto sb = mean - stdev;
    auto se = mean + stdev;

    for (uint32_t slot_index = 0; slot_index < VECTOR_DIMENSION; slot_index++) {
      float current_val = vector[slot_index];
      int8_t quantization = quantized_vector[slot_index];

      // Initialize value for 2^(B-1) - 1. This is the quantization for
      // when current_val < (mean - stdev)
      auto expected_quantization_value = -((1 << 7) - 1);
      if (current_val < sb) {
        continue;
      } else if (current_val > se) {
        expected_quantization_value = -expected_quantization_value;
      } else {
        auto temp_val = floor((1 << 7) * ((current_val - mean) / stdev));

        expected_quantization_value = temp_val;
        expected_quantization_value = temp_val > MAX_INT8 ? MAX_INT8 : temp_val;
        expected_quantization_value = temp_val < MIN_INT8 ? MIN_INT8 : temp_val;
      }

      ASSERT_EQ(expected_quantization_value, quantization);
    }
  }
}

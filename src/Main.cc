#include "LPQ.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using lpq::LowPrecisionQuantizer;

std::vector<std::vector<float>> getTestingVectors() {
  uint32_t num_vectors = 10000;
  uint32_t vector_size = 50000;

  std::default_random_engine generator;
  std::normal_distribution<float> distribution(20.0, 3.0);

  std::vector<std::vector<float>> output;
  output.reserve(num_vectors);

  for (uint32_t index = 0; index < num_vectors; index++) {
    std::vector<float> current_vector;
    current_vector.reserve(vector_size);
    for (uint32_t i = 0; i < vector_size; i++) {
      float random_number = distribution(generator);

      current_vector.push_back(random_number);
    }
    output.push_back(std::move(current_vector));
  }
  return output;
}

int main(int argc, char **argv) {
  auto testing_vectors = getTestingVectors();

  std::unique_ptr<LowPrecisionQuantizer<int8_t>> quantizer =
      std::make_unique<LowPrecisionQuantizer<int8_t>>();

  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  auto quantized_vectors = quantizer->quantizeVectors(testing_vectors);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> latency = end - start;
  std::cout << "[LATENCY] = " << latency.count() << " seconds" << std::endl;

  return 1;
}
#include "LPQ.h"
#include <iostream>

using lpq::LowPrecisionQuantizer;

int main(int argc, char **argv) {
  auto quantizer = LowPrecisionQuantizer<int8_t>(/* bit_width =*/8);
  return 1;
}
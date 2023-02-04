#include "ExactSearch.h"
#include <memory.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <src/LPQ.h>

namespace lpq::python {

namespace py = pybind11;

using lpq::LowPrecisionQuantizer;
using lpq::index::ExactSearchIndex;

void defineIndexSubmodule(py::module_ &index_submodule) {
  py::class_<ExactSearchIndex<int_least8_t>,
             std::shared_ptr<ExactSearchIndex<int_least8_t>>>(
      index_submodule, "ExactSearchIndex");
}

void defineQuantizationSubmodule(py::module_ &quantization_submodule) {
  // TODO: Refactor the bindigs using a templated function to avoid
  //  having to repeat the exact same code twice.
  py::class_<LowPrecisionQuantizer<int_least8_t>,
             std::shared_ptr<LowPrecisionQuantizer<int_least8_t>>>(
      quantization_submodule, "LowPrecisionQuantizerInt8")
      .def(py::init<>(), "Initializes a low-precision quantizer (int8) object.")
      .def("quantize_vectors",
           &LowPrecisionQuantizer<int_least8_t>::quantizeVectors,
           py::arg("vectors"),
           "Quantizes input vectors based on the low precision quantization "
           "rule.")

      .def_property_readonly("bit_width",
                             &LowPrecisionQuantizer<int_least8_t>::getBitWidth,
                             "Gets the bit width used by the quantizer");

  py::class_<LowPrecisionQuantizer<int_least16_t>,
             std::shared_ptr<LowPrecisionQuantizer<int_least16_t>>>(
      quantization_submodule, "LowPrecisionQuantizerInt16")
      .def(py::init<>(),
           "Initializes a low-precision quantizer (int16) object.")
      .def("quantize_vectors",
           &LowPrecisionQuantizer<int_least16_t>::quantizeVectors,
           py::arg("vectors"),
           "Quantizes input vectors based on the low precision quantization "
           "rule.")

      .def_property_readonly("bit_width",
                             &LowPrecisionQuantizer<int_least16_t>::getBitWidth,
                             "Gets the bit width used by the quantizer");
}

PYBIND11_MODULE(lpq, module) {

  auto index_submodule = module.def_submodule("index");
  auto quantization_submodule = module.def_submodule("quantization");

  defineQuantizationSubmodule(quantization_submodule);
  defineIndexSubmodule(index_submodule);
}

} // namespace lpq::python
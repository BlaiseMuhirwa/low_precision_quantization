#include <memory.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <src/LPQ.h>

namespace lpq::python {

namespace py = pybind11;

using lpq::LowPrecisionQuantizer;

PYBIND11_MODULE(lpq, module) {

  // TODO: Refactor the bindigs using a templated function to avoid
  //  having to repeat the exact same code twice.

  py::class_<lpq::LowPrecisionQuantizer<int_least8_t>,
             std::shared_ptr<lpq::LowPrecisionQuantizer<int_least8_t>>>(
      module, "LowPrecisionQuantizerInt8")
      .def(py::init<>(), "Initializes a low-precision quantizer (int8) object.")
      .def("quantize_vectors",
           &LowPrecisionQuantizer<int_least8_t>::quantizeVectors,
           py::arg("vectors"),
           "Quantizes input vectors based on the low precision quantization "
           "rule.")

      .def_property_readonly("bit_width",
                             &LowPrecisionQuantizer<int_least8_t>::getBitWidth,
                             "Gets the bit width used by the quantizer");

  py::class_<lpq::LowPrecisionQuantizer<int_least16_t>,
             std::shared_ptr<lpq::LowPrecisionQuantizer<int_least16_t>>>(
      module, "LowPrecisionQuantizerInt16")
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

} // namespace lpq::python
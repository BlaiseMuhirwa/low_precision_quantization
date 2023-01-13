#include <memory.h>
#include <pybind11/pybind11.h>
#include <src/LPQ.h>

namespace lpq::python {

namespace py = pybind11;

using lpq::LowPrecisionQuantizer;

PYBIND11_MODULE(low_precision_quantizer, m) {
  py::class_<LowPrecisionQuantizer<int8_t>,
             std::shared_ptr<LowPrecisionQuantizer<int8_t>>>(
      m, "LowPrecisionQuantizer")
      .def(py::init<>())
      .def("quantize_vectors", &LowPrecisionQuantizer<int8_t>::quantizeVectors,
           py::arg("vectors"))
      .def_property_readonly("bit_width",
                             &LowPrecisionQuantizer<int8_t>::getBitWidth);
}

} // namespace lpq::python
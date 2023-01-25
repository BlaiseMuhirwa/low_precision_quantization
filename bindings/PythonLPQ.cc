#include <memory.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <src/LPQ.h>

namespace lpq::python {

namespace py = pybind11;

using lpq::LowPrecisionQuantizer;
using InputVectors = std::vector<std::vector<float>>;

PYBIND11_MODULE(lpq, m) {
  py::class_<lpq::LowPrecisionQuantizer<int8_t>,
             std::shared_ptr<lpq::LowPrecisionQuantizer<int8_t>>>(
      m, "LowPrecisionQuantizer")
      .def(py::init<>())
      .def("quantize_vectors", &LowPrecisionQuantizer<int8_t>::quantizeVectors,
           py::arg("vectors"))
      .def(
          "quantize_numpy",
          [](LowPrecisionQuantizer<int8_t> &self, InputVectors &numpy_vectors) {
            std::vector<std::vector<int8_t>> quantized_vectors =
                self.quantizeVectors(numpy_vectors);
            py::object output = py::cast(quantized_vectors);
            return output;
          })
      .def_property_readonly("bit_width",
                             &LowPrecisionQuantizer<int8_t>::getBitWidth);
}

} // namespace lpq::python
#include "matrix.h"
#include "neural_net.cpp" // Include cpp directly or header if separated
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector conversion

namespace py = pybind11;

PYBIND11_MODULE(nn_engine, m) {
  m.doc() = "Neural Network Engine built with C++ and AVX2";

  py::class_<Matrix>(m, "Matrix")
      .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
      .def(py::init<int, int, std::vector<float>>(), py::arg("rows"),
           py::arg("cols"), py::arg("data"))
      .def("print", &Matrix::print)
      .def("get_rows", &Matrix::get_rows)
      .def("get_columns", &Matrix::get_columns);

  py::class_<NeuralNet>(m, "NeuralNet")
      .def(py::init<std::vector<int>>(), py::arg("topology"))
      .def("forward", &NeuralNet::forward, py::arg("input"));
}

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using TensorShape = std::vector<py::ssize_t>;
using TensorStrides = std::vector<py::ssize_t>;
using TensorSize = std::vector<py::ssize_t>;

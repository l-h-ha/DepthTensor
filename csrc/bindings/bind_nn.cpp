#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nn/activations.h"

namespace py = pybind11;
using namespace depthtensor::nn;

template <typename T>
void py_relu_forward_cpu(py::array_t<T> tensor) {
    constexpr int contiguous_flags = py::array::c_style | py::array::f_style;
    if (!tensor.writeable() || !(tensor.flags() & contiguous_flags)) {
        throw std::runtime_error("Tensor must be writable and contiguous.");
    }

    py::buffer_info buf = tensor.request();
    T *ptr = static_cast<T *>(buf.ptr);

    relu_forward_cpu_impl(ptr, buf.size);
}

void init_nn(py::module_ &m) {
    py::module_ nn_m = m.def_submodule("nn", "Neural network ops.");

    nn_m.def("relu_forward_cpu", &py_relu_forward_cpu<float>, py::arg("tensor").noconvert(),
             "ReLU for float32");
    nn_m.def("relu_forward_cpu", &py_relu_forward_cpu<double>, py::arg("tensor").noconvert(),
             "ReLU for float64");
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "nn/activations.h"

namespace py = pybind11;
using namespace depthtensor::nn;

template <typename T>
void py_relu_forward_cpu(py::array_t<T> tensor)
{
    if (!tensor.writeable() || !(tensor.flags() & py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_))
    {
        throw std::runtime_error("Tensor memory must be writable and C-contiguous");
    }

    py::buffer_info buf = tensor.request();
    T *ptr = static_cast<T *>(buf.ptr);

    relu_forward_cpu_impl(ptr, buf.size);
}

void init_nn(py::module_ &m)
{
    // submodule for nn: depthtensor._ext.nn
    py::module_ nn_m = m.def_submodule("nn", "Neural Network operations");

    nn_m.def("relu_forward_cpu", &py_relu_forward_cpu<float>, "ReLU for float32");
    nn_m.def("relu_forward_cpu", &py_relu_forward_cpu<double>, "ReLU for float64");
}
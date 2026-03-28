#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_nn(py::module_ &m);

PYBIND11_MODULE(_ext, m)
{
    m.doc() = "DepthTensor C++ Backend Engine";

    init_nn(m);
}
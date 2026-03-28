#pragma once
#include <cstddef>

namespace depthtensor
{
    namespace nn
    {
        template <typename T>
        void relu_forward_cpu_impl(T *data, size_t size);
    }
}
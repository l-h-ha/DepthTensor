#include "nn/activations.h"

namespace depthtensor
{
    namespace nn
    {
        template <typename T>
        void relu_forward_cpu_impl(T *data, size_t size)
        {
            T zero = static_cast<T>(0);
            for (size_t i = 0; i < size; i++)
            {
                if (data[i] < zero)
                {
                    data[i] = zero;
                }
            }
        }

        template void relu_forward_cpu_impl<float>(float *data, size_t size);
        template void relu_forward_cpu_impl<double>(double *data, size_t size);
        template void relu_forward_cpu_impl<int>(int *data, size_t size);
    }
}
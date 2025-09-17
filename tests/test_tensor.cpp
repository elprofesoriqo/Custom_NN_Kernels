#include <cassert>
#include <iostream>
#include <vector>

#include "core/Tensor.h"

using namespace nnk;

int main() {
    {
        auto t = Tensor::zeros({2, 3}, DeviceType::CPU);
        assert(t.device() == DeviceType::CPU);
        assert(t.shape().size() == 2);
        assert(t.shape()[0] == 2 && t.shape()[1] == 3);
        assert(t.numel() == 6);
        float* ptr = t.data();
        for (size_t i = 0; i < t.numel(); ++i) {
            assert(ptr[i] == 0.0f);
        }
    }

#if TT_WITH_CUDA
    {
        std::vector<float> v = {1,2,3,4};
        auto t = Tensor::from_vector({2,2}, v, DeviceType::CUDA);
        auto back = t.to_host();
        assert(back.device() == DeviceType::CPU);
        assert(back.numel() == 4);
        const float* p = back.data();
        for (int i = 0; i < 4; ++i) {
            assert(p[i] == v[i]);
        }
    }
#endif

    std::cout << "test_tensor OK\n";
    return 0;
}



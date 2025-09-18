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

    {
        auto t = Tensor::from_vector({2, 2}, {1.0f,2.0f,3.0f,4.0f}, DeviceType::CPU);
        // Copy constructor
        Tensor cpy = t;
        assert(cpy.numel() == t.numel());
        cpy.data()[0] = 10.0f;
        // Original must remain unchanged
        assert(t.data()[0] == 1.0f);

        // Move constructor
        Tensor mv = std::move(cpy);
        assert(mv.numel() == 4);
        // Moved-from should be in a valid but empty state
        assert(cpy.numel() == 0);

        // to_device to same CPU should deep-copy
        Tensor t2 = t.to_device(DeviceType::CPU);
        assert(t2.numel() == t.numel());
        t2.data()[1] = 20.0f;
        assert(t.data()[1] == 2.0f);
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



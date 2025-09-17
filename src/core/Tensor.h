#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Device.h"

namespace nnk {

class Tensor {
public:
    Tensor();

    static Tensor zeros(const std::vector<int64_t>& shape, DeviceType device = DeviceType::CPU);
    static Tensor from_vector(const std::vector<int64_t>& shape, const std::vector<float>& data, DeviceType device = DeviceType::CPU);

    size_t numel() const noexcept { return num_elements_; }
    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    DeviceType device() const noexcept { return device_type_; }

    // CPU data accessors. Throws if tensor is not on CPU.
    float* data();
    const float* data() const;

    // Copy tensor to target device and return a new Tensor.
    Tensor to_device(DeviceType target) const;
    Tensor to_host() const { return to_device(DeviceType::CPU); }

private:
    static size_t compute_numel(const std::vector<int64_t>& dims);
    static void validate_shape(const std::vector<int64_t>& dims);

    std::vector<int64_t> shape_{};
    size_t num_elements_ = 0;
    DeviceType device_type_ = DeviceType::CPU;

    // Host storage
    std::vector<float> host_storage_{};

#if TT_WITH_CUDA
    // Device storage (CUDA)
    float* device_storage_ = nullptr;
#endif
};

}



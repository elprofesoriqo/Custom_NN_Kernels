#include "core/Tensor.h"

#include <algorithm>
#include <numeric>

#if TT_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace nnk {

Tensor::Tensor() = default;

Tensor::~Tensor() {
#if TT_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && device_storage_ != nullptr) {
        cudaFree(device_storage_);
        device_storage_ = nullptr;
    }
#endif
}

Tensor::Tensor(const Tensor& other) {
    shape_ = other.shape_;
    num_elements_ = other.num_elements_;
    device_type_ = other.device_type_;
    if (device_type_ == DeviceType::CPU) {
        host_storage_ = other.host_storage_;
    }
#if TT_WITH_CUDA
    if (device_type_ == DeviceType::CUDA) {
        if (num_elements_ == 0) {
            device_storage_ = nullptr;
        } else {
            cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&device_storage_), num_elements_ * sizeof(float));
            if (st != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed in Tensor copy ctor");
            }
            st = cudaMemcpy(device_storage_, other.device_storage_, num_elements_ * sizeof(float), cudaMemcpyDeviceToDevice);
            if (st != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy D2D failed in Tensor copy ctor");
            }
        }
    }
#endif
}

Tensor::Tensor(Tensor&& other) noexcept {
    shape_ = std::move(other.shape_);
    num_elements_ = other.num_elements_;
    device_type_ = other.device_type_;
    host_storage_ = std::move(other.host_storage_);
#if TT_WITH_CUDA
    device_storage_ = other.device_storage_;
    other.device_storage_ = nullptr;
#endif
    other.num_elements_ = 0;
    other.device_type_ = DeviceType::CPU;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
#if TT_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && device_storage_ != nullptr) {
        cudaFree(device_storage_);
        device_storage_ = nullptr;
    }
#endif
    shape_ = other.shape_;
    num_elements_ = other.num_elements_;
    device_type_ = other.device_type_;
    host_storage_.clear();
    host_storage_ = other.host_storage_;
#if TT_WITH_CUDA
    if (device_type_ == DeviceType::CUDA) {
        if (num_elements_ == 0) {
            device_storage_ = nullptr;
        } else {
            cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&device_storage_), num_elements_ * sizeof(float));
            if (st != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed in Tensor copy assign");
            }
            st = cudaMemcpy(device_storage_, other.device_storage_, num_elements_ * sizeof(float), cudaMemcpyDeviceToDevice);
            if (st != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy D2D failed in Tensor copy assign");
            }
        }
    }
#endif
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
#if TT_WITH_CUDA
    if (device_type_ == DeviceType::CUDA && device_storage_ != nullptr) {
        cudaFree(device_storage_);
        device_storage_ = nullptr;
    }
#endif
    shape_ = std::move(other.shape_);
    num_elements_ = other.num_elements_;
    device_type_ = other.device_type_;
    host_storage_ = std::move(other.host_storage_);
#if TT_WITH_CUDA
    device_storage_ = other.device_storage_;
    other.device_storage_ = nullptr;
#endif
    other.num_elements_ = 0;
    other.device_type_ = DeviceType::CPU;
    return *this;
}

size_t Tensor::compute_numel(const std::vector<int64_t>& dims) {
    if (dims.empty()) {
        return 0;
    }
    size_t total = 1;
    for (int64_t d : dims) {
        if (d < 0) {
            throw std::invalid_argument("Tensor shape cannot contain negative dimensions");
        }
        total *= static_cast<size_t>(d);
    }
    return total;
}

void Tensor::validate_shape(const std::vector<int64_t>& dims) {
    for (int64_t d : dims) {
        if (d < 0) {
            throw std::invalid_argument("Invalid shape: negative dimension");
        }
    }
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DeviceType device) {
    validate_shape(shape);
    Tensor t;
    t.shape_ = shape;
    t.num_elements_ = compute_numel(shape);
    t.device_type_ = device;

    if (device == DeviceType::CPU) {
        t.host_storage_.assign(t.num_elements_, 0.0f);
        return t;
    }

#if TT_WITH_CUDA
    if (device == DeviceType::CUDA) {
        t.host_storage_.clear();
        t.host_storage_.shrink_to_fit();
        cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&t.device_storage_), t.num_elements_ * sizeof(float));
        if (st != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed in Tensor::zeros");
        }
        st = cudaMemset(t.device_storage_, 0, t.num_elements_ * sizeof(float));
        if (st != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed in Tensor::zeros");
        }
        return t;
    }
#else
    (void)t; // suppress unused warning if CUDA disabled
#endif

    throw std::invalid_argument("Unsupported device type in Tensor::zeros");
}

Tensor Tensor::from_vector(const std::vector<int64_t>& shape, const std::vector<float>& data, DeviceType device) {
    if (compute_numel(shape) != data.size()) {
        throw std::invalid_argument("from_vector: data size does not match shape");
    }
    if (device == DeviceType::CPU) {
        Tensor t;
        t.shape_ = shape;
        t.num_elements_ = data.size();
        t.device_type_ = DeviceType::CPU;
        t.host_storage_ = data;
        return t;
    }

#if TT_WITH_CUDA
    if (device == DeviceType::CUDA) {
        Tensor t;
        t.shape_ = shape;
        t.num_elements_ = data.size();
        t.device_type_ = DeviceType::CUDA;
        cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&t.device_storage_), t.num_elements_ * sizeof(float));
        if (st != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed in from_vector");
        }
        st = cudaMemcpy(t.device_storage_, data.data(), t.num_elements_ * sizeof(float), cudaMemcpyHostToDevice);
        if (st != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy H2D failed in from_vector");
        }
        return t;
    }
#endif

    throw std::invalid_argument("Unsupported device type in Tensor::from_vector");
}

float* Tensor::data() {
    if (device_type_ != DeviceType::CPU) {
        throw std::runtime_error("Tensor::data() is only available for CPU tensors");
    }
    return host_storage_.data();
}

const float* Tensor::data() const {
    if (device_type_ != DeviceType::CPU) {
        throw std::runtime_error("Tensor::data() const is only available for CPU tensors");
    }
    return host_storage_.data();
}

Tensor Tensor::to_device(DeviceType target) const {
    if (target == device_type_) {
        // Shallow copy on same device
        if (device_type_ == DeviceType::CPU) {
            return Tensor::from_vector(shape_, host_storage_, DeviceType::CPU);
        }
#if TT_WITH_CUDA
        if (device_type_ == DeviceType::CUDA) {
            // Copy device memory to new tensor on device
            Tensor t;
            t.shape_ = shape_;
            t.num_elements_ = num_elements_;
            t.device_type_ = DeviceType::CUDA;
            cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&t.device_storage_), t.num_elements_ * sizeof(float));
            if (st != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed in to_device (same device)");
            }
            st = cudaMemcpy(t.device_storage_, device_storage_, t.num_elements_ * sizeof(float), cudaMemcpyDeviceToDevice);
            if (st != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy D2D failed in to_device (same device)");
            }
            return t;
        }
#endif
    }

    if (target == DeviceType::CPU) {
        // target is CPU
        if (device_type_ == DeviceType::CPU) {
            return Tensor::from_vector(shape_, host_storage_, DeviceType::CPU);
        }
#if TT_WITH_CUDA
        // CUDA -> CPU
        std::vector<float> host(num_elements_);
        cudaError_t st = cudaMemcpy(host.data(), device_storage_, num_elements_ * sizeof(float), cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D2H failed in to_device");
        }
        return Tensor::from_vector(shape_, host, DeviceType::CPU);
#else
        throw std::runtime_error("CUDA support disabled: cannot move tensor from device to host");
#endif
    }

    if (target == DeviceType::CUDA) {
#if TT_WITH_CUDA
        if (device_type_ == DeviceType::CPU) {
            // CPU -> CUDA
            return Tensor::from_vector(shape_, host_storage_, DeviceType::CUDA);
        }
        // CUDA -> CUDA handled above
        throw std::runtime_error("Unexpected path in to_device CUDA");
#else
        throw std::runtime_error("CUDA support disabled: cannot move tensor to CUDA");
#endif
    }

    throw std::invalid_argument("Unsupported target device in to_device");
}

}


